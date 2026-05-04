import jax.numpy as jnp
import jax
import equinox as eqx
import optax

import argparse

import scipy.special

from functools import partial
from frozendict import frozendict

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=0, type=int)

parser.add_argument("--n", default=8, type=int)

parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--random_trajlen", default=100, type=int)

parser.add_argument("--train_steps", default=50000, type=int)

parser.add_argument("--reg_coef", default=0.0001, type=float)

parser.add_argument("--hidden_size", default=64, type=int)

parser.add_argument("--eval_batch_size", default=256, type=int)
parser.add_argument("--eval_max_length", default=600, type=int)

parser.add_argument("--print_every", default=1000, type=int)

parser.add_argument("--lr", default=3e-4, type=float)

parser.add_argument("--eps_explore", default=0.0, type=float)

parser.add_argument("--topk", default=1, type=int)
parser.add_argument("--beam_k", default=1, type=int)

parser.add_argument("--use_test_set", default=1, type=int)
parser.add_argument("--test_set_path", default="rubik2_test.npy", type=str)

parser.add_argument("--task", default="rubik2", type=str)

parser.add_argument("--save_path", default="", type=str)
parser.add_argument("--load_path", default="", type=str)

parser.add_argument("--clip_grad_norm", default=100.0, type=float)


class ResMLPPolicy(eqx.Module):
    inp: eqx.nn.Linear
    
    fc1: eqx.nn.Linear
    ln1: eqx.nn.LayerNorm
    fc2: eqx.nn.Linear
    ln2: eqx.nn.LayerNorm
    
    fc3: eqx.nn.Linear
    ln3: eqx.nn.LayerNorm
    fc4: eqx.nn.Linear
    ln4: eqx.nn.LayerNorm

    fc5: eqx.nn.Linear
    ln5: eqx.nn.LayerNorm
    fc6: eqx.nn.Linear
    ln6: eqx.nn.LayerNorm
    
    outp: eqx.nn.Linear
    n_bwd_actions: int
    n_fwd_actions: int

    task: str

    def __init__(
        self,
        n,
        n_bwd_actions,
        n_fwd_actions,
        hidden_size,
        rng_key,
        task
    ):
        self.n_bwd_actions = n_bwd_actions
        self.n_fwd_actions = n_fwd_actions
        self.task = task

        rng_key, inp_key = jax.random.split(rng_key)
        if self.task == "rubik2" or self.task == "rubik3":
            self.inp = eqx.nn.Linear(in_features=n*6, out_features=hidden_size, key=inp_key)
        elif self.task == "swap":
            self.inp = eqx.nn.Linear(in_features=n*n, out_features=hidden_size, key=inp_key)

        rng_key, fc1_key = jax.random.split(rng_key)
        self.fc1 = eqx.nn.Linear(in_features=hidden_size, out_features=hidden_size, key=fc1_key)
        self.ln1 = eqx.nn.LayerNorm(shape=(hidden_size,))

        rng_key, fc2_key = jax.random.split(rng_key)
        self.fc2 = eqx.nn.Linear(in_features=hidden_size, out_features=hidden_size, key=fc2_key)
        self.ln2 = eqx.nn.LayerNorm(shape=(hidden_size,))

        rng_key, fc3_key = jax.random.split(rng_key)
        self.fc3 = eqx.nn.Linear(in_features=hidden_size, out_features=hidden_size, key=fc3_key)
        self.ln3 = eqx.nn.LayerNorm(shape=(hidden_size,))

        rng_key, fc4_key = jax.random.split(rng_key)
        self.fc4 = eqx.nn.Linear(in_features=hidden_size, out_features=hidden_size, key=fc4_key)
        self.ln4 = eqx.nn.LayerNorm(shape=(hidden_size,))

        rng_key, fc5_key = jax.random.split(rng_key)
        self.fc5 = eqx.nn.Linear(in_features=hidden_size, out_features=hidden_size, key=fc5_key)
        self.ln5 = eqx.nn.LayerNorm(shape=(hidden_size,))

        rng_key, fc6_key = jax.random.split(rng_key)
        self.fc6 = eqx.nn.Linear(in_features=hidden_size, out_features=hidden_size, key=fc6_key)
        self.ln6 = eqx.nn.LayerNorm(shape=(hidden_size,))

        rng_key, outp_key = jax.random.split(rng_key)
        self.outp = eqx.nn.Linear(in_features=hidden_size, out_features=n_bwd_actions + n_fwd_actions, key=outp_key)

    def __call__(self, x):
        if self.task == "rubik2" or self.task == "rubik3":
            x0 = self.inp(jax.nn.one_hot(x, num_classes=6).flatten())
        elif self.task == "swap":
            x0 = self.inp(jax.nn.one_hot(x , num_classes=x.shape[0]).flatten())

        act = jax.nn.relu

        x1 = self.ln1(x0)
        x1 = self.fc1(x1)
        x1 = act(x1 + x0)
        x2 = self.ln2(x1)
        x2 = self.fc2(x2)
        x2 = act(x2 + x1)

        x3 = self.ln3(x2)
        x3 = self.fc3(x3)
        x3 = act(x3 + x2)
        x4 = self.ln4(x3)
        x4 = self.fc4(x4)
        x4 = act(x4 + x3)

        x5 = self.ln5(x4)
        x5 = self.fc5(x5)
        x5 = act(x5 + x4)
        x6 = self.ln6(x5)
        x6 = self.fc6(x6)
        x6 = act(x6 + x5)

        result = self.outp(x6 + x0)

        backward_logits, forward_logits = jnp.split(result, [self.n_bwd_actions], axis=-1)
        return {
            "backward_logits": backward_logits,
            "forward_logits": forward_logits,
        }


@jax.jit
def apply_action(states, actions, action_perms):
    return jax.vmap(lambda x, y: x[y])(states, jnp.take(action_perms, actions, axis=0))


@eqx.filter_jit
def mask_forward_logits(state, forward_logits, bwd_action_perms, fwd_action_perms, params):
    div = 1 if params["task"] == "swap" else (4 if params["task"] == "rubik2" else 9)
    mask = (state[:, None, :] == (bwd_action_perms[None, :, :] // div)).all(axis=-1)
    mask = jnp.concatenate([mask, jnp.zeros((state.shape[0], 1), dtype=bool)], axis=1)
    return mask


@eqx.filter_jit
def sample_forward_trajectories(key, model, bwd_action_perms, fwd_action_perms, init_state, params):
    def sample_step(val, x):
        step_key, state, prev_action = val
        
        logits = jax.vmap(model)(state)['forward_logits']
        
        step_key, explore_key = jax.random.split(step_key)
        do_explore = jax.random.bernoulli(explore_key, params["eps_explore"], shape=(state.shape[0],))
        logits = jnp.where(do_explore[..., jnp.newaxis], 0, logits)
        
        mask = mask_forward_logits(state, logits, bwd_action_perms, fwd_action_perms, params)
        logits = jnp.where(mask, -float("inf"), logits)
        logits = logits[:, :-1]
        
        step_key, action_key = jax.random.split(step_key)
        action = jax.random.categorical(action_key, logits)
        
        prev_state = apply_action(state, action, fwd_action_perms)
        
        return (step_key, prev_state, action), {"states": prev_state, "actions": action}
    
    trajectories_batch = jax.lax.scan(
        f=sample_step,
        init=(key, init_state, jnp.zeros((init_state.shape[0],), dtype=jnp.int32) - 1),
        xs=None,
        length=params['random_trajlen']
    )[1]

    states, actions = trajectories_batch["states"], trajectories_batch["actions"]
    return {
        "states": jax.numpy.concatenate([init_state[None, :, :], states], axis=0),
        "actions": jax.numpy.concatenate([actions, jnp.zeros((1, actions.shape[1]), dtype=jnp.int32)], axis=0)
    }
    

@eqx.filter_jit
def process_logits(state, logits, bwd_action_perms, fwd_action_perms, params):
    forward_logits = logits['forward_logits']
    backward_logits = logits['backward_logits']
    
    mask = mask_forward_logits(state, forward_logits, bwd_action_perms, fwd_action_perms, params)
    log_pfs = jax.nn.log_softmax(forward_logits, axis=-1, where=~mask) 
    
    log_pbs = jax.nn.log_softmax(backward_logits, axis=-1)
    
    log_flows = -log_pfs[:, -1]
    
    return log_pbs, log_pfs, log_flows


@eqx.filter_jit
def tb_loss(model, train_iter, trajectories_batch, bwd_action_perms, fwd_action_perms, params, debug_info=False):
    states, actions = trajectories_batch["states"], trajectories_batch["actions"]
    
    s0, s1, s2 = states.shape # s0 - trajlen, s1 - batch size, s2 - state size
    states = states.reshape((s0 * s1, s2))
    logits = jax.vmap(model)(states)
    log_pbs, log_pfs, log_flows = process_logits(states, logits, bwd_action_perms, fwd_action_perms, params)
    log_pbs, log_pfs, log_flows = log_pbs.reshape((s0, s1, -1)), log_pfs.reshape((s0, s1, -1)), log_flows.reshape((s0, s1))
    
    log_forward_policy = jnp.take_along_axis(log_pfs[:-1, :, :], actions[:-1, :, None], axis=-1).squeeze()
    log_backward_policy = jnp.take_along_axis(log_pbs[1:, :, :], actions[:-1, :, None], axis=-1).squeeze()

    log_forward_prefix = jnp.cumulative_sum(log_forward_policy, axis=0, include_initial=True)
    log_backward_prefix = jnp.cumulative_sum(log_backward_policy, axis=0, include_initial=True)

    reg_loss = params['reg_coef'] * jnp.exp(jax.nn.logsumexp((log_flows)[1:, :], axis=0)).mean()

    step_losses = (params['true_log_z'] + log_forward_prefix - log_backward_prefix - log_flows) ** 2
    tb_loss = step_losses.mean()

    return tb_loss + reg_loss


@eqx.filter_jit
def goal_state(params):
    if params['task'] == "rubik2":
        return jnp.arange(params['n']) // 4
    elif params['task'] == "rubik3":
        return jnp.arange(params['n']) // 9
    elif params['task'] == "swap":
        return jnp.arange(params['n'])
        

@eqx.filter_jit
def train_step(train_iter, key, bwd_action_perms, fwd_action_perms, model, opt_state, params):
    init_state = jnp.stack([goal_state(params) for i in range(params['batch_size'])])
    
    key, sample_key = jax.random.split(key)
    trajectories_batch = sample_forward_trajectories(
        sample_key, 
        model,
        bwd_action_perms,
        fwd_action_perms,
        init_state, 
        params
    )
    trajectories_batch = jax.lax.stop_gradient(trajectories_batch)

    loss_value, grads = eqx.filter_value_and_grad(tb_loss)(
        model, train_iter, trajectories_batch, bwd_action_perms, fwd_action_perms, params
    )
    
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


@eqx.filter_jit
def check_finished(state, params):
    return (state == goal_state(params)[None, :]).all(axis=-1)


@eqx.filter_jit
def single_state_beam_search(init_state, bwd_action_perms, fwd_action_perms, model, params, beam_k):
    def beam_search_step(state, score, done, sum_len, min_len):
        st, idx, cnt = jax.numpy.unique(state, return_index=True, size=state.shape[0], axis=0, fill_value=state[0], return_counts=True)
        state = state[idx]
        score = score[idx]
        score = jnp.where(cnt > 0, score, -100000.0)
        done = done[idx]
        sum_len = sum_len[idx]
        
        logits = jax.vmap(model)(state)
        log_pbs, _, _ = process_logits(state, logits, bwd_action_perms, fwd_action_perms, params)
        log_pbs = jnp.concatenate([log_pbs for i in range(len(bwd_action_perms))], axis=0)
        
        action = jnp.arange(state.shape[0] * len(bwd_action_perms)) // state.shape[0]
        state = jnp.concatenate([state for i in range(len(bwd_action_perms))], axis=0)
        score = jnp.concatenate([score for i in range(len(bwd_action_perms))], axis=0)
        done = jnp.concatenate([done for i in range(len(bwd_action_perms))], axis=0)
        sum_len = jnp.concatenate([sum_len for i in range(len(bwd_action_perms))], axis=0)

        score = score + log_pbs[jnp.arange(state.shape[0]), action]

        _, next_idx = jax.lax.top_k(score, min(beam_k, score.shape[0]))

        next_state = apply_action(state[next_idx], action[next_idx], bwd_action_perms)
        next_score = score[next_idx]
        done = done[next_idx]
        sum_len = sum_len[next_idx]

        done = done | check_finished(next_state, params)
        sum_len = sum_len + (~done)

        min_len = jnp.where(done.any(), jnp.minimum(min_len, sum_len.min()), min_len)

        return next_state, next_score, done, sum_len, min_len

    state = jnp.stack([init_state for i in range(1)])
    score = jnp.zeros((state.shape[0],))
    done = jnp.zeros((state.shape[0],), dtype=bool)
    sum_len = jnp.zeros((state.shape[0],), dtype=bool) + 1
    min_len = jnp.array(1000000.0)

    while state.shape[0] < beam_k:
        state, score, done, sum_len, min_len = beam_search_step(state, score, done, sum_len, min_len)

    def sample_step(i, val):
        state, score, done, sum_len, min_len = val
        state, score, done, sum_len, min_len = beam_search_step(state, score, done, sum_len, min_len)
        return (state, score, done, sum_len, min_len)
        
    _, _, _, _, min_len = jax.lax.fori_loop(
        lower=0, 
        upper=params['eval_max_length'], 
        body_fun=sample_step, 
        init_val=(state, score, done, sum_len, min_len)
    )

    return min_len


@eqx.filter_jit
def scramble(key, bwd_action_perms, fwd_action_perms, init_state, params):
    def sample_step(i, val):
        step_key, state = val
        step_key, action_key = jax.random.split(step_key)
        logits = jnp.zeros((state.shape[0], fwd_action_perms.shape[0]))
        action = jax.random.categorical(action_key, logits)
        prev_state = apply_action(state, action, fwd_action_perms)
        return (step_key, prev_state)
    
    result = jax.lax.fori_loop(lower=0, upper=1000, body_fun=sample_step, init_val=(key, init_state))[1]
    return result


@eqx.filter_jit
def create_random_eval_state(key, params, test_set):
    if params['use_test_set']:
        init_state = test_set
        if init_state.shape[0] > params['eval_batch_size']:
            key, subset_key = jax.random.split(key)
            init_state = jax.random.choice(subset_key, init_state, shape=(params['eval_batch_size'],), replace=False)
    else:
        if params['task'] == "rubik2" or params['task'] == "rubik3":
            init_state = jnp.stack([goal_state(params) for i in range(params['eval_batch_size'])])
            key, scramble_key = jax.random.split(key)
            init_state = scramble(scramble_key, bwd_action_perms, fwd_action_perms, init_state, params)
        elif params['task'] == "swap":
            init_state = jnp.stack([goal_state(params) for i in range(params['eval_batch_size'])])
            key, shuffle_key = jax.random.split(key)
            shuffle_keys = jax.random.split(shuffle_key, init_state.shape[0])
            init_state = jax.vmap(jax.random.permutation)(shuffle_keys, init_state)

    return init_state


@eqx.filter_jit
def eval_step_beam(key, bwd_action_perms, fwd_action_perms, model, test_set, params, beam_k):
    init_state = create_random_eval_state(key, params, test_set)

    sum_len = jax.vmap(single_state_beam_search, in_axes=(0, None, None, None, None, None))(
        init_state, bwd_action_perms, fwd_action_perms, model, params, beam_k
    )

    return {
        "avg_length": sum_len.mean(),
        "solve_rate": (sum_len < params['eval_max_length'] + 1).mean(), 
    }


@eqx.filter_jit
def eval_step(train_iter, key, bwd_action_perms, fwd_action_perms, model, test_set, params, greedy=False, topk=1):
    init_state = create_random_eval_state(key, params, test_set)
    
    init_state = jnp.concatenate([init_state for idx in range(topk)], axis=0)
    done = jnp.zeros((init_state.shape[0],), dtype=bool)
    
    def sample_step(i, val):
        step_key, state, done, sum_len = val
        
        step_key, action_key = jax.random.split(step_key)
        
        logits = jax.vmap(model)(state)['backward_logits']
        if not greedy:
            action = jax.random.categorical(action_key, logits)
        else:
            action = jnp.argmax(logits, axis=-1)
            
        next_state = apply_action(state, action, bwd_action_perms)
        
        done = done | check_finished(next_state, params)
        sum_len = sum_len + (~done)
        
        return (step_key, next_state, done, sum_len)
    
    _, _, done, sum_len = jax.lax.fori_loop(
        lower=0, 
        upper=params['eval_max_length'], 
        body_fun=sample_step, 
        init_val=(key, init_state, done, jnp.zeros((init_state.shape[0],), dtype=int) + 1)
    )

    topk_sum_len = [sum_len[idx * params['eval_batch_size']: (idx+1) * params['eval_batch_size']] for idx in range(topk)]
    topk_sum_len = jnp.stack(topk_sum_len).min(axis=0)
    
    return {
        "avg_length": topk_sum_len.mean(), 
        "solve_rate": (topk_sum_len < params['eval_max_length'] + 1).mean(), 
    }

    
def model_save_fn(model_idx, path):
    model, idx = model_idx
    eqx.tree_serialise_leaves(str(idx) + "_" + path, model)

    
@eqx.filter_jit
def train_run(key, bwd_action_perms, fwd_action_perms, model, opt_state, test_set, params):
    model_params, model_static = eqx.partition(model, eqx.is_array)
    
    def one_iter(idx, val):
        key, bwd_action_perms, fwd_action_perms, model_params, opt_state = val
        model = eqx.combine(model_params, model_static)
        
        key, step_key = jax.random.split(key)
        model, opt_state, loss_value = train_step(
            idx,
            step_key, 
            bwd_action_perms, 
            fwd_action_perms, 
            model, 
            opt_state, 
            params
        )
        
        key, eval_key = jax.random.split(key)
        eval_key_true, eval_key_greedy = jax.random.split(eval_key)
        eval_key_greedy, eval_key_topk = jax.random.split(eval_key_greedy)
        eval_key_topk, eval_key_beam = jax.random.split(eval_key_topk)
        def eval_and_print():
            jax.debug.print("{bar}", bar=idx)
            
            metrics = eval_step(
                idx,
                eval_key_true, 
                bwd_action_perms,
                fwd_action_perms, 
                model, 
                test_set,
                params,
                greedy=False,
                topk=1
            )
            jax.debug.print("true eval {bar}", bar=metrics)
            
            metrics = eval_step(
                idx,
                eval_key_greedy, 
                bwd_action_perms,
                fwd_action_perms, 
                model, 
                test_set,
                params,
                greedy=True,
                topk=1
            )
            jax.debug.print("greedy eval {bar}", bar=metrics)

            metrics = eval_step_beam(
                eval_key_beam,
                bwd_action_perms,
                fwd_action_perms, 
                model, 
                test_set,
                params,
                beam_k=params['beam_k']
            )
            jax.debug.print("beam search eval {k} {bar}", k=params['beam_k'], bar=metrics)

            if len(params['save_path']) > 0:
                model_params, model_static = eqx.partition(model, eqx.is_array)
                model_save_fn_with_path = partial(model_save_fn, path=params['save_path'])
                jax.experimental.io_callback(model_save_fn_with_path, None, (model_params, idx))
            
        jax.lax.cond(idx % params['print_every'] == 0, eval_and_print, lambda: None)
        
        model_params, _ = eqx.partition(model, eqx.is_array)
        return (key, bwd_action_perms, fwd_action_perms, model_params, opt_state)          
    
    result = jax.lax.fori_loop(
        lower=0, 
        upper=params['train_steps'], 
        body_fun=one_iter, 
        init_val=(key, bwd_action_perms, fwd_action_perms, model_params, opt_state)
    )
    

if __name__ == '__main__':
    args = parser.parse_args()

    if args.task == "rubik2":
        raw_actions = [
            [0, 1, 19, 17, 6, 4, 7, 5, 2, 9, 3, 11, 12, 13, 14, 15, 16, 20, 18, 21, 10, 8, 22, 23], 
            [18, 16, 2, 3, 4, 5, 6, 7, 8, 0, 10, 1, 13, 15, 12, 14, 22, 17, 23, 19, 20, 21, 11, 9], 
            [0, 5, 2, 7, 4, 21, 6, 23, 10, 8, 11, 9, 3, 13, 1, 15, 16, 17, 18, 19, 20, 14, 22, 12], 
            [4, 1, 6, 3, 20, 5, 22, 7, 8, 9, 10, 11, 12, 2, 14, 0, 17, 19, 16, 18, 15, 21, 13, 23], 
            [0, 1, 2, 3, 4, 5, 18, 19, 8, 9, 6, 7, 12, 13, 10, 11, 16, 17, 14, 15, 22, 20, 23, 21], 
            [1, 3, 0, 2, 16, 17, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 20, 21, 22, 23], 
            [0, 1, 8, 10, 5, 7, 4, 6, 21, 9, 20, 11, 12, 13, 14, 15, 16, 3, 18, 2, 17, 19, 22, 23], 
            [9, 11, 2, 3, 4, 5, 6, 7, 8, 23, 10, 22, 14, 12, 15, 13, 1, 17, 0, 19, 20, 21, 16, 18], 
            [0, 14, 2, 12, 4, 1, 6, 3, 9, 11, 8, 10, 23, 13, 21, 15, 16, 17, 18, 19, 20, 5, 22, 7], 
            [15, 1, 13, 3, 0, 5, 2, 7, 8, 9, 10, 11, 12, 22, 14, 20, 18, 16, 19, 17, 4, 21, 6, 23], 
            [0, 1, 2, 3, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 16, 17, 6, 7, 21, 23, 20, 22], 
            [2, 0, 3, 1, 8, 9, 6, 7, 12, 13, 10, 11, 16, 17, 14, 15, 4, 5, 18, 19, 20, 21, 22, 23]
        ]
        assert len(raw_actions) == 12
        bwd_action_perms = jnp.array(raw_actions)
        fwd_action_perms = jnp.array(raw_actions[len(raw_actions) // 2:] + raw_actions[:len(raw_actions) // 2])
    elif args.task == "rubik3":
        raw_actions = [
            [6, 3, 0, 7, 4, 1, 8, 5, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 47, 21, 22, 50, 24, 25, 53, 27, 28, 38, 30, 31, 41, 33, 34, 44, 36, 37, 20, 39, 40, 23, 42, 43, 26, 45, 46, 29, 48, 49, 32, 51, 52, 35], 
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 12, 9, 16, 13, 10, 17, 14, 11, 36, 19, 20, 39, 22, 23, 42, 25, 26, 45, 28, 29, 48, 31, 32, 51, 34, 35, 27, 37, 38, 30, 40, 41, 33, 43, 44, 18, 46, 47, 21, 49, 50, 24, 52, 53], 
            [44, 43, 42, 3, 4, 5, 6, 7, 8, 45, 46, 47, 12, 13, 14, 15, 16, 17, 24, 21, 18, 25, 22, 19, 26, 23, 20, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 11, 10, 9, 0, 1, 2, 48, 49, 50, 51, 52, 53], 
            [0, 1, 2, 3, 4, 5, 51, 52, 53, 9, 10, 11, 12, 13, 14, 38, 37, 36, 18, 19, 20, 21, 22, 23, 24, 25, 26, 33, 30, 27, 34, 31, 28, 35, 32, 29, 8, 7, 6, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 15, 16, 17], 
            [0, 1, 35, 3, 4, 34, 6, 7, 33, 20, 10, 11, 19, 13, 14, 18, 16, 17, 2, 5, 8, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 9, 12, 15, 42, 39, 36, 43, 40, 37, 44, 41, 38, 45, 46, 47, 48, 49, 50, 51, 52, 53], 
            [24, 1, 2, 25, 4, 5, 26, 7, 8, 9, 10, 27, 12, 13, 28, 15, 16, 29, 18, 19, 20, 21, 22, 23, 17, 14, 11, 6, 3, 0, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 51, 48, 45, 52, 49, 46, 53, 50, 47], 
            [2, 5, 8, 1, 4, 7, 0, 3, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 38, 21, 22, 41, 24, 25, 44, 27, 28, 47, 30, 31, 50, 33, 34, 53, 36, 37, 29, 39, 40, 32, 42, 43, 35, 45, 46, 20, 48, 49, 23, 51, 52, 26], 
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 17, 10, 13, 16, 9, 12, 15, 45, 19, 20, 48, 22, 23, 51, 25, 26, 36, 28, 29, 39, 31, 32, 42, 34, 35, 18, 37, 38, 21, 40, 41, 24, 43, 44, 27, 46, 47, 30, 49, 50, 33, 52, 53], 
            [45, 46, 47, 3, 4, 5, 6, 7, 8, 44, 43, 42, 12, 13, 14, 15, 16, 17, 20, 23, 26, 19, 22, 25, 18, 21, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 2, 1, 0, 9, 10, 11, 48, 49, 50, 51, 52, 53], 
            [0, 1, 2, 3, 4, 5, 38, 37, 36, 9, 10, 11, 12, 13, 14, 51, 52, 53, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 32, 35, 28, 31, 34, 27, 30, 33, 17, 16, 15, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 6, 7, 8], 
            [0, 1, 18, 3, 4, 19, 6, 7, 20, 33, 10, 11, 34, 13, 14, 35, 16, 17, 15, 12, 9, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 8, 5, 2, 38, 41, 44, 37, 40, 43, 36, 39, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53], 
            [29, 1, 2, 28, 4, 5, 27, 7, 8, 9, 10, 26, 12, 13, 25, 15, 16, 24, 18, 19, 20, 21, 22, 23, 0, 3, 6, 11, 14, 17, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 50, 53, 46, 49, 52, 45, 48, 51]
        ]
        assert len(raw_actions) == 12
        bwd_action_perms = jnp.array(raw_actions)
        fwd_action_perms = jnp.array(raw_actions[len(raw_actions) // 2:] + raw_actions[:len(raw_actions) // 2])
    elif args.task == "swap":
        raw_actions = []
        n = args.n
        for i in range(n - 1):
            perm = list(range(n))
            perm[i], perm[i + 1] = perm[i + 1], perm[i]
            raw_actions.append(perm)
    
        bwd_action_perms = jnp.array(raw_actions)
        fwd_action_perms = jnp.array(raw_actions)

    key = jax.random.key(args.seed)
    
    key, model_key = jax.random.split(key)
    model = ResMLPPolicy(
        n=len(raw_actions[0]),
        n_bwd_actions=len(raw_actions),
        n_fwd_actions=len(raw_actions) + 1,
        hidden_size=args.hidden_size,
        rng_key=model_key,
        task=args.task
    )
    
    if len(args.load_path) > 0:
        model_params, model_static = eqx.partition(model, eqx.is_array)
        model_params = eqx.tree_deserialise_leaves(args.load_path, model_params)
        model = eqx.combine(model_params, model_static)

    clipper = optax.clip_by_global_norm(max_norm=args.clip_grad_norm)
    optim = optax.chain(
        clipper,
        optax.adamw(learning_rate=args.lr, weight_decay=1e-5)
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    print("device:", bwd_action_perms.device)
    parameters, static = eqx.partition(model, eqx.is_array)
    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y.size, parameters, 0)
    print("number of model parameters:", total_params)

    if args.use_test_set:
        test_set = jnp.load(args.test_set_path)
    else:
        test_set = None
    if args.task == "rubik2":
        true_log_z = float(jnp.log(3674160.0*24))
    elif args.task == "rubik3":
        true_log_z = float(jnp.log(43252003274489856000.0))
    elif args.task == "swap":
        true_log_z = scipy.special.gammaln(args.n+1)
    print(true_log_z)
        
    params = frozendict({
        'n': len(raw_actions[0]),
        'batch_size': args.batch_size, 
        'random_trajlen': args.random_trajlen, 
        'reg_coef': args.reg_coef,
        'eval_batch_size': args.eval_batch_size if not args.use_test_set else min(len(test_set), args.eval_batch_size),
        'eval_max_length': args.eval_max_length,
        'true_log_z': true_log_z,
        'train_steps': args.train_steps,
        'print_every': args.print_every,
        'eps_explore': args.eps_explore,
        'topk': args.topk,
        'beam_k': args.beam_k,
        'use_test_set': args.use_test_set,
        'task': args.task,
        'save_path': args.save_path
    })
        
    train_run(
        key, 
        bwd_action_perms, 
        fwd_action_perms,
        model, 
        opt_state, 
        test_set,
        params
    )
    
    
