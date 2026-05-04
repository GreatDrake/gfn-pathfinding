from uai_train import *

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=0, type=int)

parser.add_argument("--n", default=8, type=int)

parser.add_argument("--hidden_size", default=64, type=int)

parser.add_argument("--eval_max_length", default=600, type=int)

parser.add_argument("--beam_k", default=1, type=int)

parser.add_argument("--test_set_path", default="rubik2_test.npy", type=str)

parser.add_argument("--task", default="rubik2", type=str)

parser.add_argument("--load_path", default="", type=str)

parser.add_argument("--test_split_parts", default=1, type=int)


@eqx.filter_jit
def eval_step_beam(test_state, key, bwd_action_perms, fwd_action_perms, model, test_set, params, beam_k):
    sum_len = jax.vmap(single_state_beam_search, in_axes=(0, None, None, None, None, None))(
        test_state, bwd_action_perms, fwd_action_perms, model, params, beam_k
    )

    return {
        "avg_length": sum_len.mean(),
        "solve_rate": (sum_len < params['eval_max_length']).mean(), 
    }


def eval(key, bwd_action_perms, fwd_action_perms, model, test_set, params):
    key, eval_key_beam = jax.random.split(key, 2)

    test_state = create_random_eval_state(key, params, test_set)

    beam_results = []
    for i in range(params['test_split_parts']):
        jax.debug.print("{bar}", bar=i)
        metrics = eval_step_beam(
            test_state[i * (test_state.shape[0] // params['test_split_parts']) : (i + 1) * (test_state.shape[0] // params['test_split_parts'])],
            eval_key_beam,
            bwd_action_perms,
            fwd_action_perms, 
            model, 
            test_set,
            params,
            beam_k=params['beam_k']
        )
        beam_results.append(metrics)
    jax.debug.print("beam eval {n} {bar}", n=params['beam_k'], bar=jax.tree.map(lambda *a: jnp.mean(jnp.array(a)), *beam_results))
    

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
    model_params, model_static = eqx.partition(model, eqx.is_array)
    model_params = eqx.tree_deserialise_leaves(args.load_path, model_params)
    model = eqx.combine(model_params, model_static)

    test_set = jnp.load(args.test_set_path)
        
    params = frozendict({
        'n': len(raw_actions[0]),
        'eval_batch_size': len(test_set),
        'eval_max_length': args.eval_max_length,
        'beam_k': args.beam_k,
        'use_test_set': 1,
        'task': args.task,
        'test_split_parts': args.test_split_parts
    })

    eval(
        key, 
        bwd_action_perms, 
        fwd_action_perms,
        model, 
        test_set,
        params
    )
    
