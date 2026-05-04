# Learning Shortest Paths with Generative Flow Networks

Official code for the paper [Learning Shortest Paths with Generative Flow Networks](https://arxiv.org/abs/2603.01786).

Nikita Morozov, Ian Maksimov, Daniil Tiapkin, Sergey Samsonov.

## Rubik’s Cubes

### 2x2x2 Rubik’s Cube

Example of training a model to solve 2x2x2 Rubik’s Cube:

```
python train.py --reg_coef 0.01 --hidden_size 1024 --eval_batch_size 100 --use_test_set 1 --test_set_path datasets/rubik2_test.npy --task rubik2 --eval_max_length 100 --beam_k 256 --batch_size 128 --train_steps 510000 --random_trajlen 12 --print_every 50000 --save_path model.eqx
```

Evaluating a pre-trained model:

```
python eval.py --test_set_path datasets/rubik2_test.npy --test_split_parts 1 --task rubik2 --eval_max_length 20 --beam_k 256 --load_path weights/rubik2_model.eqx --hidden_size 1024
```

### 3x3x3 Rubik’s Cube

Example of training a model to solve 3x3x3 Rubik’s Cube:

```
python train.py --reg_coef 0.0000005 --hidden_size 2048 --eval_batch_size 1000 --use_test_set 1 --test_set_path datasets/rubik3_test.npy --task rubik3 --eval_max_length 100 --beam_k 512 --batch_size 2048 --train_steps 2000000 --random_trajlen 24 --print_every 50000 --save_path model.eqx
```

Evaluating a pre-trained model:

```
python eval.py --test_set_path datasets/rubik3_test.npy --test_split_parts 1 --task rubik3 --eval_max_length 30 --beam_k 512 --load_path weights/rubik3_model.eqx --hidden_size 2048
```

In case of limited GPU memory, try using a larger number for `--test_split_parts`.

### General training tips

- If the solution length improves too slowly, try increasing `--reg_coef`
- If the solve rate is stuck at zero from the beginning, try decreasing `--reg_coef`
- In some rare cases, the training procedure can be sensitive to neural network initialization, so you can also try starting a training run from a different initialization.

## Swap

Example of training a model for `n=20`:

```
python train.py --reg_coef 0.0001 --hidden_size 1024 --eval_batch_size 500 --use_test_set 1 --test_set_path datasets/swap20_test.npy --task swap --n 20 --eval_max_length 500 --beam_k 4 --batch_size 128 --train_steps 100000 --random_trajlen 60 --print_every 5000
```

Example of training a model for `n=15`:

```
python3 train.py --reg_coef 0.001 --hidden_size 1024 --eval_batch_size 500 --use_test_set 1 --test_set_path datasets/swap15_test.npy --task swap --n 15 --eval_max_length 500 --beam_k 4 --batch_size 128 --train_steps 100000 --random_trajlen 35 --print_every 5000 
```

## Citation

```
@article{morozov2026learning,
  title={Learning Shortest Paths with Generative Flow Networks},
  author={Morozov, Nikita and Maksimov, Ian and Tiapkin, Daniil and Samsonov, Sergey},
  journal={arXiv preprint arXiv:2603.01786},
  year={2026}
}
```
