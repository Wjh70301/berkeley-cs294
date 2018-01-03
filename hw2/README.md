# Commands with more informative arguments

Slightly different API from
http://rll.berkeley.edu/deeprlcourse/f17docs/hw2_final.pdf

Five repeats for each experiment

```
for i in {1..5}; do
    echo "python train_pg.py CartPole-v0 --n_iter 100 --seed 1 --batch_size 1000 --experiment_id=${i} --exp_name sb_no_rtg_no_na"
    echo "python train_pg.py CartPole-v0 --n_iter 100 --seed 2 --batch_size 5000 --experiment_id=${i} --exp_name lb_no_rtg_no_na"

    echo "python train_pg.py CartPole-v0 --n_iter 100 --seed 3 --batch_size 1000 --experiment_id=${i} --exp_name sb_rtg_no_na --reward_to_go"
    echo "python train_pg.py CartPole-v0 --n_iter 100 --seed 4 --batch_size 5000 --experiment_id=${i} --exp_name lb_rtg_no_na --reward_to_go"

    echo "python train_pg.py CartPole-v0 --n_iter 100 --seed 5 --batch_size 1000 --experiment_id=${i} --exp_name sb_rtg_na --reward_to_go --normalize_advantages"
    echo "python train_pg.py CartPole-v0 --n_iter 100 --seed 6 --batch_size 5000 --experiment_id=${i} --exp_name lb_rtg_na --reward_to_go --normalize_advantages"

    echo "python train_pg.py CartPole-v0 --n_iter 100 --seed 5 --batch_size 1000 --experiment_id=${i} --exp_name sb_rtg_na_nb --reward_to_go --normalize_advantages --nn_baseline"
    echo "python train_pg.py CartPole-v0 --n_iter 100 --seed 6 --batch_size 5000 --experiment_id=${i} --exp_name lb_rtg_na_nb --reward_to_go --normalize_advantages --nn_baseline"
done
```

* `sb`: small batch
* `rtg`: rewart-to-go
* `na`: normalize_advantages


## Caveat

* My implementation for continuous action is not 100% sure correct, although it learns
* My implementation of neural network baseline may still be incorrect
