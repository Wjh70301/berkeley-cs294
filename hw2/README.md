# Commands with more informative arguments

Adopted from http://rll.berkeley.edu/deeprlcourse/f17docs/hw2_final.pdf

```
python train_pg.py CartPole-v0 --n_iter 100 --batch_size 1000 --n_experiments 5      -dna --exp_name sb_no_rtg_dna
python train_pg.py CartPole-v0 --n_iter 100 --batch_size 1000 --n_experiments 5 -rtg -dna --exp_name sb_rtg_dna
python train_pg.py CartPole-v0 --n_iter 100 --batch_size 1000 --n_experiments 5 -rtg      --exp_name sb_rtg_na
python train_pg.py CartPole-v0 --n_iter 100 --batch_size 5000 --n_experiments 5      -dna --exp_name lb_no_rtg_dna
python train_pg.py CartPole-v0 --n_iter 100 --batch_size 5000 --n_experiments 5 -rtg -dna --exp_name lb_rtg_dna
python train_pg.py CartPole-v0 --n_iter 100 --batch_size 5000 --n_experiments 5 -rtg      --exp_name lb_rtg_na
```

* `sb`: small batch
* `rtg`: rewart-to-go
* `dna`: don't normalize advantages
