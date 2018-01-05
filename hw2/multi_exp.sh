for i in {1..5}; do
    echo "python train_pg.py RoboschoolInvertedPendulum-v1 --n_iter 100 --seed 5 --batch_size 1000 --experiment_id=${i} --exp_name sb_rtg_na --reward_to_go --normalize_advantages"
    echo "python train_pg.py RoboschoolInvertedPendulum-v1 --n_iter 100 --seed 6 --batch_size 5000 --experiment_id=${i} --exp_name lb_rtg_na --reward_to_go --normalize_advantages"
done
