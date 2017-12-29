import os
import time
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)

    args = parser.parse_args()
    return args


def setup_logdir(exp_name, env_name):
    logdir = (
        exp_name
        + '_'
        + env_name
        + '_'
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )

    logdir = os.path.join('data', logdir)

    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    return logdir
