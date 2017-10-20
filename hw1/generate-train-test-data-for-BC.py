import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import gym


import tf_util
import load_policy

# task = 'Ant-v1'
task = 'Humanoid-v1'
num_rollouts = 500
max_timesteps = 2000
train_ratio = 0.8
out_pkl = './section-3-behavioral-cloning/humanoid_train_test-50-rollouts.pkl'

policy_fn = load_policy.load_policy('./experts/{0}.pkl'.format(task))

sess = tf.InteractiveSession()

env = gym.make(task)
X_all, y_all = [], []

for i in range(num_rollouts):
    print(i, end=',')
    obs = env.reset()
    action = policy_fn(obs[None, :])
    X_all.append(obs)
    y_all.append(action)   
    for j in range(max_timesteps):
        obs, r, done, _ = env.step(action)
        action = policy_fn(obs[None, :])
        X_all.append(obs)
        y_all.append(action)
        
X_all = np.array(X_all)
# concat instead of array is necessary to keep array in the right shape
y_all = np.concatenate(y_all)

# split all data into train and test set
idx = int(X_all.shape[0] * train_ratio)

X_tv = X_all[:idx, :]
X_test = X_all[idx:, :]
y_tv = y_all[:idx, :]
y_test = y_all[idx:, :]

print(X_tv.shape, X_test.shape, y_tv.shape, y_test.shape)

with open(out_pkl, 'wb') as opf:
    pickle.dump([X_tv, y_tv, X_test, y_test], opf)

