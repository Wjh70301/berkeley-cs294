import os
import sys
sys.path.insert(0, '..')
import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import gym
import load_policy

tf.logging.set_verbosity(tf.logging.INFO)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def build_nn(input_size, output_size):
    x_plh = tf.placeholder(tf.float32, shape=[None, input_size])
    y_plh = tf.placeholder(tf.float32, shape=[None, output_size])

    with tf.name_scope('fc1'):
        Wh_var = weight_variable([x_plh.shape.dims[1].value, hidden_layer_size])
        bh_var = bias_variable([hidden_layer_size])
        hh = tf.nn.sigmoid(tf.matmul(x_plh, Wh_var) + bh_var)

    for i in range(hidden_layer_num - 1):
        with tf.name_scope('fc{0}'.format(i + 2)):
            Wh_var = weight_variable([hidden_layer_size, hidden_layer_size])
            bh_var = bias_variable([hidden_layer_size])
            hh = tf.nn.sigmoid(tf.matmul(hh, Wh_var) + bh_var)

    with tf.name_scope('out'):
        W_var = weight_variable([hidden_layer_size, y_plh.shape.dims[1].value])
        b_var = bias_variable([y_plh.shape.dims[1].value])
        y_pred = tf.matmul(hh, W_var) + b_var

    with tf.name_scope('mse'):
        mse = tf.losses.mean_squared_error(labels=y_plh, predictions=y_pred)
        mse = tf.cast(mse, tf.float32)

    with tf.name_scope('adam_optimizer'):
        train_op = tf.train.AdamOptimizer(1e-4).minimize(mse)

    return x_plh, y_plh, y_pred, mse, train_op


def train(train_op, metric, X_tv, y_tv, batch_size, num_epochs):
    mse_tv, mse_test = [], []
    for k in range(num_epochs): # num. epochs
        for i in range(X_tv.shape[0] // batch_size):
            _x = X_tv[i * batch_size: (i+1) * batch_size, :]
            _y = y_tv[i * batch_size: (i+1) * batch_size, :]
            train_op.run(feed_dict={x_plh: _x, y_plh: _y})

        if (k + 1) % 10 == 1:
            tf.logging.info('epoch: {0}/{1}'.format(k, num_epochs))
        mse_tv.append(metric.eval(feed_dict={x_plh: X_tv, y_plh: y_tv}))
        mse_test.append(metric.eval(feed_dict={x_plh: X_test, y_plh: y_test}))

    # tf.logging.info(mse_tv)
    saver.save(sess, model_prefix)
    return mse_tv, mse_test


def dagger(task, max_timesteps):
    env = gym.make(task)
    obs = env.reset()

    totalr = 0
    done = False
    X_dagger, y_dagger = [], []
    for k in range(max_timesteps):
        if (k + 1) % 10 == 0:
            tf.logging.info(k + 1)
        action = pred_action(obs)
        # action = pred_action_by_expert(obs)
        action_expert = pred_action_by_expert(obs)
        X_dagger.append(obs)
        y_dagger.append(action_expert.ravel())

        obs, r, done, _ = env.step(action)
        totalr += r
    #     env.render()
    # env.render(close=True)
    # dagger_rewards.append(np.mean(totalr))

    mean_r = np.mean(totalr)
    nX_dagger = np.array(X_dagger)
    ny_dagger = np.array(y_dagger)
    return mean_r, nX_dagger, ny_dagger


if __name__ == "__main__":
    task = 'Walker2d-v1'
    hidden_layer_num = 2
    hidden_layer_size = 30
    expert_pkl = '../experts/Walker2d-v1.pkl'
    training_data_pkl = '../train_test_data/Walker2d-10-rollouts-200.pkl'

    model_prefix = './lele/a'

    tf.logging.info('loading training data ...')
    with open(training_data_pkl, 'rb') as inf:
        X_tv, y_tv, X_test, y_test = pickle.load(inf)
    tf.logging.info('{0}, {1}, {2}, {3}'.format(
        X_tv.shape, X_test.shape, y_tv.shape, y_test.shape
    ))

    tf.logging.info('buidling model to train ...')
    input_size = X_tv.shape[1]
    output_size = y_tv.shape[1]
    x_plh = tf.placeholder(tf.float32, shape=[None, input_size])
    y_plh = tf.placeholder(tf.float32, shape=[None, output_size])

    with tf.name_scope('fc1'):
        Wh_var = weight_variable([x_plh.shape.dims[1].value, hidden_layer_size])
        bh_var = bias_variable([hidden_layer_size])
        hh = tf.nn.sigmoid(tf.matmul(x_plh, Wh_var) + bh_var)

    for i in range(hidden_layer_num - 1):
        with tf.name_scope('fc{0}'.format(i + 2)):
            Wh_var = weight_variable([hidden_layer_size, hidden_layer_size])
            bh_var = bias_variable([hidden_layer_size])
            hh = tf.nn.sigmoid(tf.matmul(hh, Wh_var) + bh_var)

    with tf.name_scope('out'):
        W_var = weight_variable([hidden_layer_size, y_plh.shape.dims[1].value])
        b_var = bias_variable([y_plh.shape.dims[1].value])
        y_pred = tf.matmul(hh, W_var) + b_var

    with tf.name_scope('mse'):
        mse = tf.losses.mean_squared_error(labels=y_plh, predictions=y_pred)
        mse = tf.cast(mse, tf.float32)

    with tf.name_scope('adam_optimizer'):
        train_op = tf.train.AdamOptimizer(1e-4).minimize(mse)

    policy_fn = load_policy.load_policy(expert_pkl)
    saver = tf.train.Saver()
    mse_tv_list, mse_test_list = [], []
    dagger_rewards = []
    tf.logging.info('init session ...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def pred_action(obs):
            return y_pred.eval(feed_dict={x_plh: obs.reshape(1, -1)})

        def pred_action_by_expert(obs):
            return policy_fn(obs.reshape(1, -1))

        tf.logging.info('start train-daggering ...')
        while True:
            tf.logging.info('training ...')
            mse_tv, mse_test = train(
                train_op, mse, X_tv, y_tv, batch_size=128, num_epochs=100)
            mse_tv_list.extend(mse_tv)
            mse_test_list.extend(mse_test)

            tf.logging.info('daggering ...')
            mean_r, nX_dagger, ny_dagger = dagger(task, max_timesteps=100)
            dagger_rewards.append(mean_r)
            X_tv = np.concatenate([X_tv, nX_dagger])
            y_tv = np.concatenate([y_tv, ny_dagger])

            # break


    # out_f = os.path.join('.', 'test_mse.csv'.format(hidden_layer_size))
        # with open(out_f, 'wt') as opf:
        #     opf.write('{0}\t{1}\n'.format(
        #         mse.eval(feed_dict={x_plh: X_tv, y_plh: y_tv}),
        #         mse.eval(feed_dict={x_plh: X_test, y_plh: y_test}),
        #     ))
