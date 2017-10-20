import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys

import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


hidden_lay_size = int(sys.argv[1])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


with open('./train_test_data/humanoid_train_test.pkl', 'rb') as inf:
    X_tv, y_tv, X_test, y_test = pickle.load(inf)

print(X_tv.shape, X_test.shape, y_tv.shape, y_test.shape)

x_plh = tf.placeholder(tf.float32, shape=[None, X_tv.shape[1]])
y_plh = tf.placeholder(tf.float32, shape=[None, y_tv.shape[1]])

with tf.name_scope('fc1'):
    Wh_var = weight_variable([x_plh.shape.dims[1].value, hidden_lay_size])
    bh_var = bias_variable([hidden_lay_size])
    hh = tf.nn.sigmoid(tf.matmul(x_plh, Wh_var) + bh_var)

with tf.name_scope('out'):
    W_var = weight_variable([hidden_lay_size, y_plh.shape.dims[1].value])
    b_var = bias_variable([y_plh.shape.dims[1].value])
    y_pred = tf.matmul(hh, W_var) + b_var

with tf.name_scope('mse'):
    mse = tf.losses.mean_squared_error(labels=y_plh, predictions=y_pred)
    mse = tf.cast(mse, tf.float32)

with tf.name_scope('adam_optimizer'):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(mse)


print('starting session ...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('initialization ready')
    mse_tv, mse_test = [], []
    
    bs = 128                    # batch size
    for k in range(100):          # num. epochs
        print(k, end=',')
        for i in range(X_tv.shape[0] // bs):
            _x = X_tv[i * bs : (i+1) * bs, :]
            _y = y_tv[i * bs : (i+1) * bs, :]
            train_op.run(feed_dict={x_plh: _x, y_plh: _y})

            # if i % 10 == 0:
            #     mse_tv.append(mse.eval(feed_dict={x_plh: X_tv, y_plh: y_tv}))
            #     mse_test.append(mse.eval(feed_dict={x_plh: X_test, y_plh: y_test}))

# plt.plot(mse_tv, label='tv')
# plt.plot(mse_test, label='test')
# plt.legend()
# plt.xlabel('# iterations')
# plt.ylabel('MSE')
# plt.grid()
# plt.savefig('lele.png')

# print(mse_tv[-1], mse_test[-1])

saver = tf.train.Saver()
outdir = os.path.join('/tmp/model-hls-{0}'.format(hidden_lay_size))
if not os.path.exists(outdir):
    os.mkdir(outdir)
saver.restore(sess, outdir)

