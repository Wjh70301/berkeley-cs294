import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import pickle

# import matplotlib.pyplot as plt
# import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


hidden_layer_num = int(sys.argv[1])
hidden_layer_size = int(sys.argv[2])
training_data_pkl = sys.argv[3]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

with open(training_data_pkl, 'rb') as inf:
    X_tv, y_tv, X_test, y_test = pickle.load(inf)

tf.logging.info('{0}, {1}, {2}, {3}'.format(
    X_tv.shape, X_test.shape, y_tv.shape, y_test.shape
))

x_plh = tf.placeholder(tf.float32, shape=[None, X_tv.shape[1]])
y_plh = tf.placeholder(tf.float32, shape=[None, y_tv.shape[1]])

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


saver = tf.train.Saver()
outdir = os.path.join('./models/hl-num/{0}/hl-size/{1}'.format(
    hidden_layer_num, hidden_layer_size
))

if not os.path.exists(outdir):
    os.makedirs(outdir)

tf.logging.info('starting session ...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.logging.info('initialization ready')
    mse_tv, mse_test = [], []

    bs = 128                    # batch size
    num_epochs = 1000
    tf.logging.info('train for {0} epochs with batch size {1}'.format(num_epochs, bs))
    for k in range(num_epochs):        # num. epochs
        tf.logging.info('Epoch: {0}'.format(k + 1))
        for i in range(X_tv.shape[0] // bs):
            _x = X_tv[i * bs: (i+1) * bs, :]
            _y = y_tv[i * bs: (i+1) * bs, :]
            train_op.run(feed_dict={x_plh: _x, y_plh: _y})

            # if i % 10 == 0:
            #     mse_tv.append(mse.eval(feed_dict={x_plh: X_tv, y_plh: y_tv}))
            #     mse_test.append(mse.eval(feed_dict={x_plh: X_test, y_plh: y_test}))

    prefix = os.path.join(outdir, '{0}-{1}'.format(
        hidden_layer_num, hidden_layer_size))
    saver.save(sess, prefix)

    out_f = os.path.join(outdir, 'test_mse.csv'.format(hidden_layer_size))
    with open(out_f, 'wt') as opf:
        opf.write('{0}\t{1}\t{2}\t{3}\n'.format(
            hidden_layer_num,
            hidden_layer_size,
            mse.eval(feed_dict={x_plh: X_tv, y_plh: y_tv}),
            mse.eval(feed_dict={x_plh: X_test, y_plh: y_test}),
        ))

# plt.plot(mse_tv, label='tv')
# plt.plot(mse_test, label='test')
# plt.legend()
# plt.xlabel('# iterations')
# plt.ylabel('MSE')
# plt.grid()
# plt.savefig('lele.png')

# print(mse_tv[-1], mse_test[-1])
