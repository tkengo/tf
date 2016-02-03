import tensorflow as tf
import numpy as np

true_parameter = np.array([3, 10]).reshape(2, 1)
train_x = np.hstack((np.ones([100, 1]), np.random.rand(100, 1)))
train_y = np.dot(train_x, true_parameter) + np.random.randn(100, 1)

x = tf.placeholder(tf.float32, [100, 2])

theta = tf.Variable(tf.zeros([2, 1]))
hypothesis = tf.matmul(x, theta)
cost = tf.reduce_mean(tf.square(hypothesis - train_y))
train = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

tf.scalar_summary('cost', cost)
tf.histogram_summary('theta', theta)

init_op = tf.initialize_all_variables()
summary = tf.merge_all_summaries()

with tf.Session() as sess:
    writer = tf.train.SummaryWriter("./rg_logs", sess.graph_def)
    sess.run(init_op)

    for i in xrange(1000):
        sess.run(train, feed_dict = { x: train_x })

        if i % 100 == 0:
            print(sess.run(theta).reshape(1, 2))
            summary_result = sess.run(summary, feed_dict = { x: train_x })
            writer.add_summary(summary_result, i)
