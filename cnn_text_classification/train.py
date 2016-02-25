import os
import data_helper
import numpy as np
import tensorflow as tf
from data_helper import log
from constants import *

if not os.path.exists(CHECKPOINTS_DIR):
    os.makedirs(CHECKPOINTS_DIR)

x, y, d = data_helper.load_data_and_labels_and_dictionaries()

# Split original data into two groups for training and testing.
train_x, train_y = x[:-NUM_TESTS], y[:-NUM_TESTS]
test_x,  test_y  = x[-NUM_TESTS:], y[-NUM_TESTS:]

# Property for dropout. This is probability of keeping cell.
keep = tf.placeholder(tf.float32)

# ----------------------------------------------------------
# Build Convolutional Neural Network for text classification
# ----------------------------------------------------------
# Define input layer.
x_dim = train_x.shape[1]
input_x = tf.placeholder(tf.int32,   [ None, x_dim       ])
input_y = tf.placeholder(tf.float32, [ None, NUM_CLASSES ])

# Define 2nd layer (Word embedding layer).
with tf.name_scope('embedding'):
    w  = tf.Variable(tf.random_uniform([ len(d), EMBEDDING_SIZE ], -1.0, 1.0), name='weight')
    e  = tf.nn.embedding_lookup(w, input_x)
    ex = tf.expand_dims(e, -1)

# Define 3rd and 4th layer (Temporal 1-D convolutional and max-pooling layer).
p_array = []
for filter_size in FILTER_SIZES:
    with tf.name_scope('conv-%d' % filter_size):
        w  = tf.Variable(tf.truncated_normal([ filter_size, EMBEDDING_SIZE, 1, NUM_FILTERS ], stddev=0.02), name='weight')
        b  = tf.Variable(tf.constant(0.1, shape=[ NUM_FILTERS ]), name='bias')
        c0 = tf.nn.conv2d(ex, w, [ 1, 1, 1, 1 ], 'VALID')
        c1 = tf.nn.relu(tf.nn.bias_add(c0, b))
        c2 = tf.nn.max_pool(c1, [ 1, x_dim - filter_size + 1, 1, 1 ], [ 1, 1, 1, 1 ], 'VALID')
        p_array.append(c2)

p = tf.concat(3, p_array)

# Define output layer (Fully-connected layer).
with tf.name_scope('fc'):
    total_filters = NUM_FILTERS * len(FILTER_SIZES)
    w = tf.Variable(tf.truncated_normal([ total_filters, NUM_CLASSES ], stddev=0.02), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=[ NUM_CLASSES ]), name='bias')
    h0 = tf.nn.dropout(tf.reshape(p, [ -1, total_filters ]), keep)
    predict_y = tf.nn.softmax(tf.matmul(h0, w) + b)

# ----------------------------------------------------------
# Create optimizer.
# ----------------------------------------------------------
# Use cross entropy for softmax as a cost function.
xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_y, input_y))

# Add L2 regularization term in order to avoid overfitting.
loss = xentropy + L2_LAMBDA * tf.nn.l2_loss(w)

# Create optimizer for my cost function.
global_step = tf.Variable(0, name="global_step", trainable=False)
train = tf.train.AdamOptimizer(0.0001).minimize(loss, global_step=global_step)

# ----------------------------------------------------------
# Measurement of accuracy and summary for TensorBoard.
# ----------------------------------------------------------
predict  = tf.equal(tf.argmax(predict_y, 1), tf.argmax(input_y, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

loss_sum   = tf.scalar_summary('train loss', loss)
accr_sum   = tf.scalar_summary('train accuracy', accuracy)
t_loss_sum = tf.scalar_summary('general loss', loss)
t_accr_sum = tf.scalar_summary('general accuracy', accuracy)

saver = tf.train.Saver()

# ----------------------------------------------------------
# Start TensorFlow Session.
# ----------------------------------------------------------
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter(SUMMARY_LOG_DIR, sess.graph_def)

    train_x_length = len(train_x)
    batch_count = int(train_x_length / NUM_MINI_BATCH) + 1

    log('Start training.')
    log('     epoch: %d' % NUM_EPOCHS)
    log('mini batch: %d' % NUM_MINI_BATCH)
    log('train data: %d' % train_x_length)
    log(' test data: %d' % len(test_x))
    log('We will loop %d count per an epoch.' % batch_count)

    # Start training. We will loop some epochs.
    for epoch in xrange(NUM_EPOCHS):
        # Randomize training data every epoch in order to converge training more quickly.
        random_indice = np.random.permutation(train_x_length)

        # Split training data into mini batch for SGD.
        log('Start %dth epoch.' % (epoch + 1))
        for i in xrange(batch_count):
            # Take mini batch from training data.
            mini_batch_x = []
            mini_batch_y = []
            for j in xrange(min(train_x_length - i * NUM_MINI_BATCH, NUM_MINI_BATCH)):
                mini_batch_x.append(train_x[random_indice[i * NUM_MINI_BATCH + j]])
                mini_batch_y.append(train_y[random_indice[i * NUM_MINI_BATCH + j]])

            # TRAINING.
            _, v1, v2, v3, v4 = sess.run(
                [ train, loss, accuracy, loss_sum, accr_sum ],
                feed_dict={ input_x: mini_batch_x, input_y: mini_batch_y, keep: 0.5 }
            )
            log('%4dth mini batch complete. LOSS: %f, ACCR: %f' % (i + 1, v1, v2))

            # Write out loss and accuracy value into summary logs for TensorBoard.
            current_step = tf.train.global_step(sess, global_step)
            writer.add_summary(v3, current_step)
            writer.add_summary(v4, current_step)

            # Save all variables to a file every checkpoints.
            if current_step % CHECKPOINTS_EVERY == 0:
                saver.save(sess, CHECKPOINTS_DIR + '/model', global_step=current_step)
                log('Checkout was completed.')

            # Evaluate the model by test data every evaluation point.
            if current_step % EVALUATE_EVERY == 0:
                random_test_indice = np.random.permutation(100)
                random_test_x = test_x[random_test_indice]
                random_test_y = test_y[random_test_indice]

                v1, v2, v3, v4 = sess.run(
                    [ loss, accuracy, t_loss_sum, t_accr_sum ],
                    feed_dict={ input_x: random_test_x, input_y: random_test_y, keep: 1.0 }
                )
                log('Testing... LOSS: %f, ACCR: %f' % (v1, v2))
                writer.add_summary(v3, current_step)
                writer.add_summary(v4, current_step)

    # Save the model before the program is finished.
    saver.save(sess, CHECKPOINTS_DIR + '/model-last')
