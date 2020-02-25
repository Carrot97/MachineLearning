import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def batch(data, label, batch_size):
    index = [i for i in range(0, len(label))]
    np.random.shuffle(index)
    batch_data = []
    batch_label = []
    for i in range(0, batch_size):
        batch_data.append(data[index[i]])
        batch_label.append(label[index[i]])
    return batch_data, batch_label

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# placeholder
xs = tf.placeholder(tf.float32, [None, 784])/255.
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# add layer
# fc1
W_fc1 = weight_variable([784, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(xs, W_fc1) + b_fc1)

# fc2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

time_start = time.time()
# train
for i in range(10001):
    batch_train_x, batch_train_y = batch(mnist.train.images, mnist.train.labels, 200)
    sess.run(train_step, feed_dict={xs: batch_train_x, ys: batch_train_y, keep_prob: 0.5})
    if i % 50 == 0:
        print('step =', i, '  ', compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))

# end
time_end = time.time()
print('totally cost', time_end-time_start)

# 0.98