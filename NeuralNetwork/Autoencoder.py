import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

EPOCH = 1000
INPUT_NUM = 28 * 28
HIDDEN_NUM1 = 256
HIDDEN_NUM2 = 128
OUTPUT_NUM = INPUT_NUM
BATCH_SIZE = 128
LEARNING_RATE = 0.001

xp = tf.placeholder(tf.float32, [None, INPUT_NUM])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([INPUT_NUM, HIDDEN_NUM1])),
    'encoder_h2': tf.Variable(tf.random_normal([HIDDEN_NUM1, HIDDEN_NUM2])),
    'decoder_h1': tf.Variable(tf.random_normal([HIDDEN_NUM2, HIDDEN_NUM1])),
    'decoder_h2': tf.Variable(tf.random_normal([HIDDEN_NUM1, OUTPUT_NUM])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([HIDDEN_NUM1])),
    'encoder_b2': tf.Variable(tf.random_normal([HIDDEN_NUM2])),
    'decoder_b1': tf.Variable(tf.random_normal([HIDDEN_NUM1])),
    'decoder_b2': tf.Variable(tf.random_normal([OUTPUT_NUM])),
}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.matmul(x, weights['encoder_h1']) + biases['encoder_b1'])
    layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights['encoder_h2']) + biases['encoder_b2'])
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.matmul(x, weights['decoder_h1']) + biases['decoder_b1'])
    layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights['decoder_h2']) + biases['decoder_b2'])
    return layer_2

feature = encoder(xp)
prediction = decoder(feature)
cost = tf.losses.mean_squared_error(prediction, xp)
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
j = 0
iteration = mnist.train.num_examples // BATCH_SIZE
for e in range(EPOCH):
    for i in range(iteration):
        xs, ys = mnist.train.next_batch(BATCH_SIZE)
        feed_dict = {xp: xs}
        loss, _ = sess.run([cost, train_step], feed_dict=feed_dict)
        if j % 100 == 0:
            print('epoch:{}'.format(e),
                  'iteration:{}'.format(j),
                  'cost:{}'.format(loss))
        j += 1

SHOW_NUM = random.randint(0, 1000)
feed_dict = {xp: mnist.test.images[SHOW_NUM: SHOW_NUM + 10]}
show = sess.run(prediction, feed_dict=feed_dict)
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(mnist.test.images[SHOW_NUM + i], (28, 28)), cmap='gray')
    a[1][i].imshow(np.reshape(show[i], (28, 28)), cmap='gray')
plt.show()