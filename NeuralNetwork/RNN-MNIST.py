import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

learning_rate = 0.0001
training_time = 10001
batch_size = 200
n_input = 28
n_step = 28
n_hidden_units = 128
n_class = 10

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

xs = tf.placeholder(tf.float32, [None, n_step, n_input])
ys = tf.placeholder(tf.float32, [None, n_class])

weights = {
    'in': tf.Variable(tf.random_normal([n_input, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_class]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_class, ]))
}

def RNN(x, weights, biases):
    # input layer
    #####################################
    # 3D-->2D
    x = tf.reshape(x, [-1, n_input])
    x_in = tf.matmul(x, weights['in']) + biases['in']
    # 2D-->3D
    x_in = tf.reshape(x_in, [-1, n_step, n_hidden_units])

    # cell
    ######################################
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)

    # output layer
    ######################################
    result = tf.matmul(states[1], weights['out']) + biases['out']
    return result

prediction = RNN(xs, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

time_start = time.time()
# train
for step in range(training_time):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape(batch_size, n_step, n_input)
    sess.run([train_step], feed_dict={xs: batch_xs, ys: batch_ys})
    if step % 20 == 0:
        print('step =', step, '  ', compute_accuracy(mnist.test.images[:batch_size].reshape(-1, n_step, n_input),
                                                     mnist.test.labels[:batch_size]))

# end
time_end = time.time()
print('totally cost', time_end-time_start)












