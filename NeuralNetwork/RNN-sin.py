import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

# 1 parameters
TRAINING_NUM = 20000
TESTING_NUM = 1000
SAMPLE_GAP = 0.01
TIME_STEP = 20
INPUT_NUM = 1
HIDDEN_NUM = 30
OUTPUT_NUM = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCH = 20
################################################################

# 2 function
def generate_data(sep):
    x = []
    y = []
    for i in range(len(sep) - TIME_STEP - 1):
        x.append(sep[i: i + TIME_STEP])
        y.append(sep[i + TIME_STEP])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

def get_batch(x, y, batch_size, i):
    start_point = i * batch_size
    if (start_point + batch_size) < len(x):
        end_point = start_point + batch_size
    return x[start_point: end_point], y[start_point: end_point]
#####################################################################

# 3 input data
testing_start_point = TRAINING_NUM * SAMPLE_GAP
testing_end_point = testing_start_point + TESTING_NUM * SAMPLE_GAP

train_x, train_y = generate_data(np.sin(np.linspace(0, testing_start_point, TRAINING_NUM)))
test_x, test_y = generate_data(np.sin(np.linspace(testing_start_point, testing_end_point, TESTING_NUM)))
###########################################################################

# 4 placeholder
xp = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_NUM])
yp = tf.placeholder(tf.float32, [None, OUTPUT_NUM])
##################################################################

# 5 weights and biases
weights = {
    'in': tf.Variable(tf.random_normal([INPUT_NUM, HIDDEN_NUM])),
    'out': tf.Variable(tf.random_normal([HIDDEN_NUM * TIME_STEP, OUTPUT_NUM]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[HIDDEN_NUM, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[OUTPUT_NUM, ]))
}
######################################################################

# 6 model
def RNN(x, weights, biases):
    # input layer
    x = tf.reshape(x, [-1, INPUT_NUM])
    x_in = tf.matmul(x, weights['in']) + biases['in']
    x_in = tf.reshape(x_in, [-1, TIME_STEP, HIDDEN_NUM])

    # cell # single LSTM
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_NUM, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    outputs, final_states = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)

    # output layer
    outputs = tf.reshape(outputs, [BATCH_SIZE, -1])
    result = tf.matmul(outputs, weights['out']) + biases['out']
    return result
#######################################################################

# 7 train step
prediction = RNN(xp, weights, biases)
cost = tf.losses.mean_squared_error(prediction, yp)
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
######################################################################

# 8 init
sess = tf.Session()
sess.run(tf.global_variables_initializer())
iteration = (TRAINING_NUM - TIME_STEP) // BATCH_SIZE - 1
time_start = time.time()
k = 0
print('start')
##################################################################

# 9 training start
for e in range(EPOCH):
    for i in range(iteration):
        xs, ys = get_batch(train_x, train_y, BATCH_SIZE, i)
        feed_dict = {xp: xs[:, :, None], yp: ys[:, None]}
        loss, _ = sess.run([cost, train_step], feed_dict=feed_dict)
        if k % 10 == 0:
            print('Epoch:{}'.format(e),
                'Iteration:{}'.format(k),
                'Train loss: {:.8f}'.format(loss))
        k += 1
# end
time_end = time.time()
print('totally cost', round(time_end-time_start, 2))
######################################################################

# 10 test and plt
jteration = (TESTING_NUM - TIME_STEP) // BATCH_SIZE - 1
for j in range(jteration):
    xs, ys = get_batch(test_x, test_y, BATCH_SIZE, j)
    feed_dict = {xp: xs[:, :, None]}
    results = sess.run(prediction, feed_dict=feed_dict)
    plt.plot(test_y[j * BATCH_SIZE: (j + 1) * BATCH_SIZE], 'g--', label='real sin')
    plt.plot(results, 'r', label='predicted')
    plt.legend()
    plt.show()