from Maze import Maze
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Params:
    def __init__(self,
            actions,
            features,
            learningRate=0.01,
            rewardDecay=0.9,
            eGreedy=0.9,
            replaceStepNum=300,
            memorySize=500,
            batchSize=32,
            ):
        self.actions = actions
        self.features = features
        self.lr = learningRate
        self.gamma = rewardDecay
        self.epsilon = eGreedy
        self.rsn = replaceStepNum
        self.ms = memorySize
        self.bs = batchSize
        self.costList = []
        self.learnStep = 0
        self.memory = np.zeros((self.ms, features*2+2))
        self.NN()
        oParams = tf.get_collection('old net parameters')
        nParams = tf.get_collection('new net parameters')
        self.replace_target_op = [tf.assign(o, n) for o, n in zip(oParams, nParams)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def NN(self):
        self.s = tf.placeholder(tf.float32, [None, self.features], name='s')
        self.Qtarget = tf.placeholder(tf.float32, [None, self.actions], name='Qtarget')
        with tf.variable_scope('newnet'):
            cNames, units, wInit, bInit = ['new net parameters', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                                          tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers


            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.features, units], initializer=wInit, collections=cNames)
                b1 = tf.get_variable('b1', [1, units], initializer=bInit, collections=cNames)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)


            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [units, self.actions], initializer=wInit, collections=cNames)
                b2 = tf.get_variable('b2', [1, self.actions], initializer=bInit, collections=cNames)
                self.Qpredict = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.Qtarget, self.Qpredict))
        with tf.variable_scope('train'):
            self.trainOP = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


        self.ns = tf.placeholder(tf.float32, [None, self.features], name='newState')
        with tf.variable_scope('oldnet'):
            cNames = ['old net parameters', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.features, units], initializer=wInit, collections=cNames)
                b1 = tf.get_variable('b1', [1, units], initializer=bInit, collections=cNames)
                l1 = tf.nn.relu(tf.matmul(self.ns, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [units, self.actions], initializer=wInit, collections=cNames)
                b2 = tf.get_variable('b2', [1, self.actions], initializer=bInit, collections=cNames)
                self.QnextStatePredict = tf.matmul(l1, w2) + b2

    def store_transition(self, state, action, reward, newState):
        if not hasattr(self, 'memoryCounter'):
            self.memoryCounter = 0

        transition = np.hstack((state, [action, reward], newState))

        index = self.memoryCounter % self.ms
        self.memory[index, :] = transition

        self.memoryCounter += 1

    def chooseAction(self, state):
        state = state[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            actionValue = self.sess.run(self.Qpredict, feed_dict={self.s: state})
            action = np.argmax(actionValue)
        else:
            action = np.random.randint(0, self.actions)
        return action

    def learn(self):
        if self.learnStep % self.rsn == 0:
            self.sess.run(self.replace_target_op)

        if self.memoryCounter > self.ms:
            sampleIndex = np.random.choice(self.ms, size=self.bs)
        else:
            sampleIndex = np.random.choice(self.memoryCounter, size=self.bs)
        batchMemory = self.memory[sampleIndex, :]

        QnextStatePredict, Qpredict = self.sess.run(
            [self.QnextStatePredict, self.Qpredict],
            feed_dict={
                self.ns: batchMemory[:, -self.features:],
                self.s: batchMemory[:, :self.features],
            })

        Qtarget = Qpredict.copy()

        batchIndex = np.arange(self.bs, dtype=np.int32)
        actionIndex = batchMemory[:, self.features].astype(int)
        reward = batchMemory[:, self.features + 1]

        Qtarget[batchIndex, actionIndex] = reward + self.gamma * np.max(QnextStatePredict, axis=1)
        
        _, self.cost = self.sess.run([self.trainOP, self.loss],
                                     feed_dict={self.s: batchMemory[:, :self.features],
                                                self.Qtarget: Qtarget})
        self.costList.append(self.cost)

        # increasing epsilon
        self.learnStep += 1

    def showCost(self):
        plt.plot(np.arange(len(self.costList)), self.costList)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

def update():
    step = 0
    for e in range(300):
        state = maze.reset()
        while True:
            maze.fresh()
            action = para.chooseAction(state)
            newState, reward, done = maze.step(action)
            para.store_transition(state, action, reward, newState)
            if step > 200 and step % 5 == 0:
                para.learn()
            state = newState
            if done:
                break
            step += 1
    maze.destroy()
    
if __name__ == "__main__":
    maze = Maze()
    para = Params(maze.n_actions, maze.n_features)
    maze.after(100, update)
    maze.mainloop()
    para.showCost()