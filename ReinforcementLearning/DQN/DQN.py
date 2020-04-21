from Maze import Maze
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

"""
游戏规则：
冒险家（红）从地图左上角出发，每次向上下左右移动一格，直到找到宝藏（黄）或坠入深渊（黑）。
找到宝藏加一分，坠入深渊减一分
"""

class QParam:
    def __init__(self,
                 actions,
                 features,
                 learningRate=0.01,
                 rewardDecay=0.9,
                 eGreedy=0.1,
                 epoch=100,
                 learnStepNum=5,
                 batchSize=32,
                 memoryCapacity=500,
                 repleaceStepNum=200
                 ):
        # 常量
        self.actions = actions
        self.features = features    # 棋盘格子数量
        self.gamma = rewardDecay
        self.epsilon = eGreedy
        self.epoch = epoch
        self.lr = learningRate      # 神经网络学习率
        self.lsn = learnStepNum     # 每隔多少步real net学习一次
        self.bs = batchSize         # 神经网络real net训练的batch size
        self.mc = memoryCapacity    # 离线数据容量
        self.rsn = repleaceStepNum  # 每隔多少步更新一次predict net的参数

        # 变量
        self.memory = np.zeros((self.mc, self.features*2+2))  # 缓存（每行为state, action, reward, newState）
        self.learnStep = 0  # 训练步数计数器
        self.cost = []  # 记录神经网络cost

        # 操作
        self.neuralNetwork()  # 建立模型

        oldPara = tf.get_collection('old net parameters')
        newPara = tf.get_collection('main net parameters')
        self.replacePara = [tf.assign(p, r) for p, r in zip(oldPara, newPara)]  # 将real net的参数复制给predict net

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())  # 神经网络变量初始化

    def neuralNetwork(self):

        # 建立主神经网络
        self.s = tf.placeholder(tf.float32, [None, self.features], name='state')
        self.t = tf.placeholder(tf.float32, [None, self.actions], name='Qtarget')
        with tf.variable_scope('mainnet'):
            # 建立参数集合
            cNames, unitNum, wInit, bInit = ['main net parameters', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                                            tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)

            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w1', [self.features, unitNum], initializer=wInit, collections=cNames)
                b1 = tf.get_variable('b1', [1, unitNum], initializer=bInit, collections=cNames)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('layer2'):
                w2 = tf.get_variable('w2', [unitNum, self.actions], initializer=wInit, collections=cNames)
                b2 = tf.get_variable('b2', [1, self.actions], initializer=bInit, collections=cNames)
                self.Qpredict = tf.matmul(l1, w2) + b2
            
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.t, self.Qpredict))
        with tf.variable_scope('train'):
            self.train = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # 建立旧神经网络
        self.ns = tf.placeholder(tf.float32, [None, self.features], name='newState')
        with tf.variable_scope('oldnet'):
            cNames = ['old net parameters', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w1', [self.features, unitNum], initializer=wInit, collections=cNames)
                b1 = tf.get_variable('b1', [1, unitNum], initializer=bInit, collections=cNames)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('layer2'):
                w2 = tf.get_variable('w2', [unitNum, self.actions], initializer=wInit, collections=cNames)
                b2 = tf.get_variable('b2', [1, self.actions], initializer=bInit, collections=cNames)
                self.QnextStatePredict = tf.matmul(l1, w2) + b2

    def chooseAction(self, state):
        state = state[np.newaxis, :]  # 适应神经网络输入
        if np.random.uniform() < self.epsilon:       # epsilon贪婪
            action = np.random.choice(self.actions)  # 小于e时随机选取一个动作
        else:
            stateAction = self.sess.run(self.Qpredict, feed_dict={self.s: state})
            action = np.argmax(stateAction)  # 大于e时选取Q值最大的动作
        return action

    def getInMemory(self, state, newState, action, reward):
        if not hasattr(self, 'memoryCounter'):  # 若当前类中无该变量则创建
            self.memoryCouner = 0
        row = np.hstack((state, [action, reward], newState))  # 建立行向量
        index = self.memoryCouner % self.mc  # 超出缓存容量后覆盖最前面的行
        self.memory[index, :] = row
        self.memoryCouner += 1

    def learn(self):
        if self.learnStep % self.rsn == 0:
            self.sess.run(self.replacePara)

        # 从缓存中抽取部分行向量学习
        if self.memoryCouner > self.bs:
            index = np.random.choice(self.mc, size=self.bs)
        else:
            index = np.random.choice(self.memoryCouner, size=self.bs)
        batchMemory = self.memory[index, :]

        QnextStatePredict, Qpredict = self.sess.run([self.QnextStatePredict, self.Qpredict],
                                        feed_dict={self.ns: batchMemory[:, -self.features:],
                                                   self.s: batchMemory[:, :self.features]})
        index = np.arange(self.bs, dtype=np.int32)
        action = batchMemory[:, self.features].astype(int)  # 获取向量中的动作
        reward = batchMemory[:, self.features + 1]          # 获取向量中的奖励
        Qtarget = Qpredict.copy()
        Qtarget[index, action] = reward + self.gamma * np.max(QnextStatePredict, axis=1)
        _, cost = self.sess.run([self.train, self.loss],                             # 主神经网络训练
                                feed_dict={self.s: batchMemory[:, :self.features],
                                           self.t: Qtarget})
        self.cost.append(cost)
        self.learnStep += 1

    def showCost(self):
        plt.plot(np.arange(len(self.cost)), self.cost)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

def update():
    stepCounter = 0
    for e in range(para.epoch):
        state = maze.reset()  # 初始化
        while True:
            maze.fresh()                                           # 刷新环境（相当于显示）
            action = para.chooseAction(state)                 # 基于策略选择一个行动
            newState, reward, isDone = maze.step(action)           # 获得新棋盘状态，奖励和游戏是否结束
            para.getInMemory(state, newState, action, reward)      # 将过程存入缓存以便离线学习
            if stepCounter > 200 and stepCounter % para.lsn == 0:  # 每隔几步学习以此，抵消参数间的相关性
                para.learn()
            if isDone:
                break
            state = newState
            stepCounter += 1
    maze.destory()

if __name__ == "__main__":
    maze = Maze()  # 建立迷宫环境
    para = QParam(maze.n_actions, maze.n_features)  # 初始化QL参数缺省值参数采用默认值
    maze.after(para.epoch, update)  # 循环100次
    maze.mainloop()
    para.showCost()