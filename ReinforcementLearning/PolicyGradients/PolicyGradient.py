import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)

class Params:
    def __init__(self,
                 n_feature,
                 n_action,
                 epoch=3000,
                 learning_rate=0.02,
                 reward_decay=0.99):
        self.n_feature = n_feature
        self.n_action = n_action
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        self.round_states = []  # 每个回合的所有状态
        self.round_actions = []
        self.round_rewards = []
        self.cost = []

        self.neural_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def neural_network(self):
        with tf.name_scope('data'):
            self.s = tf.placeholder(tf.float32, [None, self.n_feature], name='state')
            self.a = tf.placeholder(tf.int32, [None, ], name='action')
            self.v = tf.placeholder(tf.float32, [None, ], name='action_value')

        hidden_layer = tf.layers.dense(inputs=self.s, units=10, activation=tf.nn.tanh,
                                       kernel_initializer=tf.random_normal_initializer(0.0, 0.3),
                                       bias_initializer=tf.constant_initializer(0.1),
                                       name='layer1')
        output = tf.layers.dense(inputs=hidden_layer, units=self.n_action, activation=None,
                                 kernel_initializer=tf.random_normal_initializer(0.0, 0.3),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 name='layer2')
        self.actions_prob = tf.nn.softmax(output, name='actions_probability')

        with tf.name_scope('loss'):
            # 每次更新先增强该次选择的动作
            prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.a)
            # 将增强效果乘动作反馈值，即分开好坏动作
            self.loss = tf.reduce_mean(prob * self.v)
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def choose_action(self, state):
        # 直接利用softmax选取动作
        actions_prob = self.sess.run(self.actions_prob, feed_dict={self.s: state[np.newaxis, :]})
        action = np.random.choice(range(actions_prob.shape[1]), p=actions_prob.ravel())  # 根据概率选动作
        return action

    def store_state(self, state, action, reward):
        self.round_states.append(state)
        self.round_actions.append(action)
        self.round_rewards.append(reward)

    def learn(self):
        round_reward_ppc = self.round_reward_preprocess()
        self.sess.run(self.train_op,
                      feed_dict={self.s: np.vstack(self.round_states),
                                 self.a: np.array(self.round_actions),
                                 self.v: round_reward_ppc, })
        self.round_states = []
        self.round_actions = []
        self.round_rewards = []
        return round_reward_ppc

    def round_reward_preprocess(self):
        discounted_rounds_reward = np.zeros_like(self.round_rewards)
        gain = 0
        # 奖励递减
        for i in reversed(range(0, len(self.round_rewards))):
            gain = gain * self.reward_decay + self.round_rewards[i]
            discounted_rounds_reward[i] = gain

        discounted_rounds_reward -= np.mean(discounted_rounds_reward)
        discounted_rounds_reward /= np.std(discounted_rounds_reward)
        return discounted_rounds_reward