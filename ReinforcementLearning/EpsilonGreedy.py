import numpy as np

def getReward(k):
    """
    获取该摇臂的奖励值
    :param k: 摇臂号数
    :return: 返回一个取值为正态分布（均值为k，方差为1 ）的随机样本，即从1至k号摇臂奖励逐渐增大
    """
    return np.random.randn() + k

def epsilonGreedy(K, T, epsilon):
    reward = 0
    Q = np.zeros((K, 1))
    count = np.zeros((K, 1))
    for t in range(T):
        if np.random.random() < epsilon:
            k = np.random.choice(np.arange(K))
        else:
            k = np.argmax(Q)
        print('choose %d' % (k+1))  # 观察该算法每次选择的摇臂号
        v = getReward(k)
        reward += v
        Q[k] = (Q[k]*count[k] + v) / (count[k] + 1)
        count[k] += 1
    return reward

if __name__ == "__main__":
    reward = epsilonGreedy(5, 100, 0.3)
    print(reward)