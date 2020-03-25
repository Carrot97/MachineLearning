import numpy as np

def getReward(k):
    """
    获取该摇臂的奖励值
    :param k: 摇臂号数
    :return: 返回一个取值为正态分布（均值为k，方差为1 ）的随机样本，即从1至k号摇臂奖励逐渐增大
    """
    return np.random.randn() + k

def epsilonGreedy(K, epsilon, Q):
    if np.random.random() < epsilon:
        k = np.random.choice(np.arange(K))
    else:
        k = np.argmax(Q)
    return k

def softmax(K, tol, Q):
    P = np.exp(Q / tol)
    P = P / np.sum(P)  # 计算各摇臂被选择概率
    choose = np.random.random()  # 生成一个0-1的随机数
    count = 0
    for k in range(K):
        count += P[k]
        if count > choose:  # 累加概率与随机数比较，超过则说明随机数落在第k摇臂的区间内
            return k

def KarmedBandit(K, T, epsilon, tol):
    reward = 0
    Q = np.zeros((K, 1))
    count = np.zeros((K, 1))
    for t in range(T):
        k = epsilonGreedy(K, epsilon, Q)  # epsilonGreedy方法选取k值
        # k = softmax(K, tol, Q)          # softmax方法选取k值
        print('choose %d' % (k+1))        # 观察该算法每次选择的摇臂号
        v = getReward(k)
        reward += v
        Q[k] = (Q[k]*count[k] + v) / (count[k] + 1)
        count[k] += 1
    return reward

if __name__ == "__main__":
    reward = KarmedBandit(5, 100, 0.3, 1)  # softmax算法依赖tol值设置
    print(reward)