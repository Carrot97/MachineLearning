import numpy as np
from scipy.stats import multivariate_normal

class Param(object):  # 参数过多，定义结构体便于传参
    def __init__(self, data, labels, labeledDataNum):
        self.m, self.n = data.shape
        self.l = labeledDataNum
        self.u = self.m - self.l
        self.labeledData = np.mat(data[:self.l, :])
        self.labels = labels
        self.unlabeledData = np.mat(data[self.l:, :])
        self.k = len(list(set(self.labels.copy().tolist())))  # 获取数据类型数量,深拷贝后利用set去重
        # 利用有标签数据进行初始化
        self.mu = np.array(np.tile(np.sum(self.unlabeledData, axis=0)/self.u, (self.k, 1)))
        self.cov = np.array([(self.unlabeledData - self.mu[0, :]).T * (self.unlabeledData - self.mu[0, :]) / self.u] * self.k)
        self.alpha = np.array([1.0 / self.k] * self.k)
        self.gamma = np.mat(np.zeros((self.u, self.k)))

def loadDataSet(filename):
    """
    加载散点数据集
    :param filename: 文件名
    :return: 数据，标签
    """
    dataSet = np.loadtxt(filename)
    data = dataSet[:, :-1]
    labels = dataSet[:50, -1]  # 取前50行作为有标签数据
    return data, labels

def phi(data, mu, cov):
    """
    获得指定的高斯分布
    :param data: 数据
    :param mu: 均值
    :param cov: 方差
    :return: 特定高斯分布对于数据data的输出
    """
    norm = multivariate_normal(mean=mu, cov=cov)
    return norm.pdf(data)

def getExpectation(pa):
    """
    E步（按公式编写）
    :param pa: 训练参数
    :return: 无
    """
    prob = np.zeros((pa.u, pa.k))
    for i in range(pa.k):
        prob[:, i] = phi(pa.unlabeledData, pa.mu[i], pa.cov[i])
    prob = np.mat(prob)
    for i in range(pa.k):
        pa.gamma[:, i] = pa.alpha[i] * prob[:, i]
    for i in range(pa.u):
        pa.gamma[i, :] /= np.sum(pa.gamma[i, :])

def maximize(pa):
    """
    M步（按公式编写）
    :param pa: 训练参数
    :return: 无
    """
    pa.mu = np.zeros((pa.k, pa.n))
    pa.cov = []
    pa.alpha = np.zeros(pa.k)
    for i in range(pa.k):
        labeled_data_i = np.array([np.array(pa.labeledData)[d] for d in range(pa.l) if pa.labels[d] == i+1])
        li = len(labeled_data_i)
        Nk = np.sum(pa.gamma[:, i]) + li
        for j in range(pa.n):
            pa.mu[i, j] = (np.sum(np.multiply(pa.gamma[:, i], pa.unlabeledData[:, j])) + np.sum(labeled_data_i[:, j])) / Nk
        cov_i = np.mat(np.zeros((pa.n, pa.n)))
        for j in range(pa.u):
            cov_i += pa.gamma[j, i] * (pa.unlabeledData[j] - pa.mu[i]).T * (pa.unlabeledData[j] - pa.mu[i])
        for j in range(pa.l):
            cov_i += (pa.labeledData[j] - pa.mu[i]).T * (pa.labeledData[j] - pa.mu[i])
        pa.cov.append(cov_i / Nk)
        pa.alpha[i] = Nk / pa.m

def semi_GMM(trainData, trainLabels, iterTimes):
    pa = Param(trainData, trainLabels, 50)
    for i in range(iterTimes):
        getExpectation(pa)
        maximize(pa)
    return pa.mu, np.array(pa.cov), pa.alpha

if __name__ == '__main__':
    trainData, trainLabels = loadDataSet('scatter_with_label.txt')
    mu, cov, alpha = semi_GMM(trainData, trainLabels, 100)
    print(mu, cov, alpha)