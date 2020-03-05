import numpy as np
from scipy.stats import multivariate_normal

def load_dataset(filename):
    """
    加载散点数据集
    :param filename: 文件名
    :return: 数据，标签
    """
    dataset = np.loadtxt(filename)
    data = dataset[:, :-1]
    labels = dataset[:50, -1]  # 取前50行作为有标签数据
    return data, labels

def get_label_num(train_labels):
    """
    获取数据类型数量
    :param train_labels: 数据类型标签
    :return: 数据类型数量
    """
    label_list = list(set(train_labels.copy().tolist()))  # 深拷贝后利用set去重
    return len(label_list)

def init(train_data):


def phi(data, mu, cov):
    norm = multivariate_normal(mean=mu, cov=cov)
    return norm.pdf(data)

def get_expectation(data, mu, cov, alpha, k):
    m = data.shape[0]
    gamma = np.mat(np.zeros((m, k)))
    prob = np.zeros((m, k))
    for i in range(k):
        prob[:, i] = phi(data, mu[i], cov[i])
    prob = np.mat(prob)
    for i in range(k):
        gamma[:, i] = alpha[i] * prob[:, i]
    for i in range(m):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma

def maximize(data, gamma, k):
    m, n = data.shape
    mu = np.zeros((k, n))
    cov = []
    alpha = np.zeros(k)
    for i in range(k):
        Nk = np.sum(gamma[:, i])
        for j in range(n):
            mu[i, j] = np.sum(np.multiply(gamma[:, i], data[:, j])) / Nk
        cov_i = np.mat(np.zeros((n, n)))
        for j in range(m):
            cov_i += gamma[j, i] * (data[j] - mu[i]).T * (data[j] - mu[i]) / Nk
        cov.append(cov_i)
        alpha[i] = Nk / m
    return mu, np.array(cov), alpha

def semi_GMM(train_data, train_labels, iter_times):
    m, n = train_data.shape
    l = len(train_labels)
    u = m - l
    k = get_label_num(train_labels)
    mu, cov = init(train_data)
    alpha = np.array([1.0 / k] * k)
    for i in range(iter_times):
        gamma = get_expectation(data, mu, cov, alpha, k)
        mu, cov, alpha = maximize(data, gamma, k)
    return mu, cov, alpha

if __name__ == '__main__':
    train_data, train_labels = load_dataset('scatter_with_label.txt')
    mu, cov, alpha = semi_GMM(train_data, train_labels, 100)