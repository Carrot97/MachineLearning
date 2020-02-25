import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

def normlization(data):
    feature_num = data.shape[1]
    for i in range(feature_num):
        feature_max = max(data[:, i])
        feature_min = min(data[:, i])
        data[:, i] = (data[:, i] - feature_min) / (feature_max - feature_min)
    return data

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

def GMM(data, k, iter_times):
    data = normlization(data)
    m, n = data.shape
    mu = np.random.rand(k, n)
    cov = np.array([np.eye(n)] * k)
    alpha = np.array([1.0 / k] * k)
    for i in range(iter_times):
        gamma = get_expectation(data, mu, cov, alpha, k)
        mu, cov, alpha = maximize(data, gamma, k)
    return mu, cov, alpha

if __name__ == '__main__':
    dataset = np.loadtxt('dataset.txt')
    data = np.mat(dataset)
    m, n = data.shape
    mu, cov, alpha = GMM(data, 2, 100)
    gamma = get_expectation(data, mu, cov, alpha, 2)
    labels = gamma.argmax(axis=1).flatten().tolist()[0]
    # sk_GMM = GaussianMixture(n_components=2).fit(dataset)
    # labels = sk_GMM.predict(dataset)
    class1 = np.array([dataset[i] for i in range(m) if labels[i] == 0])
    class2 = np.array([dataset[i] for i in range(m) if labels[i] == 1])
    plt.scatter(class1[:, 0], class1[:, 1], label="class1")
    plt.scatter(class2[:, 0], class2[:, 1], label="class2")
    plt.show()