import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def init_cent(data, k):
    random.shuffle(data)
    return np.array(data[:k])

def k_means(data, k):
    mu = init_cent(data, k)
    data = np.array(data)
    m, n = data.shape
    mu_update = np.zeros((k, n))
    iteration = True
    count = 0
    while iteration:
        labels = np.zeros(m)
        for i in range(m):
            dist = []
            for mu_j in mu:
                dist.append(np.linalg.norm(data[i, :] - mu_j))
            labels[i] = np.argsort(dist)[0]
        for i in range(k):
            labels_i = np.array([data[j] for j in range(m) if labels[j] == i])
            mu_update[i, :] = (np.sum(labels_i, axis=0, keepdims=True)) / len(labels_i)
        if (mu == mu_update).all():
            count += 1
            if count == 3:
                iteration = False
        else:
            count = 0
            mu = mu_update
    return labels

if __name__ == '__main__':
    dataset = np.loadtxt('dataset.txt')
    dataset = list(set(tuple(data) for data in dataset))
    k = 2
    m = len(dataset)
    labels = k_means(dataset, k)
    # labels = KMeans(n_clusters=2).fit_predict(dataset)
    class1 = np.array([dataset[i] for i in range(m) if labels[i] == 0])
    class2 = np.array([dataset[i] for i in range(m) if labels[i] == 1])
    plt.title('k_means')
    plt.scatter(class1[:, 0], class1[:, 1], label="class1")
    plt.scatter(class2[:, 0], class2[:, 1], label="class2")
    plt.show()
