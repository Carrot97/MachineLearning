import numpy as np
import matplotlib.pyplot as plt

def normlization(data):
    feature_num = data.shape[1]
    for i in range(feature_num):
        feature_max = max(data[:, i])
        feature_min = min(data[:, i])
        data[:, i] = (data[:, i] - feature_min) / (feature_max - feature_min)
    return data

def dist_min(data, C1, C2):
    m1 = len(C1)
    m2 = len(C2)
    min = 100
    for i in range(m1):
        for j in range(m2):
            dist = np.linalg.norm(data[C1[i]] - data[C2[j]])
            if dist < min:
                min = dist
    return min

def AGNES(data, k):
    m = len(data)
    C = [[i] for i in range(m)]
    M = (np.ones((m, m))) * 2
    for i in range(m):
        for j in range(m):
            M[i, j] = np.linalg.norm(data[i] - data[j])
            if M[i, j] == 0:
                M[i, j] = 2
    q = m
    while q > k:
        row, column = np.where(M == np.min(M))
        row = row[0]
        column = column[0]
        C[row].extend(C[column])
        del C[column]
        M = np.delete(M, column, 0)
        M = np.delete(M, column, 1)
        q -= 1
        for j in range(q):
            M[row, j] = dist_min(data, C[row], C[j])
            if M[row, j] == 0:
                M[row, j] = 2
            M[j, row] = M[row, j]
    return C

if __name__ == '__main__':
    dataset = np.loadtxt('dataset.txt')
    dataset = normlization(np.array(list(set(tuple(data) for data in dataset))))
    k = 2
    m = len(dataset)
    labels = AGNES(dataset, k)
    class1 = np.array(dataset[labels[0]])
    class2 = np.array(dataset[labels[1]])
    plt.title('AGNES')
    plt.scatter(class1[:, 0], class1[:, 1], label="class1")
    plt.scatter(class2[:, 0], class2[:, 1], label="class2")
    plt.show()