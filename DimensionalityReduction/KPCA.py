import numpy as np

def load_xihua():
    dataset = [[1,1,1,1,1,1],
               [2,1,2,1,1,1],
               [2,1,1,1,1,1],
               [1,1,2,1,1,1],
               [3,1,1,1,1,1],
               [1,1,1,1,2,2],
               [2,1,1,2,2,2],
               [2,1,1,1,2,1],
               [2,1,2,2,2,1],
               [1,2,3,1,3,2],
               [3,2,3,3,3,1],
               [3,1,1,3,3,2],
               [1,1,1,2,1,1],
               [3,1,2,2,1,1],
               [2,1,1,1,2,2],
               [3,1,1,3,3,1],
               [1,1,2,2,2,1]]
    labels = [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2]
    return np.array(dataset), np.array(labels)

def prehandle(dataset):
    _, n = np.shape(dataset)
    means = [np.mean(dataset[:, i]) for i in range(n)]
    norm_dataset = dataset - np.array(means)
    return norm_dataset

def kernel_cal(X):
    m, _ = np.shape(X)
    K = np.mat(np.zeros((m, m)))
    for i in range(m):
        for j in range(m):
            delta = X[i, :] - X[j, :]
            K[i, j] = np.dot(delta.T, delta)
    K = np.exp(K / (-2 * (1.3 ** 2)))
    return K

def KPCA(dataset, k):
    K = kernel_cal(dataset)
    val, vec = np.linalg.eig(K)
    val_sorted_index = np.argsort(-val)
    vec = vec[:, val_sorted_index]
    vec = vec[:, :k]
    Z = np.dot(K, vec)
    return Z

if __name__ == '__main__':
    dataset, labels = load_xihua()
    norm_dataset = prehandle(dataset)
    pca_dataset = KPCA(norm_dataset, 3)
    print(pca_dataset)