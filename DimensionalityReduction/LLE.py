import numpy as np
from sklearn.datasets import make_s_curve
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt

def generate_curve_data():
    dataset, labels = make_s_curve(n_samples=500, noise=0.1, random_state=42)
    return dataset, labels

# def make_swiss_roll(n_samples=100, noise=0.0, random_state=None):
#     #Generate a swiss roll dataset.
#     t = 1.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
#     x = t * np.cos(t)
#     y = 83 * np.random.rand(1, n_samples)
#     z = t * np.sin(t)
#     X = np.concatenate((x, y, z))
#     X += noise * np.random.randn(3, n_samples)
#     X = X.T
#     t = np.squeeze(t)
#     return X, t

def cal_euc_dist(dataset):
    sum_x = np.sum(np.square(dataset), 1)
    dists = np.add(np.add(-2 * np.dot(dataset, dataset.T), sum_x).T, sum_x)
    dists[dists < 0] = 0
    dists = dists ** 0.5
    return dists

def LLE(dataset, n_neighbors=15, n_components=2):
    k = n_neighbors
    euc_dists = cal_euc_dist(dataset)
    dists_sorted_index = np.argsort(euc_dists, axis=1)
    m, n = np.shape(dataset)
    w = np.zeros((k, m))
    column_I = np.ones((k, 1))
    if k > n:
        tol = 1e-3
    else:
        tol = 0
    for i in range(m):
        Xi = np.tile(dataset[i, :], (k, 1)).T
        Ni = dataset[dists_sorted_index[i, 1:k+1]].T
        Si = np.dot((Xi-Ni).T, (Xi-Ni))
        # 正则化???
        Si = Si + np.eye(k) * tol * np.trace(Si)
        # _______________________________________
        Si_inv = np.linalg.inv(Si)
        wi = np.dot(Si_inv, column_I) / (np.dot(np.dot(column_I.T, Si_inv), column_I))
        w[:, i] = wi[:, 0]

    W = np.zeros((m, m))
    for i in range(m):
        index = dists_sorted_index[i, 1:k+1]
        for j in range(k):
            W[index[j], i] = w[j, i]
    I = np.eye(m)
    M = np.dot((I - W), (I - W).T)
    val, vec = np.linalg.eig(M)
    val_sorted_index = np.argsort(np.abs(val))[1: n_components+1]
    Z = vec[:, val_sorted_index]
    return Z

if __name__ == '__main__':
    dataset, labels = generate_curve_data()
    dataset_sklearn_LLE = LocallyLinearEmbedding(n_neighbors=30, n_components=2).fit_transform(dataset)
    dataset_my_LLE = LLE(dataset, 30, 2)
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title('my_LLE')
    plt.scatter(dataset_my_LLE[:, 0], dataset_my_LLE[:, 1], c=labels)

    plt.subplot(122)
    plt.title('sklearn_LLE')
    plt.scatter(dataset_sklearn_LLE[:, 0], dataset_sklearn_LLE[:, 1], c=labels)
    # plt.savefig('LLE.png')
    plt.show()