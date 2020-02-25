import numpy as np
from mpl_toolkits import mplot3d
from sklearn.datasets import make_s_curve
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

def generate_curve_data():
    dataset, labels = make_s_curve(n_samples=500, noise=0.1, random_state=42)
    return dataset, labels

def neighbor_floyd(D, n_neighbors):
    Max = np.max(D) * 1000
    m, _ = np.shape(D)
    k = n_neighbors
    D1 = np.ones((m, m)) * Max
    D_sorted = np.argsort(D, axis=1)
    for i in range(m):
        D1[i, D_sorted[i, 0:k+1]] = D[i, D_sorted[i, 0:k+1]]
    for k in range(m):
        for i in range(m):
            for j in range(m):
                if D1[i, k] + D1[k, j] < D1[i, j]:
                    D1[i, j] = D1[i, k] + D1[k, j]
    return D1

def cal_euc_dist(dataset):
    # method 1
    # m, n = np.shape(dataset)
    # dists = np.zeros((m, m))
    # for i in range(m):
    #     for j in range(m):
    #         dists[i][j] = np.linalg.norm(dataset[i] - dataset[j])
    # method 2 (advanced)
    # (a-b)^2 = a^2 + b^2 - 2ab
    sum_x = np.sum(np.square(dataset), 1)
    dists = np.add(np.add(-2 * np.dot(dataset, dataset.T), sum_x).T, sum_x)
    dists[dists < 0] = 0
    dists = dists ** 0.5
    return dists

def MDS(dists, n):
    m, _ = np.shape(dists)
    B = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            B[i][j] = - (np.square(dists[i][j]) - np.square(dists[i, :]).sum()/m -
                         np.square(dists[:, j]).sum()/m + np.square(dists.reshape(1, -1).sum()/(m**2))) / 2
    val, vec = np.linalg.eigh(B)
    val_sorted_index = np.argsort(-val)
    val = val[val_sorted_index]
    vec = vec[:, val_sorted_index]
    val_mat = np.diag(val[:n])
    vec = vec[:, :n]
    Z = np.dot(np.sqrt(val_mat), vec.T).T
    return Z

def ISOMAP(dataset, n_neighbors=15, n=2):
    euc_dists = cal_euc_dist(dataset)
    dists = neighbor_floyd(euc_dists, n_neighbors)
    dataset_dim_red = MDS(dists, n)
    return dataset_dim_red

if __name__ == '__main__':
    dataset, labels = generate_curve_data()
    plt.figure(figsize=(6, 5))
    plt.subplot(111)
    plt.title('original')
    ax = plt.axes(projection='3d')
    ax.scatter3D(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=labels)
    ax.view_init(20, -70)
    plt.savefig('curve_dataset.png')
    plt.show()

    dataset_my_isomap = ISOMAP(dataset, 15, 2)
    dataset_sklearn_isomap = Isomap(n_neighbors=15, n_components=2).fit_transform(dataset)
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title('my_isomap')
    plt.scatter(dataset_my_isomap[:, 0], dataset_my_isomap[:, 1], c=labels)

    plt.subplot(122)
    plt.title('sklearn_isomap')
    plt.scatter(dataset_sklearn_isomap[:, 0], dataset_sklearn_isomap[:, 1], c=labels)
    plt.savefig('isomap.png')
    plt.show()