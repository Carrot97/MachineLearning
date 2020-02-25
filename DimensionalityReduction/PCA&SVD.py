import numpy as np
# from sklearn.decomposition import PCA

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

def PCA(dataset, k):
    m, n = np.shape(dataset)
    cov = 1/m * np.dot(dataset.T, dataset)
    # factorization
    val, vec = np.linalg.eig(cov)
    val_sorted_index = np.argsort(-val)
    vec = vec[:, val_sorted_index]
    vec = vec[:, :k]
    Z = np.dot(dataset, vec)
    # _________________________________
    # SVD
    # U, sigma, V = np.linalg.svd(cov)
    # val_v, V = np.linalg.eig(np.dot(dataset.T, dataset))
    # sigma = np.sqrt(val_v)
    # sigma_sorted_index = np.argsort(-sigma)
    # V = V[:, sigma_sorted_index]
    # V = V[:, :k]
    # Z = np.dot(dataset, V)
    return Z

def PCA_auto(dataset):
    m, n = np.shape(dataset)
    cov = 1/m * np.dot(dataset.T, dataset)
    val, vec = np.linalg.eig(cov)
    val_sorted_index = np.argsort(-val)
    val = val[val_sorted_index]
    vec = vec[:, val_sorted_index]
    tr_count = 0
    tr_index = 0
    for i, v in enumerate(val):
        tr_count += v
        if tr_count >= 0.95 * sum(val):
            tr_index = i
            break
    vec = vec[:, :tr_index]
    Z = np.dot(dataset, vec)
    return Z

if __name__ == '__main__':
    dataset, labels = load_xihua()
    norm_dataset = prehandle(dataset)
    pca_dataset = PCA(norm_dataset, 3)
    # pca_dataset = PCA_auto(norm_dataset)
    # pca = PCA(n_components=3)
    # pca.fit(norm_dataset)
    print(pca_dataset)