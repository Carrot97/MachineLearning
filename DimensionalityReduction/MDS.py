import numpy as np
# from sklearn.manifold import MDS
from sklearn.neighbors import KNeighborsClassifier as KNN
from os import listdir

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

def load_simple_mnist(addr):
    labels = []
    img_names = listdir(addr)
    m = len(img_names)
    data = np.zeros((m, 1024))
    for i, img_name in enumerate(img_names):
        label = int(img_name.split('_')[0])
        labels.append(label)
        data[i, :] = data_transform('%s/%s' % (addr, img_name))
    return data, labels

def data_transform(filename):
    transformed_img = []
    with open(filename) as f:
        for line in f:
            line = line.replace('\n', '')
            for i in line:
                transformed_img.append(int(i))
    return np.array(transformed_img)

def MDS(dataset, n):
    m, _ = np.shape(dataset)
    dists = np.zeros((m, m))
    B = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            dists[i][j] = np.linalg.norm(dataset[i] - dataset[j])
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

def MDS_auto(dataset):
    m, n = np.shape(dataset)
    dists = np.zeros((m, m))
    B = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            dists[i][j] = np.linalg.norm(dataset[i] - dataset[j])
    for i in range(m):
        for j in range(m):
            B[i][j] = - (np.square(dists[i][j]) - np.square(dists[i, :]).sum()/m -
                         np.square(dists[:, j]).sum()/m + np.square(dists.reshape(1, -1).sum()/(m**2))) / 2
    val, vec = np.linalg.eigh(B)
    val_sorted_index = []
    # threshold
    for i, v in enumerate(val):
        if v > 7:
            val_sorted_index.append(i)
    val = val[val_sorted_index]
    vec = vec[:, val_sorted_index]
    # _______________________________
    # proportion
    # val_sorted_index = np.argsort(-val)
    # val = val[val_sorted_index]
    # vec = vec[:, val_sorted_index]
    # tr_count = 0
    # tr_index = 0
    # for i, v in enumerate(val):
    #     tr_count += v
    #     if tr_count >= 0.95 * sum(val):
    #         tr_index = i
    #         break
    # val_mat = np.diag(val[:tr_index])
    # vec = vec[:, :tr_index]
    val_mat = np.diag(val)
    Z = np.dot(np.sqrt(val_mat), vec.T).T
    return Z

if __name__ == '__main__':
    dataset, labels = load_xihua()
    test_data = np.array([[2,1,1,1,2,2],
                          [3,1,1,3,3,1],
                          [1,1,2,2,2,1]])
    test_labels = np.array([2,2,2])
    dataset_dim_red = MDS(np.array(dataset), 4)
    test_data_dim_red = dataset_dim_red[-3:, :]
    knn1 = KNN(n_neighbors=3)
    knn1.fit(dataset[:-3, :], labels[:-3])
    knn2 = KNN(n_neighbors=3)
    knn2.fit(dataset_dim_red[:-3, :], labels[:-3])
    for i in range(len(test_labels)):
        predict1 = knn1.predict(test_data[i].reshape(1, -1))
        predict2 = knn2.predict(test_data_dim_red[i].reshape(1, -1))
        print('真实结果：%s  普通样本预测：%s  降维样本预测：%s' % (test_labels[i], predict1, predict2))
    # dataset = load_simple_mnist('trainingDigits')
    # mds = MDS()
    # mds.fit(dataset)
    # dataset_dim_red = MDS(dataset, 200)
    # dataset_dim_red = MDS_auto(dataset)
    # print(np.shape(dataset_dim_red))