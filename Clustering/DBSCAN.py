import random
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN

def normlization(data):
    feature_num = data.shape[1]
    for i in range(feature_num):
        feature_max = max(data[:, i])
        feature_min = min(data[:, i])
        data[:, i] = (data[:, i] - feature_min) / (feature_max - feature_min)
    return data

def get_cent_vec(data, epsilon, MinPts):
    cent_vec_index = []
    m = len(data)
    dist = np.zeros((m, m))
    for i in range(m):
        count = 0
        for j in range(m):
            dist[i, j] = np.linalg.norm(data[i] - data[j])
            if dist[i, j] <= epsilon and dist[i, j] != 0:
                count += 1
        if count >= MinPts:
            cent_vec_index.append(i)
    return cent_vec_index, dist

def DBSCAN(data, epsilon, MinPts):
    m = len(data)
    data = np.array(data)
    labels = np.zeros(m)
    cent_vec, dist = get_cent_vec(data, epsilon, MinPts)
    k = 0
    unvisit = [i for i in range(m)]
    visited = []
    while unvisit:
        p = random.choice(unvisit)
        unvisit.remove(p)
        visited.append(p)
        N = []
        if p in cent_vec:
            k += 1
            labels[p] = k
            for i, d in enumerate(dist[p]):
                if d <= epsilon and d != 0:
                    N.append(i)
            for pi in N:
                if pi in unvisit:
                    unvisit.remove(pi)
                    visited.append(pi)
                    if pi in cent_vec:
                        for i, d in enumerate(dist[pi]):
                            if d <= epsilon and d != 0:
                                N.append(i)
                    if labels[pi] == 0:
                        labels[pi] = k
    return labels

def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            dataset.append([float(line[0]), float(line[1])])
    return dataset

if __name__ == '__main__':
    # dataset = load_dataset('xigua4.0.txt')
    dataset = np.loadtxt('dataset.txt')
    dataset = list(set(tuple(data) for data in dataset))
    dataset = normlization(np.array(dataset))
    m = len(dataset)
    labels = DBSCAN(dataset, 0.11, 4)
    labels = DBSCAN(eps=0.11, min_samples=5).fit_predict(dataset)
    class1 = np.array([dataset[i] for i in range(m) if labels[i] == 1])
    class2 = np.array([dataset[i] for i in range(m) if labels[i] == 2])
    plt.title('DBSCAN')
    plt.scatter(class1[:, 0], class1[:, 1], label="class1")
    plt.scatter(class2[:, 0], class2[:, 1], label="class2")
    plt.show()

    # dataset = np.insert(dataset, 2, values=np.array(labels), axis=1)
    # np.savetxt('scatter_with_label.txt', dataset)
