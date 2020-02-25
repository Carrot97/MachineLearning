import random
import numpy as np
import matplotlib.pyplot as plt

def normlization(data):
    data = np.array(data)
    feature_num = data.shape[1]
    for i in range(feature_num):
        feature_max = max(data[:, i])
        feature_min = min(data[:, i])
        data[:, i] = (data[:, i] - feature_min) / (feature_max - feature_min)
    return data

def init_vec(data, labels, vec_labels):
    index = []
    for label in vec_labels:
        while True:
            i = random.randint(0, len(data))
            if labels[i] == label and i not in index:
                index.append(i)
                break
    return data[index]

def find_NN(data, vecs):
    dist = []
    for vec in vecs:
        dist.append(np.linalg.norm(data - vec))
    return np.argsort(dist)[0]

def LVQ(data, eta):
    data = np.array(data)
    m, n = data.shape
    labels = np.zeros(m)
    labels[100:200] = 1
    vec_labels = [0, 1]
    pro_vecs = init_vec(data, labels, vec_labels)
    count = 0
    iter_times = 0
    iteration = True
    while iteration:
        print('第%d次迭代' % iter_times)
        sample_i = random.randint(0, len(data) - 1)
        vec_i = find_NN(data[sample_i, :], pro_vecs)
        if vec_labels[vec_i] == labels[sample_i]:
            vec_update = pro_vecs[vec_i, :] + eta * (data[sample_i, :] - pro_vecs[vec_i, :])
        else:
            vec_update = pro_vecs[vec_i, :] + eta * (data[sample_i, :] - pro_vecs[vec_i, :])
        if abs(vec_update[0] - pro_vecs[vec_i, 0]) < 3e-6:
            count += 1
            if count == 6:
                iteration = False
        else:
            count = 0
            pro_vecs[vec_i, :] = vec_update
        iter_times += 1
    return pro_vecs, vec_labels

def classify(dataset, pro_vecs, vec_labels):
    labels = []
    for data in dataset:
        index = find_NN(data, pro_vecs)
        labels.append(vec_labels[index])
    return labels

if __name__ == '__main__':
    dataset = np.loadtxt('dataset.txt')
    dataset = list(set(tuple(data) for data in dataset))
    dataset = normlization(dataset)
    k = 2
    eta = 1e-4
    m = len(dataset)
    pro_vecs, vec_labels = LVQ(dataset, eta)
    labels = classify(dataset, pro_vecs, vec_labels)
    class1 = np.array([dataset[i] for i in range(m) if labels[i] == 0])
    class2 = np.array([dataset[i] for i in range(m) if labels[i] == 1])
    plt.title('LVQ')
    plt.scatter(class1[:, 0], class1[:, 1], label="class1")
    plt.scatter(class2[:, 0], class2[:, 1], label="class2")
    plt.show()
