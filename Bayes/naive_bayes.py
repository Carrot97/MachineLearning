import numpy as np
import pandas as pd
from functools import reduce

def load_xigua_dataset(filename):
    xigua = pd.read_csv(filename)
    dataset = np.array(xigua)
    train_data = dataset[:-1, 1:-1]
    train_labels = dataset[:-1, -1]
    test_data = dataset[-1, 1:-1]
    test_labels = dataset[-1, -1]
    return train_data, train_labels, test_data, test_labels

def naive_bayes(data, labels):
    m, n = np.shape(data)
    pa = sum(labels) / m
    # p0_num = np.zeros(n)
    # p1_num = np.zeros(n)
    # p0_base = 0
    # p1_base = 0
    # ______________________
    # Laplacian correction
    p0_num = np.ones(n)
    p1_num = np.ones(n)
    p0_base = 2
    p1_base = 2
    for i in range(m):
        if labels[i] == 1:
            p1_num += data[i]
            p1_base += sum(data[i])
        else:
            p0_num += data[i]
            p0_base += sum(data[i])
    p1_a_vec = p1_num / p1_base
    p0_a_vec = p0_num / p0_base
    return p0_a_vec, p1_a_vec, pa

def classify(data, p0_a_vec, p1_a_vec, pa):
    p1_a = reduce(lambda x, y: x*y, data * p1_a_vec) * pa
    p0_a = reduce(lambda x, y: x*y, data * p0_a_vec) * (1 - pa)
    # ______________________________________________________________
    # avoid round-off error
    # p1_a = sum(np.log(data * p1_a_vec + 1e-5)) + np.log(pa)
    # p0_a = sum(np.log(data * p0_a_vec + 1e-5)) + np.log(1 - pa)

    if p1_a > p0_a:
        return 1
    else:
        return 0

# AODE
def simi_naive_bayes(data, labels):
    m, n = np.shape(data)
    pcxi = np.zeros(n)
    for j in range(n):
        index = []
        for i in range(m):
            if data[i][j] == 1:
                index.append(i)
        pcxi[j] = sum(labels[index]) / len(index)

    p1_a_vec = np.zeros((n, n))
    p0_a_vec = np.zeros((n, n))
    for j in range(n):
        p0_num = np.ones(n)
        p1_num = np.ones(n)
        p0_base = 2
        p1_base = 2
        for i in range(m):
            if data[i][j] == 1:
                if labels[i] == 1:
                    p1_num += data[i]
                    p1_base += sum(data[i])
                else:
                    p0_num += data[i]
                    p0_base += sum(data[i])
        p1_a_vec[j] = p1_num / p1_base
        p0_a_vec[j] = p0_num / p0_base
    return p0_a_vec, p1_a_vec, pcxi

def classify_simi(data, p0_a_vec, p1_a_vec, pcxi):
    p1_a = 0
    p0_a = 0
    for i in range(len(pcxi)):
        p1_a += reduce(lambda x, y: x*y, data * p1_a_vec[i]) * pcxi[i]
        p0_a += reduce(lambda x, y: x*y, data * p0_a_vec[i]) * (1 - pcxi[i])

    if p1_a > p0_a:
        return 1
    else:
        return 0

def translate(label):
    if label:
        label = '好瓜'
    else:
        label = '坏瓜'
    return label

if __name__ == '__main__':
    train_data, train_labels, test_data, test_label = load_xigua_dataset('xigua_shuju3.1.csv')

    p0_a_vec, p1_a_vec, pa = naive_bayes(train_data, train_labels)
    predict = translate(classify(test_data, p0_a_vec, p1_a_vec, pa))
    # p0_a_vec, p1_a_vec, pcxi = naive_bayes(train_data, train_labels)
    # predict = translate(classify(test_data, p0_a_vec, p1_a_vec, pcxi))

    test_label = translate(test_label)
    print('预测结果：%s  真实结果：%s' % (predict, test_label))