import re
import random
import numpy as np

def text_split(emails):
    word_list = re.split(r'\W+', emails)
    return [word.lower() for word in word_list if len(word) >= 2]

def get_voca_vec(word_mat):
    voca_vec = set([])
    for word_list in word_mat:
        voca_vec = voca_vec | set(word_list)
    return list(voca_vec)

def vec_transform(voca_vec, word_list):
    m = len(voca_vec)
    word_vec = np.zeros(m)
    for word in word_list:
        if word in voca_vec:
            word_vec[voca_vec.index(word)] += 1
    return word_vec

def naive_bayes(data, labels):
    m, n = np.shape(data)
    pa = sum(labels) / m
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
    p1_a = sum(np.log(data * p1_a_vec + 1e-5)) + np.log(pa)
    p0_a = sum(np.log(data * p0_a_vec + 1e-5)) + np.log(1 - pa)

    if p1_a > p0_a:
        return 1
    else:
        return 0

if __name__ == '__main__':
    word_mat = []
    label_list = []
    for i in range(1, 26):
        word_list = text_split(open('email/spam/%d.txt' % i, 'r').read())
        word_mat.append(word_list)
        label_list.append(1)
        word_list = text_split(open('email/ham/%d.txt' % i, 'r').read())
        word_mat.append(word_list)
        label_list.append(0)
    voca_vec = get_voca_vec(word_mat)
    word_vec_mat = []
    for word_list in word_mat:
        word_vec_mat.append(vec_transform(voca_vec, word_list))
    test_data = []
    test_labels = []
    for i in range(10):
        test_index = int(random.uniform(0, len(word_vec_mat)))
        test_data.append(word_vec_mat[test_index])
        test_labels.append(label_list[test_index])
        del word_vec_mat[test_index], label_list[test_index]
    train_data = word_vec_mat
    train_labels = label_list
    p0_a_vec, p1_a_vec, pa = naive_bayes(train_data, train_labels)
    count = 0
    for i, data in enumerate(test_data):
        predict = classify(data, p0_a_vec, p1_a_vec, pa)
        if predict == test_labels[i]:
            count += 1
    print('正确率为：%f' % (count / len(test_data)))