import os
import jieba
import random
import operator
import numpy as np
from sklearn.naive_bayes import MultinomialNB

def load_data(path, n=20):
    with open(path + 'stopwords_cn.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read()
    path = path + 'Sample'
    folder_list = os.listdir(path)
    dataset = []
    labels = []
    for folder in folder_list:
        subpath = os.path.join(path, folder)
        file_list = os.listdir(subpath)
        for file in file_list:
            with open(os.path.join(subpath, file), 'r', encoding='utf-8') as f:
                txt = f.read()
            word_list = list(jieba.cut(txt, cut_all=False))
            dataset.append(word_list)
            labels.append(folder)
    # 构建所有新闻的词典
    word_dic = {}
    for word_list in dataset:
        for i, word in enumerate(word_list):
            if i > 400:
                break
            if word not in word_dic.keys():
                word_dic[word] = 0
            word_dic[word] += 1
    # 删除连词
    for word in list(word_dic.keys()):
        if word in stopwords:
            del word_dic[word]
    # 删除前20高频词
    word_dic_sorted = sorted(word_dic.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(n):
        del word_dic[word_dic_sorted[i][0]]
    # 构建词语向量
    word_vec = list(word_dic.keys())
    data_mat = np.zeros((len(dataset), len(word_vec)))
    for i, word_list in enumerate(dataset):
        for word in word_list:
            if i > 400:
                break
            if word in word_vec:
                data_mat[i][word_vec.index(word)] += 1
    # 分开训练集和测试集
    test_data = []
    test_labels = []
    for i in range(10):
        test_index = int(random.uniform(0, len(data_mat)))
        test_data.append(data_mat[test_index])
        test_labels.append(labels[test_index])
        data_mat = np.delete(data_mat, test_index, axis=0)
        del labels[test_index]
    train_data = data_mat
    train_labels = labels
    return train_data, train_labels, np.array(test_data), test_labels

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = load_data('./SogouNews/', 100)
    naive_bayes = MultinomialNB().fit(train_data, train_labels)
    print(naive_bayes.score(test_data, test_labels))