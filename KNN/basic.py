import numpy as np
import operator
from os import listdir

def load_dataset(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():  # 逐行读取，滤除空格等
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 添加数据
        labelMat.append(float(lineArr[2]))  # 添加标签
    return dataMat, labelMat

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

def classify(data, dataset, labels, k):
    m, _ = np.shape(dataset)
    delta_list = np.tile(data, (m, 1)) - dataset
    distances_square = (delta_list ** 2).sum(axis=1)
    sort_index = distances_square.argsort()
    class_count = {}
    for i in range(k):
        label = labels[sort_index[i]]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
    sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sort_class_count[0][0]

if __name__ == '__main__':
    # linear dataset
    dataset, labels = load_dataset('linear_trainset.txt')
    testset, test_labels = load_dataset('linear_testset.txt')
    # ____________________________________________________________
    # mnist Handwriting
    # dataset, labels = load_simple_mnist('testDigits')
    # testset, test_labels = load_simple_mnist('trainingDigits')
    count = 0
    for i in range(len(test_labels)):
        predict = classify(testset[i], dataset, labels, 3)
        if predict == test_labels[i]:
            count += 1
    print('准确率为：%s' % (float(count) / len(test_labels)))