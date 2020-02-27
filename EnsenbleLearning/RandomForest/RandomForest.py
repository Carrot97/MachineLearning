# -*- coding:utf-8 -*-
# Author: Li
# Date: 2020/2/27

import random
import operator
import numpy as np
import pandas as pd
import TreeofForest

def load_dataset(filename):
    """
    加载数据集
    :param filename: 待加载文件名
    :return: 训练集，测试集, 属性名
    """
    training_set_name = filename + 'train.csv'
    test_set_name = filename + 'test.csv'
    training_set = pd.read_csv(training_set_name)
    test_set = pd.read_csv(test_set_name)
    feature_name = training_set.keys().tolist()[1:]  # 提取数据集中各属性的名称
    return training_set.values[:, 1:].tolist(), test_set.values[:, 1:].tolist(), feature_name[:-1]  # 返回形式为列表，删除序号列

def create_training_set(dataset):
    """
    以自主采样法创建新训练集
    :param dataset: 原数据集
    :return: 新训练集
    """
    train_set = []
    for i in range(len(dataset)):
        train_set.append(random.choice(dataset))  # 每次从原数据集中随机选择一个数据加入新训练集，直至新训练集与原数据集规模一致
    return train_set

def random_forest(dataset, feature_name, T):
    """
    随机森林算法主干
    :param dataset: 训练集（数据+标签）
    :param feature_name: 训练集各属性名称
    :param T: 训练轮数
    :return: T个决策树的集合
    """
    forest = []
    for i in range(T):
        train_set = create_training_set(dataset)  # 创建一组新的训练集
        tree = TreeofForest.createTree(train_set, feature_name)  # 调用之前写过的决策树函数
        forest.append(tree)  # 将生成的决策树加入森林集合
    return forest

def classify(forest, feature_name, test_set):
    """
    集成学习器分类函数
    :param forest: 决策树集合
    :param feature_name: 测试集各属性名称
    :param test_set: 测试集（数据+标签）
    :return: 无
    """
    ensemble_predict = []  # 初始化集成学习器预测值
    data_num = len(test_set)
    tree_num = len(forest)
    predict = [[] for i in range(tree_num)]  # 初始化基学习器预测值
    for i, tree in enumerate(forest):        # 遍历所有基学习器
        for test_vec in test_set:            # 遍历所有测试集数据
            predict[i].append(TreeofForest.classify(tree, feature_name, test_vec))  # 计算预测值
    predict = np.array(predict)
    for i in range(data_num):
        count = {}  # 初始化计数器
        for j in range(tree_num):
            if predict[j, i] not in count.keys():        # 若计数器中无此标签则创建该标签
                count[predict[j, i]] = 0
            count[predict[j, i]] += 1
        sorted_count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)  # 按照各标签出现次数排序，sorted_count[0][0]即为集成学习器预测值
        if sorted_count[0][0] == 'None':                 # 可能出现基分类器无法分类的情况
            ensemble_predict.append(sorted_count[1][0])  # 若无法分类的情况占据第一的位置则输出第二位置上的标签
        else:
            ensemble_predict.append(sorted_count[0][0])
    print(ensemble_predict)
    count = 0
    for i, test_vec in enumerate(test_set):
        if ensemble_predict[i] == test_vec[-1]:          # 比较预测值与真实值
            count += 1
    print('集成学习器准确率为:%.3f' % (float(count) / data_num))

if __name__ == '__main__':
    training_set, test_set, feature_name = load_dataset('xigua_2.0_')
    forest = random_forest(training_set, feature_name, 5)
    classify(forest, feature_name, test_set)