# -*- coding:utf-8 -*-
# Author: Li
# Date: 2020/2/26

import numpy as np
from sklearn.model_selection import train_test_split  # 拆分数据集工具
from sklearn.datasets import load_breast_cancer  # 乳腺癌数据集

def load_dataset():
    """
    加载乳腺癌数据集
    :return: 训练、测试的数据和标签
    """
    dataset = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], test_size=0.3, random_state=0)
    for i in range(len(y_train)):  # 将标签中的0转换为-1
        if y_train[i] == 0:
            y_train[i] = -1
    for i in range(len(y_test)):
        if y_test[i] == 0:
            y_test[i] = -1
    return x_train, x_test, y_train, y_test

def stump_classify(x, i, threshold, k):
    """
    弱分类器分类
    :param x: 数据
    :param i: 最优特征
    :param threshold: 特征阈值
    :param k: 正负类划分标识
    :return: 预测值
    """
    predict = np.ones((len(x), 1))
    if k == 'lt':
        predict[x[:, i] <= threshold] = -1.0  # 如果小于阈值,则赋值为-1
    else:
        predict[x[:, i] > threshold] = -1.0  # 如果大于阈值,则赋值为-1
    return predict

def build_tree(x_trian, y_train, D):
    """
    训练弱分类器（决策树）
    :param x_trian: 训练集数据
    :param y_train: 训练集标签
    :param D: 样本权重
    :return: 弱分类器，误差，预测值
    """
    x_trian = np.mat(x_trian)
    y_train = np.mat(y_train).T
    m, n = np.shape(x_trian)
    step = 10.0
    tree = {}
    best_predict = np.mat(np.zeros((m, 1)))
    min_error = 1000.0                          
    for i in range(n):
        feature_min = x_trian[:, i].min()
        feature_max = x_trian[:, i].max()                # 找到特征中最小的值和最大值
        step_size = (feature_max - feature_min) / step   # 计算步长
        for j in range(-1, int(step) + 1):
            for k in ['lt', 'gt']:          # 大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshold = (feature_min + float(j) * step_size)    # 计算阈值
                predict = stump_classify(x_trian, i, threshold, k)  # 计算分类结果
                error_arr = np.mat(np.ones((m, 1)))                 # 初始化误差矩阵
                error_arr[predict == y_train] = 0                   # 分类正确的,赋值为0
                error = D * error_arr                               # 计算误差
                if error < min_error:                               # 找到误差最小的分类方式
                    min_error = error
                    best_predict = predict.copy()
                    tree['dim'] = i
                    tree['thresh'] = threshold
                    tree['ineq'] = k
    return tree, min_error, best_predict

def get_ensemble_error(ensemble_predict, alpha, predict, y_train):
    """
    计算集成学习器误差
    :param ensemble_predict: 集成学习器预测值
    :param alpha: 当前训练的弱分类器权重
    :param predict: 当前训练的弱分类器预测值
    :param y_train: 训练集标签
    :return: 集成学习器误差
    """
    ensemble_predict += np.multiply(alpha, predict.reshape(1, -1))            # 所有弱分类器分类结果加权和
    ensemble_error = np.multiply(np.sign(ensemble_predict), np.mat(y_train))  # 分类结果与真实值相乘，分类正确为1，分类错误为-1
    return np.array(ensemble_error)

def Adaboost(x_train, y_train, T):
    """
    Adaboost算法主干
    :param x_train: 训练集数据
    :param y_train: 训练集标签
    :param T: 训练轮数
    :return: 弱分类器
    """
    weak_classifier = []
    m = len(x_train)
    D = np.mat(np.ones(m) / m)                                          # 初始化样本权重为1/m
    ensemble_predict = np.mat(np.zeros(m))                              # 初始化集成分类器的预测结果
    for i in range(T):
        tree, error, predict = build_tree(x_train, y_train, D)          # 训练弱分类器
        # if error > 0.5:                                               # 弱分类器误差过大则舍弃
        #     continue
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-10)))  # 计算弱分类器权重
        tree['alpha'] = alpha                                           # 储存alpha的值
        weak_classifier.append(tree)                                    # 储存弱分类器
        expon = np.multiply(-1 * alpha * np.mat(y_train), predict.T)    # 更新样本权重
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        ensemble_error = get_ensemble_error(ensemble_predict, alpha, predict, y_train)  # 计算集成学习器的误差
        if (ensemble_error == 0).all():                                 # 弱误差为0则结束训练
            break
    return weak_classifier

def classify(weak_classifier, x_test, y_test):
    """
    集成分类器分类
    :param weak_classifier: 弱分类器集合
    :param x_test: 测试集数据
    :param y_test: 测试集标签
    :return: 无
    """
    x_test = np.mat(x_test)
    m = np.shape(x_test)[0]
    ensemble_predict = np.mat(np.zeros((m, 1)))
    for i in range(len(weak_classifier)):
        predict = stump_classify(x_test, weak_classifier[i]['dim'], weak_classifier[i]['thresh'], weak_classifier[i]['ineq'])
        ensemble_predict += weak_classifier[i]['alpha'] * predict
    labels = np.array(np.sign(ensemble_predict)).reshape(1, -1)[0]
    count = 0
    for i in range(m):
        if labels[i] == y_test[i]:
            count += 1
    print('测试集准确率为:%f' % (float(count) / m))

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_dataset()
    weak_classifier = Adaboost(x_train, y_train, 30)
    classify(weak_classifier, x_test, y_test)
