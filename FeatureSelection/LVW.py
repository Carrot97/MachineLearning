from sklearn.model_selection import cross_val_score
from sklearn import svm
import pandas as pd
import numpy as np
import random

def load_dataset(filename):
    """
    加载西瓜数据集
    :param filename: 文件名
    :return: 数据集，特征名
    """
    dataset = pd.read_csv(filename)
    data = dataset.values[:, 1:]
    feature_name = dataset.keys().tolist()
    return np.array(data), feature_name[1:-1]

def get_random_feature(dataset, feature):
    """
    随机选取特征子集
    :param dataset: 数据集
    :param feature: 特征索引
    :return: 特征子集，包含特征子集的数据集，特征子集特征数量
    """
    new_dataset = dataset.copy()  # 深拷贝
    new_d = random.randint(2, 5)  # 随机产生特征数量
    random_feature = sorted(random.sample(feature, new_d))  # 随机产生new_d个特征的索引，并按大小排序
    new_dataset = new_dataset[:, random_feature]            # 提取包含特征子集的数据集
    return random_feature, new_dataset, new_d

def LVW(dataset, feature_name, T):
    """
    LVW算法主体
    :param dataset: 数据集
    :param feature_name: 特征名
    :param T: 终止条件控制参数
    :return: 特征子集
    """
    best_accuracy = 0                   # 初始化参数
    d = len(feature_name)
    feature = [i for i in range(d)]
    best_feature = feature
    iteration = 0
    while iteration < T:
        random_feature, new_dataset, new_d = get_random_feature(dataset, feature)
        clf = svm.SVC(kernel='rbf', C=1, gamma='auto')                               # 使用支持向量机作为分类器
        accuracy = sum(cross_val_score(clf, dataset[:, :-1], dataset[:, -1], cv=8))  # 采用交叉验证法
        if accuracy > best_accuracy or (accuracy == best_accuracy and new_d < d):    # 若新特征子集分类准确高（或准确率相等但数量少）则替换
            iteration = 0
            best_accuracy = accuracy
            d = new_d
            best_feature = random_feature
        else:
            iteration += 1
    best_feature_name = np.array(feature_name)[best_feature]  # 将索引替换为名称
    return best_feature_name

if __name__ == '__main__':
    dataset, feature_name = load_dataset('xigua_shuju2.0.csv')
    feature_name_selected = LVW(dataset, feature_name, 50)
    print(feature_name_selected)