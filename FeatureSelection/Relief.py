import numpy as np
import pandas as pd

def load_dataset(filename):
    """
    加载西瓜数据集
    :param filename: 文件名
    :return: 数据，标签，属性名
    """
    dataset = pd.read_csv(filename)
    data = dataset.values[:, 1:-1]
    labels = dataset.values[:, -1]
    feature_name = dataset.keys().tolist()
    return np.array(data), np.array(labels), feature_name[1:-1]

def get_near(data, labels):
    """
    获得各样本的邻近样本索引
    :param data: 数据
    :param labels: 标签
    :return: 邻近样本集
    """
    sum_x = np.sum(np.square(data), 1)  # 计算样本间欧氏距离
    dists = np.add(np.add(-2 * np.dot(data, data.T), sum_x).T, sum_x)  # (a-b)^2 = a^2+b^2-2ab
    dists = dists ** 0.5
    dists[dists == 0] = 100  # 自己的距离不计入后续运算
    dists_sorted = np.argsort(dists, axis=0).transpose()  # 每行数据按大小排列后的索引
    m = len(data)
    near = np.zeros((m, 2), dtype=int)  # 2列，存储猜中邻近和猜错临近的索引（注意：数据类型必须为int）
    for i in range(m):       # 找出猜中邻近和猜错临近
        for j in range(m):
            if labels[i] == labels[j]:
                near[i, 0] = dists_sorted[i, j]
                break
        for j in range(m):
            if labels[i] != labels[j]:
                near[i, 1] = dists_sorted[i, j]
                break
    return near

def diff(vec1, vec2):
    """
    计算diff（按位比较两向量是否相等，相等为0，不等为1）
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 计算结果
    """
    m = len(vec1)
    result = np.zeros(m, dtype=int)
    for i in range(m):
        if vec1[i] != vec2[i]:
            result[i] = 1
    return result

def Relief(data, labels, feature_name):
    """
    Relief算法主体
    :param data: 数据
    :param labels: 标签
    :param feature_name: 属性名
    :return: 选取的属性名
    """
    near = get_near(data, labels)
    m, n = data.shape
    delta = np.zeros(n)
    for i in range(m):
        delta += diff(data[i], data[near[i, 1]]) - diff(data[i], data[near[i, 0]])  # 计算各属性得分
    feature_name_selected = [feature_name[i] for i in range(n) if delta[i] > 0]     # 选取分数大于0的属性
    return feature_name_selected

if __name__ == '__main__':
    data, labels, feature_name = load_dataset('xigua_shuju2.0.csv')
    feature_name_selected = Relief(data, labels, feature_name)