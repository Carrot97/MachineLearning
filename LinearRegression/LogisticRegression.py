import numpy as np
import matplotlib.pyplot as plt

def load_dataset(filename):
    """
    加载散点数据集
    :param filename: 文件名
    :return: 数据，标签
    """
    dataset = np.loadtxt(filename)
    data = dataset[:, :-1]
    data = np.insert(data, 2, 1, axis=1)  # 加入b偏置列
    labels = dataset[:, -1]
    labels[labels == 2] = 0  # 正类为1，负类为0
    return data, labels

def sigmoid(x):
    """
    sigmoid函数
    :param x: 输入值
    :return: 1/1+e^-x
    """
    return 1.0 / (1 + np.exp(-x))

def logistic_regression(train_data, train_labels):
    """
    逻辑回归算法主体
    :param train_data: 训练集数据
    :param train_labels: 训练集标签
    :return: 分割线权重矩阵
    """
    data_mat = np.mat(train_data)                    # 转换为矩阵利于之后的矩阵计算
    label_mat = np.mat(train_labels).transpose()
    alpha = 0.001  # 学习率
    n = data_mat.shape[1]
    weights = np.ones((n, 1))  # 初始化权重为全1矩阵
    iteration = 10000  # 设置最大迭代次数
    for i in range(iteration):
        print(i)
        h = sigmoid(data_mat * weights)  # new_theta = theta + alpha * X.T *(y - sigmoid(theta * X))
        loss = label_mat - h
        new_weight = weights + alpha * data_mat.T * loss
        if (abs(new_weight - weights) < 1e-3).all():  # 终止条件
            break
        weights = new_weight  # 更新权重
    return np.array(weights)

if __name__ == '__main__':
    train_data, train_labels = load_dataset('scatter_with_label.txt')  # 使用聚类中DBSCAN算法打标签的数据集
    weights = logistic_regression(train_data, train_labels)
    m = len(train_data)
    class1 = np.array([train_data[i] for i in range(m) if train_labels[i] == 1])
    class2 = np.array([train_data[i] for i in range(m) if train_labels[i] == 0])
    plt.title('logistic_regression')
    plt.scatter(class1[:, 0], class1[:, 1], label="class1")
    plt.scatter(class2[:, 0], class2[:, 1], label="class2")
    x = np.arange(0, 1, 0.2)
    y = -(weights[0] * x + weights[2]) / weights[1]
    plt.plot(x, y, 'purple')
    plt.show()