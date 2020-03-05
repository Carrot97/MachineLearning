import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def load_dataset(filename):
    """
    加载西瓜数据集
    :param filename: 文件名
    :return: 数据集，特征名
    """
    dataset = pd.read_csv(filename)
    data = dataset.values[:, 1:-1]
    labels = dataset.values[:, -1]
    feature_name = dataset.keys().tolist()
    return np.array(data), np.array(labels), feature_name[1:-1]

def sigmoid(x):
    """
    sigmoid函数
    :param x: 输入值
    :return: 1/1+e^-x
    """
    return 1.0 / (1 + np.exp(-x))

def gradient_ascent(train_data, train_labels):
    """
    逻辑回归算法主体
    :param train_data: 训练集数据
    :param train_labels: 训练集标签
    :return: 分割线权重向量
    """
    data_mat = np.mat(train_data)                    # 转换为矩阵利于之后的矩阵计算
    label_mat = np.mat(train_labels).transpose()
    alpha = 0.001  # 学习率
    beta = 0.05   # 惩罚因子
    m, n = data_mat.shape
    weights = np.ones((n, 1))  # 初始化权重为全1矩阵
    iteration = 10000          # 设置最大迭代次数
    for i in range(iteration):
        h = sigmoid(data_mat * weights)  # new_theta = theta + alpha *(X.T *(y - sigmoid(theta * X)) + beta * 正则化项)
        loss = label_mat - h
        l1 = np.sign(weights)  # 正则化项(由于|x|不可导，采用次梯度)
        new_weight = weights + 1/m * alpha * (data_mat.T * loss + beta * l1)  # 近似解
        if (abs(new_weight - weights) < 1e-5).all():  # 终止条件
            break
        weights = new_weight   # 更新权重
    return np.array(weights)

def soft_threshold1(x, t):
    """
    软判决函数1
    :param x: 输入向量
    :param t: 学习率
    :return: sign(x)*max(|x|-t, 0)
    """
    m = len(x)
    max_x = np.mat(np.zeros((m, 1)))
    for i in range(m):
        max_x[i, 0] = max(abs(x[i])-t, 0)
    return np.multiply(np.sign(x), max_x)

def soft_threshold2(x, t):
    """
    软判决函数1
    :param x: 输入向量
    :param t: 学习率
    :return: x(k+1)={ z(k)-t, t<z(k)
                      0,      |z(k)|<=t
                      z(k)+t, z(k)<-t
    """
    m = len(x)
    result = np.mat(np.zeros((m, 1)))
    for i in range(m):
        if x[i, 0] > t:
            result[i, 0] = x[i, 0] - t
        elif abs(x[i, 0]) <= t:
            result[i, 0] = 0
        else:
            result[i, 0] = x[i, 0] + t
    return result

def proximal_gradient_method(train_data, train_labels):
    """
    近端梯度下降算法
    :param train_data: 训练集数据
    :param train_labels: 训练集标签
    :return: 分割线权重向量
    """
    data_mat = np.mat(train_data)
    label_mat = np.mat(train_labels).transpose()
    t = 1        # 初始化步长
    beta = 0.5   # 步长衰减因子
    gamma = 8    # 惩罚因子
    m, n = data_mat.shape
    weights = np.ones((n, 1))
    iteration = 4000         # 设置最大迭代次数
    for i in range(iteration):
        h = sigmoid(data_mat * weights)
        loss = label_mat - h
        grad_fx = data_mat.T * loss
        z = weights - t * grad_fx  # z = x-step*grad[f(x)]

        """
        此处存在问题，fx,fz: m*1   z,weight: n*1   无法计算！！！！
        
        fz = np.multiply(label_mat, np.log(sigmoid(data_mat * z))) + np.multiply((label_mat - 1), np.log(sigmoid(data_mat * z) - 1))
        fx = np.multiply(label_mat, np.log(sigmoid(data_mat * weights))) + np.multiply((label_mat - 1), np.log(sigmoid(data_mat * weights) - 1))
        if fz <= fx + (z - weights)*grad_fx + 1/(2*t)*((z - weights)**2):
            break
        """

        # weights = soft_threshold1(z, t * gamma)  # x(k+1) = prox[z(k)]
        weights = soft_threshold2(z, t * gamma)
        t *= beta  # 缩减步长
    return weights

if __name__ == '__main__':
    train_data, train_labels, feature_name = load_dataset('xigua_shuju2.0.csv')
    clf = LogisticRegression(penalty='l1')
    clf.fit(train_data, train_labels)
    weights = proximal_gradient_method(train_data, train_labels)
    print(weights.transpose())
    print(clf.coef_)
    """
    使用sklearn中逻辑回归算法做对比，有很大差距（我的算法不会出现负数？），目前还没有验证我的算法的正确性。
    """