import math
import numpy as np
import matplotlib.pyplot as plt

"""
本算法问题严重，泛用性很差
"""

class Param(object):
    def __init__(self, labeledData, labels, unlabeledData):
        self.l, self.n = labeledData.shape
        self.u = len(unlabeledData)
        self.m = self.u + self.l
        self.labeledData = np.mat(labeledData)
        self.labels = labels
        self.unlabeledData = np.mat(unlabeledData)
        self.k = len(np.unique(labels))  # 获得标签总个数

# def loadDataSet(filename):
#     """
#     加载散点数据集
#     :param filename: 文件名
#     :return: 数据，标签
#     """
#     dataSet = np.loadtxt(filename)     # 散点数据集出现问题，无法正确分类，原因未知！！！！
#     data = dataSet[:, :-1]
#     labels = dataSet[:, -1]
#     labels[labels == 2] = 0
#     return data, labels

def loadCircleData(num_data):
    """
    生成双圆环数据集
    :param num_data: 散点个数
    :return: 有标签数据及其标签， 无标签数据
    """
    center = np.array([5.0, 5.0])
    radiu_inner = 2
    radiu_outer = 4
    num_inner = num_data // 3
    num_outer = num_data - num_inner
    data = []
    theta = 0.0
    for i in range(num_inner):
        pho = (theta % 360) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radiu_inner * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radiu_inner * math.sin(pho) + np.random.rand(1) + center[1]
        data.append(tmp)
        theta += 2
    theta = 0.0
    for i in range(num_outer):
        pho = (theta % 360) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radiu_outer * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radiu_outer * math.sin(pho) + np.random.rand(1) + center[1]
        data.append(tmp)
        theta += 1
    labeledData = np.zeros((2, 2), np.float32)
    labeledData[0] = center + np.array([-radiu_inner + 0.5, 0])
    labeledData[1] = center + np.array([-radiu_outer + 0.5, 0])
    labels = [0, 1]
    unlabeledData = np.vstack(data)
    return labeledData, labels, unlabeledData

def createTransitonMat(pa, alpha, neighborNum):
    """
    生成转移矩阵P（rbf和函数和knn两种方法）
    :param pa: 训练参数
    :param alpha: 核函数参数
    :param neighborNum: 临近点个数
    :return: 转移矩阵P
    """
    allData = np.vstack((np.array(pa.labeledData), np.array(pa.unlabeledData)))
    squareDelta = np.zeros((pa.m, pa.m))
    P = np.zeros((pa.m, pa.m))
    for i in range(pa.m):
        for j in range(pa.m):
                squareDelta[i, j] = np.linalg.norm(allData[i, :] - allData[j, :])
    if neighborNum == None:
        P = np.exp(P / (-2.0 * alpha ** 2))
        rowSum = np.sum(P, axis=1)
        P /= rowSum
    else:
        for i in range(pa.m):
            neighbors = np.argsort(squareDelta[i, :])[1:neighborNum+1]
            P[i][neighbors] = 1.0 / neighborNum
    return P

def createLabelMat(pa):
    """
    生成标签矩阵F
    :param pa: 训练参数
    :return: 标签矩阵F
    """
    Fl = np.zeros((pa.l, pa.k))
    Fu = - np.ones((pa.u, pa.k))
    a = pa.labels
    for i in range(pa.l):
        Fl[i, int(pa.labels[i])] = 1.0
    F = np.vstack((Fl, Fu))
    return F

def labelPropagation(labeledData, labels, unlabeledData, iteration, alpha=None, neighborNum=None):
    """
    标签传播算法主体
    :param labeledData: 有标签数据
    :param labels: 有标签数据标签
    :param unlabeledData: 无标签数据
    :param iteration: 迭代次数
    :param alpha: 核函数参数
    :param neighborNum: 临近点个数
    :return: 无标签数据预测值
    """
    pa = Param(labeledData, labels, unlabeledData)
    P = createTransitonMat(pa, alpha, neighborNum)
    F = createLabelMat(pa)
    for i in range(iteration):
        newF = np.dot(P, F)
        newF[:pa.l, :] = F[:pa.l, :]
        delta = abs(F - newF).sum()
        F = newF
        print('iteration:%d, delta:%f' % (i, delta))
        if delta < 0.1:
            break
    print(F)
    predict = np.argmax(F[pa.l:, :], axis=1)
    return predict

# def calAccuracy(labels, predict):
#     """
#     计算分类准确率
#     :param labels: 真实标签
#     :param predict: 预测值
#     :return: 无
#     """
#     m = len(labels)
#     count = 0
#     for i in range(m):
#         if predict[i] == labels[i]:
#             count += 1
#     print('准确率为：%.3f' % (count / m))

def show(labeledData, labels, unlabeledData, predict):
    for i in range(labeledData.shape[0]):
        if int(labels[i]) == 0:
            plt.plot(labeledData[i, 0], labeledData[i, 1], 'Dr')
        elif int(labels[i]) == 1:
            plt.plot(labeledData[i, 0], labeledData[i, 1], 'Db')
        else:
            plt.plot(labeledData[i, 0], labeledData[i, 1], 'Dy')
    for i in range(unlabeledData.shape[0]):
        if int(predict[i]) == 0:
            plt.plot(unlabeledData[i, 0], unlabeledData[i, 1], 'or')
        elif int(predict[i]) == 1:
            plt.plot(unlabeledData[i, 0], unlabeledData[i, 1], 'ob')
        else:
            plt.plot(unlabeledData[i, 0], unlabeledData[i, 1], 'oy')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(0.0, 12.)
    plt.ylim(0.0, 12.)
    plt.show()

if __name__ == '__main__':
    # trainData, trainLabels = loadDataSet('scatter_with_label.txt')
    # labeledDataNum = 5
    # predict = labelPropagation(trainData, trainLabels, labeledDataNum, 400, alpha=1.5)
    labeledData, labels, unlabeledData = loadCircleData(800)
    predict = labelPropagation(labeledData, labels, unlabeledData, 400, neighborNum=10)
    show(labeledData, labels, unlabeledData, predict)

