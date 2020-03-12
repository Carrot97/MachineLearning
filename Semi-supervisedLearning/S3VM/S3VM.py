import numpy as np
import basicSVM
import matplotlib.pyplot as plt

class Param(object):
    def __init__(self, data, labels, labeledDataNum):
        self.m, self.n = data.shape
        self.l = labeledDataNum
        self.u = self.m - self.l
        self.labeledData = np.mat(data[:self.l, :])
        self.labels = labels[:self.l]
        self.unlabeledData = np.mat(data[self.l:, :])
        self.predictedLabels = np.zeros(self.u)
        self.Cl = 0.6
        self.Cu = self.Cl / 100

def loadDataSet(filename):
    """
    加载散点数据集
    :param filename: 文件名
    :return: 数据，标签
    """
    dataSet = np.loadtxt(filename)
    data = dataSet[:, :-1]
    labels = dataSet[:, -1]
    labels[labels == 2] = -1
    return data, labels

def predictLabel(pa, alpha0, b0):
    """
    使用SVM预测样本类型
    :param pa: 所用参数结构体
    :param alpha0: 各样本权重
    :param b0: 分割线偏置
    :return: 预测值
    """
    supVecIndex = np.nonzero(np.mat(alpha0).A > 0)[0]  # 选取支持向量索引
    supVecAlpha = alpha0[supVecIndex]
    supVecX = pa.labeledData[supVecIndex]
    supVecY = pa.labels[supVecIndex]
    predict = basicSVM.classify(pa.unlabeledData, supVecAlpha, supVecX, supVecY, b0)
    return predict

def existPair(pa, xi):
    """
    检测是否有符合标准的样本对出现（出现则对调两样本标签）
    :param pa: 所用参数结构体
    :param xi: 松弛变量
    :return: 循环控制
    """
    for i in range(pa.u):
        for j in range(pa.u):
            if pa.predictedLabels[i]*pa.predictedLabels[j] < 0 and (xi[i] > 0) and (xi[j] > 0) and (xi[i] + xi[j]) > 2:
                pa.predictedLabels[i] = - pa.predictedLabels[i]
                pa.predictedLabels[j] = - pa.predictedLabels[j]
                return True
    return False

def showClassifer(pa, w, b):
    """
    绘制样本点
    :param pa: 所用参数结构体
    :param w: 分割线权重
    :param b: 分割线偏置
    :return: 无
    """
    labeled_data_plus = []  # 带标签正样本
    labeled_data_minus = []  # 带标签负样本
    unlabeled_data_plus = []  # 不带标签正样本
    unlabeled_data_minus = []  # 不带标签负样本
    # 筛选样本
    for i in range(pa.l):
        if pa.labels[i] > 0:
            labeled_data_plus.append(pa.labeledData[i])
        else:
            labeled_data_minus.append(pa.labeledData[i])
    for i in range(pa.u):
        if pa.predictedLabels[i] > 0:
            unlabeled_data_plus.append(pa.unlabeledData[i])
        else:
            unlabeled_data_minus.append(pa.unlabeledData[i])
    # 转换为numpy矩阵
    labeled_data_plus_np = np.array(labeled_data_plus)
    labeled_data_minus_np = np.array(labeled_data_minus)
    unlabeled_data_plus_np = np.array(unlabeled_data_plus)
    unlabeled_data_minus_np = np.array(unlabeled_data_minus)
    plt.scatter(np.transpose(labeled_data_plus_np)[0], np.transpose(labeled_data_plus_np)[1], s=30, alpha=0.7)  # 正样本散点图
    plt.scatter(np.transpose(labeled_data_minus_np)[0], np.transpose(labeled_data_minus_np)[1], s=30, alpha=0.7)  # 负样本散点图
    plt.scatter(np.transpose(unlabeled_data_plus_np)[0], np.transpose(unlabeled_data_plus_np)[1], s=30, alpha=0.7)
    plt.scatter(np.transpose(unlabeled_data_minus_np)[0], np.transpose(unlabeled_data_minus_np)[1], s=30, alpha=0.7)
    # 绘制分割线
    x1 = max(np.array(pa.labeledData)[:, 0])
    x2 = min(np.array(pa.labeledData)[:, 0])
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    plt.show()

def S3VM(data, labels, labeledDataNum):
    """
    半监督支持向量机主体
    :param data: 训练样本
    :param labels: 训练标签
    :param labeledDataNum: 带标签样本个数
    :return: 未带标签样本预测值
    """
    pa = Param(data, labels, labeledDataNum)
    alpha0, w0, b0 = basicSVM.smoSimple(pa.labeledData, pa.labels, Cl=pa.Cl)  # 用有标签样本训练SVM
    pa.predictedLabels = predictLabel(pa, alpha0, b0)  # 基于训练的SVM预测未带标签样本的类型
    # showClassifer(pa, w0, b0)
    newData = np.vstack((np.array(pa.labeledData), np.array(pa.unlabeledData)))  # 合并样本
    while pa.Cu < pa.Cl:
        newLabels = np.hstack((pa.labels, pa.predictedLabels))  # 合并标签
        xi, w, b = basicSVM.smoSimple(newData, newLabels, Cl=pa.Cl, Cu=pa.Cu)
        while existPair(pa, xi):
            newLabels = np.hstack((pa.labels, pa.predictedLabels))
            xi, w, b = basicSVM.smoSimple(newData, newLabels, Cl=pa.Cl, Cu=pa.Cu)
        pa.Cu = min(2 * pa.Cu, pa.Cl)
    showClassifer(pa, w, b)
    return pa.predictedLabels

def calAccuracy(labels, predict):
    """
    计算分类准确率
    :param labels: 真实标签
    :param predict: 预测值
    :return: 无
    """
    m = len(labels)
    count = 0
    for i in range(m):
        if predict[i] == labels[i]:
            count += 1
    print('准确率为：%.3f' % (count / m))

if __name__ == '__main__':
    data, labels = loadDataSet('scatter_with_label.txt')
    l = 50  # 假设前50个数据带标签
    predict = S3VM(data, labels, l)
    calAccuracy(labels[l:], predict)