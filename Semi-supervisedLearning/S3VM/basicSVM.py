import matplotlib.pyplot as plt
import numpy as np
import random

def loadDataSet(filename):
    """
    加载散点数据集
    :param filename: 文件名
    :return: 数据，标签
    """
    dataSet = np.loadtxt(filename)
    data = dataSet[:50, :-1]
    labels = dataSet[:50, -1]  # 取前50行作为有标签数据
    labels[labels == 2] = -1
    return data, labels

def selectJrand(i, m):
    j = i
    if i < 50:
        while (j == i):
            j = int(random.uniform(0, 49))
    else:
        while (j == i):
            j = int(random.uniform(50, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, Cl=0.6, Cu=0, toler=0.001, maxiter_num=200):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter_num_num = 0
    alphaPairsChanged = 0
    while (iter_num_num < maxiter_num):
        C = Cl
        for i in range(m):
            if i >= 50:
                C = Cu
            # 步骤1：计算误差Ei
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            # 优化alpha，更设定一定的容错率。
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 随机选择另一个与alpha_i成对优化的alpha_j
                j = selectJrand(i, m)
                # 步骤1：计算误差Ej
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                # 保存更新前的aplpha值，使用深拷贝
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 步骤2：计算上下界L和H
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    # print("L==H")
                    continue
                # 步骤3：计算eta
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    # print("eta>=0")
                    continue
                # 步骤4：更新alpha_j
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                # 步骤5：修剪alpha_j
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    # print("alpha_j变化太小")
                    continue
                # 步骤6：更新alpha_i
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 步骤7：更新b_1和b_2
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                # 步骤8：根据b_1和b_2更新b
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # 统计优化次数
                alphaPairsChanged += 1
                # 打印统计信息
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num_num, i, alphaPairsChanged))
        # 更新迭代次数
        iter_num_num += 1
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMatIn), np.array(classLabels)
    labelMat = labelMat.reshape(1, -1)
    w = np.dot((np.tile(labelMat.T, (1, 2)) * dataMat).T, alphas)
    if Cu == 0:
        return alphas, w, b
    else:
        C = Cl
        xi = np.zeros((m))
        for i in range(m):
            if i >= 50:
                C = Cu
            if alphas[i] < C:
                xi[i] = 0
            else:
                xi[i] = 1 - labelMat[0, i] * (np.dot(dataMat[i, :], w)[0] + b)
        return xi, w, b

def classify(test_data, sup_vec_alphas, sup_vec_x, sup_vec_y, b):
    sup_vec_alphas = np.mat(sup_vec_alphas)
    sup_vec_x = np.mat(sup_vec_x).transpose()
    sup_vec_y = np.mat(sup_vec_y).transpose()
    test_data = np.mat(test_data)
    m = len(test_data)
    result = []
    for i in range(m):
        predict = test_data[i, :] * sup_vec_x * np.multiply(sup_vec_y, sup_vec_alphas) + b
        result.append(np.sign(float(predict)))
    return np.array(result)