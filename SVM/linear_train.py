import matplotlib.pyplot as plt
import numpy as np
import random

class train_parameters:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn                                 # 数据矩阵
        self.labelMat = classLabels                        # 数据标签
        self.C = C                                         # 松弛变量
        self.tol = toler                                   # 容错率
        self.m = np.shape(dataMatIn)[0]                    # 数据矩阵行数
        self.alphas = np.mat(np.zeros((self.m, 1)))        # 根据矩阵行数初始化alpha参数为0
        self.b = 0                                         # 初始化b参数为0
        self.eCache = np.mat(np.zeros((self.m, 2)))        # 根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值

def load_dataset(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():                                     # 逐行读取，滤除空格等
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])      # 添加数据
        labelMat.append(float(lineArr[2]))                          # 添加标签
    return dataMat, labelMat

def cal_E(tp, i):
    return float(np.multiply(tp.alphas, tp.labelMat).T * (tp.X * tp.X[i, :].T) + tp.b) - float(tp.labelMat[i])

def select_j_rand(i, m):
    j = i                                                # 选择一个不等于i的j
    while j == i:
        j = int(random.uniform(0, m))
    return j

def select_j(i, tp, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0                                                  # 初始化
    tp.eCache[i] = [1,Ei]                                   # 根据Ei更新误差缓存
    validEcacheList = np.nonzero(tp.eCache[:,0].A)[0]       # 返回误差不为0的数据的索引值
    if (len(validEcacheList)) > 1:                          # 有不为0的误差
        for k in validEcacheList:                           # 遍历,找到最大的Ek
            if k == i:
                continue                                    # 不计算i,浪费时间
            Ek = cal_E(tp, k)                               # 计算Ek
            deltaE = abs(Ei - Ek)                           # 计算|Ei-Ek|
            if (deltaE > maxDeltaE):                        # 找到maxDeltaE
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:                                                   # 没有不为0的误差
        j = select_j_rand(i, tp.m)                          # 随机选择alpha_j的索引值
        Ej = cal_E(tp, j)                                   # 计算Ej
    return j, Ej

def update_E(tp, i):
    Ei = cal_E(tp, i)                    # 计算Ek
    tp.eCache[i] = [1, Ei]               # 更新误差缓存

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def main_process(i, tp):
    # 步骤1：计算误差Ei
    Ei = cal_E(tp, i)
    # 优化alpha,设定一定的容错率。
    if ((tp.labelMat[i] * Ei < -tp.tol) and (tp.alphas[i] < tp.C)) or ((tp.labelMat[i] * Ei > tp.tol) and (tp.alphas[i] > 0)):
        # 使用内循环启发方式2选择alpha_j,并计算Ej
        j, Ej = select_j(i, tp, Ei)
        # 保存更新前的aplpha值，使用深拷贝
        alphaIold = tp.alphas[i].copy()
        alphaJold = tp.alphas[j].copy()
        # 步骤2：计算上下界H和L
        if (tp.labelMat[i] != tp.labelMat[j]):
            L = max(0, tp.alphas[j] - tp.alphas[i])
            H = min(tp.C, tp.C + tp.alphas[j] - tp.alphas[i])
        else:
            L = max(0, tp.alphas[j] + tp.alphas[i] - tp.C)
            H = min(tp.C, tp.alphas[j] + tp.alphas[i])
        if L == H:
            # print("L==H")
            return 0
        # 步骤3：计算eta
        eta = 2.0 * tp.X[i, :] * tp.X[j, :].T - tp.X[i, :] * tp.X[i, :].T - tp.X[j, :] * tp.X[j, :].T
        if eta >= 0:
            # print("eta>=0")
            return 0
        # 步骤4：更新alpha_j
        tp.alphas[j] -= tp.labelMat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        tp.alphas[j] = clipAlpha(tp.alphas[j], H, L)
        # 更新Ej至误差缓存
        update_E(tp, j)
        if (abs(tp.alphas[j] - alphaJold) < 0.00001):
            # print("alpha_j变化太小")
            return 0
        # 步骤6：更新alpha_i
        tp.alphas[i] += tp.labelMat[j] * tp.labelMat[i] * (alphaJold - tp.alphas[j])
        # 更新Ei至误差缓存
        update_E(tp, i)
        # 步骤7：更新b1和b2
        b1 = tp.b - Ei - tp.labelMat[i] * (tp.alphas[i] - alphaIold) * tp.X[i, :] * tp.X[i, :].T - tp.labelMat[j] * \
             (tp.alphas[j] - alphaJold) * tp.X[i, :] * tp.X[j, :].T
        b2 = tp.b - Ej - tp.labelMat[i] * (tp.alphas[i] - alphaIold) * tp.X[i, :] * tp.X[j, :].T - tp.labelMat[j] * \
             (tp.alphas[j] - alphaJold) * tp.X[j, :] * tp.X[j, :].T
        # 步骤8：根据b1和b2更新b
        if (0 < tp.alphas[i]) and (tp.C > tp.alphas[i]):
            tp.b = b1
        elif (0 < tp.alphas[j]) and (tp.C > tp.alphas[j]):
            tp.b = b2
        else:
            tp.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def integrated_smo(dataMatIn, classLabels, C, toler, max_iter_num):
    tp = train_parameters(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)         # 初始化数据结构
    iter_num = 0                                                                                # 初始化当前迭代次数
    entire_set = True
    alphaPairsChanged = 0
    while (iter_num < max_iter_num) and ((alphaPairsChanged > 0) or entire_set):                # 遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
        alphaPairsChanged = 0
        if entire_set:                                                                          # 遍历整个数据集
            for i in range(tp.m):
                alphaPairsChanged += main_process(i, tp)                                              # 使用优化的SMO算法
                print("全样本遍历 第%d次迭代 样本:%d alpha优化次数:%d" % (iter_num + 1, i + 1, alphaPairsChanged))
            iter_num += 1
        else:                                                                                   # 遍历非边界值
            non_bound_index = np.nonzero((tp.alphas.A > 0) * (tp.alphas.A < C))[0]              # 遍历不在边界0和C的alpha
            for i in non_bound_index:
                alphaPairsChanged += main_process(i, tp)
                print("非边界遍历 第%d次迭代 样本:%d alpha优化次数:%d" % (iter_num + 1, i + 1, alphaPairsChanged))
            iter_num += 1
        if entire_set:                                                                          # 遍历一次后改为非边界遍历
            entire_set = False
        elif alphaPairsChanged == 0:                                                            # 如果alpha没有更新,计算全样本遍历
            entire_set = True
        print("迭代次数: %d" % iter_num)
    return tp.b, tp.alphas                                                                      # 返回SMO算法计算的b和alphas

def show_classifer(dataMat, classLabels, w, b):
    # 绘制样本点
    data_plus = []                                  # 正样本
    data_minus = []                                 # 负样本
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)            # 转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)    # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)  # 负样本散点图
    # 绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()

def get_w(alphas,dataArr,classLabels):
    x = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(x)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], x[i, :].T)
    return w

if __name__ == '__main__':
    dataArr, classLabels = load_dataset('linear_dataset.txt')
    b, alphas = integrated_smo(dataArr, classLabels, 0.6, 0.001, 20)
    w = get_w(alphas, dataArr, classLabels)
    show_classifer(dataArr, classLabels, w, b)