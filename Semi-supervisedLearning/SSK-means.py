import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    """
    加载数据集
    :param filename: 文件名
    :return: 数据坐标，M必连点对，C勿连点对
    """
    dataSet = np.loadtxt(filename, delimiter='\t')
    M = [[3, 24], [24, 3], [11, 19], [19, 11], [13, 16], [16, 13]]
    C = [[1, 20], [20, 1], [12, 22], [22, 12], [18, 22], [22, 18]]
    return dataSet, M, C

def init(data, k):
    """
    随机选取k个点
    :param data: 数据集
    :param k: 聚类中心个数
    :return: 随机的k个点
    """
    np.random.shuffle(data)  # 打乱数组顺序（使用random.shuffle会出现问题）
    return np.array(data[:k])

def checkConstraint(i, newLabel, labels, M, C):
    """
    检查是否违反约束条件
    :param i: 当前点索引
    :param newLabel: 当前点新预测值
    :param labels: 所有点的预测值
    :param M: 必连点对
    :param C: 勿连点对
    :return: 是否违反
    """
    for m in M:
        if i == m[0] and labels[m[1]] != -1 and newLabel != labels[m[1]]:  # 注意第二个条件
            return True
    for c in C:
        if i == c[0] and newLabel == labels[c[1]]:
            return True
    return False

def SSKmeans(data, M, C, kNum, iteration):
    """
    SSKmeans算法主体
    :param data: 数据集
    :param M: 必连点对
    :param C: 勿连点对
    :param kNum: 聚类中心个数
    :param iteration: 迭代次数
    :return: 预测值
    """
    m = len(data)
    mu = init(data, kNum)
    for iter in range(iteration):
        print('第%d次迭代：' % (iter+1))
        labels = - np.ones((m, 1))  # 初始化标签为全-1
        for i in range(m):          # 对各点分开计算
            dist = []
            for mu_j in mu:
                dist.append(np.linalg.norm(data[i, :] - mu_j))  # 计算该点到各聚类中的距离
            K = [k for k in range(kNum)]
            isMerged = False
            indexCount = 0   # 设置计数器（若最近中心违反条件则自增1，即选取第二近中心）
            while not isMerged:
                newLabel = np.argsort(dist)[indexCount]      # 选择下标在K中的距离最近的聚类中心
                isVoulated = checkConstraint(i, newLabel, labels, M, C)  # 检查是否违反规则
                if not isVoulated:
                    labels[i] = newLabel
                    isMerged = True
                else:
                    indexCount += 1
                    if indexCount >= kNum:  # 当计数器大于等于中心个数时跳出循环
                        print("data[%d]无满足约束条件组合" % (i+1))
                        break
        # 更新聚类中心向量
        for i in range(kNum):
            labels_i = np.array([data[j] for j in range(m) if labels[j] == i])
            mu[i, :] = (np.sum(labels_i, axis=0, keepdims=True)) / len(labels_i)
    return labels

def show(data, labels, k, M, C):
    """
    画图
    """
    M = M[:, 0]
    C = C[:, 0]
    m = len(data)
    for i in range(k):
        labels_i = np.array([data[j] for j in range(m) if labels[j] == i])
        plt.scatter(labels_i[:, 0], labels_i[:, 1])
    # 检查勿连必连点对情况
    # plt.scatter((dataSet[M])[:, 0], (dataSet[M])[:, 1], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    # plt.scatter((dataSet[C])[:, 0], (dataSet[C])[:, 1], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='green')
    plt.show()

if __name__ == '__main__':
    dataSet, M, C = loadDataSet('xigua4.0.txt')
    labels = SSKmeans(dataSet, M, C, 3, 100)
    show(dataSet, labels, 3, np.array(M), np.array(C))