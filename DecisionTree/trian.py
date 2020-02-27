from math import log
import operator
import pandas as pd
import pickle

def calcShannonEnt(dataSet):  # 计算数据的熵(entropy)
    numEntries = len(dataSet)  # 数据条数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 每行数据的最后一个字（类别）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 统计有多少个类以及每个类的数量
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算单个类的熵值
        shannonEnt -= prob * log(prob, 2)  # 累加每个类的熵值
    return shannonEnt

def cal_Gini(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Gini = 1
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        Gini -= prob ** 2
    return Gini

def splitDataSet(dataSet,axis,value): # 按某个特征分类后的数据
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):  # 选择最优的分类特征
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)  # 原始的熵
    bestInfoGain = 0
    bestFeature = -1
    # IV = 0
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        Gini_index = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            # IV -= prob * log(prob, 2)
            newEntropy += prob * calcShannonEnt(subDataSet)  # 按特征分类后的熵
# _______________________________________________________________________________________
        # 基尼系数 CART
        #     Gini_index += prob * cal_Gini(subDataSet)
# _______________________________________________________________________________________
        # 信息增益
        infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
# ————————————————————————————————————————————————————————————————————————————————————————
        # 信息率 C4.5
        # inforate = infoGain / IV

        if (infoGain > bestInfoGain):   # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    return sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

def createTree(train_dataSet, test_dataset, labels):
    classList = [example[-1] for example in train_dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(train_dataSet[0]) == 1:
        return (majorityCnt(classList))[0][0]
    bestFeat = chooseBestFeatureToSplit(train_dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    if cal_whether_cut(myTree, train_dataSet, test_dataset, labels, bestFeat, bestFeatLabel) == 0:
        # 多变量决策
        del labels[bestFeat]
        # ————————————————————————————————————————————————————————————————————————————————
        featValues = [example[bestFeat] for example in train_dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = createTree(splitDataSet(train_dataSet, bestFeat, value),
                                                      splitDataSet(test_dataset, bestFeat, value), subLabels)
    else:
        del myTree[bestFeatLabel], labels[bestFeat]
        myTree = (majorityCnt(classList))[0][0]
        print('cut ' + str(bestFeatLabel))
    return myTree

def cal_whether_cut(tree, train_dataSet, test_dataset, labels, best_feature, best_feature_label):
    class_list = [example[-1] for example in test_dataset]
    major_num = (majorityCnt(class_list))[0][1]
    feature_accuracy = major_num / len(test_dataset)
    featValues = [example[best_feature] for example in train_dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        classList = []
        for example in train_dataSet:
            if example[best_feature] == value:
                classList.append(example[-1])
        tree[best_feature_label][value] = (majorityCnt(classList))[0][0]
    new_accuracy = test(tree, test_dataset, labels)
    print(new_accuracy, feature_accuracy)
    if new_accuracy > feature_accuracy:
        return 0
    else:
        return 1

def test(inputTree, test_data, labels):
    accurate_num = 0
    for example in test_data:
        test_vec = example[:-1]
        firstStr = next(iter(inputTree))
        secondDict = inputTree[firstStr]
        featIndex = labels.index(firstStr)
        for key in secondDict.keys():
            if test_vec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = test(secondDict[key], labels, test_vec)
                else:
                    classLabel = secondDict[key]
        if classLabel == example[-1]:
            accurate_num += 1
    return accurate_num / len(test_data)

def load_xigua_dataset(filename):
    xigua_dataset = pd.read_csv(filename)
    labels = xigua_dataset.keys().tolist()[1: -1]
    dataset = xigua_dataset.values[:, 1:].tolist()
    return dataset, labels

def discretize_dataset(dataset, labels):
    for column in range(len(dataset[0]) - 1):
        featList = [row[column] for row in dataset]
        uniqueVals = set(featList)
        if len(uniqueVals) > 3:
            # print('Feature ' + str(labels[column]) + ' discretizing...')
            breakpoint = get_breakpoint(dataset, column)
            for row in dataset:
                if row[column] >= breakpoint:
                    row[column] = 'higher than ' + str(breakpoint)
                else:
                    row[column] = 'lower than ' + str(breakpoint)
    return dataset

def get_breakpoint(dataset, column):
    sorted_dataset = sorted(dataset, key=operator.itemgetter(column))
    sorted_value = sorted(set([row[column] for row in sorted_dataset]))
    points = []
    best_Entropy = 100
    best_t = 0
    for prob in range(len(sorted_value) - 1):
        points.append((sorted_value[prob] + sorted_value[prob + 1]) / 2)
    for point in points:
        label_count = 0
        for value in sorted_value:
            if value < point:
                label_count += 1
        newEntropy = label_count / float(len(dataset)) * calcShannonEnt(sorted_dataset[:label_count]) + \
                     len(dataset) - label_count / float(len(dataset)) * calcShannonEnt(sorted_dataset[label_count:])
        if newEntropy < best_Entropy:
            best_Entropy = newEntropy
            best_t = point
    return best_t

def store_tree(tree):
    with open('decision_tree.txt', 'wb') as f:
        pickle.dump(tree, f)

train_data = 'xigua_2.0_train.csv'
test_data = 'xigua_2.0_test.csv'

if __name__ == '__main__':
    train_dataSet, labels = load_xigua_dataset(train_data)
    test_dataset, _ = load_xigua_dataset(test_data)
    original_labels = tuple(labels)
    tree = createTree(train_dataSet, test_dataset, labels)
    print(tree)
    store_tree(tree)