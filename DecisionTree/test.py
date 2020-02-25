import pandas as pd
import pickle

def load_test_dataset():
    xigua_dataset = pd.read_csv('xigua_2.0_test.csv')
    labels = xigua_dataset.keys().tolist()[1: -1]
    dataset = xigua_dataset.values[:, 1:].tolist()
    return dataset, labels

def load_tree():
    with open('decision_tree.txt', 'rb') as f:
        return pickle.load(f)

def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

if __name__ == '__main__':
    dataSet, labels = load_test_dataset()
    tree = load_tree()
    accurate_num = 0
    for test_vec in dataSet:
        result = classify(tree, labels, test_vec[:-1])
        print(result)
        if result == test_vec[-1]:
            accurate_num += 1
    print(accurate_num / len(dataSet))