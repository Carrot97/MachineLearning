import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN

def load_simple_mnist(addr):
    labels = []
    img_names = listdir(addr)
    m = len(img_names)
    data = np.zeros((m, 1024))
    for i, img_name in enumerate(img_names):
        label = int(img_name.split('_')[0])
        labels.append(label)
        data[i, :] = data_transform('%s/%s' % (addr, img_name))
    return data, labels

def data_transform(filename):
    transformed_img = []
    with open(filename) as f:
        for line in f:
            line = line.replace('\n', '')
            for i in line:
                transformed_img.append(int(i))
    return np.array(transformed_img)

if __name__ == '__main__':
    # mnist Handwriting
    dataset, labels = load_simple_mnist('testDigits')
    testset, test_labels = load_simple_mnist('trainingDigits')
    count = 0
    knn = KNN(n_neighbors=3, algorithm='auto')
    knn.fit(dataset, labels)
    for i in range(len(test_labels)):
        predict = knn.predict(testset[i].reshape(1, -1))
        print('预测结果：%s   真实结果：%s' % (int(predict), test_labels[i]))
        if predict == test_labels[i]:
            count += 1
    print('准确率为：%s' % (float(count) / len(test_labels)))