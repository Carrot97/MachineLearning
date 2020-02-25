import numpy as np
from os import listdir
from sklearn.svm import SVC

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
    train_data, train_labels = load_simple_mnist('trainingDigits')
    svm = SVC(C=100, kernel='rbf')
    svm.fit(train_data, train_labels)
    test_data, test_labels = load_simple_mnist('testDigits')
    Count = 0
    m, n = np.shape(test_data)
    for i in range(m):
        result = svm.predict(test_data[i, :].reshape(1, -1))
        if result == test_labels[i]:
            Count += 1
    print("测试集准确率: %s" % (float(Count) / m))