import numpy as np
from os import listdir

def kernel_cal(X, X_exe, sigma):
    m, _ = np.shape(X)
    k = np.mat(np.zeros((m, 1)))
    for j in range(m):
        delta = X[j, :] - X_exe
        k[j] = delta * delta.T
    k = np.exp(k/(-2 * (sigma ** 2)))
    return k

def load_dataset(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def load_simple_mnist(addr):
    labels = []
    img_names = listdir(addr)
    m = len(img_names)
    data = np.zeros((m, 1024))
    for i, img_name in enumerate(img_names):
        label = int(img_name.split('_')[0])
        if label == 9:
            labels.append(-1)
        else:
            labels.append(1)
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

def load_sup_vec_pm(filename):
    sup_vec_alphas = []
    sup_vec_y = []
    sup_vec_x_uncut = []
    f = open(filename)
    for line in f.readlines():
        line = line.replace('[', '').replace(']', '').strip().split('\t')
        sup_vec_x_uncut.append(line[0])
        try:
            sup_vec_y.append(float(line[1]))
        except:
            break
        sup_vec_alphas.append(float(line[2]))
    b = sup_vec_x_uncut[-1]
    sup_vec_x_uncut = sup_vec_x_uncut[:-1]
    m = len(sup_vec_x_uncut)
    sup_vec_x = np.zeros((m, 1024))
    for line in sup_vec_x_uncut:
        line = line.replace('   ', '  ').replace('  ', ' ').split(' ')
        print(line)
        for i in range(1024):
            sup_vec_x.append([float(line[0]), float(line[1])])
    return np.array(sup_vec_x), np.array(sup_vec_y), np.array(sup_vec_alphas), float(b)

def classify(test_data, test_labels, sup_vec_alphas, sup_vec_x, sup_vec_y, b):
    sup_vec_alphas = np.mat(sup_vec_alphas).transpose()
    sup_vec_x = np.mat(sup_vec_x)
    sup_vec_y = np.mat(sup_vec_y).transpose()
    test_data = np.mat(test_data)
    test_labels = np.mat(test_labels).transpose()
    count = 0
    m, _ = np.shape(test_data)
    for i in range(m):
        kernel_val = kernel_cal(sup_vec_x, test_data[i, :], 1.3)
        predict = kernel_val.T * np.multiply(sup_vec_y, sup_vec_alphas) + b
        if np.sign(predict) == np.sign(test_labels[i]):
            count += 1
    print("测试集准确率: %.2f%%" % ((float(count) / m) * 100))

if __name__ == '__main__':
    test_data, test_labels = load_dataset('nonlinear_testset.txt')
    sup_vec_x, sup_vec_y, sup_vec_alphas, b = load_sup_vec_pm('sup_vec_pm.txt')
    classify(test_data, test_labels, sup_vec_alphas, sup_vec_x, sup_vec_y, b)