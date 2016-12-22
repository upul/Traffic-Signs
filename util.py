import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import pickle


class DataSet:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.pointer = 0
        self.dataset_length = len(y)

    def next_batch(self, size):
        next_indices = np.arange(self.pointer, self.pointer + size) % self.dataset_length
        self.pointer += size
        self.pointer = self.pointer % self.dataset_length

        return self.X[next_indices], self.y[next_indices]

    def length(self):
        return self.dataset_length


def train_dev_split(X, y, test_size=0.1):
    train_features, dev_features, train_labels, dev_labels = train_test_split(X,
                                                                              y,
                                                                              test_size=test_size,
                                                                              random_state=1024)
    return train_features, dev_features, train_labels, dev_labels


def one_hot_encoding(y_train, y_test):
    labelBinarizer = LabelBinarizer()
    labelBinarizer.fit(y_train)

    y_train_one_hot = labelBinarizer.transform(y_train)
    y_test_one_hot = labelBinarizer.transform(y_test)
    return y_train_one_hot, y_test_one_hot


def center_normalization(X_train, X_test):
    x_train = X_train.astype(np.float32)
    x_train -= np.mean(x_train, axis=0)
    x_train /= np.std(x_train, axis=0)

    x_test = X_test.astype(np.float32)
    x_test -= np.mean(x_test, axis=0)
    x_test /= np.std(x_test, axis=0)

    return x_train, x_test


def min_max_normalization(data, a=-0.5, b=0.5):
    data_max = np.max(data)
    data_min = np.min(data)
    return a + (b - a) * ((data - data_min) / (data_max - data_min))


def conv2d(X, W, b, stride=1):
    X = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME')
    X = tf.nn.bias_add(X, b)
    X = tf.nn.relu(X)
    return X


def maxpool2d(X, k=2):
    return tf.nn.max_pool(
        X, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME'
    )


def plot_learning_curves(training_losses, training_accuracies, dev_losses, dev_accuracies):
    import seaborn as sbs;
    sbs.set()
    epochs = np.arange(len(training_losses))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, training_losses, color='#dd1c77', linewidth=2.0, label='training')
    plt.plot(epochs, dev_losses, color='#c994c7', linewidth=2.0, label='dev')

    # plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, training_accuracies, color='#dd1c77', linewidth=2.0, label='training')
    plt.plot(epochs, dev_accuracies, color='#c994c7', linewidth=2.0, label='dev')

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.savefig('learning_curves.jpg')
    plt.show()


def transform_image(img, ang_range, shear_range, trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    shear_M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))

    return img


if __name__ == '__main__':
    #a = np.random.random_sample(100)
    #b = np.random.random_sample(100)
    #c = np.random.random_sample(100)
    #d = np.random.random_sample(100)

    #plot_learning_curves(a, b, c, d)

    TRAINING_FILE = './data/train.p'

    TESTING_FILE = './data/test.p'

    with open(TRAINING_FILE, mode='rb') as f:
        train = pickle.load(f)
    with open(TESTING_FILE, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']

    X_train_transformed = np.zeros_like(X_train)
    y_train_transformed = np.zeros_like(y_train)
    for i in range(X_train_transformed.shape[0]): #
        X_train_transformed[i] = transform_image(X_train[i], 20, 10, 5)
        y_train_transformed[i] = y_train[i]

    X_train_transformed.dump('augmented_data_x.hkl')
    y_train_transformed.dump('augmented_data_y.hkl')

    x = np.load('augmented_data_x.hkl')
    y = np.load('augmented_data_y.hkl')
    print(x.shape)
    print(y.shape)



