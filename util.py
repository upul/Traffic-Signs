import numpy as np
import  tensorflow as tf

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


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

    def dataset_length(self):
        return self.dataset_length


def train_dev_split(X, y, test_size=0.01):
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
    return y_test_one_hot, y_test_one_hot


def center_normalization(X_train, X_test):
    x_train = X_train.astype(np.float32)
    x_train -= np.mean(x_train, axis=0)
    x_train /= np.std(x_train, axis=0)

    x_test = X_test.astype(np.float32)
    x_test -= np.mean(x_test, axis=0)
    x_test /= np.std(x_test, axis=0)

    return x_train, x_test


def conv2d(X, W, b, stride=1):
    X = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME')
    X = tf.nn.bias_add(X, b)
    X = tf.nn.relu(X)
    return X


def maxpool2d(X, k=2):
    return tf.nn.max_pool(
    X, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding='SAME'
    )
