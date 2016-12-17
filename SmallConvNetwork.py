import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import pickle

from util import conv2d, maxpool2d

NUM_CLASSES = 43

BATCH_SIZE = 128

LEARNING_RATE = 0.001

NUM_EPOCHS = 1024

FILTER_SIZE = 5

TRAINING_FILE = './data/train.p'
TESTING_FILE  = './data/test.p'

with open(TRAINING_FILE, mode='rb') as f:
    train = pickle.load(f)
with open(TESTING_FILE, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


def build_inference_graph(images, conv_depth = 8, fc_size = 256):
    with tf.name_scope('conv_layer'):
        con_weight = tf.Variable(tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, 3, conv_depth],
                                                     stddev=0.01), name = 'con_weight')
        con_biases = tf.Variable(tf.zeros(FILTER_SIZE), name = 'con_bias')

        conv = conv2d(images, con_weight, con_biases)

    with tf.name_scope('fc_layer'):
        fc = flatten(conv)
        fc_num_unites = fc.get_shape().as_list()[-1]
        fc_weight = tf.Variable(tf.truncated_normal([fc_num_unites, NUM_CLASSES], stddev = 0.01),
                                name='fc_weight')
        fc_biases = tf.Variable(tf.zeros(NUM_CLASSES))

        logits = tf.matmul(fc, fc_weight) + fc_biases

    return logits


