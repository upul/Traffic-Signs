import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import pickle

import util

NUM_CLASSES = 43

BATCH_SIZE = 128

LEARNING_RATE = 0.001

NUM_EPOCHS = 1024

FILTER_SIZE = 5

TRAINING_FILE = './data/train.p'
TESTING_FILE = './data/test.p'

REGULARIZATION_PARAM = 0.001

with open(TRAINING_FILE, mode='rb') as f:
    train = pickle.load(f)
with open(TESTING_FILE, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


def build_inference_graph(images, conv1_depth=8,  conv2_depth=8, fc1_size=1024, fc2_size = 512):
    with tf.name_scope('conv1_layer'):
        con_weight = tf.Variable(tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, 3, conv1_depth],
                                                     stddev=0.01), name='con_weight')
        con_biases = tf.Variable(tf.zeros(conv1_depth), name='con_bias')

        conv1 = util.conv2d(images, con_weight, con_biases)
        conv1 = util.maxpool2d(conv1, 2)

    with tf.name_scope('conv2_layer'):
        con2_weight = tf.Variable(tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, conv1_depth, conv2_depth],
                                                     stddev=0.01), name='con_weight')
        con2_biases = tf.Variable(tf.zeros(conv2_depth), name='con_bias')

        conv2 = util.conv2d(conv1, con2_weight, con2_biases)
        conv2 = util.maxpool2d(conv2, 2)

    with tf.name_scope('fc1_layer'):
        fc1 = flatten(conv2)
        fc1_num_unites = fc1.get_shape().as_list()[-1]
        fc1_weight = tf.Variable(tf.truncated_normal([fc1_num_unites, fc1_size], stddev=0.01),
                                name='fc1_weight')
        fc1_biases = tf.Variable(tf.zeros(fc1_size))
        fc1 = tf.matmul(fc1, fc1_weight) + fc1_biases
        fc1 = tf.nn.relu(fc1)

    with tf.name_scope('fc2_layer'):
        fc2_weight = tf.Variable(tf.truncated_normal([fc1_size, fc2_size], stddev=0.01),
                                name='fc2_weight')
        fc2_biases = tf.Variable(tf.zeros(fc2_size))
        fc2 = tf.matmul(fc1, fc2_weight) + fc2_biases
        fc2 = tf.nn.relu(fc2)

    with tf.name_scope('softmax_layer'):
        fc2_weight = tf.Variable(tf.truncated_normal([fc2_size, NUM_CLASSES], stddev=0.01),
                                 name='softmax_weight')
        fc2_biases = tf.Variable(tf.zeros(NUM_CLASSES))
        logits = tf.matmul(fc2, fc2_weight) + fc2_biases

    return logits


#X_train_centered, X_test_centered = util.center_normalization(X_train, X_test)
X_train_centered = util.min_max_normalization(X_train)
X_test_centered = util.min_max_normalization(X_test)
y_train, y_test = util.one_hot_encoding(y_train, y_test)
train_features, dev_features, train_labels, dev_labels = util.train_dev_split(X_train_centered, y_train, 0.1)

training_dataset = util.DataSet(train_features, train_labels)
dev_dataset = util.DataSet(dev_features, dev_labels)
testing_dataset = util.DataSet(X_test_centered, y_test)


def build_training_graph(logits, labels, learning_rate, regularization):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return train_op, loss, accuracy_op


small_cov_graph = tf.Graph()
with small_cov_graph.as_default():
    x = tf.placeholder("float", [None, 32, 32, 3])
    y = tf.placeholder("float", [None, NUM_CLASSES])

    logits = build_inference_graph(x, 8, 16, 1024, 512)
    train_op, loss, accuracy_op = build_training_graph(logits, y, LEARNING_RATE, REGULARIZATION_PARAM)

    init = tf.initialize_all_variables()

with tf.Session(graph=small_cov_graph) as session:
    session.run(init)
    steps_per_epoch = len(train_features) // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE

    for i in range(NUM_EPOCHS):
        training_accuracies = []
        training_losses = []
        for step in range(steps_per_epoch):
            batch_x, batch_y = training_dataset.next_batch(BATCH_SIZE)
            _, loss_val, accuracy_val = session.run([train_op, loss, accuracy_op], feed_dict={x: batch_x, y: batch_y})
            training_accuracies.append(accuracy_val)
            training_losses.append(loss_val)

        print('Epoch: {}'.format((i)))
        print('Training loss: {},  training accuracy: {}'.format(
            sum(training_losses) / len(training_losses), sum(training_accuracies) / len(training_accuracies)))

        dev_loss, dev_acc = session.run([loss, accuracy_op], feed_dict={x: dev_features, y: dev_labels})
        # dev_loss, dev_acc = eval_data(dev_dataset)
        print('Dev loss: {:5f} accuracy: {:5f}'.format(dev_loss, dev_acc))
        print('')

        test_loss, test_acc = session.run([loss, accuracy_op], feed_dict={x: X_test_centered, y: y_test})
        # test_loss, test_acc = eval_data(testing_dataset)
        print('Dev loss: {:5f} accuracy: {:5f}'.format(test_loss, test_acc))
        print('-----------------------------------\n')
