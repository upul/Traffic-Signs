"""
My final model for the German Traffic Sign Classification System
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
import pickle

import util
import numpy as np

NUM_CLASSES = 43

LEARNING_RATE = 1e-3 #looks like too much

EPOCHS = 120

BATCH_SIZE = 128

TRAINING_FILE = './data/train.p'

TESTING_FILE = './data/test.p'

REGULARIZATION_PARAM = 1e-6

MODEL_NAME = './checkpoint_64_64_128_1024_1024/LeNetPlusPlus_64_64_128_1024_1024.ckpt'


def LeNetPlusPlus(x):

    conv1_w = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 64), stddev=0.01))
    conv1_b = tf.Variable(tf.zeros(64))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv2_w = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), stddev=0.01))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv3_w = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), stddev=0.01))
    conv3_b = tf.Variable(tf.zeros(128))
    conv3 = tf.nn.conv2d(conv2, conv3_w, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    fc1 = flatten(conv3)
    fc1_shape = (fc1.get_shape().as_list()[-1], 1024)
    fc1_w = tf.Variable(tf.truncated_normal(shape=(fc1_shape), stddev=0.01))
    fc1_b = tf.Variable(tf.zeros(1024))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    fc2_w = tf.Variable(tf.truncated_normal(shape=(1024, 1024), stddev=0.01))
    fc2_b = tf.Variable(tf.zeros(1024))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    fc3_w = tf.Variable(tf.truncated_normal(shape=(1024, 43), stddev=0.01))
    fc3_b = tf.Variable(tf.zeros(43))
    return (tf.matmul(fc2, fc3_w) + fc3_b), [conv1_w, conv2_w, fc1_w, fc2_w, fc3_w]


def regularization_cost(weights, regularization):
    reg_cost = 0.0
    for weight in weights:
        reg_cost += regularization * tf.nn.l2_loss(weight)
    return reg_cost


# consists of 32x32x3, color images
x = tf.placeholder("float", [None, 32, 32, 3])
tf.add_to_collection("x", x)  # Remember this Op.

# Classify over 43 traffic signs
y = tf.placeholder("float", [None, 43])
tf.add_to_collection("y", y) 

keep_prob = tf.placeholder(tf.float32)
tf.add_to_collection("keep_prob", keep_prob) 

logit, weights = LeNetPlusPlus(x)
tf.add_to_collection("logit", logit)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logit, y)) + \
          regularization_cost(weights, REGULARIZATION_PARAM)
tf.add_to_collection('loss_op', loss_op)

opt = tf.train.AdamOptimizer(LEARNING_RATE)
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.add_to_collection('accuracy_op', accuracy_op)


def eval_data(dataset):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    steps_per_epoch = dataset.length() // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    for step in range(steps_per_epoch):
        batch_x, batch_y = dataset.next_batch(BATCH_SIZE)
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss / num_examples, total_acc / num_examples


if __name__ == '__main__':

    with open(TRAINING_FILE, mode='rb') as f:
        train = pickle.load(f)
    with open(TESTING_FILE, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']

    X_train_transformed = np.zeros_like(X_train)
    y_train_transformed = np.zeros_like(y_train)
    for i in range(X_train_transformed.shape[0]):
        X_train_transformed[i] = util.transform_image(X_train[i], 20, 10, 5)
        y_train_transformed[i] = y_train[i]

    X_train = np.vstack((X_train, X_train_transformed))
    y_train = np.hstack((y_train, y_train_transformed))
    y_train = y_train.astype(int)

    X_train_centered = util.min_max_normalization(X_train)
    X_test_centered = util.min_max_normalization(X_test)
    y_train, y_test = util.one_hot_encoding(y_train, y_test)
    train_features, dev_features, train_labels, dev_labels = util.train_dev_split(X_train_centered, y_train, 0.1)

    training_dataset = util.DataSet(train_features, train_labels)
    dev_dataset = util.DataSet(dev_features, dev_labels)
    testing_dataset = util.DataSet(X_test_centered, y_test)

    saver = tf.train.Saver()
    best_dev_acc = 1e-10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        steps_per_epoch = len(train_features) // BATCH_SIZE
        num_examples = steps_per_epoch * BATCH_SIZE

        training_accuracies = []
        dev_accuracies = []

        training_losses = []
        dev_losses = []

        # Train the model
        print('Model building starts...')
        for epoch in range(EPOCHS):
            for step in range(steps_per_epoch):
                batch_x, batch_y = training_dataset.next_batch(BATCH_SIZE) 
                loss, accur = sess.run([train_op, accuracy_op],
                                   feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        
        
            # training loss and accuracy after an epoch
            loss_tr, acc_tr = sess.run([loss_op, accuracy_op], 
                                   feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})        
            training_losses.append(loss_tr)
            training_accuracies.append(acc_tr)

            loss_dev, acc_dev = eval_data(dev_dataset)
            dev_accuracies.append(acc_dev)
            dev_losses.append(loss_dev)

            if acc_dev > best_dev_acc:                
                saver.save(sess, MODEL_NAME, global_step=step)
                print('current dev acc: {:.4f} best dev acc: {:.4f} hence save this model.'.format(acc_dev, best_dev_acc))
                best_dev_acc = acc_dev
        
            print('Epoch:{:d} train loss:{:.4f} dev loss:{:.4f} train accu:{:.4f} dev accu:{:.4f}'.format(
                    epoch, loss_tr, loss_dev, acc_tr, acc_dev))  

            loss_test, acc_test = eval_data(testing_dataset)  
            print('testing loss:{:.4f} and testing acc: {:4f}'.format(loss_test, acc_test))  
    
        util.plot_learning_curves(training_losses, training_accuracies, dev_losses, dev_accuracies)