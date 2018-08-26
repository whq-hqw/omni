import tensorflow as tf
import numpy as np
import os

sess = tf.Session()

a = tf.placeholder(tf.float16, shape = (None, 224, 224, 3), name="img")

def resnet(input, layers = 50):
    NET_DEF = {50:[3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}

    with tf.variable_scope("conv1_x"):
        conv = tf.layers.conv2d(inputs=input, filters=64, kernel_size=7, padding="same", activation=tf.nn.relu)
        net = tf.layers.batch_normalization(conv)

    with tf.variable_scope("conv2_x"):
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2)
        for _ in range(NET_DEF[layers][0]):
            conv = tf.layers.conv2d(inputs=net, filters=64, kernel_size=1, padding="same", activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
            conv = tf.layers.conv2d(inputs=conv, filters=128, kernel_size=3, padding="same", activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
            conv = tf.layers.conv2d(inputs=conv, filters=256, kernel_size=1, padding="same", activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
            net = tf.add(conv, net)

    with tf.variable_scope("conv3_x"):
        for _ in range(NET_DEF[layers][1]):
            conv = tf.layers.conv2d(inputs=net, filters=128, kernel_size=1, padding="same", activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
            conv = tf.layers.conv2d(inputs=conv, filters=128, kernel_size=3, padding="same", activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
            conv = tf.layers.conv2d(inputs=conv, filters=256, kernel_size=1, padding="same", activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
            net = tf.add(conv, net)

    with tf.variable_scope("conv4_x"):
        for _ in range(NET_DEF[layers][2]):
            conv = tf.layers.conv2d(inputs=net, filters=256, kernel_size=1, padding="same", activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
            conv = tf.layers.conv2d(inputs=conv, filters=256, kernel_size=3, padding="same", activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
            conv = tf.layers.conv2d(inputs=conv, filters=1024, kernel_size=1, padding="same", activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
            net = tf.add(conv, net)

    with tf.variable_scope("conv5_x"):
        for _ in range(NET_DEF[layers][3]):
            conv = tf.layers.conv2d(inputs=net, filters=512, kernel_size=1, padding="same", activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
            conv = tf.layers.conv2d(inputs=conv, filters=512, kernel_size=3, padding="same", activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
            conv = tf.layers.conv2d(inputs=conv, filters=2048, kernel_size=1, padding="same", activation=tf.nn.relu)
            conv = tf.layers.batch_normalization(conv)
            net = tf.add(conv, net)

    with tf.variable_scope("average_pool"):
        net = tf.layers.dense(net, 1000, activation=tf.nn.softmax)
    return net


net = resnet(a)
result = sess.run(net, feed_dict={a:np.random.normal(size=(1, 224, 224, 3))})

