import tensorflow as tf
import numpy as np

def block_conv_1(input):
    with tf.variable_scope("conv1_x"):
        return tf.layers.conv2d(inputs=input, filters=64, kernel_size=7, padding="same", activation=tf.nn.relu)

def block_con2x(input, repeat=3):
    with tf.variable_scope("conv2_x"):
        tmp =  tf.layers.max_pooling2d(inputs=input, pool_size=[3,3], strides=2)
        for _ in range(repeat):
            tmp = tf.layers.conv2d(inputs=tmp, filters=64, kernel_size=1, padding="same", activation=tf.nn.relu)
            tmp = tf.layers.conv2d(inputs=tmp, filters=128, kernel_size=3, padding="same", activation=tf.nn.relu)
            tmp = tf.layers.conv2d(inputs=tmp, filters=256, kernel_size=1, padding="same", activation=tf.nn.relu)
    return tmp

def block_con3x(input, repeat=4):
    tmp = input
    with tf.variable_scope("conv3_x"):
        for _ in range(repeat):
            tmp = tf.layers.conv2d(inputs=tmp, filters=128, kernel_size=1, padding="same", activation=tf.nn.relu)
            tmp = tf.layers.conv2d(inputs=tmp, filters=128, kernel_size=3, padding="same", activation=tf.nn.relu)
            tmp = tf.layers.conv2d(inputs=tmp, filters=256, kernel_size=1, padding="same", activation=tf.nn.relu)
    return tmp

def block_con4x(input, repeat=6):
    tmp = input
    with tf.variable_scope("conv4_x"):
        for _ in range(repeat):
            tmp = tf.layers.conv2d(inputs=tmp, filters=256, kernel_size=1, padding="same", activation=tf.nn.relu)
            tmp = tf.layers.conv2d(inputs=tmp, filters=256, kernel_size=3, padding="same", activation=tf.nn.relu)
            tmp = tf.layers.conv2d(inputs=tmp, filters=1024, kernel_size=1, padding="same", activation=tf.nn.relu)
    return tmp

def block_con5x(input, repeat=3):
    tmp = input
    with tf.variable_scope("conv5_x"):
        for _ in range(repeat):
            tmp = tf.layers.conv2d(inputs=tmp, filters=512, kernel_size=1, padding="same", activation=tf.nn.relu)
            tmp = tf.layers.conv2d(inputs=tmp, filters=512, kernel_size=3, padding="same", activation=tf.nn.relu)
            tmp = tf.layers.conv2d(inputs=tmp, filters=2048, kernel_size=1, padding="same", activation=tf.nn.relu)
    return tmp


a = tf.layers.dense(inputs=1)