import tensorflow as tf
import numpy as np

def block_conv_1(input):
    with tf.variable_scope():
        return tf.layers.conv2d(inputs=input, filters=64, kernel_size=7, padding="same", activation=tf.nn.relu)

    