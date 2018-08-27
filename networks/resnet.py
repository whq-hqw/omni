import tensorflow as tf
import numpy as np
import networks.blocks as block
import os

sess = tf.Session()

a = tf.placeholder(tf.float16, shape = (None, 224, 224, 3), name="img")

def resnet_50(input):
    net = block.resnet_block(input, scope="conv1_x", filters=[64], kernel_sizes=[7])
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2)

    net = block.resnet_block(net, scope="conv2_x", filters=[64, 128, 256], kernel_sizes=[1, 3, 1])
    net = block.resnet_block(net, scope="conv3_x", filters=[128, 128, 256], kernel_sizes=[1, 3, 1])
    net = block.resnet_block(net, scope="conv4_x", filters=[256, 256, 1024], kernel_sizes=[1, 3, 1])
    net = block.resnet_block(net, scope="conv5_x", filters=[512, 512, 2048], kernel_sizes=[1, 3, 1])

    net = tf.layers.dense(net, 1000, activation=tf.nn.softmax, name="average_pool")
    return net

net = resnet_50(a)
result = sess.run(net, feed_dict={a:np.random.normal(size=(1, 224, 224, 3))})

