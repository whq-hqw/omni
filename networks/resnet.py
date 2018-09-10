import tensorflow as tf
import numpy as np
import networks.blocks as block
from datasets.load_data import get_ilsvrc_dataset
from networks.train_op import build_train_op


def evaluation(prediction, ground_truth):
    correct = tf.nn.in_top_k(predictions=prediction, targets=ground_truth, k=1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def calculate_loss(input, ground_truth):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(input, ground_truth))
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([loss] + reg_loss, name='total_loss')
    return total_loss

def build(input, ground_truth, learning_rate, architecture):
    prediction = architecture(input)
    loss = calculate_loss(prediction, ground_truth)

    global_step = tf.Variable(0, trainable=False)

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    train_op = build_train_op(loss, "adam", learning_rate, tf.trainable_variables(), global_step)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=3)

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    #summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Operation to initialize the variables.
    #sess.run(init)
    return train_op

def build_18(input):
    net = block.resnet_block(input, scope="conv1_x", filters=[64], kernel_sizes=[7])
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2)

    net = block.resnet_block(net, scope="conv2_x", filters=[64, 64], kernel_sizes=[3, 3], repeat=2)
    net = block.resnet_block(net, scope="conv3_x", filters=[128, 128], kernel_sizes=[3, 3], repeat=2)
    net = block.resnet_block(net, scope="conv4_x", filters=[256, 256], kernel_sizes=[3, 3], repeat=2)
    net = block.resnet_block(net, scope="conv5_x", filters=[512, 512], kernel_sizes=[3, 3], repeat=2)

    net = tf.layers.dense(net, 1000, activation=tf.nn.softmax, name="average_pool")
    return net

def build_34(input):
    net = block.resnet_block(input, scope="conv1_x", filters=[64], kernel_sizes=[7])
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2)

    net = block.resnet_block(net, scope="conv2_x", filters=[64, 64], kernel_sizes=[3, 3], repeat=3)
    net = block.resnet_block(net, scope="conv3_x", filters=[128, 128], kernel_sizes=[3, 3], repeat=4)
    net = block.resnet_block(net, scope="conv4_x", filters=[256, 256], kernel_sizes=[3, 3], repeat=6)
    net = block.resnet_block(net, scope="conv5_x", filters=[512, 512], kernel_sizes=[3, 3], repeat=3)

    net = tf.layers.dense(net, 1000, activation=tf.nn.softmax, name="average_pool")
    return net

def build_50(input):
    net = block.resnet_block(input, scope="conv1_x", filters=[64], kernel_sizes=[7])
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2)

    net = block.resnet_block(net, scope="conv2_x", filters=[64, 128, 256], kernel_sizes=[1, 3, 1], repeat=3)
    net = block.resnet_block(net, scope="conv3_x", filters=[128, 128, 256], kernel_sizes=[1, 3, 1], repeat=4)
    net = block.resnet_block(net, scope="conv4_x", filters=[256, 256, 1024], kernel_sizes=[1, 3, 1], repeat=6)
    net = block.resnet_block(net, scope="conv5_x", filters=[512, 512, 2048], kernel_sizes=[1, 3, 1], repeat=3)

    net = tf.layers.dense(net, 1000, activation=tf.nn.softmax, name="average_pool")
    return net

def build_101(input):
    net = block.resnet_block(input, scope="conv1_x", filters=[64], kernel_sizes=[7])
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2)

    net = block.resnet_block(net, scope="conv2_x", filters=[64, 128, 256], kernel_sizes=[1, 3, 1], repeat=3)
    net = block.resnet_block(net, scope="conv3_x", filters=[128, 128, 256], kernel_sizes=[1, 3, 1], repeat=4)
    net = block.resnet_block(net, scope="conv4_x", filters=[256, 256, 1024], kernel_sizes=[1, 3, 1], repeat=23)
    net = block.resnet_block(net, scope="conv5_x", filters=[512, 512, 2048], kernel_sizes=[1, 3, 1], repeat=3)

    net = tf.layers.dense(net, 1000, activation=tf.nn.softmax, name="average_pool")
    return net

def build_152(input):
    net = block.resnet_block(input, scope="conv1_x", filters=[64], kernel_sizes=[7])
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2)

    net = block.resnet_block(net, scope="conv2_x", filters=[64, 128, 256], kernel_sizes=[1, 3, 1], repeat=3)
    net = block.resnet_block(net, scope="conv3_x", filters=[128, 128, 256], kernel_sizes=[1, 3, 1], repeat=8)
    net = block.resnet_block(net, scope="conv4_x", filters=[256, 256, 1024], kernel_sizes=[1, 3, 1], repeat=36)
    net = block.resnet_block(net, scope="conv5_x", filters=[512, 512, 2048], kernel_sizes=[1, 3, 1], repeat=3)

    net = tf.layers.dense(net, 1000, activation=tf.nn.softmax, name="average_pool")
    return net

if __name__ == "__main__":
    dataset = get_ilsvrc_dataset(path="~/Pictures/dataset")
    

