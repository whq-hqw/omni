#coding=utf-8

import os, random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import data
import data.data_loader as loader
from data.affine_transform import AffineTransform
import networks.blocks as block
from networks.train_op import build_train_op
from options.base_options import BaseOptions


class SimoSerra():
    def __init__(self, args):
        self.opt = args
        
    def initialize(self):
        # 这里的变量决定了这个神经网络将要读取什么样数据(shape)，什么类型的数据(dtype)
        # 以及输出什么样的数据(output_shape)
        self.image_paths_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="image_paths")
        self.ground_truth_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="ground_truth")
        self.input_queue = tf.FIFOQueue(capacity=args.capacity, shapes=[(1,), (1,)],
                                        dtypes=[tf.string, tf.string])
        self.enqueue_op = self.input_queue.enqueue_many([self.image_paths_placeholder,
                                                         self.ground_truth_placeholder])
        self.output_shape = [(args.img_size, args.img_size, args.img_channel),
                             (args.img_size, args.img_size, args.img_channel)]
        self.image_batch, self.label_batch = data.create_batch_from_queue(args, self.input_queue,
                                                                          self.output_shape,
                                                                          functions=[data.load_images] * 2,
                                                                          dtypes=[None] * 2)
        
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.global_step = tf.Variable(0, trainable=False)

    def build_model(self, args, network, loss_function):
        # 这里的network是一个函数形参数，一般是将网络结构的信息传递进来
        self.prediction = network(self.image_batch, args)
        # 损失函数的计算
        self.loss = loss_function(self.prediction, self.label_batch)
        # 设定根据损失函数进行优化的优化器
        self.train_op = build_train_op(self.loss, args.optimizer, args.learning_rate,
                                  tf.trainable_variables(), self.global_step)
        # Build the summary Tensor based on the TF collection of Summaries.
        self.summary = tf.summary.merge_all()
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver(max_to_keep=3)

    def create_graph(self, args):
        with tf.Graph().as_default():
            self.initialize()
            # Network Architecture and Train_op Graph
            self.build_model(args, network=simoserra_net, loss_function=calculate_loss)
            # Training Configuration
            if args.gpu_id is not None:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
                self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            else:
                self.sess = tf.Session()
            # Initialize variables
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            #summary_writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=self.sess)

    def fit(self):
        dataset = loader.img2img_dataset(path=self.opt.path)
        image_paths = np.expand_dims(np.array(list(dataset.keys())), axis=1)
        labels = np.expand_dims(np.array([dataset[_] for _ in dataset.keys()]), axis=1)
        feed_dict = {net.image_paths_placeholder: image_paths, net.ground_truth_placeholder: labels}
        for i in range(self.opt.epoch_num):
            if i % 10 is 0:
                # Update the queue for each 100 epochs
                subset = random.sample(list(dataset.items()), self.opt.capacity)
                path = [element[1] for element in subset]
                cls = [element[0] for element in subset]
                self.sess.run(self.enqueue_op, {self.image_paths_placeholder: path,
                                                self.ground_truth_placeholder: cls})
            # Get Training Data
            self.sess.run([self.image_batch, self.label_batch], feed_dict={})


def calculate_loss(prediction, ground_truth):
    loss = tf.reduce_mean(tf.losses.mean_squared_error(ground_truth, prediction))
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([loss] + reg_loss, name='total_loss')
    tf.summary.scalar('total_loss', total_loss)
    return total_loss

def simoserra_net(input, args):
    if args.do_affine:
        affine = AffineTransform(translation=args.translation, scale=args.scale, shear=args.shear,
                                 rotation=args.rotation, project=args.project, mean=args.imgaug_mean,
                                 stddev=args.imgaug_stddev)
        affine_mat = affine.to_transform_matrix()
        input = tf.contrib.image.transform(input, affine_mat)
    net = block.conv_block(input, "block_1", filters=[48, 128, 128], kernel_sizes=[5, 3, 3], stride=[2, 1, 1])
    net = block.conv_block(net, "block_2", filters=[256, 256, 256], kernel_sizes=[3, 3, 3], stride=[2, 1, 1])
    net = block.conv_block(net, "block_3", filters=[256, 512, 1024, 1024, 1024, 512, 256], kernel_sizes=[3]*7,
                           stride=[2, 1, 1, 1, 1, 1, 1])
    net = block.conv_block(net, "block_4", filters=[256, 256, 128], kernel_sizes=[4, 3, 3], stride=[0.5, 1, 1])
    net = block.conv_block(net, "block_5", filters=[128, 128, 48], kernel_sizes=[4, 3, 3], stride=[0.5, 1, 1])
    net = block.conv_block(net, "block_6", filters=[48, 24, args.img_channel], kernel_sizes=[4, 3, 3], stride=[0.5, 1, 1])
    return input


if __name__ == "__main__":
    """
        Test code of img2img dataset loading
    """
    from scipy.misc import imsave
    args = BaseOptions().initialize()

    dataset = loader.img2img_dataset(path="~/Pictures/dataset/buddha")
    image_paths = np.expand_dims(np.array(list(dataset.keys())), axis=1)
    labels = np.expand_dims(np.array([dataset[_] for _ in dataset.keys()]), axis=1)

    net = SimoSerra(args)
    net.create_graph(args)

    feed_dict = {net.image_paths_placeholder: image_paths, net.ground_truth_placeholder: labels}
    net.sess.run(net.enqueue_op, feed_dict=feed_dict)
    for e in range(10):
        img_batch = net.sess.run([net.image_batch])
        #img_batch, gt_batch= net.sess.run([net.image_batch, net.label_batch])
        for i in range(args.batch_size):
            imsave(os.path.expanduser("~/Pictures/test/img_in_batch_" + str(i) + ".jpg"), img_batch[i])
            #imsave(os.path.expanduser("~/Pictures/test/img_out_batch_" + str(i) + ".jpg"), gt_batch[i])
    
