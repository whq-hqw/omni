# coding=utf-8

import os, random
import tensorflow as tf

import datasets.load_data as load
import datasets.miscellaneous as misc
import networks.blocks as block
import networks.utils as util
from networks.train_op import build_train_op
from options.BaseOptions import BaseOptions
import numpy as np


class SimoSerra_GAN:
    def __init__(self, args):
        self.opt = args

    def initialize(self):
        """
        Here we will define an FIFOQueue defines what kind(shape, dtype) of data we are going to read.
            i.e. Placeholders, Input_queue, Enqueue_op

        And what kind of the output is expected.
            i.e. Output_shape

        DO NOT PLACE THIS IN THE __init__ METHOD,
        OTHERWISE VARIABLES BELOW CANNOT BE DEFINED IN THE SANE GRAPH
        """
        #---------------------------------INPUT--------------------------------------
        # Budda 3D rendering Images(東京藝術大学, trainA)
        self.image_paths_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="image_paths")
        # Buddha Line Drawing(東京藝術大学, trainB)
        self.ground_truth_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="ground_truth")
        # Unconstrained 3D rendering Images(爬虫数据, gan_A)
        self.sketch_images_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="sketch_paths")
        # Unconstrained Line Drawing(爬虫数据, gan_B)
        self.line_images_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="line_paths")
        self.input_queue = tf.FIFOQueue(capacity=args.capacity, shapes=[(1,), (1,), (1,), (1,)],
                                        dtypes=[tf.string, tf.string, tf.string, tf.string])
        self.enqueue_op = self.input_queue.enqueue_many([self.image_paths_placeholder,
                                                         self.ground_truth_placeholder,
                                                         self.sketch_images_placeholder,
                                                         self.line_images_placeholder])
        # --------------------------------OUTPUT--------------------------------------
        self.output_shape = [(args.img_size, args.img_size, args.img_channel),
                             (args.img_size, args.img_size, args.img_channel),
                             (args.img_size, args.img_size, args.img_channel),
                             (args.img_size, args.img_size, args.img_channel)]
        # --------------------------------OTHERS--------------------------------------
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.global_step = tf.Variable(0, trainable=False)

    def build_model(self, args):
        # 这里的network是一个函数形参数，一般是将网络结构的信息传递进来
        print("Creating Computational Graph...")
        img_pred = simoserra_net(self.image_batch, args)
        sketch_pred = simoserra_net(self.sketch_batch, args, reuse=True)
        self.gan_prediction1 = simoserra_gan(img_pred)
        self.gan_prediction2 = simoserra_gan(self.label_batch, reuse=True)
        self.gan_prediction3 = simoserra_gan(self.line_batch, reuse=True)
        self.gan_prediction4 = simoserra_gan(sketch_pred, reuse=True)
        # 损失函数的计算
        self.g_loss, self.d_loss = calculate_loss(img_pred, self.label_batch, self.gan_prediction1,
                                                  self.gan_prediction2, self.gan_prediction3, self.gan_prediction4)
        # 获取可训练的变量
        print("Creating Training Operation...")
        self.vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.vars_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        # 设定根据损失函数进行优化的优化器
        self.train_op_gen = build_train_op(self.g_loss, args.optimizer, args.learning_rate,
                                           self.vars_gen, self.global_step)
        self.train_op_dis = build_train_op(self.d_loss, args.optimizer, args.learning_rate,
                                           self.vars_dis, self.global_step)
        print("Completing...")
        # TODO: Under development
        # Build the summary Tensor based on the TF collection of Summaries.
        self.summary = tf.summary.merge_all()
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver(max_to_keep=3)

    def create_graph(self, args):
        """
        Here we create the graph for the SimoSerra_GAN.
        :param args: args from options/BaseOptions.py
        :return:
        """
        with tf.Graph().as_default():
            self.initialize()
            # Data Load Graph
            print("Creating Data Load Graph...")
            self.image_batch, self.label_batch, self.sketch_batch, self.line_batch = \
                load.data_load_graph(args, self.input_queue, self.output_shape, [load.load_images]*4)
            # Network Architecture and Train_op Graph
            self.build_model(args)
            # Training Configuration
            if args.gpu_id is not None:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
                self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            else:
                self.sess = tf.Session()
            # TODO: Under development
            # Initialize variables
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            # summary_writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=self.sess)

    def fit(self):
        """
        Train the defined network.
        :return:
        """
        # TODO: Under development
        dataset = get_dataset(self.opt)
        #feed_dict = {self.image_paths_placeholder: images, net.ground_truth_placeholder: labels,
                     #self.sketch_images_placeholder: sketches, self.line_images_placeholder: lines}
        for i in range(self.opt.epoch_num):
            if i % 10 is 0:
                # Update the queue for each 10 epochs
                subset = random.sample(list(dataset.items()), self.opt.capacity)
                path = [element[1] for element in subset]
                cls = [element[0] for element in subset]
                self.sess.run(self.enqueue_op, {self.image_paths_placeholder: path,
                                                self.ground_truth_placeholder: cls})
            # Get Training Data
            self.sess.run([self.image_batch, self.label_batch], feed_dict={})


def get_dataset(opt):
    # 其中opt,path 就是args.path/--path 中的输入的数据集的所在位置
    dataset = load.arbitrary_dataset(path=opt.path,
                                     folder_names=[("trainA", "trainB", "gan_A", "gan_B")],
                                     data_load_funcs=[misc.load_path_from_folder], dig_level=[0, 0, 0, 0])
    # 获取4个文件夹中拥有图片最多的文件夹的图片数量
    dim = max(len(dataset["A"][0]), len(dataset["A"][1]), len(dataset["A"][2]), len(dataset["A"][3]))
    # 使文件夹"trainA"，"trainB" 中的图片路径能够一一对应
    # 前提是"trainA"，"trainB" 相对应的图片的名称必须相同
    dataset["A"][0].sort()
    dataset["A"][1].sort()
    # 将读取到的图片路径信息全部转化为tensorflow可以读取的形式
    images = np.expand_dims(np.array(util.compliment_dim(dataset["A"][0], dim)), axis=1)
    labels = np.expand_dims(np.array(util.compliment_dim(dataset["A"][1], dim)), axis=1)
    sketches = np.expand_dims(np.array(util.compliment_dim(dataset["A"][2], dim)), axis=1)
    lines = np.expand_dims(np.array(util.compliment_dim(dataset["A"][3], dim)), axis=1)
    return images, labels, sketches, lines

def calculate_loss(prediction, ground_truth, gan1, gan2, gan3, gan4):
    # MSE Loss
    mse_loss = tf.reduce_mean(tf.losses.mean_squared_error(ground_truth, prediction))
    # mse_reg_loss = tf.losses.get_regularization_losses(scope="generator")
    g_loss = tf.add_n([mse_loss], name="g_loss")
    # GAN Loss
    loss_G_1 = tf.reduce_mean(tf.log(gan1))
    loss_G_2 = tf.reduce_mean(tf.log(gan4))
    loss_D_1 = -tf.reduce_mean(tf.log(gan2) - tf.log(1 - gan1))
    loss_D_2 = -tf.reduce_mean(tf.log(gan3))
    # gan_reg_loss = tf.losses.get_regularization_losses(scope="discriminator")
    d_loss = tf.add_n([loss_G_1, loss_G_2, loss_D_1, loss_D_2], name="d_loss")

    tf.summary.scalar('generator_loss', g_loss)
    tf.summary.scalar('discriminator_loss', d_loss)
    return g_loss, d_loss


def simoserra_net(input, args, reuse=False):
    """
    Structures from below papers:
    Learning to Simplify: Fully Convolutional Networks for Rough Sketch Cleanup(2016)
    https://esslab.jp/~ess/publications/SimoSerraSIGGRAPH2016.pdf
    """
    with tf.variable_scope("generator"):
        net = block.conv_block(input, name="block_1", filters=[48, 128, 128], kernel_sizes=[5, 3, 3],
                               stride=[2, 1, 1], reuse=reuse)
        net = block.conv_block(net, name="block_2", filters=[256, 256, 256], kernel_sizes=[3, 3, 3],
                               stride=[2, 1, 1], reuse=reuse)
        net = block.conv_block(net, name="block_3", filters=[256, 512, 1024, 1024, 1024, 512, 256],
                               kernel_sizes=[3] * 7, stride=[2, 1, 1, 1, 1, 1, 1], reuse=reuse)
        net = block.conv_block(net, name="block_4", filters=[256, 256, 128], kernel_sizes=[4, 3, 3],
                               stride=[0.5, 1, 1], reuse=reuse)
        net = block.conv_block(net, name="block_5", filters=[128, 128, 48], kernel_sizes=[4, 3, 3],
                               stride=[0.5, 1, 1], reuse=reuse)
        net = block.conv_block(net, name="block_6", filters=[48, 24, args.img_channel],
                               kernel_sizes=[4, 3, 3], stride=[0.5, 1, 1], reuse=reuse)
    return net


def simoserra_gan(input, reuse=False):
    # TODO: Under development
    with tf.variable_scope("discriminator"):
        net = block.conv_block(input, name="gan_block1", filters=[48, 128, 128], kernel_sizes=[5, 3, 3],
                               stride=[2, 1, 1], reuse=reuse)
        net = block.conv_block(net, name="block_2", filters=[256, 256, 256], kernel_sizes=[3, 3, 3],
                               stride=[2, 1, 1], reuse=reuse)
    return net


if __name__ == "__main__":
    args = BaseOptions().initialize()

    img, gt, sketch, lines = get_dataset()

    net = SimoSerra_GAN(args)
    net.create_graph(args)

    feed_dict = {net.image_paths_placeholder: img, net.ground_truth_placeholder: gt,
                 net.sketch_images_placeholder: sketch, net.line_images_placeholder: lines}
    net.sess.run(net.enqueue_op, feed_dict=feed_dict)
    img_batch, gt_batch, sketch_batch, line_batch = net.sess.run(
        [net.image_batch, net.label_batch, net.sketch_batch, net.line_batch])
    pass