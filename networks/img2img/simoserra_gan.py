#coding=utf-8

import os, random
import tensorflow as tf
import datasets.load_data as load
import datasets.image_set as load_func
import networks.blocks as block
from networks.train_op import build_train_op
from options.BaseOptions import BaseOptions
import numpy as np

class SimoSerra_GAN():
    def __init__(self, args):
        self.opt = args
        
    def initialize(self):
        # 这里的变量决定了这个神经网络将要读取什么样数据(shape)，什么类型的数据(dtype)
        # 以及输出什么样的数据(output_shape)
        self.image_paths_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="image_paths")
        self.ground_truth_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="ground_truth")
        self.sketch_images_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="sketch_paths")
        self.line_images_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="line_paths")
        self.input_queue = tf.FIFOQueue(capacity=args.capacity, shapes=[(1,), (1,), (1,), (1,)],
                                        dtypes=[tf.string, tf.string, tf.string, tf.string])
        self.enqueue_op = self.input_queue.enqueue_many([self.image_paths_placeholder,
                                                         self.ground_truth_placeholder,
                                                         self.sketch_images_placeholder,
                                                         self.line_images_placeholder])
        self.output_shape = [(args.img_size, args.img_size, args.img_channel),
                             (args.img_size, args.img_size, args.img_channel),
                             (args.img_size, args.img_size, args.img_channel),
                             (args.img_size, args.img_size, args.img_channel)]
        self.learning_rate = tf.placeholder(tf.float16, name="learning_rate")

        self.global_step = tf.Variable(0, trainable=False)

    def build_model(self, args):
        # 这里的network是一个函数形参数，一般是将网络结构的信息传递进来
        I2I_prediction = simoserra_net(self.image_batch, args)
        sketch_pred = simoserra_net(self.sketch_batch, args, reuse=True)
        self.gan_prediction1 = simoserra_gan(I2I_prediction)
        self.gan_prediction2 = simoserra_gan(self.label_batch, reuse=True)
        self.gan_prediction3 = simoserra_gan(self.line_batch, reuse=True)
        self.gan_prediction4 = simoserra_gan(sketch_pred, reuse=True)
        # 损失函数的计算
        self.g_loss, self.d_loss = calculate_loss(I2I_prediction, self.label_batch, self.gan_prediction1,
                                                  self.gan_prediction2, self.gan_prediction3, self.gan_prediction4)
        # 获取可训练的变量
        vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        vars_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        # 设定根据损失函数进行优化的优化器
        self.train_op_gen = build_train_op(self.g_loss, args.optimizer, args.learning_rate,
                                           vars_gen, self.global_step)
        self.train_op_dis = build_train_op(self.d_loss, args.optimizer, args.learning_rate,
                                           vars_dis, self.global_step)
        # Build the summary Tensor based on the TF collection of Summaries.
        self.summary = tf.summary.merge_all()
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver(max_to_keep=3)

    def create_graph(self, args):
        with tf.Graph().as_default():
            self.initialize()
            # Data Load Graph
            output_batch= tf.unstack(load.data_load_graph(args, self.input_queue, self.output_shape))
            self.image_batch = output_batch[0]
            self.label_batch = output_batch[1]
            self.sketch_batch = output_batch[2]
            self.line_batch = output_batch[3]
            # Network Architecture and Train_op Graph
            #self.build_model(args)
            # Training Configuration
            if args.gpu_id is not None:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
                self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            else:
                self.sess = tf.Session()
            # Initialize variables
            #self.sess.run(tf.global_variables_initializer())
            #self.sess.run(tf.local_variables_initializer())
            #summary_writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)
            #coord = tf.train.Coordinator()
            #tf.train.start_queue_runners(coord=coord, sess=self.sess)

    def fit(self):
        dataset = load.img2img_dataset(path=self.opt.path)
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


def calculate_loss(prediction, ground_truth, gan1, gan2, gan3, gan4):
    # MSE Loss
    mse_loss = tf.reduce_mean(tf.losses.mean_squared_error(ground_truth, prediction))
    #mse_reg_loss = tf.losses.get_regularization_losses(scope="generator")
    g_loss = tf.add_n([mse_loss], name="g_loss")
    # GAN Loss
    loss_G_1 = tf.reduce_mean(tf.log(gan1))
    loss_G_2 = tf.reduce_mean(tf.log(gan4))
    loss_D_1 = -tf.reduce_mean(tf.log(gan2) - tf.log(1 - gan1))
    loss_D_2 = -tf.reduce_mean(tf.log(gan3))
    #gan_reg_loss = tf.losses.get_regularization_losses(scope="discriminator")
    d_loss = tf.add_n([loss_G_1, loss_G_2, loss_D_1, loss_D_2], name="d_loss")
    
    tf.summary.scalar('generator_loss', g_loss)
    tf.summary.scalar('discriminator_loss', d_loss)
    return g_loss, d_loss

def simoserra_net(input, args, reuse=False):
    with tf.variable_scope("generator"):
        net = block.conv_block(input, name="block_1", filters=[48, 128, 128], kernel_sizes=[5, 3, 3],
                               stride=[2, 1, 1], reuse=reuse)
        net = block.conv_block(net, name="block_2", filters=[256, 256, 256], kernel_sizes=[3, 3, 3],
                               stride=[2, 1, 1], reuse=reuse)
        net = block.conv_block(net, name="block_3", filters=[256, 512, 1024, 1024, 1024, 512, 256],
                               kernel_sizes=[3]*7, stride=[2, 1, 1, 1, 1, 1, 1], reuse=reuse)
        net = block.conv_block(net, name="block_4", filters=[256, 256, 128], kernel_sizes=[4, 3, 3],
                               stride=[0.5, 1, 1], reuse=reuse)
        net = block.conv_block(net, name="block_5", filters=[128, 128, 48], kernel_sizes=[4, 3, 3],
                               stride=[0.5, 1, 1], reuse=reuse)
        net = block.conv_block(net, name="block_6", filters=[48, 24, args.img_channel],
                               kernel_sizes=[4, 3, 3], stride=[0.5, 1, 1], reuse=reuse)
    return net

def simoserra_gan(input, reuse=False):
    with tf.variable_scope("discriminator"):
        net = block.conv_block(input, name="gan_block1", filters=[48, 128, 128], kernel_sizes=[5, 3, 3],
                               stride=[2, 1, 1], reuse=reuse)
        net = block.conv_block(net, name="block_2", filters=[256, 256, 256], kernel_sizes=[3, 3, 3],
                               stride=[2, 1, 1], reuse=reuse)
    return net

def compliment_dim(input, dim):
    assert len(input) <= dim, "length of input should not larger than dim."
    if len(input) == dim:
        return input
    else:
        repeat = dim // len(input)
        input = input * repeat + [input[_] for _ in range(dim-len(input)*repeat)]
        return input


if __name__ == "__main__":
    dataset = load.arbitrary_dataset(path="~/Pictures/dataset/buddha",
                                     folder_names=[("trainA", "trainB", "testA", "testB")],
                                     functions=[load_func.load_path_from_folder], dig_level=[0, 0, 0, 0])
    dim = max(len(dataset["A"][0]), len(dataset["A"][1]), len(dataset["A"][2]), len(dataset["A"][3]))
    dataset["A"][0].sort()
    dataset["A"][1].sort()
    img = np.expand_dims(np.array(compliment_dim(dataset["A"][0], dim)), axis=1)
    gt = np.expand_dims(np.array(compliment_dim(dataset["A"][1], dim)), axis=1)
    sketch = np.expand_dims(np.array(compliment_dim(dataset["A"][2], dim)), axis=1)
    lines = np.expand_dims(np.array(compliment_dim(dataset["A"][3], dim)), axis=1)

    args = BaseOptions().initialize()

    net = SimoSerra_GAN(args)
    net.create_graph(args)
    #net.fit()

    feed_dict={net.image_paths_placeholder: img, net.ground_truth_placeholder: gt,
               net.sketch_images_placeholder: sketch, net.line_images_placeholder: lines}
    net.sess.run(net.enqueue_op, feed_dict=feed_dict)
    b = net.sess.run(net.label_batch)
    img_batch, gt_batch, sk_batch, ln_batch = net.sess.run([net.image_batch, net.label_batch, net.sketch_batch, net.line_batch])
    #pred = net.sess.run(net.prediction)
    #loss = net.sess.run(net.loss)
    #pass
