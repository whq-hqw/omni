import os, random
import tensorflow as tf
import datasets.load_data as load
import networks.blocks as block
from networks.train_op import build_train_op
from options.BaseOptions import BaseOptions

class SimoSerra():
    def __init__(self, args):
        self.opt = args
        
    def initialize(self):
        self.image_paths_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string)
        self.ground_truth_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string)
        self.input_queue = tf.FIFOQueue(capacity=args.capacity, shapes=[(1,), (1,)],
                                        dtypes=[tf.string, tf.string])
        self.enqueue_op = self.input_queue.enqueue_many([self.image_paths_placeholder,
                                                         self.ground_truth_placeholder])
        self.output_shape = [(args.img_size, args.img_size, args.img_channel),
                             (args.img_size, args.img_size, args.img_channel)]
        self.global_step = tf.Variable(0, trainable=False)

    def build_model(self, args, input, ground_truth, network, loss_function):
        # Make Prediction
        prediction = network(input, args)
        # Calculate Loss
        self.loss = loss_function(prediction, ground_truth)
        # Create the gradient descent optimizer with the given learning rate.
        self.train_op = build_train_op(self.loss, args.optimizer, args.learning_rate,
                                  tf.trainable_variables(), self.global_step)
        # Build the summary Tensor based on the TF collection of Summaries.
        self.summary = tf.summary.merge_all()
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver(max_to_keep=3)

    def create_graph(self, args):
        with tf.Graph().as_default():
            tmp = tf.constant(2)
            self.initialize()
            # Data Load Graph
            self.image_batch, self.label_batch = load.data_load_graph(args, self.input_queue, self.output_shape)
            # Network Architecture and Train_op Graph
            self.build_model(args, self.image_batch, self.label_batch, network=simoserra_net,
                             loss_function=calculate_loss)
            # Training Configuration
            if args.gpu_id is not None:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
                self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            else:
                self.sess = tf.Session()
            # Initialize variables
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            summary_writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=self.sess)

    def fit(self):
        dataset = load.img2img_dataset(self.opt.path, capacity = self.opt.capacity)
        for i in range(self.opt.epoch_num):
            if i % 100 is 0:
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
    net = block.conv_block(input, "block_1", filters=[48, 128, 128], kernel_sizes=[5, 3, 3], stride=[2, 1, 1])
    net = block.conv_block(net, "block_2", filters=[256, 256, 256], kernel_sizes=[3, 3, 3], stride=[2, 1, 1])
    net = block.conv_block(net, "block_3", filters=[256, 512, 1024, 1024, 1024, 512, 256], kernel_sizes=[3]*7,
                           stride=[2, 1, 1, 1, 1, 1, 1])
    net = block.conv_block(net, "block_4", filters=[256, 256, 128], kernel_sizes=[4, 3, 3], stride=[0.5, 1, 1])
    net = block.conv_block(net, "block_5", filters=[128, 128, 48], kernel_sizes=[4, 3, 3], stride=[0.5, 1, 1])
    net = block.conv_block(net, "block_6", filters=[48, 24, args.img_channel], kernel_sizes=[4, 3, 3], stride=[0.5, 1, 1])
    return net

def simoserra_net2018(input1, input2, args):
    net = block.conv_block(input1, "block_1", filters=[48, 128, 128], kernel_sizes=[5, 3, 3], stride=[2, 1, 1])
    net = block.conv_block(net, "block_2", filters=[256, 256, 256], kernel_sizes=[3, 3, 3], stride=[2, 1, 1])
    net = block.conv_block(net, "block_3", filters=[256, 512, 1024, 1024, 1024, 512, 256], kernel_sizes=[3] * 7,
                           stride=[2, 1, 1, 1, 1, 1, 1])
    net = block.conv_block(net, "block_2", filters=[256, 256, 128], kernel_sizes=[4, 3, 3], stride=[0.5, 1, 1])
    net = block.conv_block(net, "block_2", filters=[128, 128, 48], kernel_sizes=[4, 3, 3], stride=[0.5, 1, 1])
    net = block.conv_block(net, "block_2", filters=[48, 24, args.img_channel], kernel_sizes=[4, 3, 3], stride=[0.5, 1, 1])
    return net

if __name__ == "__main__":
    args = BaseOptions().initialize()
    net = SimoSerra(args)
    net.create_graph(args)
