import os, random
import tensorflow as tf
import networks.blocks as block
import datasets.load_data as load
from networks.train_op import build_train_op

class Resnet():
    def __init__(self, args, path):
        # Declare Placeholders
        self.image_paths_placeholder = tf.placeholder(shape=(None), dtype=tf.string)
        self.ground_truth_placeholder = tf.placeholder(shape=(None), dtype=tf.int32)
        self.opt = args
        self.dataset_path = os.path.expanduser(path)

    def build_model(self, args, input, ground_truth, network, loss_function,
              global_step=tf.Variable(0, trainable=False)):
        # Make Prediction
        prediction = network(input)
        # Calculate Loss
        self.loss = loss_function(prediction, ground_truth)
        # Create the gradient descent optimizer with the given learning rate.
        self.train_op = build_train_op(self.loss, args.optimizer, args.learning_rate,
                                  tf.trainable_variables(), global_step)
        # Build the summary Tensor based on the TF collection of Summaries.
        self.summary = tf.summary.merge_all()
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver(max_to_keep=3)

    def create_graph(self, args):
        with tf.Graph().as_default():
            # Data Load Graph
            self.enqueue_op, self.image_batch, self.label_batch = load.data_load_graph(
                self.image_paths_placeholder, self.ground_truth_placeholder, args.threads,
                args.batch_size, args.output_shape)

            # Network Architecture and Train_op Graph
            self.build_model(args, self.image_batch, self.label_batch, network=resnet_50, loss_function=calculate_loss)

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
        dataset = load.ilsvrc_dataset(self.dataset_path, capacity = self.opt.capacity)
        for i in range(self.opt.epoch_num):
            if i % 100 is 0:
                # Update the queue for each 100 epochs
                subset = random.sample(list(dataset.items()), self.opt.capacity)
                path = [element[1] for element in subset]
                cls = [element[0] for element in subset]
                self.sess.run(self.enqueue_op, {self.image_paths_placeholder: path,
                                                self.ground_truth_placeholder: cls})
            self.sess.run(self.label_batch)

    def evaluation(prediction, ground_truth):
        correct = tf.nn.in_top_k(predictions=prediction, targets=ground_truth, k=1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

#@staticmethod
def calculate_loss(input, ground_truth):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(input, ground_truth))
    tf.summary.scalar('entropy_loss', loss)
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([loss] + reg_loss, name='total_loss')
    tf.summary.scalar('total_loss', total_loss)
    return total_loss

#@staticmethod
def resnet_18(input):
    net = block.resnet_block(input, scope="conv1_x", filters=[64], kernel_sizes=[7])
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2)

    net = block.resnet_block(net, scope="conv2_x", filters=[64, 64], kernel_sizes=[3, 3], repeat=2)
    net = block.resnet_block(net, scope="conv3_x", filters=[128, 128], kernel_sizes=[3, 3], repeat=2)
    net = block.resnet_block(net, scope="conv4_x", filters=[256, 256], kernel_sizes=[3, 3], repeat=2)
    net = block.resnet_block(net, scope="conv5_x", filters=[512, 512], kernel_sizes=[3, 3], repeat=2)

    net = tf.layers.dense(net, 1000, activation=tf.nn.softmax, name="average_pool")
    return net

#@staticmethod
def resnet_34(input):
    net = block.resnet_block(input, scope="conv1_x", filters=[64], kernel_sizes=[7])
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2)

    net = block.resnet_block(net, scope="conv2_x", filters=[64, 64], kernel_sizes=[3, 3], repeat=3)
    net = block.resnet_block(net, scope="conv3_x", filters=[128, 128], kernel_sizes=[3, 3], repeat=4)
    net = block.resnet_block(net, scope="conv4_x", filters=[256, 256], kernel_sizes=[3, 3], repeat=6)
    net = block.resnet_block(net, scope="conv5_x", filters=[512, 512], kernel_sizes=[3, 3], repeat=3)

    net = tf.layers.dense(net, 1000, activation=tf.nn.softmax, name="average_pool")
    return net

#@staticmethod
def resnet_50(input):
    net = block.resnet_block(input, scope="conv1_x", filters=[64], kernel_sizes=[7])
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2)

    net = block.resnet_block(net, scope="conv2_x", filters=[64, 128, 256], kernel_sizes=[1, 3, 1], repeat=3)
    net = block.resnet_block(net, scope="conv3_x", filters=[128, 128, 256], kernel_sizes=[1, 3, 1], repeat=4)
    net = block.resnet_block(net, scope="conv4_x", filters=[256, 256, 1024], kernel_sizes=[1, 3, 1], repeat=6)
    net = block.resnet_block(net, scope="conv5_x", filters=[512, 512, 2048], kernel_sizes=[1, 3, 1], repeat=3)

    net = tf.layers.dense(net, 1000, activation=tf.nn.softmax, name="average_pool")
    return net

#@staticmethod
def resnet_101(input):
    net = block.resnet_block(input, scope="conv1_x", filters=[64], kernel_sizes=[7])
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2)

    net = block.resnet_block(net, scope="conv2_x", filters=[64, 128, 256], kernel_sizes=[1, 3, 1], repeat=3)
    net = block.resnet_block(net, scope="conv3_x", filters=[128, 128, 256], kernel_sizes=[1, 3, 1], repeat=4)
    net = block.resnet_block(net, scope="conv4_x", filters=[256, 256, 1024], kernel_sizes=[1, 3, 1], repeat=23)
    net = block.resnet_block(net, scope="conv5_x", filters=[512, 512, 2048], kernel_sizes=[1, 3, 1], repeat=3)

    net = tf.layers.dense(net, 1000, activation=tf.nn.softmax, name="average_pool")
    return net

#@staticmethod
def resnet_152(input):
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


