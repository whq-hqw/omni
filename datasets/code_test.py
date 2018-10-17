import datasets
import datasets.data_loader as loader
import tensorflow as tf
from options import BaseOptions


if __name__ == "__main__":
    args = BaseOptions().initialize()
    gpu_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="gpu_paths")
    cpu_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="cpu_truth")
    iris_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="iris_paths")

    gpu_bbox_placeholder = tf.placeholder(shape=(None), dtype=tf.int32, name="gpu_bbox")
    cpu_label_placeholder = tf.placeholder(shape=(None), dtype=tf.int32, name="cpu_label")
    iris_bbox_placeholder = tf.placeholder(shape=(None), dtype=tf.int32, name="iris_bbox")

    input_queue = tf.FIFOQueue(capacity=10000, shapes=[(1,), (1,), (1,), (1,), (1,), (1,)],
                               dtypes=[tf.string, tf.string, tf.string, tf.int32, tf.int32, tf.int32])
    enqueue_op = input_queue.enqueue_many([gpu_placeholder, cpu_placeholder, iris_placeholder,
                                           gpu_bbox_placeholder, cpu_label_placeholder, iris_bbox_placeholder])
    # --------------------------------OUTPUT--------------------------------------
    output_shape = [(100, 100, 3), (100, 100, 3), (100, 100, 3), (4,), (3,)]

    gpu_batch, cpu_batch, iris_batch, gpu_bbox_batch, cpu_label_batch, iris_bbox_batch = \
        datasets.data_load_graph(args, input_queue, output_shape, [loader.load_images] * 3 + [])

    dataset = datasets.arbitrary_dataset(path="~/Pictures/dataset1",
                                children=[("child1", "child2", "child3", "annotation/child1_bbox.txt",
                                           "annotation/child2.xml", "annotation/child2_bbox.mat")],
                                data_load_funcs=[loader.load_path_from_folder]*3, dig_level=[0, 0, 0, 0, 0, 0])