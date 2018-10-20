import os

import tensorflow as tf
import numpy as np

import data
import data.data_loader as loader
import data.miscellaneous as misc
from options.base_options import BaseOptions


def load_path_from_csv(len, paths, dig_level=None):
    import csv
    dataset = {}
    name = []
    bbox = []
    # sometimes one csv file are correspond to one image file
    for path in paths:
        with open(path, "r") as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for line in readCSV:
                name.append(line[0])
                bbox.append(line[1:])
    # 1->A, 2->B, 3->C, ..., 26->Z
    key_name = misc.number_to_char(len)
    key_bbox = misc.number_to_char(len+1)
    return {key_name:name, key_bbox: bbox}

def load_easy():
    args = BaseOptions().initialize()
    
    dataset = loader.arbitrary_dataset(path=os.path.expanduser(args.path),
                                       sources=["annotation/bbox.csv"],
                                       modes=[load_path_from_csv],
                                       dig_level=[None])
    bboxes = np.array(dataset["B"])
    
    sess = tf.Session()
    
    gpu_bbox_placeholder = tf.placeholder(shape=(None, 4), dtype=tf.int32, name="gpu_bbox")

    input_queue = tf.FIFOQueue(capacity=10000, shapes=[(4,)], dtypes=[tf.int32])
    enqueue_op = input_queue.enqueue_many([gpu_bbox_placeholder])

    output_shape = [(4,)]
    
    batch = data.create_batch_from_queue(args, input_queue, output_shape, [data.pass_it], [tf.int32])

    feed_dict = {gpu_bbox_placeholder: bboxes}
    sess.run(enqueue_op, feed_dict=feed_dict)
    
    bbox_batch = sess.run(batch)
    
def load_1():
    args = BaseOptions().initialize()
    gpu_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="gpu_paths")
    cpu_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="cpu_truth")
    iris_placeholder = tf.placeholder(shape=(None, 1), dtype=tf.string, name="iris_paths")
    
    gpu_bbox_placeholder = tf.placeholder(shape=(None, 4), dtype=tf.int32, name="gpu_bbox")
    cpu_label_placeholder = tf.placeholder(shape=(None, 4), dtype=tf.int32, name="cpu_label")
    iris_bbox_placeholder = tf.placeholder(shape=(None, 4), dtype=tf.int32, name="iris_bbox")
    
    input_queue = tf.FIFOQueue(capacity=10000, shapes=[(1,), (1,), (1,), (1,), (1,), (1,)],
                               dtypes=[tf.string, tf.string, tf.string, tf.int32, tf.int32, tf.int32])
    enqueue_op = input_queue.enqueue_many([gpu_placeholder, cpu_placeholder, iris_placeholder,
                                           gpu_bbox_placeholder, cpu_label_placeholder, iris_bbox_placeholder])
    # --------------------------------OUTPUT--------------------------------------
    output_shape = [(100, 100, 3), (100, 100, 3), (100, 100, 3), (4,), (3,)]
    
    gpu_batch, cpu_batch, iris_batch, gpu_bbox_batch, cpu_label_batch, iris_bbox_batch = \
        data.data_load_graph(args, input_queue, output_shape, [loader.load_images] * 3 + [])
    
    dataset = data.arbitrary_dataset(path="~/Pictures/dataset1",
                                     children=[("child1", "child2", "child3", "annotation/child1_bbox.txt",
                                                "annotation/child2.xml", "annotation/child2_bbox.mat")],
                                     data_load_funcs=[loader.load_path_from_folder] * 3, dig_level=[0, 0, 0, 0, 0, 0])
    

if __name__ == "__main__":
    load_easy()