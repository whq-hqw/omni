
import os
import tensorflow as tf

#tf.enable_eager_execution()

a=tf.constant([1,2,3,4])
b=tf.multiply(a, 2)

filename_queue = tf.train.string_input_producer([os.path.expanduser("~/Pictures/dataset1/annotation/child1_bbox.txt")])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col0, col1, col2, col3, col4 = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])

with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print(sess.run(b))

    for i in range(3):
        # Retrieve a single instance:
        example, label = sess.run([features, col0])

    coord.request_stop()
    coord.join(threads)