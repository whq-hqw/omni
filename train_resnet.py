import tensorflow as tf

from networks.resnet import resnet_50

from tensorflow.examples.tutorials.mnist import input_data

def main():
    input = tf.placeholder(name="input", shape=[64, 224, 224, 3])
    graph = tf.Graph()
    with graph.as_default():
        net = resnet_50(input)

    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        print("Tensorflow initialized all variables.")

        preds = sess.run(nn.preds,
                         feed_dict={
                             nn.inputRGB: imgs
                         })

if __name__ == "__main__":
    tf.app.run()
