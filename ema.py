#coding=utf-8
import tensorflow as tf


# W represent the model parameter in real situation
W = tf.Variable(1.0)

ema = tf.train.ExponentialMovingAverage(0.9)

G = tf.placeholder(tf.float32, shape=(), name="Gradient")

# assign_add equals to apply_gradient/minimize op while 2.0 represent gradient
update = tf.assign_add(W, 1.0)

with tf.control_dependencies([update]):
    #返回一个op,这个op用来更新moving_average,i.e. shadow value
    ema_op = ema.apply([W])#这句和下面那句不能调换顺序

# 以 w 当作 key， 获取 shadow value 的值
ema_val = ema.average(W)#参数不能是list，有点蛋疼

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(4):
        sess.run(ema_op)
        #sess.run(ema_op, feed_dict={G: i})
        print(sess.run(ema_val))