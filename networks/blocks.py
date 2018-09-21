import tensorflow as tf

def conv_block(input, scope, filters, kernel_sizes, stride=None, padding = "same", repeat=1,
               activation = tf.nn.relu, batch_norm = True, start_op = None, end_op = None):
    assert len(filters) == len(kernel_sizes)
    if not stride:
        stride = 1
    if start_op:
        conv = start_op(input)
    else:
        conv = input
    with tf.variable_scope(scope):
        for _ in range(repeat):
            for i in range(len(filters)):
                if stride[i] >=1:
                    conv = tf.layers.conv2d(inputs=conv, filters=filters[i], kernel_size=kernel_sizes[i],
                                            strides=stride[i], padding=padding, activation=activation)
                else:
                    conv = tf.layers.conv2d_transpose(inputs=conv, filters=filters[i], kernel_size=kernel_sizes[i],
                                            strides=round(1/stride[i]), padding=padding, activation=activation)
                if batch_norm:
                    conv = tf.layers.batch_normalization(conv)
    if end_op:
        conv = end_op(conv)
    return conv

def resnet_block(input, scope, filters, kernel_sizes, stride=None, padding = "same", repeat=1,
                 activation = tf.nn.relu, batch_norm = True, start_op = None, end_op = None):
    assert len(filters) == len(kernel_sizes)
    if not stride:
        stride = (1,1)
    if start_op:
        conv = start_op(input)
    else:
        conv = input
    with tf.variable_scope(scope):
        for _ in range(repeat):
            for i in range(len(filters)):
                if stride[i] >=1:
                    conv = tf.layers.conv2d(inputs=conv, filters=filters[i], kernel_size=kernel_sizes[i],
                                            strides=stride[i], padding=padding, activation=activation)
                else:
                    conv = tf.layers.conv2d_transpose(inputs=conv, filters=filters[i], kernel_size=kernel_sizes[i],
                                            strides=round(1/stride[i]), padding=padding, activation=activation)
                if batch_norm:
                    conv = tf.layers.batch_normalization(conv)
            net = tf.add(conv, input)
    if end_op:
        net = end_op(net)
    return net