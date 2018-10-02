import tensorflow as tf

def conv_block(input, name, filters, kernel_sizes, stride=None, padding = "same", repeat=1,
               activation = tf.nn.relu, batch_norm = True, reuse = False, start_op = None,
               end_op = None):
    assert len(filters) == len(kernel_sizes)
    if not stride:
        stride = 1
    if start_op:
        conv = start_op(input)
    else:
        conv = input
    for _ in range(repeat):
        for i in range(len(filters)):
            name = name + "_" + str(repeat) + "_" + str(i)
            if stride[i] >=1:
                conv = tf.layers.conv2d(inputs=conv, filters=filters[i], kernel_size=kernel_sizes[i],
                                        strides=stride[i], padding=padding, activation=activation,
                                        name=name, reuse=reuse)
            else:
                conv = tf.layers.conv2d_transpose(inputs=conv, filters=filters[i], kernel_size=kernel_sizes[i],
                                                  strides=round(1/stride[i]), padding=padding,
                                                  activation=activation, name=name, reuse=reuse)
            if batch_norm:
                conv = tf.layers.batch_normalization(conv, name=name+"_batch_norm", reuse=reuse)
    if end_op:
        conv = end_op(conv)
    return conv

def resnet_block(input, name, filters, kernel_sizes, stride=None, padding = "same", repeat=1,
                 activation = tf.nn.relu, batch_norm = True, reuse = False, start_op = None,
                 end_op = None):
    assert len(filters) == len(kernel_sizes)
    if not stride:
        stride = (1,1)
    if start_op:
        conv = start_op(input)
    else:
        conv = input
    for _ in range(repeat):
        for i in range(len(filters)):
            if stride[i] >=1:
                conv = tf.layers.conv2d(inputs=conv, filters=filters[i], kernel_size=kernel_sizes[i],
                                        strides=stride[i], padding=padding, activation=activation,
                                        name=name, reuse=reuse)
            else:
                conv = tf.layers.conv2d_transpose(inputs=conv, filters=filters[i], kernel_size=kernel_sizes[i],
                                                  strides=round(1/stride[i]), padding=padding, activation=activation,
                                                  name=name, reuse=reuse)
            if batch_norm:
                conv = tf.layers.batch_normalization(conv, name=name+"_batch_norm", reuse=reuse)
        net = tf.add(conv, input)
    if end_op:
        net = end_op(net)
    return net