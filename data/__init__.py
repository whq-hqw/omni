import os, warnings
import tensorflow as tf
from data.affine_transform import AffineTransform
import data.miscellaneous as misc

def create_batch_from_queue(args, input_queue, output_shape, functions, dtypes):
    cell = []
    for _ in range(args.threads):
        # Create multi-thread loading processes
        dequeue_obj = input_queue.dequeue()
        core=[]
        for i, component in enumerate(tf.unstack(dequeue_obj)):
            load_func = functions[i]
            # load data based on paths, load_func are defined below
            # or defined specifically in the network
            data = load_func(args, component, dtypes[i])
            core.append(data)
        cell.append(core)
    # The actual shape of cell is:
    #    [cell:  (core numbers = args.threads)
    #       number of data = len(dequeue_obj)
    #       [core_01: [data_01], [data_02], ......],
    #       [core_02: [data_01], [data_02], ......],
    #       ......
    #    ]
    #    Where data_01 and data_02 can be arbitrary shapes
    output_batch = tf.train.batch_join(cell, batch_size=args.batch_size,
                                       capacity=4 * args.batch_size * args.threads,
                                       shapes=output_shape, enqueue_many=True,
                                       allow_smaller_final_batch=True)
    return output_batch


def load_images(args, paths, dtype):
    # TODO: maybe we can use **options here to accept more parameters for flexibility
    images = []
    for path in tf.unstack(paths):
        img_byte = tf.read_file(path)
        image = tf.image.decode_image(img_byte)

        # --------------------------------Image Augmentation--------------------------------
        if args.do_affine:
            affine = AffineTransform(translation=args.translation, scale=args.scale, shear=args.shear,
                                     rotation=args.rotation, project=args.project, mean=args.imgaug_mean,
                                     stddev=args.imgaug_stddev, order=args.imgaug_order)
            affine_mat = affine.to_transform_matrix()
            image = tf.contrib.image.transform(image, affine_mat)
        if args.random_brightness:
            image = tf.image.random_brightness(image, max_delta=args.imgaug_max_delta)
        # TODO: random contrast, random blur
        if args.random_noise:
            noise = tf.add(tf.random_normal(shape=tf.shape(image), mean=args.imgaug_mean,
                                            stddev=args.imgaug_stddev), 1)
            image = tf.matmul(image, noise)
        if args.random_crop:
            image = tf.random_crop(image, [args.img_size, args.img_size, 3])
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, args.img_size, args.img_size)
        if args.random_flip:
            image = tf.image.random_flip_left_right(image)
        # --------------------------------Image Augmentation--------------------------------

        image.set_shape((args.img_size, args.img_size, 3))
        images.append(tf.image.per_image_standardization(image))
    return images

def pass_it(args, input, dtype):
    # TODO: maybe we can use **options here to accept more parameters for flexibility
    data = []
    for datum in tf.unstack(input):
        data.append(tf.constant(datum, dtype=dtype))
    return data