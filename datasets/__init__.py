import tensorflow as tf
import numpy as np
import warnings

def load_images(args, paths, seed):
    images = []
    for path in tf.unstack(paths):
        img_byte = tf.read_file(path)
        image = tf.image.decode_image(img_byte)
        # TODO: Image Augumentation
        if args.random_crop:
            image = tf.random_crop(image, [args.img_size, args.img_size, 3], seed=seed)
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, args.img_size, args.img_size)
        if args.random_flip:
            image = tf.image.random_flip_left_right(image, seed=seed)

        image.set_shape((args.img_size, args.img_size, 3))
        images.append(tf.image.per_image_standardization(image))
    return images

def data_load_graph(args, input_queue, output_shape, functions):
    cell = []
    for _ in range(args.threads):
        seed = np.random.randint(4096)
        print("data_load_graph random seed:{}".format(seed))
        dequeue_obj = input_queue.dequeue()
        component=[]
        for i, paths in enumerate(dequeue_obj):
            try:
                load_func = functions[i]
            except (TypeError, IndexError):
                warnings.warn("Error Encounted in datasets/__init__.py: data_load_graph")
                load_func = load_images
            data = load_func(args, paths, seed)
            component.append(data)
            cell.append(component)
    # The actual shape of cell is:
    #    [cell:  (core numbers = args.threads)
    #       [core_01: [data_01], [data_02], ......],
    #       [core_02: [data_01], [data_02], ......],
    #       ......
    #   ]
    #    Where data_01 and data_02 can be arbitrary shapes
    output_batch = tf.train.batch_join(cell, batch_size=args.batch_size,
                                               capacity=4 * args.batch_size * args.threads,
                                               shapes=output_shape,
                                               enqueue_many=True, allow_smaller_final_batch=True)
    return output_batch