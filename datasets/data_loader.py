import glob
import tensorflow as tf
from datasets.affine_transform import AffineTransform

def load_images(args, paths):
    images = []
    for path in tf.unstack(paths):
        img_byte = tf.read_file(path)
        image = tf.image.decode_image(img_byte)

        # --------------------------------Image Augmentation--------------------------------
        if args.do_affine:
            affine = AffineTransform(translation=args.translation, scale=args.scale, shear=args.shear,
                                     rotation=args.rotation, project=args.projects, mean=args.imgaug_mean,
                                     stddev=args.imgaug_stddev, order=args.imgaug_order)
            affine_mat = affine.to_transform_matrix()
            image = tf.contrib.image.transform(image, affine_mat)
        if args.random_brightness:
            image = tf.image.random_brightness(image, max_delta=args.imgaug_max_delta)
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

def load_images_from_file(args, paths):
    """
    load image and its additional information, such as bound box, based on text files.
    :param args:
    :param paths: files that contains image path and information such as txt, csv, mat, xml, ...
    :return:
    """
    images = []
    #for path in tf.unstack(paths):


def load_path_from_folder(paths, dig_level=0):
    """
    'paths' is a list or tuple, which means you want all the sub paths within 'dig_level' levels.
    'dig_level' represent how deep you want to get paths from.
    """
    output = []
    for path in paths:
        current_folders = [path]
        # Do not delete the following line, we need this when dig_level is 0.
        sub_folders = []
        while dig_level > 0:
            sub_folders = []
            for sub_path in current_folders:
                sub_folders += glob.glob(sub_path + "/*")
            current_folders = sub_folders
            dig_level -= 1
        sub_folders = []
        for _ in current_folders:
            sub_folders += glob.glob(_ + "/*")
        output.append(sub_folders)
    return output