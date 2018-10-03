import tensorflow as tf
import numpy as np
import os
import datasets.miscellaneous as misc


def ilsvrc_dataset(path, extensions=("jpg", "jpeg", "JPG", "png", "PNG", "bmp", "BMP"), verbose=False):
    dataset = {}
    path = os.path.expanduser(path)
    classes = os.listdir(path)
    classes = [cls for cls in classes if os.path.isdir(os.path.join(path, cls))]
    for i, cls in enumerate(classes):
        if verbose:
            print('Loading {}th {} class.'.format(i, cls))
        dataset.update({cls: [_ for _ in os.listdir(os.path.join(path, cls)) if
                    misc.extension_check(_, extensions)]})
    print("Dataset loading is complete.")
    return dataset

def img2img_dataset(path, A_B_folder=("trainA", "trainB"), one_to_one=True,
                        extensions=("jpg", "jpeg", "JPG", "png", "PNG", "bmp", "BMP"), verbose=False):
    dataset = {}
    path = os.path.expanduser(path)
    assert len(A_B_folder) == 2, "A_B_folder should be the name of source and target folder."
    source = os.path.join(path, A_B_folder[0])
    target = os.path.join(path, A_B_folder[1])
    assert os.path.isdir(source) and os.path.isdir(target), "one of the folder does not exist."
    if one_to_one:
        source_imgs = [os.path.join(path, A_B_folder[0], _) for _ in os.listdir(source)
                       if misc.extension_check(_, extensions)]
        target_imgs = [os.path.join(path, A_B_folder[1], _) for _ in os.listdir(target)
                       if misc.extension_check(_, extensions)]
        if verbose: print("Sorting files...")
        source_imgs.sort()
        target_imgs.sort()
        if verbose: print("Sorting completed.")
        assert len(source_imgs) == len(target_imgs)
        for i in range(len(source_imgs)):
            if verbose and i % 100 == 0:
                print("{} samples has been loaded...".format(i))
            dataset.update({target_imgs[i]: source_imgs[i]})
    else:
        dataset.update({"A": [os.path.join(path, _) for _ in os.listdir(source) if misc.extension_check(_, extensions)]})
        dataset.update({"B": [os.path.join(path, _) for _ in os.listdir(target) if misc.extension_check(_, extensions)]})
        #dataset.append([Sample(label="B", path=_) for _ in os.listdir(target) if extension_check(_, extensions)])
    print('Dataset loading is complete.')
    return dataset

def arbitrary_dataset(path, folder_names, data_load_funcs, dig_level=None):
    dataset = {}
    path = os.path.expanduser(path)
    assert len(folder_names) is len(data_load_funcs), "folder_names and functions should be same dimensions."
    for i in range(len(folder_names)):
        key = misc.number_to_char(i)
        arg = [os.path.join(path, _) for _ in folder_names[i]]
        try:
            value = data_load_funcs[i](arg, dig_level[i])
        except (TypeError, IndexError):
            value = data_load_funcs[i](arg)
        dataset.update({key: value})
    return dataset

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
        #
        core=[]
        for i, paths in enumerate(dequeue_obj):
            try:
                load_func = functions[i]
            except (TypeError, IndexError):
                load_func = load_images
            data = load_func(args, paths, seed)
            core.append(data)
            cell.append(core)
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

if __name__ == "__main__":
    dataset = arbitrary_dataset(path="~/Pictures/dataset/buddha",
                                     folder_names=[("trainA", "trainB", "testA", "testB")],
                                     functions=[misc.load_path_from_folder], dig_level=[0, 0, 0, 0])