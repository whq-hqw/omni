import os, warnings
import tensorflow as tf
import datasets.miscellaneous as misc

def data_load_graph(args, input_queue, output_shape, functions):
    cell = []
    for _ in range(args.threads):
        # Create multi-thread loading processes
        dequeue_obj = input_queue.dequeue()
        component=[]
        for i, paths in enumerate(dequeue_obj):
            load_func = functions[i]
            # load data based on paths, load_func are defined in datasets/data_loader.py
            data = load_func(args, paths)
            component.append(data)
        cell.append(component)
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


def arbitrary_dataset(path, children, data_load_funcs, dig_level=None):
    """
    :param path: dataset's root folder
    :param children: all the sub-folders or files you want to read(correspond to data_load_funcs)
    :param data_load_funcs: the way you treat your sub-folders and files(correspond to folder_names)
    :param dig_level: how deep you want to find the sub-folders
    :return: a dataset in the form of dict {'A':[...], 'B':[...], ...}
    """
    dataset = {}
    path = os.path.expanduser(path)
    assert len(children) is len(data_load_funcs), "folder_names and functions should be same dimensions."
    for i in range(len(children)):
        key = misc.number_to_char(i)
        arg = [os.path.join(path, _) for _ in children[i]]
        try:
            value = data_load_funcs[i](arg, dig_level[i])
        except (TypeError, IndexError):
            value = data_load_funcs[i](arg)
        dataset.update({key: value})
    return dataset


def ilsvrc_dataset(path, extensions=("jpg", "jpeg", "JPG", "png", "PNG", "bmp", "BMP"), verbose=False):
    """
    load all the path of an typical image recognition dataset. e.g. ILSVRC
    :param path: path of the dataset root
    :param extensions: extentions that you treat them as the images.
    :param verbose: display the detail of the process
    :return: a dataset in the form of dict {'A':[...], 'B':[...], ...}
    """
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
    """
    Load all the path of an typical image-to-image translation dataset
    :param path: path of the dataset root
    :param A_B_folder:
    :param one_to_one: if the A & B folder is one-to-one correspondence
    :param extensions: extentions that you treat them as the images.
    :param verbose: display the detail of the process
    :return: a dataset in the form of dict {'A':[...], 'B':[...], ...}
    """
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