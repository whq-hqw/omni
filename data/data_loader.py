import os, glob
import data.miscellaneous as misc


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

def arbitrary_dataset(path, sources, modes, dig_level=None):
    """
    :param path: dataset's root folder
    :param sources: all the sub-folders or files you want to read(correspond to data_load_funcs)
    :param data_load_funcs: the way you treat your sub-folders and files(correspond to folder_names)
    :param dig_level: how deep you want to find the sub-folders
    :return: a dataset in the form of dict {'A':[...], 'B':[...], ...}
    """
    dataset = {}
    path = os.path.expanduser(path)
    assert len(sources) is len(modes), "sources and modes should be same dimensions."
    input_types = len(modes)
    for i in range(input_types):
        sub_paths = [os.path.join(path, _) for _ in sources]
        if modes[i] is "path":
            dataset.update(load_path_from_folder(len(dataset), sub_paths[i], dig_level[i]))
        # We can add other modes if we want
        elif callable(modes[i]):
            # mode[i] is a function
            dataset.update(modes[i](len(dataset), sub_paths[i], dig_level[i]))
        else:
            raise NotImplementedError
    return dataset


def load_path_from_folder(len, paths, dig_level=0):
    """
    'paths' is a list or tuple, which means you want all the sub paths within 'dig_level' levels.
    'dig_level' represent how deep you want to get paths from.
    """
    output = []
    if type(paths) is str:
        paths = [paths]
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
        output += sub_folders
    # 1->A, 2->B, 3->C, ..., 26->Z
    key = misc.number_to_char(len)
    return {key: output}