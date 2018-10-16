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