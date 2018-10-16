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