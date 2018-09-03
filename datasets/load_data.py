import tensorflow as tf
import os, random

class Sample():
    def __init__(self, label, path):
        assert path is not None, "Path should not be None"
        self.label = label
        self.path = path

    def __str__(self):
        return self.label + " contains " + len(self.path) + " files."

def extension_check(file, extentions):
    if file[file.rfind("."):] in extentions:
        return True
    else:
        return False

def get_ilsvrc_dataset(path, extensions=set(["jpg", "jpeg", "JPG", "png", "PNG", "bmp", "BMP"]), verbose=False):
    dataset = []
    path = os.path.expanduser(path)
    classes = os.listdir(path)
    classes = [_ for _ in classes if os.path.isdir(os.path.join(path, classes))]
    for cls in classes:
        if verbose:
            print('Loading data, Processing {}'.format(cls))
        dataset += [Sample(label=cls, path=_) for _ in os.listdir(os.path.join(path, cls)) if
                    extension_check(_, extensions)]
    print("Dataset loading is complete.")
    return dataset

def get_img2img_dataset(path, A_B_folder=None, one_to_one=True,
                        extensions=set(["jpg", "jpeg", "JPG", "png", "PNG", "bmp", "BMP"]), verbose=False):
    dataset = []
    path = os.path.expanduser(path)
    assert len(A_B_folder) == 2, "A_B_folder should be the name of source and target folder."
    source = os.path.join(path, A_B_folder[0])
    target = os.path.join(path, A_B_folder[1])
    assert os.path.isdir(source) and os.path.isdir(target), "one of the folder does not exist."
    if one_to_one:
        source_imgs = [_ for _ in os.listdir(source) if extension_check(_, extensions)].sort()
        target_imgs = [_ for _ in os.listdir(target) if extension_check(_, extensions)].sort()
        assert len(source_imgs) == len(target_imgs)
        for i in range(len(source_imgs)):
            if verbose and i % 100 == 0:
                print("{} samples has been loaded...".format(i))
            dataset.append(Sample(label=target_imgs[i], path=source_imgs[i]))
    else:
        dataset.append([Sample(label="A", path=_) for _ in os.listdir(source) if extension_check(_, extensions)].sort())
        print("Source data loaded.")
        dataset.append([Sample(label="B", path=_) for _ in os.listdir(target) if extension_check(_, extensions)].sort())
    print("Dataset loading is complete.")
    return dataset

def get_dataset_with_addtional_info(path, extensions=None, verbose=False):
    # TODO: Implement this dataload method
    dataset = []
    path = os.path.expanduser(path)
    raise NotImplementedError
