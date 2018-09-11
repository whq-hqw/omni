import tensorflow as tf
import os, random

class Sample():
    def __init__(self, label, path):
        assert path is not None, "Path should not be None"
        self.label = label
        self.path = path

    def __str__(self):
        return self.path + ": " + self.label
    

def extension_check(file, extentions):
    if file[file.rfind(".")+1:] in extentions:
        return True
    else:
        return False

def get_ilsvrc_dataset(path, extensions=set(["jpg", "jpeg", "JPG", "png", "PNG", "bmp", "BMP"]), verbose=False):
    dataset = []
    path = os.path.expanduser(path)
    classes = os.listdir(path)
    classes = [cls for cls in classes if os.path.isdir(os.path.join(path, cls))]
    for cls in classes:
        if verbose:
            print('Loading data, Processing {}'.format(cls))
        dataset += [Sample(label=cls, path=_) for _ in os.listdir(os.path.join(path, cls)) if
                    extension_check(_, extensions)]
    print("Dataset loading is complete.")
    return dataset

def get_img2img_dataset(path, A_B_folder=["trainA", "trainB"], one_to_one=True,
                        extensions=set(["jpg", "jpeg", "JPG", "png", "PNG", "bmp", "BMP"]), verbose=False):
    dataset = []
    path = os.path.expanduser(path)
    assert len(A_B_folder) == 2, "A_B_folder should be the name of source and target folder."
    source = os.path.join(path, A_B_folder[0])
    target = os.path.join(path, A_B_folder[1])
    assert os.path.isdir(source) and os.path.isdir(target), "one of the folder does not exist."
    if one_to_one:
        source_imgs = [os.path.join(A_B_folder[0], _) for _ in os.listdir(source) if extension_check(_, extensions)]
        target_imgs = [os.path.join(A_B_folder[1], _) for _ in os.listdir(target) if extension_check(_, extensions)]
        if verbose: print("Sorting files...")
        source_imgs.sort()
        target_imgs.sort()
        if verbose: print("Sorting completed.")
        assert len(source_imgs) == len(target_imgs)
        for i in range(len(source_imgs)):
            if verbose and i % 100 == 0:
                print("{} samples has been loaded...".format(i))
            dataset.append(Sample(label=target_imgs[i], path=source_imgs[i]))
    else:
        dataset.append([Sample(label="A", path=_) for _ in os.listdir(source) if extension_check(_, extensions)])
        print("Source data loaded.")
        dataset.append([Sample(label="B", path=_) for _ in os.listdir(target) if extension_check(_, extensions)])
    print("Dataset loading is complete.")
    return dataset

def get_dataset_with_addtional_info(path, extensions=None, verbose=False):
    # TODO: Implement this dataload method
    dataset = []
    path = os.path.expanduser(path)
    raise NotImplementedError

def image_augumentation(img_tensor):
    # TODO: Image Augumentation
    return img_tensor

def get_shape(placeholder):
    try:
        result = [_ for _ in placeholder.shape.as_list() if _ is not None]
    except ValueError:
        result = ()
    return result

def data_load_graph(img_path, ground_truth, threads, batch_size, output_shape, capacity=100000):
    # Find None Dimension's location
    img_path_shape = get_shape(img_path)
    ground_truth_shape = get_shape(ground_truth)

    input_queue = tf.FIFOQueue(capacity=capacity, dtypes=[img_path.dtype, ground_truth.dtype],
                               shapes=[img_path_shape, ground_truth_shape])
    enqueue_op = input_queue.enqueue_many([img_path, ground_truth])

    images_and_labels = []
    for _ in range(threads):
        img_path, label = input_queue.dequeue()
        img = tf.read_file(img_path)
        img=image_augumentation(img)
        images_and_labels.append(img, label)
    image_batch, label_batch = tf.train.batch_join(images_and_labels, batch_size=batch_size,
                                                   capacity=4 * batch_size * threads,
                                                   shapes=output_shape,
                                                   enqueue_many=True, allow_smaller_final_batch=True)
    return enqueue_op, image_batch, label_batch

if __name__ == "__main__":
    #dataset = get_ilsvrc_dataset(path="~/Downloads/Dataset/pedestrain")
    #print(len(dataset))
    
    dataset = get_img2img_dataset(path="~/Pictures/dataset/buddha")
    print(len(dataset))
