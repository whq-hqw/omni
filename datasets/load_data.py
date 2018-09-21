import tensorflow as tf
import os, random

def extension_check(file, extentions):
    if file[file.rfind(".")+1:] in extentions:
        return True
    else:
        return False

def ilsvrc_dataset(path, extensions=set(["jpg", "jpeg", "JPG", "png", "PNG", "bmp", "BMP"]), verbose=False):
    dataset = {}
    path = os.path.expanduser(path)
    classes = os.listdir(path)
    classes = [cls for cls in classes if os.path.isdir(os.path.join(path, cls))]
    for i, cls in enumerate(classes):
        if verbose:
            print('Loading {}th {} class.'.format(i, cls))
        dataset.update({cls: [_ for _ in os.listdir(os.path.join(path, cls)) if
                    extension_check(_, extensions)]})
    print("Dataset loading is complete.")
    return dataset

def img2img_dataset(path, A_B_folder=["trainA", "trainB"], one_to_one=True,
                        extensions=set(["jpg", "jpeg", "JPG", "png", "PNG", "bmp", "BMP"]), verbose=False):
    dataset = {}
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
            dataset.update({target_imgs[i]: source_imgs[i]})
    else:
        dataset.update({"A": [_ for _ in os.listdir(source) if extension_check(_, extensions)]})
        dataset.update({"B": [_ for _ in os.listdir(target) if extension_check(_, extensions)]})
        #dataset.append([Sample(label="B", path=_) for _ in os.listdir(target) if extension_check(_, extensions)])
    print("Dataset loading is complete.")
    return dataset

def dataset_with_addtional_info(path, extensions=None, verbose=False):
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

def data_load_graph(img_path, ground_truth, threads, batch_size, output_shape, capacity):
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
    #dataset = ilsvrc_dataset(path="~/Downloads/Dataset/pedestrain")
    #print(len(dataset))
    
    dataset = img2img_dataset(path="~/Pictures/dataset/buddha")
    print(len(dataset))
