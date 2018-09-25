import tensorflow as tf
import numpy as np
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
    print('Dataset loading is complete.')
    return dataset

def dataset_with_addtional_info(path, extensions=None, verbose=False):
    # TODO: Implement this dataload method
    dataset = []
    path = os.path.expanduser(path)
    raise NotImplementedError


def load_image(args, path, seed):
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
    return tf.image.per_image_standardization(image)

def get_shape(placeholder):
    try:
        result = [_ for _ in placeholder.shape.as_list() if _ is not None]
    except ValueError:
        result = ()
    return result

def data_load_graph(args, input_queue, output_shape):
    images_and_labels = []
    for _ in range(args.threads):
        seed = np.random.randint(4096)
        print("data_load_graph random seed:{}".format(seed))
        img_paths, labels = input_queue.dequeue()
        images = []
        for path in tf.unstack(img_paths):
            images.append(load_image(args, path, seed))
        if args.img2img:
            gt_images = []
            for gt_path in tf.unstack(labels):
                gt_images.append(load_image(args, gt_path, seed))
        images_and_labels.append([images, gt_images])
    # The actual shape of images_and_labels is:
    #    [[several images], [several labels]]
    #    Where images and labels can be arbitrary shapes
    image_batch, label_batch = tf.train.batch_join(images_and_labels, batch_size=args.batch_size,
                                                   capacity=4 * args.batch_size * args.threads,
                                                   shapes=output_shape,
                                                   enqueue_many=True, allow_smaller_final_batch=True)
    return image_batch, label_batch

if __name__ == "__main__":
    #dataset = ilsvrc_dataset(path="~/Downloads/Dataset/pedestrain")
    #print(len(dataset))
    
    dataset = img2img_dataset(path="~/Pictures/dataset/buddha")
    print(len(dataset))
