"""
# Copyright (c) 2019 Wang Hanqin.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

import random, warnings, itertools
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import omni_torch.data as data
import omni_torch.utils as util
import omni_torch.data.augmentation as aug
import imgaug
from imgaug import augmenters


def read_image_with_bbox(args, items, seed, size, pre_process=None, rand_aug=None,
               bbox_loader=None, _to_tensor=True):
    """
    Default image loading function invoked by Dataset object(Arbitrary, Img2Img, ILSVRC)
    :param args:
    :param path:
    :param seed:
    :param size: size may be different especially for super-resolution research
    :param pre_process: callable functions, perform special options on images,
            ops can divide img into several patches in order to save memory
            ops can invert image color, switch channel, increase contrast
            ops can also calculate the infomation extactable brom image, e.g. affine matrix
    :return:
    """
    if bbox_loader:
        # image should be an np.ndarray
        # bbox should be an imgaug BoundingBoxesOnImage instance
        image, bbox, box_label = bbox_loader(args, items, seed, size)
    else:
        path = items
        image = load_img(args, path)
        bbox = None
    if pre_process:
        image, data = pre_process(image, args, items, seed, size)
    else:
        data = None
    # If pre-process returns some information about deterministic augmentation
    # Then initialize the deterministic augmentation based on that information
    det_aug_list = aug.prepare_deterministic_augmentation(args, data)
    aug_seq = aug.combine_augs(det_aug_list, rand_aug)
    if bbox:
        if aug_seq:
            # Do random augmentaion defined in pipline declaration
            aug_seq = aug_seq.to_deterministic()
            image = aug_seq.augment_image(image)
            bbox = aug_seq.augment_bounding_boxes([bbox])[0]
            #image_after = bbox.draw_on_image(image, thickness=2, color=[0, 0, 255])
            #cv2.imwrite("/home/wang/Pictures/tmp_after.jpg", image_after)
        # numpy-lize bbox
        coords = []
        labels = []
        h, w = image.shape[0], image.shape[1]
        for i, box in enumerate(bbox.bounding_boxes):
            condition_1 = box.x1 <= 0 and box.x2 <= 0
            condition_2 = box.y1 <= 0 and box.y2 <= 0
            condition_3 = box.x1 >= w -1 and box.x2 >= w -1
            condition_4 = box.y1 >= h -1 and box.y2 >= h -1
            if condition_1 or condition_2 or condition_3 or condition_4:
                # After aigmentation, at least one dimension of the bbox exceeded the image
                # omni_torch will ignore this bbox
                continue
            horizontal_constrain = lambda x: max(min(w, x), 0)
            vertival_constrain = lambda y: max(min(h, y), 0)
            coords.append([horizontal_constrain(box.x1)/w, vertival_constrain(box.y1)/h,
                           horizontal_constrain(box.x2)/w, vertival_constrain(box.y2)/h])
            labels.append(box_label[i])
        coords = torch.Tensor(coords)
        labels = torch.Tensor(labels)
    else:
        # With no bounding boxes, augment the image only
        if aug_seq:
            aug_seq = aug_seq.to_deterministic()
            image = aug_seq.augment_image(image)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    if _to_tensor:
        if bbox:
            return to_tensor(args, image, seed, size), coords, labels
        return to_tensor(args, image, seed, size)
    else:
        if bbox:
            return image, coords, labels
        return image


def read_image(args, items, seed, size, pre_process=None, rand_aug=None,
               bbox_loader=None, _to_tensor=True):
    """
    Default image loading function invoked by Dataset object(Arbitrary, Img2Img, ILSVRC)
    :param args:
    :param path:
    :param seed:
    :param size: size may be different especially for super-resolution research
    :param pre_process: callable functions, perform special options on images,
            ops can divide img into several patches in order to save memory
            ops can invert image color, switch channel, increase contrast
            ops can also calculate the infomation extactable brom image, e.g. affine matrix
    :return:
    """
    if type(items) is str:
        items = [items]
    images = []
    for path in items:
        images.append(load_img(args, path))
    if pre_process:
        images = pre_process(images, args, items, seed, size)
    aug_seq = augmenters.Sequential(rand_aug, random_order=False)
    if aug_seq:
        aug_seq = aug_seq.to_deterministic()
        images = aug_seq.augment_images(images)
    for i, image in enumerate(images):
        if len(image.shape) == 2:
            images[i] = np.expand_dims(image, axis=-1)
    if _to_tensor:
        tensor = [to_tensor(args, image, seed, size) for image in images]
        if len(tensor) == 1:
            return tensor[0]
        else:
            return tensor
    else:
        if len(images) == 1:
            return images[0]
        else:
            return images


def load_img(args, path):
    """
    A generalized image loading function, support n-bit, n-channel images
    :param args:
    :param path: string-path or list of string-paths
    :return:
    """
    # -1 means it adapts to any bit-depth image
    # e.g. 8-bit, 12-bit, 14-bit, 16-bit, and etc.
    image = cv2.imread(path, -1)
    if image.shape[-1] == 4:
        # RGB-A image
        if args.img_channel is 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if args.img_channel is 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[-1] == 3:
        if args.img_channel is 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[-1] == 1:
        image = np.squeeze(image)
    else:
        if len(image.shape) == 2:
            pass
        else:
            raise ValueError("Image shape should not be: %s" % (str(image.shape)))
    return image


def to_tensor(args, image, seed=None, size=None):
    image = util.normalize_image(args, image)
    trans = T.Compose([T.ToTensor()])
    return trans(image.astype("float32"))


def to_tensor_with_aug(args, image, seed, size, rand_aug):
    if args.do_imgaug:
        imgaug.seed(seed)
        image = rand_aug.augment_image(image)
    return to_tensor(args, image, seed, size)


def just_return_it(args, data, seed, size):
    """
    Because the label in cifar dataset is int
    So here it will be transfered to a torch tensor
    """
    return torch.tensor(data, dtype=torch.float)


def one_hot(label_num, index):
    assert type(label_num) is int and type(index) is int, "Parameters Error"
    return torch.eye(label_num)[index]


if __name__ == "__main__":
    import os
    img_path = os.path.expanduser("~/Pictures/sample.jpg")

