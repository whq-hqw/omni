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

import warnings
import torch
import torchvision.transforms as T
import cv2
import numpy as np
import omni_torch.data as data


def segment_image(image, args, path, seed, size, to_tensor=False):
    """
    :param image: input image
    :param args:
    :param path:
    :param seed:
    :param size: segmented image size(output), list or tuple with 2 elements
    :return: [w, h, C] => [w/slice_w, h/slice_h, C*slice_w*sliceH]
    """
    assert len(args.segment_patch_size) == 2, "image patch size should contains exactly 2 dimensions"
    # Format the pieces of slice
    if args.segments:
        assert len(args.segments) <= 2, "slice shoud no longer then 2 dimensions"
        if len(args.segments) == 1:
            slice = (args.segments[0], args.segments[0])
        else:
            slice = tuple(args.segments)
    else:
        slice = (int(image.shape[0] / args.segment_patch_size[0]), int(image.shape[1] / args.segment_patch_size[1]))

    # Confirm the aspect ratio of input image and output image
    ori_ratio = image.shape[1] / image.shape[0]
    new_ratio = args.segment_patch_size[1] * slice[1] / args.segment_patch_size[0] / slice[0]
    if ori_ratio / new_ratio > 1.3 or ori_ratio / new_ratio < 0.7:
        warnings.warn(
            "the ratio of output image is significantly different from original image, please modify the slice or output_size")

    # Perform segmentation
    image = cv2.resize(image, (args.segment_patch_size[0]*slice[0], args.segment_patch_size[1]*slice[1]))
    if len(image.shape) == 2:
        # When it was a grayscale image:
        image = np.expand_dims(image, axis=-1)
    imgs = []
    if to_tensor:
        for i in range(slice[0]):
            for j in range(slice[1]):
                img = T.ToTensor()(image[i * args.segment_patch_size[0]:(i + 1) * args.segment_patch_size[1],
                                                              j * args.segment_patch_size[1]:(j + 1) * args.segment_patch_size[1], :])
                imgs.append(torch.unsqueeze(img, 0))
        return torch.cat(imgs, dim=0)
    else:
        for i in range(slice[0]):
            for j in range(slice[1]):
                imgs.append(image[i * args.segment_patch_size[0]:(i + 1) * args.segment_patch_size[1],
                    j * args.segment_patch_size[1]:(j + 1) * args.segment_patch_size[1], :])
        return imgs
    

def contrast_normalization(image, args, path, seed, size):
    return

def inverse_image(image, args, path, seed, size):
    image = cv2.bitwise_not(image)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    image = T.ToTensor()(image)
    return image

def segment_and_inverse(image, args, path, seed, size):
    return 1- segment_image(image, args, path, seed, size)
