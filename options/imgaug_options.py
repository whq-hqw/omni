"""
# Copyright (c) 2018 Works Applications Co., Ltd.
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

import argparse

class ImgAug:
    def initialize(self):
        self.parser = argparse.ArgumentParser()
        
        # Image Loading
        self.parser.add_argument("--img_channel", type=int, default=3,
                                 help="3 stand for color image while 1 for greyscale")

        # Image Augmentation
        self.parser.add_argument("--do_imgaug", type=bool, default=True,
                                 help="do image augmentation on image or not")
        
        self.parser.add_argument("--do_affine", type=bool, default=True,
                                 help="do affine transformation on image or not")
        self.parser.add_argument("--translation", type=tuple, default=(0.0, 0.0),
                                 help="a list of two elements indicate translation on x and y axis")
        self.parser.add_argument("--scale", type=tuple, default=(1.0, 1.0),
                                 help="a list of two elements indicate scale on x and y axis")
        self.parser.add_argument("--shear", type=tuple, default=(-0.0, 0.0),
                                 help="a list of two elements indicate shear on x and y axis")
        self.parser.add_argument("--rotation", type=tuple, default=(-0.0, 0.0),
                                 help="a number indicate rotation of image")
        self.parser.add_argument("--aug_bg_color", type=int, default=255,
                                 help="background color of augmentation, 0=black, 255=white")
        
        self.parser.add_argument("--do_crop_to_fix_size", type=bool, default=True,
                                 help="randomly crop an image")
        self.parser.add_argument("--crop_size", type=tuple,default=(360, 360),
                                 help="the image will be cropped to this size")
        self.parser.add_argument("--keep_size", type=bool, default=True,
                                 help="Keep ratio or not")

        self.parser.add_argument("--do_random_flip", type=bool, default=True,
                                 help="randomly flip an image horizontally and vertically")
        self.parser.add_argument("--v_flip_prob", type=float, default=0.5,
                                 help="probability of image be flipped")
        self.parser.add_argument("--h_flip_prob", type=float, default=0.5,
                                 help="probability of image be flipped")

        self.parser.add_argument("--do_random_brightness", type=bool, default=True,
                                 help="adjust brightness on image")
        self.parser.add_argument("--brightness_vibrator", type=tuple, default=(1.0, 1.0),
                                 help="adjust brightness on image")
        self.parser.add_argument("--brightness_multiplier", type=tuple, default=(1.0, 1.0),
                                 help="adjust brightness on image")
        self.parser.add_argument("--linear_contrast", type=float, default=1.0,
                                 help="adjust contrast on image")
        self.parser.add_argument("--gamma_contrast", type=float, default=1.0,
                                 help="adjust contrast on image")
        self.parser.add_argument("--do_random_noise", type=bool, default=True,
                                 help="apply random gaussian noise on image")

        self.parser.add_argument("--random_augment_order", type=bool, default=True,
                                 help="the order of each augmentation ops will be randomly assigned.")

        self.parser.add_argument("--to_final_size", type=bool, default=True,
                                 help="resize the input image or not")
        self.parser.add_argument("--final_size", type=tuple, default=(224, 224),
                                 help="size of input images")
        self.parser.add_argument("--standardize_size", type=bool, default=True,
                                 help="make the image be able to fit the network.")
        self.parser.add_argument("--standardize_gcd", type=int, default=8,
                                 help="make the width and height of image be divided evenly by gcd")

        self.parser.add_argument("--img_mean", type=tuple, default=(0.5, 0.5, 0.5),
                                 help="mean of random coefficiency of translation")
        self.parser.add_argument("--img_std", type=tuple, default=(1.0, 1.0, 1.0),
                                 help="standard deviation of random coefficiency of translation")

        # Image Segmentation
        self.parser.add_argument("--segments", type=tuple,
                                 help="sliced in W and H dimension of an image")
        self.parser.add_argument("--segment_patch_size", type=tuple,
                                 help="the size of segmented patches from origin image")

