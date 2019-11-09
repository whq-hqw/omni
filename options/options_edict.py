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

from easydict import EasyDict as edict

def initialize():
    """
    :return: default settings
    """
    return edict({
        "code_name": None,
        "cover_exist": True,
        "create_path":True,
        "path": None,  # the directory contains the datasets
        "extensions": ["jpeg", "JPG", "jpg", "png", "PNG", "gif", "tiff"],

        "gpu_id": None,
        "output_gpu_id": 0,
        "epoch_num": 100,
        "deterministic_train": True,
        "seed": 1,

        "batch_size_per_gpu": 8,
        "cumulative_batch_size": None,
        "learning_rate":1e-4,
        "weight_decay":1e-4,

        "curr_epoch": 0,
        "epoches_per_phase": 1,
        "steps_per_epoch": None,

        "model1": None,
        "model2": None,
        "model3": None,
        "model4": None,
        "model5": None,
        
        "loading_threads": 1, # how many cpu core to use to load data, usually 4 is sufficient
        "random_order_load": False,
        
        "img_channel": 3,
        "img_mean": (0.5, 0.5, 0.5),
        "img_std": (1.0, 1.0, 1.0),
        "img_bias": (0.0, 0.0, 0.0),
        "img_bit": 8,
        
        # Below options will be deprecated in the Future
        "do_imgaug": False,
        "imgaug_order": "default",
        # default or a list, ["affine", "crop", "pad", ...], each element represent a process
        "do_affine": False, # See Documentation of imgaug affine
        # numbers in translations means pixel
        "translation_x": (0.0, 0.0),
        "translation_y": (0.0, 0.0),
        # numbers in translations means percentage
        "scale_x": (1.0, 1.0),
        "scale_y": (1.0, 1.0),
        "shear": (-0.0, 0.0),
        # numbers in translations means degree
        "rotation": (-0.0, 0.0),
        "aug_bg_color": 255,
        # Crop
        # searh for CropToFixedSize in imgaug documentation
        "do_crop_to_fix_size": False,
        "crop_size": (224, 224),
        # Pad
        # searh for PadToFixedSize in imgaug documentation
        "do_pad_to_fix_size": False,
        "padding_size": [(224, 224)],
        # in padding_position, 1 and 0 represent image start at the begining and end of
        # corresponding dimension, respectively
        # or choose uniform to randomly place the image
        "padding_position": 'uniform',
        "do_random_flip": False,
        # v represent for vertical, h represent for horizontal
        "v_flip_prob": 0.5,
        "h_flip_prob": 0.5,
        "do_random_brightness": False,
        "brightness_vibrator": (1.0, 1.0),
        "multiplier": (1.0, 1.0),
        "multiplier_per_channel": False,
        "linear_contrast": 1.0,
        "do_random_noise": False,
        "gaussian_sigma": (0, 0.1),
        "to_final_size": False,
        "final_size": (224, 224),
        "standardize_size": False,
        "standardize_gcd": 8,
    })

if __name__ == "__main__":
    opt = initialize()
    print(opt)