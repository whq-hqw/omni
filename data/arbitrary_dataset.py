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

import os, random
from misc import *

class Arbitrary_Dataset(object):
    def __init__(self, args, sources, step_1, step_2, pre_process=None, bbox_loader=None,
                 auxiliary_info=None, augmentation=None, as_generator=False, **options):
        """
        A generalized data initialization method, inherited by other data class, e.g. ilsvrc, img2img, etc.
        Arbitrary is a parent class of all other data class.

        :param args: options from terminal
        :param sources: a list of input and output source folder or file(e.g. csv, mat, xml, etc.)
        :param step_1: a list of functions to load path or other infomation from sources
        :param step_2: a list of functions to load infomation from step_1 to PyTorch readable tensor
        :param pre_process: will be invoked immediately after step_2 has loaded the data, e.g. pre process the images
        :param auxiliary_info: when loading path from a folder, it might contains some subfolders you don't
         want to load or other operations you wanted to add.
        you to load as deep as you want to.
        :param options: For Future upgrade.
        """
        assert len(sources) == len(step_1), \
            "Length of 'sources', 'step_1' must be the same."
        self.args = args
        self.sources = sources
        self.step_1 = step_1
        self.step_2 = step_2
        self.as_generator = as_generator

        num_of_data = len(step_2)
        self.auxiliary = self.standardize_input(auxiliary_info, num_of_data)
        self.pre_process = self.standardize_input(pre_process, num_of_data)
        self.bbox_loader = self.standardize_input(bbox_loader, num_of_data)
        if args.do_imgaug:
            self.augmentation = self.standardize_input(augmentation, num_of_data)
        if args.to_final_size:
            self.sizes = args.final_size
            assert len(self.sizes) == num_of_data
            for size in self.sizes:
                assert len(size) == 2, "each element inside the args.final_size have 2 dimensions, " \
                                       "height and width, respectively"
        else:
            self.sizes = [None] * num_of_data
        
    def prepare(self):
        if len(self.sources) == 1:
            print("Loading data from: %s."%(os.path.join(self.args.path, self.sources[0])))
        elif len(self.sources) > 1:
            print("Loading data from %s locations."%(len(self.sources)))
            for i, source in enumerate(self.sources):
                if type(source) is str:
                    self.show_sub_folder_data(source, i)
                elif type(source) is tuple or type(source) is list:
                    for sub_source in source:
                        self.show_sub_folder_data(sub_source, i)
                else:
                    raise TypeError
        else:
            raise ValueError("Length of the source must be larger than 0")
        self.dataset = self.load_dataset()
        print("Number of samples in dataset is: %s"%(len(self.dataset)))

    def summary(self):
        #print("data loading pipeline summarization function will be implemented soon")
        for i, source in enumerate(self.sources):
            print("Print the piprline of loading %s"%(source))
            print("1. Path and other infomation will be loaded by:")
            print(self.step_1)
            step = 1
            if self.pre_process[i] is not None:
                step += 1
                print("%s. During data loading, pre-process will be executed:"%(step))
                print(self.pre_process[i])


    def test_augmentation(self):
        from imgaug import augmenters
        aug = augmenters.Sequential(self.augmentation, random_order=False)
        for data in self.dataset:
            print(data)

    @staticmethod
    def standardize_input(input, num_of_data):
        if input is not None:
            assert len(input) == num_of_data
            return input
        else:
            return [None] * num_of_data
    
    def show_sub_folder_data(self, source, idx):
        source_path = os.path.join(self.args.path, source)
        if not os.path.exists(source_path):
            raise FileNotFoundError("%s does not exist on your disk, please check."
                                    % source_path)
        print("|==> %d. %s" % (idx + 1, os.path.join(self.args.path, source)))
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        result = []
        if self.args.random_order_load:
            items = []
            item_num = len(self.dataset[0])
            up_range = len(self.dataset)
            selected_j = []
            for i in range(item_num):
                j = random.randint(0, up_range-1)
                # avoid repeated elements
                while j in selected_j:
                    #print("repeat element ENCOUNTERED!")
                    j = random.randint(0, up_range - 1)
                selected_j.append(j)
                items.append(self.dataset[j][i])
        else:
            items = self.dataset[index]
        assert len(items) is len(self.step_2), "length of item and mode should be same."
        if self.args.deterministic_train:
            seed = index + self.args.curr_epoch
        else:
            # seed is used to keep same image augmentation manner when load things from one item
            seed = random.randint(0, 100000)
        for i in range(len(items)):
            if callable(self.step_2[i]):
                result.append(self.step_2[i](args=self.args, items=items[i], seed=seed, size=self.sizes[i],
                                             pre_process=self.pre_process[i], rand_aug=self.augmentation[i],
                                             bbox_loader=self.bbox_loader[i]))
            else:
                raise TypeError
        if self.as_generator:
            yield result
        else:
            return result
        

    def load_dataset(self):
        data = []
        path = os.path.expanduser(self.args.path)
        assert len(self.sources) is len(self.step_1), "sources and modes should be same dimensions."
        input_types = len(self.step_1)
        for i in range(input_types):
            sub_paths = []
            for source in self.sources:
                if type(source) is str:
                    sub_paths.append(os.path.join(path, source))
                elif type(source) is tuple or type(source) is list:
                    # In this case, there are multiple inputs in one source,
                    # but we want it to be considered as one file
                    sub_paths.append([os.path.join(path, _) for _ in source])
                else:
                    raise TypeError
            if callable(self.step_1[i]):
                # mode[i] is a function
                data += self.step_1[i](self.args, len(data), sub_paths[i], self.auxiliary[i])
            else:
                raise NotImplementedError
        dataset = []
        data_pieces = max([len(_) for _ in data])
        for key in range(len(data)):
            data[key] = compliment_dim(data[key], data_pieces)
        for i in range(data_pieces):
            dataset.append([_[i] for _ in data])
        return dataset

if __name__ == "__main__":
    pass