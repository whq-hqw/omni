import os, argparse
from options.imgaug_options import ImgAug

class BaseOptions(ImgAug):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        ImgAug.initialize(self)
        self.parser.add_argument("--gpu_id", type=str, default="0",
                                 help="if None, then use CPU mode.")
        self.parser.add_argument("--gpu_memory_fraction", type=float, default=0.9,
                                 help="GPU memory usage.")

        self.parser.add_argument("--epoch_num", type=int, default=100,
                                 help="Total training epoch")
        
        # -------------------------------Optimizer------------------------------------
        self.parser.add_argument("--optimizer", type=str, default="Adam")
        self.parser.add_argument("--learning_rate", type=float, default=0.0005)

        # ------------------------------Input Queue-----------------------------------
        self.parser.add_argument("--threads", type=int, default=4,
                                 help="Input data queue loading threads")
        self.parser.add_argument("--batch_size", type=int, default=8)
        self.parser.add_argument("--capacity", type=int, default=10000,
                                 help="input data queue capacity")

        # -------------------------------Input Images------------------------------------
        self.parser.add_argument("--path", type=str, required=True,
                                 help="the path of dataset")
        
        # ------------------------------Miscellaneous----------------------------------
        self.parser.add_argument("--log_dir", type=str, help="log location")
        args = self.parser.parse_args()
        return args

if __name__ == "__main__":
    opt = BaseOptions().initialize()
    print(opt)