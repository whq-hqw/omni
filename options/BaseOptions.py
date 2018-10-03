import os, argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        self.parser.add_argument("--gpu_id", type=str, default="0",
                                 help="if None, then use CPU mode.")
        self.parser.add_argument("--gpu_memory_fraction", type=float, default=0.9,
                                 help="GPU memory usage.")

        self.parser.add_argument("--epoch_num", type=int, default=100,
                                 help="Total training epoch")
        
        # -------------------------------Optimizer------------------------------------
        self.parser.add_argument("--optimizer", type=str, default="adam")
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
        self.parser.add_argument("--img_size", type=int, default=320,
                                 help="size of input images")
        self.parser.add_argument("--img_channel", type=int, default=3,
                                 help="3 stand for color image while 1 for greyscale")
        self.parser.add_argument("--random_crop", type=bool, default=True,
                                 help="randomly crop an image")
        self.parser.add_argument("--random_flip", type=bool, default=True,
                                 help="randomly flip an image")
        self.parser.add_argument("--img2img", type=bool, default=True,
                                 help="randomly flip an image")

        # ------------------------------Miscellaneous----------------------------------
        self.parser.add_argument("--log_dir", type=str, help="log location")
        args = self.parser.parse_args()
        return args

if __name__ == "__main__":
    opt = BaseOptions().initialize()
    print(opt)