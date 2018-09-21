import os, argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        self.parser.add_argument("--path", type=str, required=True,
                                 help="the path of dataset")
        self.parser.add_argument("--img_size", type=int, default=320,
                                 help="size of input images")
        self.parser.add_argument("--img_channel", type=int, default=3,
                                 help="3 stand for color image while 1 for greyscale")
        self.parser.add_argument("--gpu_id", type=str, default="0",
                                 help="if None, then use CPU mode.")

        # -------------------------------Optimizer------------------------------------
        self.parser.add_argument("--optimizer", type=str, default="adam")
        self.parser.add_argument("--learning_rate", type=float, default=0.0005)

        # ------------------------------Input Queue-----------------------------------
        self.parser.add_argument("--loading_threads", type=int, default=8,
                                 help="Input data queue loading threads")
        self.parser.add_argument("--batch_size", type=int, default=64)
        self.parser.add_argument("--output_shape", type=list, default=[(1,), (1,)],
                                 help="input data queue cell shape")
        self.parser.add_argument("--capacity", type=int, default=10000,
                                 help="input data queue capacity")

        # ------------------------------Miscellaneous----------------------------------
        self.parser.add_argument("--log_dir", type=str, help="log location")
        args = self.parser.parse_args()
        return args

if __name__ == "__main__":
    opt = BaseOptions().initialize()
    print(opt)