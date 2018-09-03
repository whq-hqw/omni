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
        