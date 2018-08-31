import os

def load_img_path(img_folder):
    folder = os.path.expanduser(img_folder)
    classes = os.listdir(folder)
