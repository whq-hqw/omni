
import os, glob

def load_path_from_folder(paths, dig_level=1):
    """
    path is a list or tuple
    dig_level represent how deep you want to get images from.
    """
    output = []
    for path in paths:
        current_folders = [path]
        while dig_level > 0:
            sub_folders = []
            for sub_path in current_folders:
                sub_folders += glob.glob(sub_path + "/*")
            current_folders = sub_folders
            dig_level -= 1
        output.append([glob.glob(_) for _ in current_folders])
    return output