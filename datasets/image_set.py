
import os, glob

def load_path_from_folder(paths, dig_level=0):
    """
    'paths' is a list or tuple, which means you want all the sub paths within 'dig_level' levels.
    'dig_level' represent how deep you want to get paths from.
    """
    output = []
    for path in paths:
        current_folders = [path]
        sub_folders = []
        while dig_level > 0:
            sub_folders = []
            for sub_path in current_folders:
                sub_folders += glob.glob(sub_path + "/*")
            current_folders = sub_folders
            dig_level -= 1
        sub_folders = []
        for _ in current_folders:
            sub_folders += glob.glob(_ + "/*")
        output.append(sub_folders)
    return output

if __name__ == "__main__":
    pass