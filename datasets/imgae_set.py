
import os

def load_path_from_folder(path, dig_level=1):
    files = os.listdir(path)
    current_path = path
    if dig_level > 1:
        pass
    for i in range(dig_level-1):
        level = dig_level - 1 - i
        sub_files = load_path_from_folder(current_path, dig_level=level)
    paths = [_ for _ in files]
    return files