import os

class Arbitrary():
    """
    Load a dataset with arbitrary form, and return a dictionary
    """
    def __init__(self, path, sub_files, data_load_funcs, dig_level=None):
        """
        :param path: dataset's root folder
        :param sub_files: all the sub-folders or files you want to read(correspond to data_load_funcs)
        :param data_load_funcs: the way you treat your sub-folders and files(correspond to folder_names)
        :param dig_level: how deep you want to find the sub-folders
        :return: a dataset in the form of dict {'A':[...], 'B':[...], ...}
        """
        assert type(path) is str, "path type error"
        self.path = path

        assert type(sub_files) is list, "folder_names type error"
        for i in sub_files:
            assert type(sub_files[i]) is str, "folder_name should be a string path"
        self.folder_names = sub_files

        assert type(data_load_funcs) is list, "folder_names type error"
        for i in data_load_funcs:
            assert callable(data_load_funcs[i]), "data_load_func should be a function(callable)"
        self.folder_names = sub_files

        if dig_level is None:
            self.dig_level = 0
        else:
            self.dig_level = dig_level


    def get_path(self):
        dataset = {}
        path = os.path.expanduser(self.path)
        assert len(self.folder_names) is len(self.data_load_funcs), "folder_names and functions should be same dimensions."
        for i in range(len(self.folder_names)):
            key = misc.number_to_char(i)
            arg = [os.path.join(path, _) for _ in self.folder_names[i]]
            try:
                value = self.data_load_funcs[i](arg, self.dig_level[i])
            except (TypeError, IndexError):
                value = self.data_load_funcs[i](arg)
            dataset.update({key: value})
        return dataset