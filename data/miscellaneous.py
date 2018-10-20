import numpy as np

def number_to_char(num):
    assert (num >= 0 and num < 26), "Max 26 kind of input are supported."
    return chr(num+65)

def extension_check(file, extensions):
    if file[file.rfind(".")+1:] in extensions:
        return True
    else:
        return False

def get_shape(placeholder):
    try:
        result = [_ for _ in placeholder.shape.as_list() if _ is not None]
    except ValueError:
        result = ()
    return result

def compliment_dim(input, dim):
    assert len(input) <= dim, "length of input should not larger than dim."
    if len(input) == dim:
        return input
    else:
        repeat = dim // len(input)
        if type(input) is list:
            input = input * repeat + input[:dim - len(input) * repeat]
        elif type(input) is np.ndarray:
            input = np.concatenate((np.tile(input, repeat), input[:dim - len(input) * repeat]))
        return input

if __name__ == "__main__":
    pass