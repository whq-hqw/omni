

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

if __name__ == "__main__":
    pass