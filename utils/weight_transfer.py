import os
import h5py, torch
import numpy as np
import keras
from numpy.random import seed as numpy_seed
from tensorflow import set_random_seed
from keras.engine.saving import load_attributes_from_hdf5_group


def initialize_with_keras_hdf5(keras_model, dict_map, torch_model,
                               model_path=None, seed=None):
    """
    :param keras_model: a keras model created by keras.models.Sequential
    :param dict_map: a dictionary maps keys from Kera => PyTorch
    :param torch_model: a PyTorch network
    :param model_path: path where h5 file located, if None, than keras will initialize a new network
    :return: PyTorch StateDict
    """
    if model_path:
        weight_dict = load_weights_from_hdf5(model_path, keras_model)
    else:
        if seed:
            numpy_seed(seed)
            set_random_seed(seed)
        keras_model.compile(keras.optimizers.adam())
        weight_dict = {}
        for layer in keras_model.layers:
            weight_dict.update({layer.name: layer.get_weights()})
    state_dict = torch_model.state_dict()
    for key in weight_dict.keys():
        destiny = dict_map[key]
        for i, item in enumerate(destiny):
            if len(weight_dict[key][i].shape) == 4:
                # Convolutional Layer
                tensor = np.transpose(weight_dict[key][i], (3, 2, 0, 1))
            elif len(weight_dict[key][i].shape) == 2:
                # Full Connection Layer
                tensor = np.transpose(weight_dict[key][i], (1, 0))
            else:
                tensor = weight_dict[key][i]
            state_dict[item] = torch.tensor(tensor)
    torch_model.load_state_dict(state_dict)
    return torch_model

def load_weights_from_hdf5(model_path, keras_model):
    f = h5py.File(model_path, 'r')['model_weights']
    layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
        else keras_model.layers
    filtered_layers = []
    for layer in layers:
        weights = layer.weights
        if weights:
            filtered_layers.append(layer)
    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        if weight_names:
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names
    if len(layer_names) != len(filtered_layers):
        raise ValueError('You are trying to load a weight file '
                         'containing ' + str(len(layer_names)) +
                         ' layers into a model with ' +
                         str(len(filtered_layers)) + ' layers.')
    weight_dict = {}
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
        weight_dict.update({name: weight_values})
    return weight_dict

if __name__ == "__main__":
    from keras.models import Sequential
    from keras.layers import *
    model_path = os.path.join(os.getcwd(), 'models', "cifar10_cnn.h5")
    model = Sequential()
    model.add(Conv2D(32, 3, padding='same', input_shape=(32, 32, 3), activation='relu'))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    weight_dict = load_weights_from_hdf5(model_path, model)
    
    print(weight_dict)