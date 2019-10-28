import numpy as np

layers = ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'dense_1', 'dense_2']
batch_layers = ['batch_normalization_1', 'batch_normalization_2', 'batch_normalization_3', 'batch_normalization_4']
channels_list = {2: 'best_model_2.h5', 18: 'best_model_18.h5', 31: 'best_model_38.h5'}


def get_weights(layer_number, shape, channels):
    file_name = '{}_{}_weights.npy'.format(channels_list[channels], layers[layer_number])
    return np.loadtxt(file_name, delimiter=",").reshape(shape)


def get_bias(layer_number, shape, channels):
    file_name = '{}_{}_bias.npy'.format(channels_list[channels], layers[layer_number])
    return np.loadtxt(file_name, delimiter=",").reshape(shape)


def get_gamma(layer_number, shape, channels):
    file_name = '{}_{}_gamma.npy'.format(channels_list[channels], batch_layers[layer_number])
    return np.loadtxt(file_name, delimiter=",").reshape(shape)


def get_beta(layer_number, shape, channels):
    file_name = '{}_{}_beta.npy'.format(channels_list[channels], batch_layers[layer_number])
    return np.loadtxt(file_name, delimiter=",").reshape(shape)


def get_mean(layer_number, shape, channels):
    file_name = '{}_{}_mean.npy'.format(channels_list[channels], batch_layers[layer_number])
    return np.loadtxt(file_name, delimiter=",").reshape(shape)


def get_variance(layer_number, shape, channels):
    file_name = '{}_{}_variance.npy'.format(channels_list[channels], batch_layers[layer_number])
    return np.loadtxt(file_name, delimiter=",").reshape(shape)
