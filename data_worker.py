import pickle
import numpy as np
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def unpack_data(data_dict):
    N = len(data_dict[b'labels'])
    Y = np.array(data_dict[b'labels'])
    data = data_dict[b'data']
    print(data.dtype)
    X = np.zeros((N, 32, 32, 3), dtype=data.dtype)

    for i in range(N):
        data_i = data[i, :]
        temp = data_i.reshape(3, 32, 32)
        X[i, :, :, :] = np.moveaxis(temp, 0, -1)

    return X, Y


def display_img(array):
    plt.imshow(array)
    plt.show()
