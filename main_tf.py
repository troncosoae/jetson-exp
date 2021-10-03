import numpy as np

from data_worker import combine_batches, split_into_batches, unpickle, \
    unpack_data, display_img
from tf_lib.Net import Net
from tf_lib.data_worker import suit4tf


batches_names = [
    'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
    'data_batch_5']


if __name__ == '__main__':
    print('running main')

    saved_weights_file = 'saved_nets/saved_torch/version1.pth'

    data_batches = [
        unpickle(f'datasets/cifar-10-batches-py/{batch_name}') for batch_name
        in batches_names]

    unpacked_batches = [
        (unpack_data(data_batch)) for data_batch
        in data_batches]

    print(unpacked_batches[0][0].shape)

    X, Y = combine_batches(unpacked_batches)

    print(X.shape, Y.shape)

    batches = split_into_batches(X, Y, 3)

    torch_batches = [(suit4tf(X, Y)) for X, Y in batches]
