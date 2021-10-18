import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_worker import combine_batches, split_into_batches, unpickle, \
    unpack_data, display_img
from torch_lib.data_worker import suit4pytorch
from torch_lib.Nets import MediumNet
from torch_lib.Interface import Interface


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

    X_torch, Y_torch = suit4pytorch(X, Y)

    net = MediumNet()
    net_interface = Interface(net)
    net_interface.load_weights(saved_weights_file)

    acc, n = net_interface.eval_acc_net(X_torch, Y_torch)

    print(f'acc: {acc*100:.2f}%, N: {n}')
