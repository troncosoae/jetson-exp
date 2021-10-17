import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_worker import combine_batches, split_into_batches, unpickle, \
    unpack_data, display_img
from torch_lib.Interface import Interface
from torch_lib.Nets import MediumNet
from torch_lib.data_worker import suit4pytorch


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

    torch_batches = [(suit4pytorch(X, Y)) for X, Y in batches]

    X_torch = torch_batches[0][0]
    Y_torch = torch_batches[0][1]

    net = MediumNet()
    net_interface = Interface(net)
    net_interface.train_net(torch_batches, 1, verbose=False, batch_size=None)
    net_interface.save_weights(saved_weights_file)

    preds = net_interface.predict_net(X_torch)
    preds = preds.detach().numpy()
    print(preds)
    print(np.argmax(preds, axis=1), Y_torch)
