import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_worker import combine_batches, split_into_batches, unpickle, \
    unpack_data, display_img
from torch_lib.Net import Net
from torch_lib.Interface import Interface
from torch_lib.data_worker import suit4pytorch


batches_names = [
    'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
    'data_batch_5']


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, out_channels=6, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            6, out_channels=16, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

    net = Net()
    net_interface = Interface(net)
    net_interface.train_net(torch_batches, 1, verbose=False, batch_size=None)
    net_interface.save_weights(saved_weights_file)

    preds = net_interface.predict_net(X_torch)
    preds = preds.detach().numpy()
    print(preds)
    print(np.argmax(preds, axis=1), Y_torch)
