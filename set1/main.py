import torch
from torch.utils.data import dataset

from Net.Net import Net

if __name__ == "__main__":
    print('running')

    batch_size = 4

    trainloader = torch.utils.data.DataLoader(
        dataset='data/cifar-10-batches-py/data_batch_2')

    testloader = torch.utils.data.DataLoader(
        dataset='data/cifar-10-batches-py/test_batch')

    print(testloader, trainloader)

    net = Net()
