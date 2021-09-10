import sys
import torch
import torchvision
import torchvision.transforms as transforms
from data_worker import import_data
from Net.Net import Net
import matplotlib.pyplot as plt
import numpy as np
import time

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    WEIGHTS_PATH = sys.argv[1]
    DEVICE = sys.argv[2]
    print(f'forward.py {WEIGHTS_PATH} {DEVICE}')

    # verifying cuda available
    print('checking cuda availability...')
    device = torch.device('cpu')
    if DEVICE == 'cuda':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    # importing data
    start_time = time.time()
    print('importing data...')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    trainloader, testloader = import_data('./data', batch_size, transform)
    print("--- %s seconds ---" % (time.time() - start_time))

    # get net and import weights
    start_time = time.time()
    print('loading network...')
    net = Net()
    net.to(device)
    net.load_state_dict(torch.load(WEIGHTS_PATH))
    print("--- %s seconds ---" % (time.time() - start_time))

    # get batch of images
    print('get test images...')
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    print('printing test images...')
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(
        '%5s' % classes[labels[j]] for j in range(4)))

    images, labels = images.to(device), labels.to(device)

    # running forward
    print('running forward...')
    start_time = time.time()
    outputs = net(images)
    print("--- %s seconds ---" % (time.time() - start_time))
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    print('running 2ndforward...')
    outputs = net(images)
    print("--- %s seconds ---" % (time.time() - start_time))
