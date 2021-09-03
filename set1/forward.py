import sys
import torch
import torchvision
import torchvision.transforms as transforms
from data_worker import import_data
from Net.Net import Net
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    WEIGHTS_PATH = sys.argv[1]
    print(f'forward.py {WEIGHTS_PATH}')

    # verifying cuda available
    print('checking cuda availability...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    # importing data
    print('importing data...')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    trainloader, testloader = import_data('./data', batch_size, transform)

    # get net and import weights
    print('loading network...')
    net = Net()
    net.to(device)
    net.load_state_dict(torch.load(WEIGHTS_PATH))

    # get batch of images
    print('get test images...')
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    print('printing test images...')
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(
        '%5s' % classes[labels[j]] for j in range(4)))

    images, labels = images.to(device),labels.to(device)

    # running forward
    print('running forward...')
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))
