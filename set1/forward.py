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
    try:
        FORWARD_LOOPS = int(sys.argv[3])
    except IndexError:
        FORWARD_LOOPS = 1
    except ValueError:
        FORWARD_LOOPS = 1
    try:
        temp = sys.argv[4]
        if temp == 'True':
            SHOW_IMAGES = True
        else:
            SHOW_IMAGES = False
    except IndexError:
        SHOW_IMAGES = False

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

    if SHOW_IMAGES:
        print('printing test images...')
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join(
            '%5s' % classes[labels[j]] for j in range(4)))

    images, labels = images.to(device), labels.to(device)

    # run forwards
    forward_counts = []
    forward_times = []
    for count in range(FORWARD_LOOPS):
        print(f'Running forward # {count}...')
        start_time = time.time()
        outputs = net(images)
        t = time.time() - start_time
        print(f"--- {t} seconds ---")
        forward_counts.append(count)
        forward_times.append(t)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                      for j in range(4)))
    if SHOW_IMAGES:
        plt.plot(forward_counts, forward_times)
        plt.show()
        plt.hist(forward_times, bins=20)
        plt.show()



    # print('running forward...')
    # start_time = time.time()
    # outputs = net(images)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # _, predicted = torch.max(outputs, 1)

    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
    #                               for j in range(4)))

    # print('running 2ndforward...')
    # start_time = time.time()
    # outputs = net(images)
    # print("--- %s seconds ---" % (time.time() - start_time))
