import numpy as np

from data_worker import combine_batches, split_into_batches, unpickle, \
    unpack_data, display_img
from tf_lib.Interface import Interface
from tf_lib.Nets import MediumNet
from tf_lib.data_worker import suit4tf


batches_names = [
    'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
    'data_batch_5']


if __name__ == '__main__':
    print('running main')

    saved_weights_file = 'saved_nets/saved_tf/medium_v1.pth'

    data_batches = [
        unpickle(f'datasets/cifar-10-batches-py/{batch_name}') for batch_name
        in batches_names]

    unpacked_batches = [
        (unpack_data(data_batch)) for data_batch
        in data_batches]

    print(unpacked_batches[0][0].shape)

    X, Y = combine_batches(unpacked_batches)

    X_tf, Y_tf = suit4tf(X, Y)

    # print(X.shape, Y.shape)

    # batches = split_into_batches(X, Y, 3)

    # tf_batches = [(suit4tf(X, Y)) for X, Y in batches]

    # X_tf = tf_batches[0][0]
    # Y_tf = tf_batches[0][1]

    # print(X_tf, Y_tf)

    net = MediumNet
    net_interface = Interface(net)
    net_interface.load_weights(saved_weights_file)

    preds = net_interface.predict_net(X_tf)
    # preds = preds.numpy()
    print(preds)
    print(np.argmax(preds, axis=1), Y_tf)

    acc, n = net_interface.eval_acc_net(X_tf, Y_tf)

    print(f'acc: {acc*100:.2f}%, N: {n}')
