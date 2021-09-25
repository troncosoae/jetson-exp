from data_worker import unpickle, unpack_data, display_img


if __name__ == '__main__':
    print('running main')

    data_batch_1 = unpickle('datasets/cifar-10-batches-py/data_batch_1')
    print(data_batch_1.keys())

    X, Y = unpack_data(data_batch_1)
    N = Y.shape[0]

    idx = 25
    display_img(X[idx, :, :, :])
