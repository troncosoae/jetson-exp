import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
import pathlib
import time
import sys

from Net.Net import Net


if __name__ == "__main__":

    DEVICE = 'gpu'
    try:
        DEVICE = sys.argv[1]
    except IndexError:
        pass
    if DEVICE == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    img_height = 180
    img_width = 180
    channels = 3
    num_classes = 5

    net = Net(num_classes, img_height, img_width, channels)
    net.load_weights('./checkpoints/my_checkpoint')

    batch_size = 32

    # dataset_url = "https://storage.googleapis.com/\
    #     download.tensorflow.org/example_images/flower_photos.tgz"
    dataset_url = "data/flower_photos.tgz"
    data_dir = keras.utils.get_file(
        'flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    train_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    for element in val_ds:
        start_time = time.time()
        estimate = net.predict(element)
        t = time.time() - start_time
        print(f"--- {t} seconds ---")
