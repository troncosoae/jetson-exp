import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
import pathlib
import time
import sys
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit



def predict(batch): # result gets copied into output
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # Execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # Syncronize threads
    stream.synchronize()
    
    return output


if __name__ == "__main__":

    img_height = 180
    img_width = 180
    channels = 3
    num_classes = 5
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

    f = open("trt_models/version1_trt.trt", "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    target_dtype = np.float32

    for element in val_ds:
        print(type(element))
        print(len(element))
        print(type(element[0]))
        print(type(element[1]))
        e0 = np.array(element[0])
        e1 = np.array(element[1])
        print(type(e0), type(e1))
        print(e0.shape, e1.shape)

        input_batch = e0.astype(np.float32)
        output = np.empty([batch_size, 1000], dtype=target_dtype)

        d_input = cuda.mem_alloc(1 * input_batch.nbytes)
        d_output = cuda.mem_alloc(1 * output.nbytes)

        bindings = [int(d_input), int(d_output)]

        stream = cuda.Stream()

        print("Warming up...")
        trt_predictions = predict(input_batch).astype(np.float32)
        print("Done warming up!")
