from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.keras.layers.core import Dense


class Net(Sequential):
    def __init__(self) -> None:
        self.num_classes = 10
        self.img_height = 32
        self.img_width = 32
        self.channels = 3

        model_list = [
            layers.Conv2D(
                filters=6, kernel_size=5, padding=0, activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.Conv2D(
                filters=16, kernel_size=5, padding=0, activation='relu'),
            layers.Flatten(),
            layers.Dense(120, activation='relu'),
            layers.Dense(84, activation='relu'),
            layers.Dense(self.num_classes)
        ]

        super().__init__(model_list)

    def train(self, dataset_batches, epochs, verbose=False, **kwargs):
        optimizer = kwargs.get('optimizer', 'adam')
        loss = kwargs.get(
            'loss',
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        metrics = kwargs.get('metrics', ['accuracy'])

        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        i = 0
        for epoch in range(epochs):
            if verbose:
                print('epoch:', epoch)
            for batch in dataset_batches:
                X, Y = batch
                self.fit(X, Y, **kwargs)

    # def save_weights(self, filepath, **kwargs):
    #     return self.save_weights(filepath, **kwargs)

    # def load_weights(self, filepath, **kwargs):
    #     return self.load_weights(filepath, **kwargs)
