from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf


class Net(Sequential):
    def __init__(self) -> None:
        self.num_classes = 10
        self.img_height = 32
        self.img_width = 32
        self.channels = 3

        model_list = [
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
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
