from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


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
