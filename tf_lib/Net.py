from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


class Net(Sequential):
    def __init__(self, num_classes, img_height, img_width, channels) -> None:
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels

        model_list = [
            layers.experimental.preprocessing.Rescaling(
                1./255,
                input_shape=(img_height, img_width, channels)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ]

        super().__init__(model_list)
