import tensorflow as tf


class Interface:
    def __init__(self, net) -> None:
        self.net = net

    def train_net(self, dataset_batches, epochs, verbose=False, **kwargs):
        optimizer = kwargs.get('optimizer', 'adam')
        loss = kwargs.get(
            'loss',
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        metrics = kwargs.get('metrics', ['accuracy'])
        self.net.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        for epoch in range(epochs):
            if verbose:
                print('epoch:', epoch)
            for batch in dataset_batches:
                X, Y = batch
                self.net.fit(X, Y, **kwargs)

    def save_weights(self, filepath, **kwargs):
        return self.net.save_weights(filepath, **kwargs)

    def load_weights(self, filepath, **kwargs):
        return self.net.load_weights(filepath, **kwargs)

    def predict_net(self, X, *args, **kwargs):
        return self.net.predict(X, *args, **kwargs)
