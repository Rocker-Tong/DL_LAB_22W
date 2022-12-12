from tensorflow import keras as k
from keras import layers, models


class CNN(k.Model):
    def __init__(self):
        super(CNN, self).__init__(name="CNN")
        model = models.Sequential()
        model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 3)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3), name='last_conv'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid', name='last_output'))
        self.model = model

    def call(self, x):
        output = self.model(x)
        return output
