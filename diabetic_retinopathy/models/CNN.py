from tensorflow import keras as k
from keras import layers, models
import tensorflow as tf


class CNN(k.Model):
    def __init__(self, classification):
        super(CNN, self).__init__(name="CNN")
        model = models.Sequential()
        model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(256, 256, 3)))
        model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 3)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3), name='last_conv'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # model.add(layers.Flatten())
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(8, activation='relu'))
        if classification == 'binary':
            model.add(layers.Dense(1, activation='sigmoid', name='last_output'))
        elif classification == 'multiple':
            model.add(layers.Dense(5, activation='softmax', name='last_output'))
        elif classification == 'regression':
            model.add(layers.Dense(1, activation='linear', name='last_output'))
        self.model = model

    def call(self, x):
        output = self.model(x)
        return output
