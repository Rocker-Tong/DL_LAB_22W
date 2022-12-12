'''import gin
import tensorflow as tf

@gin.configurable
def vgg_block(inputs, dropout_rate):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.

    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)

    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(8, 3, padding='1', stride='4', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.Conv2D(16, 3, padding='1', stride='4', activation=tf.nn.relu)(out)
    out = tf.keras.layers.MaxPool2D((3, 3))(out)
    out = tf.keras.layers.Conv2D(32, 3, padding='1', stride='4', activation=tf.nn.relu)(out)
    out = tf.keras.layers.Conv2D(32, 3, padding='1', stride='4', activation=tf.nn.relu)(out)
    out = tf.keras.layers.Flatten()
    out = tf.keras.layers.Dropout(dropout_rate)

    return out'''

import gin
import tensorflow as tf

@gin.configurable
def vgg_block(inputs, filters, kernel_size):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.
    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)
    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out


# @gin.configurable
# def con2d
