import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


def inception_resnet_v2(classification, img_size=(256, 256, 3)):
    inputs = tf.keras.Input(shape=img_size)
    preprocess_inputs = tf.keras.applications.inception_resnet_v2.preprocess_input
    # rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    base_model = tf.keras.applications.InceptionResNetV2(input_shape=img_size, include_top=False, weights='imagenet')
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    activation = tf.keras.layers.Activation(activation='linear', name='last_conv')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dense_1_layer = tf.keras.layers.Dense(8)
    if classification == 'binary':
        prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='last_output')
    elif classification == 'multiple':
        prediction_layer = tf.keras.layers.Dense(5, activation='softmax', name='last_output')
    elif classification == 'regression':
        prediction_layer = tf.keras.layers.Dense(1, activation='linear', name='last_output')
    # sigmoid_prediction = tf.keras.layers.ReLU()

    x = preprocess_inputs(inputs)
    # x = rescale(x)
    x = base_model(x)
    x = activation(x)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = dense_1_layer(x)
    outputs = prediction_layer(x)
    # outputs = sigmoid_prediction(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def mobilenet(classification, img_size=(256, 256, 3)):
    inputs = tf.keras.Input(shape=img_size)
    preprocess_inputs = tf.keras.applications.mobilenet.preprocess_input
    rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    base_model = tf.keras.applications.MobileNet(input_shape=img_size, include_top=False, weights='imagenet')
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    activation = tf.keras.layers.Activation(activation='linear', name='last_conv')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dense_1_layer = tf.keras.layers.Dense(8)
    if classification == 'binary':
        prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='last_output')
    elif classification == 'multiple':
        prediction_layer = tf.keras.layers.Dense(5, activation='softmax', name='last_output')
    elif classification == 'regression':
        prediction_layer = tf.keras.layers.Dense(1, activation='linear', name='last_output')
    # sigmoid_prediction = tf.keras.layers.ReLU()

    x = preprocess_inputs(inputs)
    x = rescale(x)
    x = base_model(x)
    x = activation(x)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = dense_1_layer(x)
    outputs = prediction_layer(x)
    # outputs = sigmoid_prediction(x)
    model = tf.keras.Model(inputs, outputs)
    return model