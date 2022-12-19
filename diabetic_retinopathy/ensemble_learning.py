import gin
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from evaluation.metrics import *
from input_pipeline import datasets
from models.architectures import *
from models.transfer_learning import inception_resnet_v2, mobilenet
from models.CNN import CNN
from models.VGG import VGG16
from models.Resnet import ResNet101



def ensemble_learning(train, validation, test, classification):
    # model_list = ['CNN', 'VGG', 'Resnet', 'inception_resnet_v2', 'mobilenet']
    model_list = ['CNN', 'VGG']

    # record the predictions and labels of each model
    binary_predictions_list = []
    multiple_predictions_list = []

    # the times that we split dataset for each model
    times = 5

    # Only use for save the original probability of predictions in multi-class case
    y_pred_with_probability = np.zeros([103, 5])


    # evaluate each model
    for model_name in model_list:
        # setup model
        if model_name == 'CNN':
            model = CNN(classification=classification)
        elif model_name == 'VGG':
            model = VGG16(classification=classification)
        elif model_name == 'Resnet':
            model = ResNet101(bottleneck_list=[3, 4, 23, 3], neurons=64, classification=classification)
        elif model_name == 'inception_resnet_v2':
            model = inception_resnet_v2(classification=classification)
        elif model_name == 'mobilenet':
            model = mobilenet(classification=classification)

        if classification == 'binary':
            opt = tf.keras.optimizers.Adam(lr=0.0005 / 10)
            model.compile(loss="BinaryCrossentropy", optimizer=opt, metrics=["accuracy"])
        elif classification == 'multiple':
            opt = tf.keras.optimizers.Adam(lr=0.0005 / 10)
            model.compile(loss="SparseCategoricalCrossentropy", optimizer=opt, metrics=["accuracy"])

        for time in range(times):

            train_split, val_split = sklearn.model_selection.train_test_split(train, test_size=0.2)
            model.fit(train_split, epochs=10, batch_size=8, validation_data=val_split)

            if classification == 'binary':
                y_pred_origin = model.predict(test)
                y_pred = np.where(y_pred_origin > 0.5, 1, 0)
                y_pred = np.ndarray.tolist(y_pred)
                y_pred = [x[0] for x in y_pred]
                if model_name == 'CNN':
                    binary_predictions_list = y_pred
                elif model_name != 'CNN':
                    binary_predictions_list = np.vstack((binary_predictions_list, y_pred))

            elif classification == 'multiple':
                y_pred_origin = model.predict(test)
                y_pred = []
                for i in range(y_pred_origin.shape[0]):
                    idx = np.argmax(y_pred_origin[i])
                    y_pred.append(idx)
                if model_name == 'CNN':
                    multiple_predictions_list = y_pred
                elif model_name != 'CNN':
                    multiple_predictions_list = np.vstack((multiple_predictions_list, y_pred))
                y_pred_with_probability += y_pred_origin


    # evaluate the ensemble learning model
    print('---Evaluation of Ensemble Learning---')

    y_pred = []
    for i in range(multiple_predictions_list.shape[1]):
        pred = np.argmax(np.bincount(multiple_predictions_list[:, i]))
        y_pred.append(pred)
    print(y_pred)
    print(len(y_pred))

    # Get the true test labels
    y_true = []
    for idx, (test_images, test_labels) in enumerate(test):
        dim = test_labels.shape[0]
        for i in range(dim):
            y_true.append(test_labels[i].numpy())
    print(y_true)
    print(len(y_true))

    if classification == 'multiple':
        y_pred_with_probability = y_pred_with_probability / times / len(model_list)
    ConfusionMatrix(y_pred_with_probability, y_pred, y_true, classification)


data_dir = "/Users/yinzheming/Desktop/Deep_Learning/Lab/IDRID_dataset"
# data_dir = "/Users/rocker/Desktop/Uni Stuttgart/DL Lab/dataset/IDRID_dataset"

# data_dir = "/home/data/IDRID_dataset"
ds_train, ds_val, ds_test, ds_info = datasets.load(name="IDRID", data_dir=data_dir, classification='multiple')
ensemble_learning(ds_train, ds_val, ds_test, classification='multiple')