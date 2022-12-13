import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as k
from keras import layers, models, regularizers
import input_pipeline.TFRecord as tfr
from input_pipeline.TFRecord import read_labels
from input_pipeline.datasets import load
from models.architectures import vgg_like
import numpy as np
import os
import wandb
import gin
from utils import utils_params, utils_misc
import logging
from PIL import Image
from evaluation.metrics import confusion_matrix_show, confusion_matrix_plot, accuracy, roc_curve_plot
from sklearn.model_selection import StratifiedKFold, cross_val_score
from models.CNN import CNN


# train_set_path = '/Users/rocker/Desktop/Uni Stuttgart/DL Lab/dataset/IDRID_dataset/train.tfrecords'
# test_set_path = '/Users/rocker/Desktop/Uni Stuttgart/DL Lab/dataset/IDRID_dataset/test.tfrecords'


data_dir = "/Users/yinzheming/Desktop/Deep_Learning/Lab/IDRID_dataset"
# data_dir = "/Users/rocker/Desktop/Uni Stuttgart/DL Lab/dataset/IDRID_dataset"

# data_dir = "/home/data/IDRID_dataset"
# train_set, test_set, val_set = load("IDRID", data_dir)
# test_labels = read_labels(data_dir, "test")

# img = Image.open(data_dir + '/images/IDRiD_001 copy.jpg')
# print(img)
# img.resize(size=(80, 256))
# img.save(data_dir + '/images/visualization/new_image.jpg')

train_set, val_set, test_set, ds_info = load("IDRID", data_dir)
y_true = []
for idx, (test_images, test_labels) in enumerate(test_set):
    dim = test_labels.shape[0]
    for i in range(dim):
        y_true.append(test_labels[i].numpy())
print(y_true)



# test_set = test_set.unbatch()
# model = CNN()
# for idx, (test_image, test_label) in enumerate(test_set):
    # if idx < 1:
        # test_image = test_image.unbatch()
        # a = test_image.shape[0]
        # img = test_image.numpy()
        # proto_tensor = tf.make_tensor_proto(test_image)  # convert `tensor a` to a proto tensor
        # img = tf.make_ndarray(test_image)
        # print(a)
        # print(img.shape)
    # else:
    #     break
# img_path = data_dir + '/images/test/IDRiD_001.jpg'
# img = tf.keras.utils.load_img(img_path)
# img = tf.keras.utils.img_to_array(img)
# print(img.shape)