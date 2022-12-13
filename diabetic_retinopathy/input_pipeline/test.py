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
from evaluation.metrics import ConfusionMatrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from train import Trainer
from input_pipeline.preprocessing import augment
import cv2
from evaluation.Grad_CAM import deep_visualize
import shutil
from evaluation.make_new_folder import make_folder
from models.transfer_learning import inception_resnet_v2, mobilenet


# train_set_path = '/Users/rocker/Desktop/Uni Stuttgart/DL Lab/dataset/IDRID_dataset/train.tfrecords'
# test_set_path = '/Users/rocker/Desktop/Uni Stuttgart/DL Lab/dataset/IDRID_dataset/test.tfrecords'

# # generate folder structures
# run_paths = utils_params.gen_run_folder()
#
# # set loggers
# utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
#
# # gin-config
# gin.parse_config_files_and_bindings(['configs/config.gin'], [])
# utils_params.save_config(run_paths['path_gin'], gin.config_str())


data_dir = "/Users/yinzheming/Desktop/Deep_Learning/Lab/IDRID_dataset"
# data_dir = "/Users/rocker/Desktop/Uni Stuttgart/DL Lab/dataset/IDRID_dataset"

# data_dir = "/home/data/IDRID_dataset"
train_set, val_set, test_set, ds_info = load("IDRID", data_dir)
test_labels = read_labels(data_dir, "test")

for images, labels in test_set:
    print(len(labels))

#
# print(type(train_set))


# vals = np.fromiter(train_set(lambda x, y: y), float)
#
# plt.hist(vals)
# plt.xticks(range(10))
# plt.title('Label Frequency')
# plt.show()

"""
Test the vgg_block
"""
# model = vgg_like(input_shape=(32,), n_classes=10)

# histogram_img = Image.open(data_dir + "/labels/histogram.png")
# histogram_img.show()

# for image, label in test_set:
#     print(label)
#     plt.figure()
#     plt.imshow(image)
#     plt.title(np.array(label))
#     plt.show()
# (256, 256, 3) tf.Tensor(0, shape=(), dtype=int64)


class CNN(k.Model):
    def __init__(self):
        super(CNN, self).__init__()
        model = models.Sequential()
        model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 3), name='conv_1'))
        model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3), name='last_conv'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), name='maxpooling'))
        model.add(layers.Flatten(name='flatten'))
        model.add(layers.Dense(128, activation='relu', name='dense_1'))
        model.add(layers.Dense(64, activation='relu', name='dense_2'))
        model.add(layers.Dense(1, activation='sigmoid', name='last_output'))
        self.model = model

    def call(self, x):
        output = self.model(x)
        # self.model.summary()
        return output

class VGG16(models.Model):

    def __init__(self):
        super(VGG16, self).__init__()

        weight_decay = 0.0001
        input_shape = (256, 256, 3)

        model = models.Sequential()

        model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='conv_1'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='conv_2'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='conv_3'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='conv_4'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='conv_5'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='conv_6'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='conv_7'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))


        model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='conv_8'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='conv_9'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='conv_10'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))


        model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='conv_11'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='conv_12'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='last_conv'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())
        model.add(layers.Dense(4096, kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='dense_1'))
        model.add(layers.BatchNormalization())

        model.add(layers.Flatten())
        model.add(layers.Dense(4096, kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='dense_2'))
        model.add(layers.BatchNormalization())

        model.add(layers.Flatten())
        model.add(layers.Dense(1000, kernel_regularizer=regularizers.l2(weight_decay), activation='relu', name='dense_3'))
        model.add(layers.BatchNormalization())

        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, kernel_regularizer=regularizers.l2(weight_decay), activation='sigmoid', name='last_output'))


        self.model = model


    def call(self, x):

        x = self.model(x)

        return x


class Bottleneck(k.Model):
    def __init__(self, neurons, stride=1, residual_path=False):
        super(Bottleneck, self).__init__()
        self.neurons = neurons

        model = models.Sequential()

        model.add(layers.Conv2D(self.neurons, (1, 1), strides=stride, activation='relu', padding='same'))
        model.add(layers.BatchNormalization(axis=3))

        model.add(layers.Conv2D(self.neurons, (3, 3), strides=1, activation='relu', padding='same'))
        model.add(layers.BatchNormalization(axis=3))

        model.add(layers.Conv2D(self.neurons*4, (1, 1), strides=stride, padding='same'))
        model.add(layers.BatchNormalization(axis=3))

        if residual_path:
            self.downsampling = models.Sequential([
                layers.Conv2D(self.neurons * 4, (1, 1), strides=stride**2, padding='SAME')
            ])

        self.model = model
        self.stride = stride
        self.residual_path = residual_path

    def call(self, x):
        identity = x
        output = self.model(x)
        if self.residual_path:
            identity = self.downsampling(x)
        output = layers.add([output, identity])
        output = tf.nn.relu(output)
        return output


class ResNet101(k.Model):
    def __init__(self, bottleneck_list, neurons):  # block_list表示每个block有几个餐叉结构
        super(ResNet101, self).__init__()
        # self.num_bottleneck = len(bottleneck_list)  # 共有几个block
        # self.bottleneck_list = bottleneck_list
        # self.neurons = neurons   # 卷积核个数

        bottleneck_model = models.Sequential()
        # 第一个CPA结构
        bottleneck_model.add(layers.Conv2D(256, (7, 7), strides=2, activation='relu', input_shape=(256, 256, 3), padding='same'))
        bottleneck_model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
        for bottleneck_id in range(len(bottleneck_list)):  # 第几个block
            for layer_id in range(bottleneck_list[bottleneck_id]):  # 第几个残差结构
                if bottleneck_id != 0 and layer_id == 0:  # 除第一个block的第一个残差结构外
                    bottleneck_model.add(Bottleneck(neurons=neurons, stride=2, residual_path=True))  # 其余3个block的第1个残差结构都用虚线连接
                else:
                    bottleneck_model.add(Bottleneck(neurons=neurons, residual_path=False))  # 其它的残差结构都用实线连接
                # model.add(bottleneck)  # 将构建好的block加入resnet
            neurons *= 2  # 下一个block的卷积核数是上一个block的2倍
        # 最后的池化层和全连接层（算1层）
        bottleneck_model.add(layers.GlobalAveragePooling2D())
        bottleneck_model.add(layers.Dense(1, activation='sigmoid'))

        self.bottleneck_model = bottleneck_model

    def call(self, x):
        output = self.bottleneck_model(x)
        # self.model.summary()
        return output


def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    print('mean:', mean, 'std:', std)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test


def train_model(name, train, validation):
    if name == 'CNN':
        model = CNN()
        opt = tf.keras.optimizers.RMSprop(0.001)

    elif name == 'VGG16':
        model = VGG16()

    elif name == 'ResNet101':
        model = ResNet101(bottleneck_list=[3, 4, 23, 3], neurons=64)

    elif name == 'inception_resnet_v2':
        model = inception_resnet_v2()

    elif name == 'mobilenet':
        model = mobilenet()

    opt = tf.keras.optimizers.Adam(lr=0.0005/10)
    # opt = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss="BinaryCrossentropy", optimizer=opt, metrics=["accuracy"])
    history = model.fit(train, epochs=10, batch_size=8, validation_data=validation)
    plt.plot(history.history["loss"])
    plt.legend(["loss"])
    plt.xticks(range(10))
    plt.xlabel("epochs")
    plt.title("Training process")
    plt.show()

    return model


def k_fold(model_name, x, y):
    seed =10
    np.random.seed(seed)
    # for image, label in dataset:
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, validation in folds.split(x, y):
        model = train_model(name=model_name, train=train)
        scores = model.evaluate(validation)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    return model


def test_model(model, train, validation, test, data_dir):
    model = train_model(name=model, train=train, validation=validation)
    make_folder(path=data_dir)
    for idx, (test_image, test_label) in enumerate(test):
            deep_visualize(model=model, images=test_image, dataset=test, step=idx, run_paths=data_dir)
    y_pred_origin = model.predict(test)
    y_pred = np.where(y_pred_origin > 0.5, 1, 0)
    y_pred = np.ndarray.tolist(y_pred)
    y_pred = [x[0] for x in y_pred]
    print(y_pred)
    print(len(y_pred))
    return y_pred_origin, y_pred, model


y_pred_origin, y_pred, model = test_model(model='inception_resnet_v2', train=train_set, validation=val_set, test=test_set, data_dir=data_dir)
print(test_labels)
print(len(test_labels))
ConfusionMatrix(y_pred, test_labels)
# cm = confusion_matrix_show(y_true=test_labels, y_pred=y_pred)
# confusion_matrix_plot(cm)
# accuracy, precision, recall, f1_score, sensitivity, specificity = accuracy(cm)
# stats_text = "\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}\nSensitivity={:0.3f}\nSpecificity={:0.3f}".format(accuracy, precision, recall, f1_score, sensitivity, specificity)
# print(stats_text)
# roc_curve_plot(y_true=test_labels, y_pred=y_pred)
