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
from models.CNN import CNN
import shutil


# train_set_path = '/Users/rocker/Desktop/Uni Stuttgart/DL Lab/dataset/IDRID_dataset/train.tfrecords'
# test_set_path = '/Users/rocker/Desktop/Uni Stuttgart/DL Lab/dataset/IDRID_dataset/test.tfrecords'


# data_dir = "/Users/yinzheming/Desktop/Deep_Learning/Lab/IDRID_dataset"
# data_dir = "/Users/rocker/Desktop/Uni Stuttgart/DL Lab/dataset/IDRID_dataset"

# data_dir = "/home/data/IDRID_dataset"
# train_set, test_set, val_set = load("IDRID", data_dir, classification='multiple')
# test_labels = read_labels(data_dir, "test")
# train_labels = read_labels(data_dir, "train")

# from sklearn.preprocessing import StandardScaler###标准化
# train = train_labels.drop(["target"], axis=1)
# standard = StandardScaler()
# standard.fit(train)
# X_scaled = standard.transform(train)
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)###为了接下来做图方便一些，就只保留数据的前两个部分，实际上这里也可以多保留些主成分
# pca.fit(X_scaled)###对数据进行拟合pca模型
# X_pca = pca.transform(X_scaled)#将数据变换到前两个主成分上
#
# import numpy as np
# result = np.c_[X_pca, train_labels["target"]]###将处理结果和数据集的目标值结合起来，这样就是一个新的数据集了。
# ###新数据集与原数据集的信息相差不大，甚至剔除了部分重叠数据造成的影响
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8,8))
# plt.rcParams['font.sans-serif'] = ['SimHei']###防止中文显示不出来
# plt.rcParams['axes.unicode_minus'] = False###防止坐标轴符号显示不出来
# result_1=result[result[:,2]==1]###target类别1
# result_0=result[result[:,2]==0]###target类别0
# plt.scatter(result_1[:,0],result_1[:,1])###画类别1的散点图
# plt.scatter(result_0[:,0],result_0[:,1])###画类别0的散点图
# plt.legend()
# plt.xlabel('第一主成分')###横坐标轴标题
# plt.ylabel('第二主成分')###纵坐标轴标题
# plt.show()



# img = Image.open(data_dir + '/images/IDRiD_001 copy.jpg')
# print(img)
# img.resize(size=(80, 256))
# img.save(data_dir + '/images/visualization/new_image.jpg')

# train_set, val_set, test_set, ds_info = load("IDRID", data_dir)
# y_true = []
# for idx, (test_images, test_labels) in enumerate(test_set):
#     dim = test_labels.shape[0]
#     for i in range(dim):
#         y_true.append(test_labels[i].numpy())
# print(y_true)

# V = tf.convert_to_tensor(np.array([[1,2,3],[2,3,5],[3,6,8],[323,623,823]]))
# P = tf.convert_to_tensor(np.array([[8,3,5],[7,20,4],[7,2,9],[564,736,275]]))
# Q = tf.convert_to_tensor(np.array([[7,8,4],[3,6,8],[4,6,3],[536,863,937]]))
# print(V + P)
# V = np.array([[0.3], [0.7], [0.8]])
# P = tf.convert_to_tensor(V)
# print(P[:, 2])
# print(P)
# Q = 1-V
# print(Q)
# y_pred = []
# print(V)
# print(len(V))
# for i in range(len(V)):
#     idx = np.argmax(V[i])
#     print(V[i, idx])
#     y_pred.append(idx)
# print(y_pred)
#
# x = '/Users/yinzheming/Desktop/Deep_Learning/Lab/IDRID_dataset/images/train/IDRiD_001.jpg'
# y = x.split('.')[0][-3:]
# print(y)


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

# indices = [0, 1, 3, 4, 2, 5, 4, 1]
# P = [4, 2, 44, 5, 2, 5, 2, 6]
# V = []
# Q = np.vstack((V, P))
# print(Q)
# print(Q.shape)
# indices = tf.convert_to_tensor(indices)
# print(indices.shape)
# depth = 5
# labels = tf.one_hot(indices, depth, on_value=1, off_value=0, axis=-1)
# print(labels)

# y_pred_with_probability = np.zeros([103, 5])
# print(y_pred_with_probability)
# print(y_pred_with_probability.shape)

