import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras as k
import numpy as np

def _parse_function(filename, label):
    image_string = tf.compat.v1.read_file(filename)
    # 将图像使用JPEG的格式解码从而得到图像对应的三维矩阵。
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # 通过tf.image.resize_images函数调整图像的大小。
    image_resized = tf.compat.v1.image.resize_images(image_decoded, [256, 256]) / 255.0
    return image_resized, label

data_dir = '/Users/yinzheming/Desktop/Deep_Learning/Lab/IDRID_dataset'
# data_dir = '/Users/rocker/Desktop/Uni Stuttgart/DL Lab/dataset/IDRID_dataset'
train_dir = data_dir + '/images/train/'
train_filenames = [train_dir + filename for filename in os.listdir(train_dir)]
files = pd.read_csv(data_dir + "/labels/train.csv")
files.dropna(inplace=True, axis='columns')
files_with_labels = files['Retinopathy grade'].values.tolist()

dataset = tf.data.Dataset.from_tensor_slices((train_filenames, files_with_labels))
dataset = dataset.map(_parse_function)  # map对数据集应用自定义函数
# dataset = dataset.batch(40)  # 设置每一批读取的数据量
# dataset = dataset.repeat(2)  # 设置可以重复读取dataset n次

iterator = iter(dataset)
while 1:
    try:
        image, _ = next(iterator)
        print(image.shape)
    except StopIteration:  # python内置的迭代器越界错误类型
        print("iterator done")
        break;

for image, label in dataset:
    print(image.shape, label)
    plt.imshow(image)
    plt.show()

