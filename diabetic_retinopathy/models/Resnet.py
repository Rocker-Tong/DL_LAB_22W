import tensorflow as tf
from tensorflow import keras as k
from keras import layers, models
import logging


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
        bottleneck_model.add(layers.Dense(1, activation='sigmoid', name='last_output'))

        self.bottleneck_model = bottleneck_model

    def call(self, x):
        output = self.bottleneck_model(x)
        return output