import math
import numpy as np
import tensorflow as tf
from functools import reduce


class ResNet:
    def __init__(self, input, calsses, TRAINING):
        self.X = input
        self.classes = calsses
        self.TRAINING = TRAINING
        self.build()



    def build(self):
        """
        load variable from npy to build the Resnet or Generate a new one
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        # Preprocessing: Turning RGB to BGR - Mean.


        self.conv1 = tf.layers.conv2d(self.X, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',name='conv1')
        self.conv_norm_1 = self.batch_norm(self.conv1)
        self.conv1_relu = tf.nn.relu(self.conv_norm_1)

        self.pool1 = self.max_pool(self.conv1_relu, 3, 2, "pool1")
        self.block1_1 = self.res_block_3_layers(self.pool1, [64, 64, 256], "block1_1", True)
        self.block1_2 = self.res_block_3_layers(self.block1_1, [64, 64, 256], "block1_2")
        self.block1_3 = self.res_block_3_layers(self.block1_2, [64, 64, 256], "block1_3")
        print(self.block1_3)
        self.block2_1 = self.res_block_3_layers(self.block1_3, [128, 128, 512], "block2_1", True, 2)
        self.block2_2 = self.res_block_3_layers(self.block2_1, [128, 128, 512], "block2_2")
        self.block2_3 = self.res_block_3_layers(self.block2_2, [128, 128, 512], "block2_3")
        self.block2_4 = self.res_block_3_layers(self.block2_3, [128, 128, 512], "block2_4")
        print(self.block2_4)
        self.block3_1 = self.res_block_3_layers(self.block2_4, [256, 256, 1024], "block3_1", True, 2)
        self.block3_2 = self.res_block_3_layers(self.block3_1, [256, 256, 1024], "block3_2")
        self.block3_3 = self.res_block_3_layers(self.block3_2, [256, 256, 1024], "block3_3")
        self.block3_4 = self.res_block_3_layers(self.block3_3, [256, 256, 1024], "block3_4")
        self.block3_5 = self.res_block_3_layers(self.block3_4, [256, 256, 1024], "block3_5")
        self.block3_6 = self.res_block_3_layers(self.block3_5, [256, 256, 1024], "block3_6")
        print(self.block3_6)
        self.block4_1 = self.res_block_3_layers(self.block3_6, [512, 512, 2048], "block4_1", True, 2)
        self.block4_2 = self.res_block_3_layers(self.block4_1, [512, 512, 2048], "block4_2")
        self.block4_3 = self.res_block_3_layers(self.block4_2, [512, 512, 2048], "block4_3")
        print(self.block4_3)
        self.pool2 = self.avg_pool(self.block4_3, 7, 1, "pool2")
        print(self.pool2)
        self.flatten = tf.layers.flatten(self.pool2, name='flatten')
        self.logits = tf.layers.dense(self.flatten, self.classes, "softmax")


        return self.pool2

    def res_block_3_layers(self, bottom, channel_list, name, change_dimension=False, block_stride=1):
        """
        bottom: input values (X)
        channel_list : number of channel in 3 layers
        name: block name
        """
        if (change_dimension):
            # short_cut_conv = self.conv_layer(bottom, 1, bottom.get_shape().as_list()[-1], channel_list[2], block_stride,
            #                                  name + "_ShortcutConv")
            short_cut_conv = tf.layers.conv2d(bottom,channel_list[2] , (1, 1),
                             strides=(block_stride, block_stride), name=name + '_ShortcutConv')
            block_conv_input = self.batch_norm(short_cut_conv)
        else:
            block_conv_input = bottom

        block_conv_1 = self.conv_layer(bottom, 1, bottom.get_shape().as_list()[-1], channel_list[0], block_stride,
                                       name + "_lovalConv1")
        block_norm_1 = self.batch_norm(block_conv_1)
        block_relu_1 = tf.nn.relu(block_norm_1)

        block_conv_2 = self.conv_layer(block_relu_1, 3, channel_list[0], channel_list[1], 1, name + "_lovalConv2")
        block_norm_2 = self.batch_norm(block_conv_2)
        block_relu_2 = tf.nn.relu(block_norm_2)

        block_conv_3 = self.conv_layer(block_relu_2, 1, channel_list[1], channel_list[2], 1, name + "_lovalConv3")
        block_norm_3 = self.batch_norm(block_conv_3)
        block_res = tf.add(block_conv_input, block_norm_3)
        relu = tf.nn.relu(block_res)

        return relu

    def conv_layer(self, bottom, kernal_size, in_channels, out_channels, stride, name):


        return tf.layers.conv2d(inputs=bottom,filters=out_channels,kernel_size=(kernal_size,kernal_size), strides=(stride, stride),padding='same', name=name)

    def batch_norm(self, inputsTensor):
        """
        Batchnorm
        """
        _BATCH_NORM_DECAY = 0.99
        _BATCH_NORM_EPSILON = 1e-12
        return tf.layers.batch_normalization(inputs=inputsTensor, axis=3, momentum=_BATCH_NORM_DECAY,
                                             epsilon=_BATCH_NORM_EPSILON, center=True, scale=True,
                                             training=self.TRAINING)

    def avg_pool(self, bottom, kernal_size=2, stride=2, name="avg"):
        """
        bottom: input values (X)
        kernal_size : n * n kernal
        stride : stride
        name : block_layer name
        """
        print(name + ":")
        print(bottom.get_shape().as_list())
        return tf.nn.avg_pool(bottom, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride, stride, 1],
                              padding='VALID', name=name)

    def max_pool(self, bottom, kernal_size=2, stride=2, name="max"):
        """
        bottom: input values (X)
        kernal_size : n * n kernal
        stride : stride
        name : block_layer name
        """
        print(name + ":")
        print(bottom.get_shape().as_list())
        return tf.nn.max_pool(bottom, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name=name)







