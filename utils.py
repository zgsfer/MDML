import tensorflow as tf
import numpy as np


def conv2d(input_, output_, k_h=1, k_w=1, d_h=1, d_w=1, bias=False, names="W"):
    stddev = np.sqrt(2.0 / (k_h * k_w * input_.get_shape().as_list()[-1] * output_))
    w = weight_variable([k_h, k_w, input_.get_shape()[-1], output_], stddev, names)
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding="SAME")
    if bias:
        bias = bias_variable([output_], "b")
        return tf.nn.bias_add(conv, bias)
    else:
        return conv


def deconv2d(input_, output_, k_h=4, k_w=4, d_h=2, d_w=2, names="deconv2d"):
    stddev = np.sqrt(2.0 / (k_h * k_w * input_.get_shape().as_list()[-1] * output_[-1]))
    w = weight_variable([k_h, k_w, output_[-1], input_.get_shape()[-1]], stddev, names)
    bias = bias_variable([output_[-1]], "b")
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_, strides=[1, d_h, d_w, 1], padding="SAME")
    return tf.nn.bias_add(deconv, bias)


def linear(input_, output_size, stddev=0.1):
    shape = input_.get_shape().as_list()
    matrix = weight_variable([shape[1], output_size], stddev, "W")
    bias = bias_variable([output_size], "b")
    return tf.matmul(input_, matrix) + bias


def Bn(conv, training):
    conv_bn = tf.layers.batch_normalization(conv, training=training, epsilon=1e-5, name="bn")
    return conv_bn


def weight_variable(shape, stddev=0.02, name=None):
    w = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    #     #正则化
    #     tf.add_to_collection("regular_loss", tf.contrib.layers.l2_regularizer(0.5)(w))
    return w


def bias_variable(shape, name=None):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))


def max_pool(x, k_h=2, k_w=2, d_h=2, d_w=2, name='pool'):
    return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, d_h, d_w, 1], padding="SAME", name=name)


def avg_pool(x, k_h=2, k_w=2, d_h=2, d_w=2, name='pool'):
    return tf.nn.avg_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, d_h, d_w, 1], padding="SAME", name=name)


def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var


def conv2d_basic(x, W, bias, d_h=1, d_w=1):
    conv = tf.nn.conv2d(x, W, strides=[1, d_h, d_w, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def global_pool(x):
    return tf.reduce_mean(x, [1, 2])


def relu(x):
    return tf.nn.relu(x, name='relu')
