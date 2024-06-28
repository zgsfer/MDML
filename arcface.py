import tensorflow as tf
import tensorflow.contrib.slim as slim

import math


W_INIT = tf.contrib.layers.xavier_initializer()
# class NormDense(tf.keras.layers.Layer):
#
#     def __init__(self, classes=1000):
#         super(NormDense, self).__init__()
#         self.classes = classes
#
#     def build(self, input_shape):
#         self.w = self.add_weight(name='norm_dense_w', shape=(input_shape[-1], self.classes),
#                                  initializer='random_normal', trainable=True)
#
#     def call(self, inputs, **kwargs):
#         norm_w = tf.nn.l2_normalize(self.w, axis=0)
#         x = tf.matmul(inputs, norm_w)
#
#         return x

def get_logits(embds, labels, class_num,logits_scale, logits_margin,w_init=W_INIT, reuse=False, scope='logits'):
    with tf.variable_scope(scope, reuse=reuse):
        weights = tf.get_variable(name='classify_weight', shape=[embds.get_shape().as_list()[-1], class_num], dtype=tf.float32, initializer=w_init, regularizer=slim.l2_regularizer(5e-6), trainable=True)
        print(weights)
        # if config['loss_type'] == 'arcface':
        return calculate_arcface_logits(embds, weights, labels, class_num, logits_scale, logits_margin)
        # elif config['loss_type'] == 'softmax':
        #     return slim.fully_connected(embds, num_outputs=config['class_num'], activation_fn=None, normalizer_fn=None, weights_initializer=w_init, weights_regularizer=slim.l2_regularizer(config['weight_decay']))
        # else:
        #     raise ValueError('Invalid loss type.')


def calculate_arcface_logits(embds, weights, labels, class_num, s, m):
    output2 = tf.matmul(embds, weights)
    embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
    weights = tf.nn.l2_normalize(weights, axis=0)

    cos_m = math.cos(m)
    sin_m = math.sin(m)

    mm = sin_m * m

    threshold = math.cos(math.pi - m)

    cos_t = tf.matmul(embds, weights, name='cos_t')

    cos_t2 = tf.square(cos_t, name='cos_2')
    sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
    sin_t = tf.sqrt(sin_t2, name='sin_t')
    cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
    cond_v = cos_t - threshold
    cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
    keep_val = s*(cos_t - mm)
    cos_mt_temp = tf.where(cond, cos_mt, keep_val)
    mask = tf.one_hot(labels, depth=class_num, name='one_hot_mask')
    inv_mask = tf.subtract(1., mask, name='inverse_mask')
    s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')
    output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_logits')

    # output2 =

    return output,output2


def arcface_loss(x, normx_cos, labels, m1=1, m2=0.2, m3=0.3, s=64.0):
    norm_x = tf.norm(x, axis=1, keepdims=True)
    cos_theta = normx_cos / norm_x
    theta = tf.acos(cos_theta)
    mask = tf.one_hot(labels, depth=normx_cos.shape[-1])
    zeros = tf.zeros_like(mask)
    cond = tf.where(tf.greater(theta * m1 + m3, math.pi), zeros, mask)
    cond = tf.cast(cond, dtype=tf.bool)
    m1_theta_plus_m3 = tf.where(cond, theta * m1 + m3, theta)
    cos_m1_theta_plus_m3 = tf.cos(m1_theta_plus_m3)
    prelogits = tf.where(cond, cos_m1_theta_plus_m3 - m2, cos_m1_theta_plus_m3) * s

    # cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # do softmax
    # loss = cce(labels, prelogits)

    return prelogits