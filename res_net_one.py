import tensorflow as tf
import numpy as np
class resnet_branch_net:
    def __init__(self, input, calsses,TRAINING,branch_size,dropout_rate):
        self.X = input
        self.classes = calsses
        self.TRAINING = TRAINING
        self.branch_size = branch_size
        self.dropout = dropout_rate
        self.create()

    def create(self):

        assert (self.X.shape == (self.X.shape[0], 224, 224, 3))

        # stage 1
        self.x = tf.layers.conv2d(self.X, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',name='conv1')
        self.x = tf.layers.batch_normalization(self.x, axis=3, name='bn_conv1', training=self.TRAINING)
        self.x = tf.nn.relu(self.x)
        self.x = tf.layers.max_pooling2d(self.x, pool_size=(3, 3), strides=(2, 2),padding='same')
        print(self.x)
        # stage 2
        self.x = self.identity_block(self.x, 3, [64, 64], stage=2, block='a')
        self.x = self.identity_block(self.x, 3, [64, 64], stage=2, block='b')
        print(self.x)
        # stage 3
        self.x = self.convolutional_block(self.x, 3, [128, 128,128], stage=3, block='a')
        self.x = self.identity_block(self.x, 3, [128, 128], stage=3, block='b')
        print(self.x)
        # stage 4
        self.x = self.convolutional_block(self.x, 3, [256, 256,256], stage=4, block='a')
        self.x = self.identity_block(self.x, 3, [256, 256], stage=4, block='b')
        # stage 5
        self.x = self.convolutional_block(self.x, 3, [512, 512,512], stage=5, block='a')
        self.x = self.identity_block(self.x, 3, [512, 512], stage=5, block='b')
        self.x = tf.layers.average_pooling2d(self.x, pool_size=(7, 7), strides=(1, 1))
        self.flatten = tf.layers.flatten(self.x, name='flatten')
        # dense1 = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
        # self.logits = tf.layers.dense(self.flatten, units=7)

        self.feature = tf.layers.dense(self.flatten, units=self.branch_size,name='feature' )

        self.feature_relu = tf.nn.dropout(tf.nn.relu(self.feature),self.dropout)
        # self.weights = tf.get_variable(name='softmax', shape=[self.feature.get_shape().as_list()[-1], 7], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=False), trainable=True)

        self.softmax = tf.layers.dense(self.feature, units=self.classes, name = 'softmax' )
        # self.weight = tf.layers.dense(self.feature, units=self.classes, name = 'softmax' )
    def convolutional_block(self,X_input, kernel_size, filters, stage, block, stride=2):

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        with tf.name_scope("conv_block_stage" + str(stage)):
            # Retrieve Filters
            filter1, filter2 ,filter3= filters

            # Save the input value
            X_shortcut = X_input
            # First component of main path
            x = tf.layers.conv2d(X_input, filter1,
                                 kernel_size=(3, 3),
                                 strides=(2, 2),
                                 name=conv_name_base + '2a',
                                 padding='same')
            x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2a', training=self.TRAINING)
            x = tf.nn.relu(x)
            # Second component of main path
            x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size), padding='same' ,name=conv_name_base + '2b')
            x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2b', training=self.TRAINING)

            # SHORTCUT PATH
            X_shortcut = tf.layers.conv2d(X_shortcut, filter3, (1, 1),
                                          strides=(stride, stride), name=conv_name_base + '1')
            X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, name=bn_name_base + '1', training=self.TRAINING)

            # Final step: Add shortcut value to main path, and pass it through a RELU activation
            X_add_shortcut = tf.add(X_shortcut, x)
            add_result = tf.nn.relu(X_add_shortcut)

        return add_result

    def identity_block(self,X_input, kernel_size, filters, stage, block):

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        with tf.name_scope("id_block_stage" + str(stage)):
            filter1, filter2 = filters
            X_shortcut = X_input

            # Second component of main path
            x = tf.layers.conv2d(X_input, filter1, (kernel_size, kernel_size),
                                 padding='same', name=conv_name_base + '2b')
            x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'2b', training=self.TRAINING)
            x = tf.nn.relu(x)

            # Third component of main path
            x = tf.layers.conv2d(x, filter2, kernel_size=(kernel_size, kernel_size), name=conv_name_base + '2c',padding='same')
            x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=self.TRAINING)

            # Final step: Add shortcut value to main path, and pass it through a RELU activation
            X_add_shortcut = tf.add(x, X_shortcut)
            add_result = tf.nn.relu(X_add_shortcut)

        return add_result