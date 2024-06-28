import tensorflow as tf
import numpy as np
class resnet_branch_net:
    def __init__(self, input, calsses,TRAINING,branch_num,branch_size,dropout_rate):
        self.X = input
        self.classes = calsses
        self.TRAINING = TRAINING
        self.branch_num = branch_num
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
        print(self.x)
        # stage 5
        self.x = self.convolutional_block(self.x, 3, [512, 512,512], stage=5, block='a')
        self.x = self.identity_block(self.x, 3, [512, 512], stage=5, block='b')
        print(self.x)
        self.x = tf.layers.average_pooling2d(self.x, pool_size=(7, 7), strides=(1, 1))
        print(self.x)
        self.flatten = tf.layers.flatten(self.x, name='flatten')
        self.sp = tf.split(self.flatten,7,0)
        print(self.sp)
        # dense1 = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
        # self.logits = tf.layers.dense(self.flatten, units=7)





        # self.attentio = tf.get_variable('attention', shape=[1,self.branch_num],
        #                             initializer=tf.contrib.layers.xavier_initializer())
        # self.attention = tf.layers.dense(self.flatten,units=3,name="attention")
        # self.sum = tf.reduce_sum(self.attention,1,keepdims=True)
        # self.attention =self.attention/self.sum
        # self.attention = tf.split(self.attention, self.branch_num,1)
        # print(self.attention)
        # # self.sum = tf.reduce_sum(self.attention, 0,keepdims=True)
        # print(self.sum)
        self.features = []
        self.features_relu = []
        self.softmaxs = []
        for i in range(1, self.branch_num + 1):
            feature_name = 'feature' + str(i)
            softmax_name = 'softmax' + str(i)
            # self.x = self.senet(self.x, 512, 16,feature_name)
            self.feature = tf.layers.dense(self.flatten, units=self.branch_size,name=feature_name )
            # self.att = tf.layers.dense(self.feature,units=1,activation=tf.nn.sigmoid,name = feature_name+"attention")
            # self.feature_noml = tf.nn.l2_normalize(self.feature, 1, 1e-10)
            self.features.append(self.feature)
            # self.attention.append(self.att)
            self.feature_relu = tf.nn.relu(self.feature)
            # self.attention = self.NonLocalBlock(feature_relu, feature_name)
            # feature_relu = tf.nn.relu(self.feature_noml)
            self.features_relu.append(self.feature_relu)
            # self.res = tf.nn.dropout(self.feature_relu, self.dropout)
            # self.softmax =tf.layers.dense(self.feature_relu, units=self.classes,name=softmax_name)
            # self.softmaxs.append(self.softmax)
        # print(self.softmaxs)
        # self.logist = tf.divide(tf.add_n(self.softmaxs),self.branch_num)
        # print(self.logist)
        # self.features_mean=tf.divide(tf.add_n(self.features_relu),self.branch_num)
        # self.add_mean=tf.nn.dropout(self.features_mean,self.dropout)
        # self.features_mean = tf.add_n(self.features_relu)
        # self.max_val = tf.reduce_max(self.features_mean, axis=[0, 1], keepdims=True)
        # print(self.max_val)
        # print(self.features_mean.get_shape())
        # # self.features_mean= tf.nn.relu(self.features_mean)

        self.concat = tf.concat(self.features_relu, axis=1)
        print(self.concat)
        self.merge = tf.nn.dropout(self.concat, self.dropout)
        print(self.merge.get_shape())
        self.softmax = tf.layers.dense(self.merge, units=self.classes,name = 'softmax')

    # def senet(self, input_x, out_dim, ratio,name):
    #     # Excitation， 采用两个全连接网络进行信息的综合
    #     print("input_x")
    #     print(input_x)
    #     excitation = tf.layers.dense(input_x, units=out_dim / ratio, activation=tf.nn.relu, name=name+'se_fully_connected1')
    #     excitation = tf.layers.dense(excitation, units=out_dim, name=name+'se_fully_connected2')
    #     excitation = tf.nn.sigmoid(excitation)  # scale  信息的归一化处理  [0, 1]
    #     # excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
    #     # input_x = tf.reshape(input_x, [-1, 1, 1, out_dim])
    #     scale = input_x * excitation
    #     # scale = tf.layers.flatten(scale)
    #     print("scale")
    #     print(scale)
    #     return scale
    # def NonLocalBlock(self,input,feature_name, subsample=False):
    #     """
    #     @Non-local Neural Networks
    #     Non-local Block
    #     """
    #     input = tf.reshape(input,[-1,1,1,1536])
    #     _, height, width, channel = input.get_shape().as_list()  # (B, H, W, C)
    #
    #     theta = tf.layers.conv2d(input, channel / 2, 1,name = "theta"+feature_name )  # (B, H, W, C // 2)
    #     theta = tf.reshape(theta, [-1, height * width, channel // 2])  # (B, H*W, C // 2)
    #
    #     phi = tf.layers.conv2d(input, channel // 2, 1,name ="phi"+feature_name)  # (B, H, W, C // 2)
    #     if subsample:
    #         phi = tf.layers.max_pooling2d(phi, 2, 2)  # (B, H / 2, W / 2, C // 2)
    #         phi = tf.reshape(phi, [-1, height * width // 4, channel // 2])  # (B, H * W / 4, C // 2)
    #     else:
    #         phi = tf.reshape(phi, [-1, height * width, channel // 2])  # (B, H*W, C // 2)
    #     phi = tf.transpose(phi, [0, 2, 1])  # (B, C // 2, H*W)
    #
    #     f = tf.matmul(theta, phi)  # (B, H*W, H*W)
    #     f = tf.nn.softmax(f)  # (B, H*W, H*W)
    #
    #     g = tf.layers.conv2d(input, channel // 2, 1,name = "g"+feature_name)  # (B, H, W, C // 2)
    #     if subsample:
    #         g = tf.layers.max_pooling2d(g, 2, 2)  # (B, H / 2, W / 2, C // 2)
    #         g = tf.reshape(g, [-1, height * width // 4, channel // 2])  # (B, H*W, C // 2)
    #     else:
    #         g = tf.reshape(g, [-1, height * width, channel // 2])  # (B, H*W, C // 2)
    #
    #     y = tf.matmul(f, g)  # (B, H*W, C // 2)
    #     y = tf.reshape(y, [-1, height, width, channel // 2])  # (B, H, W, C // 2)
    #     y = tf.layers.conv2d(y, channel, 1)  # (B, H, W, C)
    #
    #     y = tf.add(input, y)  # (B, W, H, C)
    #
    #     y =tf.layers.flatten(y)
    #     return y


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
                                 padding='same'
                                 )
            x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2a', training=self.TRAINING)
            x = tf.nn.relu(x)
            # Second component of main path
            x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size), padding='same' ,name=conv_name_base + '2b')
            x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2b', training=self.TRAINING)

            # SHORTCUT PATH
            X_shortcut = tf.layers.conv2d(X_shortcut, filter3, (1, 1),
                                          strides=(stride, stride),name=conv_name_base + '1')
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
                                 padding='same',name=conv_name_base + '2b')
            x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'2b', training=self.TRAINING)
            x = tf.nn.relu(x)

            # Third component of main path
            x = tf.layers.conv2d(x, filter2, kernel_size=(kernel_size, kernel_size), name=conv_name_base + '2c',padding='same')
            x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=self.TRAINING)

            # Final step: Add shortcut value to main path, and pass it through a RELU activation
            X_add_shortcut = tf.add(x, X_shortcut)
            add_result = tf.nn.relu(X_add_shortcut)

        return add_result