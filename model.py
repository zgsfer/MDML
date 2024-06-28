import tensorflow as tf
import numpy as np

class vgg_branch_net:
    def __init__(self,input,calsses,branch_num,branch_size,dropout_rate):
        self.X = input
        self.classes=calsses
        self.branch_num = branch_num
        self.branch_size=branch_size
        self.dropout=dropout_rate

        self.create()


    def create(self):
        self.conv1_1 = self.conv(self.X, 3,3,64,1,1,'conv1_1')
        self.conv1_2 = self.conv(self.conv1_1, 3, 3, 64, 1, 1, 'conv1_2')
        self.pool1 = self.max_pool(self.conv1_2, 2,2,2,2,'pool1')


        self.conv2_1 = self.conv(self.pool1, 3,3,128,1,1,'conv2_1')
        self.conv2_2 = self.conv(self.conv2_1,3,3,128,1,1,'conv2_2')
        self.pool2 = self.max_pool(self.conv2_2, 2,2,2,2,'pool2')


        self.conv3_1 = self.conv(self.pool2, 3,3,256,1,1,'conv3_1')
        self.conv3_2 = self.conv(self.conv3_1,3,3,256,1,1,'conv3_2')
        self.conv3_3 = self.conv(self.conv3_2,3,3,256,1,1,'conv3_3')
        self.pool3 = self.max_pool(self.conv3_3, 2,2,2,2,'pool3')


        self.conv4_1 = self.conv(self.pool3, 3,3,512,1,1,'conv4_1')
        self.conv4_2 = self.conv(self.conv4_1,3,3,512,1,1,'conv4_2')
        self.conv4_3 = self.conv(self.conv4_2,3,3,512,1,1,'conv4_3')
        self.pool4 = self.max_pool(self.conv4_3, 2,2,2,2,'pool4')


        self.conv5_1 = self.conv(self.pool4, 3,3,512,1,1,'conv5_1')
        self.conv5_2 = self.conv(self.conv5_1, 3,3,512,1,1,'conv5_2')
        self.conv5_3 = self.conv(self.conv5_2, 3,3,512,1,1,'conv5_3')
        self.pool5 = self.max_pool(self.conv5_3, 2,2,2,2,'pool5')


        dimension=np.prod(self.pool5.get_shape().as_list()[1:])
        self.pool5_reshape = tf.reshape(self.pool5, [-1, dimension])
       

        self.fc6 = self.fc(self.pool5_reshape, 4096, 'fc6')
        self.drop_fc6=tf.nn.dropout(self.fc6,self.dropout)

        self.fc7 = self.fc(self.drop_fc6, 4096, 'fc7')
        self.drop_fc7 =tf.nn.dropout(self.fc7,self.dropout)


        self.features=[]
        self.features_relu=[]

        for i in range(1,self.branch_num+1):
            feature_name='feature'+str(i)
            self.feature = self.fc(self.drop_fc7, self.branch_size, feature_name, need_relu=False)
            self.features.append(self.feature)

            feature_relu=tf.nn.relu(self.feature)
            self.features_relu.append(feature_relu)

        #self.features_mean=tf.divide(tf.add_n(self.features_relu),self.branch_num)
        #print(self.features_mean.get_shape())
        #self.add_mean=tf.nn.dropout(self.features_mean,self.dropout)

        self.merge=tf.nn.dropout(tf.concat(self.features_relu,axis=1),self.dropout)
        print (self.merge.get_shape())

        self.fc8 = self.fc(self.merge, 256, 'fc8')
        self.drop_fc8=tf.nn.dropout(self.fc8,self.dropout)

        self.fc9 = self.fc(self.drop_fc8, 256, 'fc9')
        self.drop_fc9 = tf.nn.dropout(self.fc9, self.dropout)

        self.softmax=self.fc(self.drop_fc9,self.classes,'softmax',need_relu=False)


    def conv(self, input, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
        input_channels = int(input.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights',
                                      shape=[filter_height, filter_width, input_channels, num_filters],initializer=tf.contrib.layers.xavier_initializer())

            tf.add_to_collection('all_loss', tf.contrib.layers.l2_regularizer(0.0005)(weights))

            biases = tf.get_variable('biases', shape=[num_filters],initializer=tf.zeros_initializer())

            conv= tf.nn.conv2d(input, weights, [1, stride_y, stride_x, 1], padding)
            conv_relu= tf.nn.relu(tf.nn.bias_add(conv, biases))
        return conv_relu

    def max_pool(self, input, filter_height, filter_width, stride_y, stride_x, name, padding="SAME"):
        return tf.nn.max_pool(input, [1, filter_height, filter_width, 1], [1, stride_y, stride_x, 1], name=name,padding=padding)

    def fc(self,input, num_outputs, name, need_relu=True):
        num_inputs = int(input.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[num_inputs, num_outputs],initializer=tf.contrib.layers.xavier_initializer())
            tf.add_to_collection('all_loss', tf.contrib.layers.l2_regularizer(0.0005)(weights))

            biases = tf.get_variable('biases', shape=[num_outputs],initializer=tf.zeros_initializer())
            fc= tf.nn.xw_plus_b(input, weights, biases)
            if need_relu:
                fc = tf.nn.relu(fc)
            return fc
