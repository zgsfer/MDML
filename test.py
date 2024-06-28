import tensorflow as tf
import os
from resnet_branch_net import resnet_branch_net
from tool.generate_input_data import Input_generator
import numpy as np
class_num = 7
branch_num = 7
branch_size = 256
parser.add_argument("--val_path",type=str,default='mmi')
parser.add_argument("--model_path",type=str,default='/media/D/yjy/FER/basenet/58.62/epoch24')

batch_size=config.val_path
model_path=config.model_path
val_data_generator=Input_generator(val_path,class_num,shuffle=False)
val_batch=int(np.floor(val_data_generator.data_size/1)) #np.floor 返回不大于输入参数的最大整数
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def Softmax_loss(net_out, gt):
    with tf.name_scope('softmax_loss'):      #命名空间
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=gt)
        softmax_loss = tf.reduce_mean(cross_entropy)


    return softmax_loss
def Accuracy(net_out, one_hot_label):
    with tf.name_scope('accuracy'):
        correct_predict = tf.equal(tf.argmax(net_out, 1), tf.argmax(one_hot_label, 1)) #记录每一行最大的元素的索引 判断是否正确
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy
with tf.Session()as sess:
    X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])#[batch, height, width, channels]

    Y_one_hot=tf.placeholder(dtype=tf.float32, shape=[None,class_num])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])
    TRAINING = tf.placeholder(dtype=tf.bool)
    keep_prob = tf.placeholder(dtype=tf.float32)
    model= resnet_branch_net(X,class_num,TRAINING,branch_num,branch_size,keep_prob)

    softmax_output = model.softmax
    s_loss = Softmax_loss(softmax_output, Y_one_hot)
    acc = Accuracy(softmax_output, Y_one_hot)

    restorer = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    restorer.restore(sess, model_path)

    mean_val_acc = 0
    mean_val_loss = 0
    mean_val_s_loss = 0
    mean_val_t_loss = 0
    for i in range(1, val_batch + 1):
        val_x, val_y_one_hot, val_y = val_data_generator.next_batch(1, aug_data=False)

        val_s_loss, val_acc = sess.run([s_loss , acc],
                                      feed_dict={X: val_x,
                                                 Y_one_hot: val_y_one_hot,
                                                 Y: val_y,
                                                 TRAINING: False,
                                                 keep_prob: 1.
                                                 })

        mean_val_acc += val_acc
        mean_val_s_loss += val_s_loss
    mean_val_acc /= val_batch
    mean_val_s_loss /= val_batch
    print("val_loss:{:.4f}           val_acc :{:.4f}".format(mean_val_s_loss, mean_val_acc))