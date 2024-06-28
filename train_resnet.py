import tensorflow as tf
import os
from res18 import resnet
import time
import shutil
from tool.generate_input_data import Input_generator
import numpy as np
import argparse
import sys

train_path = ''
val_path = ''
#创建一个解析器
parser=argparse.ArgumentParser()

config=parser.parse_args(sys.argv[1:])

notrestore_layer = ['dense']
# Learning params
os.environ['CUDA_VISIBLE_DEVICES'] ='1'
learning_rate=1e-4
epochs=70
batch_size=80
class_num=7
train_num=0


model_path='./models_'+'%d'%train_num
# 是否有文件，没有则创建
if os.path.exists(model_path):
     shutil.rmtree(model_path)
os.mkdir(model_path)

# 进行批次处理
train_data_generator=Input_generator(train_path,class_num,shuffle=True)
train_batch=int(np.floor(train_data_generator.data_size/batch_size)) #np.floor 返回不大于输入参数的最大整数

val_data_generator=Input_generator(val_path,class_num,shuffle=False)
val_batch=int(np.floor(val_data_generator.data_size/1)) #np.floor 返回不大于输入参数的最大整数




def Softmax_loss(net_out, gt):
    with tf.name_scope('softmax_loss'):      #命名空间
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=gt)
        softmax_loss = tf.reduce_mean(cross_entropy)
        l2_loss = tf.losses.get_regularization_loss()
        tf.add_to_collection('losses', softmax_loss)#将元素softmax_loss添加到列表losses中
        all_loss = tf.add_n(tf.get_collection('losses'))+l2_loss#返回名称为losses的列表
        tf.summary.scalar('losses', all_loss)  #画图


    return  all_loss


def Accuracy(net_out, one_hot_label):
    with tf.name_scope('accuracy'):
        correct_predict = tf.equal(tf.argmax(net_out, 1), tf.argmax(one_hot_label, 1)) #记录每一行最大的元素的索引 判断是否正确
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy



# config=tf.ConfigProto()
# config.gpu_options.allow_growth=True  #动态申请显存


with tf.Session()as sess:

    X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])#[batch, height, width, channels]

    Y_one_hot=tf.placeholder(dtype=tf.float32, shape=[None,class_num])
    TRAINING = tf.placeholder(dtype=tf.bool)
    model= resnet(X,class_num,TRAINING)

    softmax_output = model.logits

    loss=Softmax_loss(softmax_output,Y_one_hot)
    acc=Accuracy(softmax_output,Y_one_hot)

    with tf.name_scope('train'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op=optimizer.minimize(loss)



    merged_summary = tf.summary.merge_all()
    var_restore = [var for var in tf.trainable_variables() ]
    print(var_restore)
    model_saver = tf.train.Saver(var_list=tf.global_variables(),max_to_keep=1)  # 每次保留上一次训练的模型
    # restorer = tf.train.Saver(var_restore)
    restorer = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    print(learning_rate)
    print ("Starting training")
    min_acc = 0
    max_acc = 0
    mval = []
    for epoch in range(1, epochs + 1):
        print("{}      Epoch number : {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch))

        mean_train_loss = 0
        mean_train_acc = 0
        for step in range(1, train_batch + 1):

            train_x, train_y_one_hot,train_y= train_data_generator.next_batch(batch_size,aug_data=True)
            _,train_loss, train_acc,summary_train = sess.run([train_op,loss, acc,merged_summary],feed_dict={X: train_x,Y_one_hot:train_y_one_hot, TRAINING: True})

            mean_train_loss += train_loss
            mean_train_acc += train_acc

        mean_train_loss /= train_batch
        mean_train_acc /= train_batch

        print("Train_loss: {:.4f}       Train_acc :{:.4f}".format(mean_train_loss,mean_train_acc))
        mean_val_acc = 0
        mean_val_loss = 0
        for i in range(1, val_batch + 1):
            val_x, val_y_one_hot,val_y= val_data_generator.next_batch(1,aug_data=False)
            # val_x,val_y_one_hot = minst.train.next_batch(batch_size)
            val_loss,val_acc, summary_test = sess.run([loss,acc, merged_summary],
                                             feed_dict={X: val_x,Y_one_hot: val_y_one_hot, TRAINING: False})

            mean_val_acc+= val_acc
            mean_val_loss += val_loss
        mean_val_acc /= val_batch
        mean_val_loss/=val_batch

        print("val_loss:{:.4f}           val_acc :{:.4f}".format(mean_val_loss, mean_val_acc))
        if mean_val_acc >= max_acc:
             checkpoint_name = os.path.join(model_path, 'epoch' + str(epoch))
             model_saver.save(sess, checkpoint_name)
             max_acc = mean_val_acc
             follow_train_acc = mean_train_acc
             follow_epoch = epoch


        print(max_acc)
        train_data_generator.reset_pointer()
        val_data_generator.reset_pointer()

sess.close()

print ('Training Done')

