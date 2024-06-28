import tensorflow as tf
import os
from res_net_one import resnet_branch_net

import time
import shutil
from tool.generate_input_data import Input_generator
import numpy as np
import argparse
import sys
from triplet_loss import batch_hard_triplet_loss

#创建一个解析器
parser=argparse.ArgumentParser()
parser.add_argument("--cuda_device",type=str,default='1')
parser.add_argument("--train_path",type=str,default='/media/F/data/data/SFEW/SFEW_COLOR/train_aligment_aug/train_aligment_aug_mess1.txt')
parser.add_argument("--val_path",type=str,default='/media/F/data/data/SFEW/SFEW_COLOR/val_aligment/val_aligment1.txt')
parser.add_argument("--val_path1",type=str,default='/media/F/data/data/SFEW/SFEW_COLOR/val_aligment/val_aligment1.txt')
parser.add_argument("--batch_size",type=int,default=96)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--class_num",type=int,default=7)
parser.add_argument("--train_name",type=str,default='fer')
parser.add_argument("--l_r",type=float,default=5e-6)
parser.add_argument("--branch_margin",type=float,default=0.65)
parser.add_argument("--branch_size",type=int,default=256)
parser.add_argument("--recordname",type=str,default=1)
parser.add_argument("--istriplet_loss",type=int,default=1)
parser.add_argument("--ifscale",type=bool,default=True)



#把parser中设置的所有"add_argument"给返回到config子类实例当中， 那么parser中增加的属性内容都会在config实例中
config=parser.parse_args(sys.argv[1:])
# os.environ['CUDA_VISIBLE_DEVICES'] =config.cuda_device
# train_num=config.num

restore_layer=['conv1','bn_conv1','res2a_branch2b','bn2a_branch2b','res2a_branch2c','bn2a_branch2c','res2b_branch2b','bn2b_branch2b',
               'res2b_branch2c','bn2b_branch2c','res3a_branch2a','bn3a_branch2a','res3a_branch2b','bn3a_branch2b','res3a_branch1','bn3a_branch1','res3b_branch2b','bn3b_branch2b',
               'res3b_branch2c','bn3b_branch2c','res4a_branch2a','bn4a_branch2a','res4a_branch2b','res4a_branch2b','bn4a_branch2b','res4a_branch1','bn4a_branch1','res4b_branch2b',
               'bn4b_branch2b','res4b_branch2c','bn4b_branch2c','res5a_branch2a','bn5a_branch2a','res5a_branch2b','bn5a_branch2b','res5a_branch1','bn5a_branch1','res5b_branch2b',
               'bn5b_branch2b','res5b_branch2c','bn5b_branch2c']

notrestore_layer = ['dense']

os.environ['CUDA_VISIBLE_DEVICES'] =config.cuda_device
learning_rate=config.l_r
epochs=config.epochs
batch_size=config.batch_size
class_num=config.class_num
branch_margin=config.branch_margin
branch_size=config.branch_size
weight=0.5
train_name=config.train_name
recordname = config.recordname
train_path = config.train_path
val_path = config.val_path
val_path1 = config.val_path1
istriplet_loss = config.istriplet_loss
ifscale = config.ifscale
model_path='./iccv_record/res_one'+'/%s'%recordname+'/%s'%train_name
record_path = './iccv_record/res_one'+'/%s'%recordname+'/%s'%train_name



record_path=record_path+'/%s.txt'%recordname
#进行批次处理
train_data_generator=Input_generator(train_path,class_num,shuffle=True,ifscale=False)
if train_data_generator.data_size % batch_size == 0:
    train_batch = int(np.floor(train_data_generator.data_size/ batch_size))
else:
    train_batch = int(np.floor(train_data_generator.data_size / batch_size)) + 1
# train_batch=int(np.floor(train_data_generator.data_size/batch_size)) #np.floor 返回不大于输入参数的最大整数

val_data_generator=Input_generator(val_path,class_num,shuffle=False)
val_batch=int(np.floor(val_data_generator.data_size/1)) #np.floor 返回不大于输入参数的最大整数

val_data_generator1=Input_generator(val_path1,class_num,shuffle=False)
val_batch1=int(np.floor(val_data_generator1.data_size/1)) #np.floor 返回不大于输入参数的最大整数


def Softmax_loss(net_out, gt):
    with tf.name_scope('softmax_loss'):      #命名空间
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=gt)
        softmax_loss = tf.reduce_mean(cross_entropy)


    return softmax_loss

def New_triple_loss(net_out, gt,m,index):
    with tf.name_scope('new_triple_loss'):
        new_triple_loss= batch_hard_triplet_loss(net_out, gt, m,index, squared=False)
        tf.summary.scalar('new_triple_loss', new_triple_loss)

    return new_triple_loss

def Sum_loss(s_loss,t_loss,w):
    loss = s_loss+w*t_loss
    tf.add_to_collection('all_loss',loss)
    all_loss = tf.add_n(tf.get_collection('all_loss'))

    return all_loss

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
    Y = tf.placeholder(dtype=tf.float32, shape=[None])
    TRAINING = tf.placeholder(dtype=tf.bool)
    keep_prob = tf.placeholder(dtype=tf.float32)
    model= resnet_branch_net(X,class_num,TRAINING,branch_size,keep_prob)

    softmax_output = model.softmax

    s_loss=Softmax_loss(softmax_output,Y_one_hot)
    acc=Accuracy(softmax_output,Y_one_hot)

    t_loss = 0
    count = 0
    print(weight)

    embedding = tf.nn.l2_normalize(model.feature, 1, 1e-10)
    print (embedding)
    triple_loss = New_triple_loss(embedding, Y, branch_margin, count)
    t_loss += triple_loss
    if istriplet_loss == 1:
        loss = Sum_loss(s_loss,t_loss,weight)
    if istriplet_loss == 0:
        loss = s_loss
    with tf.name_scope('train'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op=optimizer.minimize(loss)



    merged_summary = tf.summary.merge_all()
    var_restore = [var for var in tf.trainable_variables() if var.name.split('/')[0] in restore_layer]
    print(var_restore)
    model_saver = tf.train.Saver(var_list=tf.global_variables(),max_to_keep=1)  # 每次保留上一次训练的模型
    restorer = tf.train.Saver(var_restore)
    sess.run(tf.global_variables_initializer())
    restorer.restore(sess, '/media/D/yjy/FER/basenet/58.62/epoch24')

    print(learning_rate)
    print(branch_margin)
    print(train_name)
    print(config.cuda_device)
    print ("Starting training")
    min_acc = 0
    max_acc = 0
    max_acc1 = 0
    mval = []
    max_train_acc = 0
    max_val_acc = 0
    follow_train_acc = 0
    follow_epoch = 0
    for epoch in range(1, epochs + 1):
        print("{}      Epoch number : {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch))

        mean_train_loss = 0
        mean_train_acc = 0
        mean_s_loss = 0
        mean_t_loss = 0
        for step in range(1, train_batch + 1):
            if step == train_batch:
                bt = train_data_generator.data_size % batch_size
            else:
                bt = batch_size
            train_x, train_y_one_hot,train_y= train_data_generator.next_batch(bt,aug_data=True)

            _,train_loss,train_s_loss,train_t_loss, train_acc,summary_train = sess.run([train_op,loss,s_loss,t_loss, acc,merged_summary],
                                                                                        feed_dict={
                                                                                            X: train_x,
                                                                                            Y_one_hot:train_y_one_hot,
                                                                                            Y:train_y,
                                                                                            TRAINING: True,
                                                                                            keep_prob:0.5})

            mean_train_loss += train_loss
            mean_train_acc += train_acc
            mean_s_loss += train_s_loss
            mean_t_loss += train_t_loss
        mean_train_loss /= train_batch
        mean_train_acc /= train_batch
        mean_s_loss /= train_batch
        mean_t_loss /= train_batch
        if mean_train_acc>max_train_acc:
            max_train_acc=mean_train_acc
        # print("Train_loss: {:.4f}       Train_acc :{:.4f}".format(mean_train_loss,mean_train_acc))
        print("train_loss:{:.4f}      train_acc :{:.4f}      max_train_acc:{:.4f}".format(mean_train_loss,mean_train_acc,max_train_acc))

        mean_val_acc = 0
        mean_val_loss = 0
        mean_val_s_loss = 0
        mean_val_t_loss = 0
        for i in range(1, val_batch + 1):
            val_x, val_y_one_hot,val_y= val_data_generator.next_batch(1,aug_data=False)
            val_s_loss,val_loss,val_acc, summary_test = sess.run([s_loss,loss,acc, merged_summary],
                                             feed_dict={X: val_x,
                                                        Y_one_hot: val_y_one_hot,
                                                        Y: val_y,
                                                        TRAINING: False,
                                                        keep_prob :1.
                                                        })

            mean_val_acc+= val_acc
            mean_val_loss += val_loss
            mean_val_s_loss += val_s_loss
        mean_val_acc /= val_batch
        mean_val_loss/=val_batch
        mean_val_s_loss /= val_batch
        print("val_loss:{:.4f}           val_acc :{:.4f}".format(mean_val_s_loss, mean_val_acc))
        if mean_val_acc >= max_acc:
             max_acc = mean_val_acc
             follow_train_acc = mean_train_acc
             follow_epoch = epoch
             # f = open(record_path, 'a')
             # a = ('max_train_acc:%.4f , ' % max_train_acc)
             # b = ('max_val_acc:%.4f , ' % max_acc)
             # c = ('train_acc:%.4f , ' % follow_train_acc)
             # d = ('epoch:%d' % follow_epoch)
             # f.write(a)
             # f.write(b)
             # f.write(c)
             # f.write(d)
             # f.write('\n')

        mean_val_acc1 = 0
        mean_val_loss1 = 0
        mean_val_s_loss1 = 0
        mean_val_t_loss1 = 0
        for i in range(1, val_batch1 + 1):
            val_x, val_y_one_hot, val_y = val_data_generator1.next_batch(1, aug_data=False)
            val_s_loss, val_loss, val_acc, summary_test = sess.run([s_loss, loss, acc, merged_summary],
                                                                   feed_dict={X: val_x,
                                                                              Y_one_hot: val_y_one_hot,
                                                                              Y: val_y,
                                                                              TRAINING: False,
                                                                              keep_prob: 1.
                                                                              })

            mean_val_acc1 += val_acc
            mean_val_loss1 += val_loss
            mean_val_s_loss1 += val_s_loss
        mean_val_acc1 /= val_batch1
        mean_val_loss1 /= val_batch1
        mean_val_s_loss1 /= val_batch1
        print("val_loss:{:.4f}           val_acc :{:.4f}".format(mean_val_s_loss, mean_val_acc1))
        if mean_val_acc1 >= max_acc1:
            checkpoint_name = os.path.join(model_path, 'epoch' + str(epoch))
            # model_saver.save(sess, checkpoint_name)
            max_acc1 = mean_val_acc1
            follow_train_acc = mean_train_acc
            follow_epoch = epoch
            # f = open(record_path, 'a')
            # a = ('max_train_acc:%.4f , ' % max_train_acc)
            # b = ('max_val_acc1:%.4f , ' % max_acc1)
            # c = ('train_acc:%.4f , ' % follow_train_acc)
            # d = ('epoch:%d' % follow_epoch)
            # f.write(a)
            # f.write(b)
            # f.write(c)
            # f.write(d)
            # f.write('\n')
        # print("                       val_s_loss:{:.4f}                                  val_acc : {:.4f}       max_test._acc:{:.4f}      train_acc:{:.4f}      epoch:{:d}".format(
        #         mean_val_s_loss, mean_val_acc, max_acc, follow_train_acc, follow_epoch))
        print('max_acc:',max_acc)
        print('max_acc1:', max_acc1)
        train_data_generator.reset_pointer()
        val_data_generator.reset_pointer()
        val_data_generator1.reset_pointer()
sess.close()

print ('Training Done')

