import tensorflow as tf
import os
from resnet_branch_net import resnet_branch_net
import time
import shutil
from tool.generate_input_data import Input_generator
import numpy as np
import argparse
import sys
from triplet_loss import batch_hard_triplet_loss
#创建一个解析器
parser=argparse.ArgumentParser()
parser.add_argument("--cuda_device",type=str,default='2')
parser.add_argument("--train_path",type=str)
parser.add_argument("--val_path",type=str,default='')
parser.add_argument("--batch_size",type=int,default=96)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--class_num",type=int,default=7)
parser.add_argument("--train_name",type=str,default='7_bran_fer')
parser.add_argument("--l_r",type=float,default=5e-6)
parser.add_argument("--branch_num",type=int,default=7)
parser.add_argument("--branch_size",type=int,default=256)
parser.add_argument("--recordname",type=str,default='sfew')
parser.add_argument("--pretrain",type=str,default='')
parser.add_argument("--same",type=int,default='0')

#把parser中设置的所有"add_argument"给返回到config子类实例当中， 那么parser中增加的属性内容都会在config实例中
config=parser.parse_args(sys.argv[1:])
# os.environ['CUDA_VISIBLE_DEVICES'] =config.cuda_device
# train_num=config.num

restore_layer=['conv1','bn_conv1','res2a_branch2b','bn2a_branch2b','res2a_branch2c','bn2a_branch2c','res2b_branch2b','bn2b_branch2b',
               'res2b_branch2c','bn2b_branch2c','res3a_branch2a','bn3a_branch2a','res3a_branch2b','bn3a_branch2b','res3a_branch1','bn3a_branch1','res3b_branch2b','bn3b_branch2b',
               'res3b_branch2c','bn3b_branch2c','res4a_branch2a','bn4a_branch2a','res4a_branch2b','res4a_branch2b','bn4a_branch2b','res4a_branch1','bn4a_branch1','res4b_branch2b',
               'bn4b_branch2b','res4b_branch2c','bn4b_branch2c','res5a_branch2a','bn5a_branch2a','res5a_branch2b','bn5a_branch2b','res5a_branch1','bn5a_branch1','res5b_branch2b',
               'bn5b_branch2b','res5b_branch2c','bn5b_branch2c']

notrestore_layer = ['softmax','excitation','excitation1']
os.environ['CUDA_VISIBLE_DEVICES'] =config.cuda_device
learning_rate=config.l_r
epochs=config.epochs
batch_size=config.batch_size
class_num=config.class_num
branch_num = config.branch_num
# branch_margin=config.branch_margin
branch_size=config.branch_size
weight=0.5/branch_num
train_name=config.train_name
recordname = config.recordname
train_path = config.train_path
val_path = config.val_path
pretrain_path = config.pretrain
same = config.same
model_path='./iccv_record/mulititask'+'/%s'%recordname+'/%s'%train_name
record_path = './iccv_record/mulititask'+'/%s'%recordname+'/%s'%train_name
if branch_num == 7:
    branch_margin = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
if branch_num == 13:
    branch_margin = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
if branch_num == 4:
    branch_margin = [0.15, 0.35, 0.55, 0.75]
if same == 1:
    branch_margin = [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]
if same == 2:
    branch_margin = [0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65]
if same == 3:
    branch_margin = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
# 是否有文件，没有则创建
# if os.path.exists(model_path):
#      shutil.rmtree(model_path)
# os.mkdir(model_path)
#
#
# if os.path.exists(record_path):
#     shutil.rmtree(record_path)
# os.makedirs(record_path)

record_path=record_path+'/%s.txt'%recordname
#进行批次处理
train_data_generator=Input_generator(train_path,class_num,shuffle=True)
if train_data_generator.data_size % batch_size == 0:
    train_batch = int(np.floor(train_data_generator.data_size/ batch_size))
else:
    train_batch = int(np.floor(train_data_generator.data_size / batch_size)) + 1
train_batch=int(np.floor(train_data_generator.data_size/batch_size)) #np.floor 返回不大于输入参数的最大整数

val_data_generator=Input_generator(val_path,class_num,shuffle=False)
val_batch=int(np.floor(val_data_generator.data_size/1)) #np.floor 返回不大于输入参数的最大整数




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
    all_loss = all_loss

    return all_loss

def Accuracy(net_out, one_hot_label):
    with tf.name_scope('accuracy'):
        correct_predict = tf.equal(tf.argmax(net_out, 1), tf.argmax(one_hot_label, 1)) #记录每一行最大的元素的索引 判断是否正确
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

def Voting_acc(voting,one_hot_label):
    pre_label = np.argmax(voting, 1)
    true_label = np.argmax(one_hot_label, 1)

    correct_predict = np.equal(pre_label, true_label)
    accuracy = np.mean(correct_predict)


    return accuracy


# config=tf.ConfigProto()
# config.gpu_options.allow_growth=True  #动态申请显存


with tf.Session()as sess:
    X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])#[batch, height, width, channels]

    Y_one_hot=tf.placeholder(dtype=tf.float32, shape=[None,class_num])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])
    TRAINING = tf.placeholder(dtype=tf.bool)
    keep_prob = tf.placeholder(dtype=tf.float32)
    model= resnet_branch_net(X,class_num,TRAINING,branch_num,branch_size,keep_prob)

    softmax_output = model.softmaxs
    s_loss = 0
    for i in range(7):
        s_loss += Softmax_loss(softmax_output[i], Y_one_hot)
    s_loss = s_loss/7
    t_loss = 0
    count = 0
    print(weight)
    for feature, margin in zip(model.features, branch_margin):
        count += 1
        embedding = tf.nn.l2_normalize(feature, 1, 1e-10)
        # embedding = feature
        triple_loss = New_triple_loss(embedding, Y, margin, count)
        t_loss += triple_loss
    loss = s_loss*7

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
    restorer.restore(sess, pretrain_path)
    print(learning_rate)
    print(config.cuda_device)
    print(branch_margin)
    print(branch_num)
    print(train_name)
    print(batch_size)
    print(branch_size)
    print ("Starting training")
    min_acc = 0
    max_acc = 0
    mval = []
    max_train_acc = 0
    max_val_acc = 0
    follow_train_acc = 0
    follow_epoch = 0
    for epoch in range(1, epochs + 1):
        # if (epoch == 10 or epoch == 18 or epoch == 25 or epoch == 32):
        #     learning_rate = learning_rate / 10
        #     print(learning_rate)
        print("{}      Epoch number : {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch))

        mean_train_loss = 0
        mean_train_acc = 0
        mean_s_loss = 0
        mean_t_loss = 0
        brn_acc = [0,0,0,0,0,0,0]
        for step in range(1, train_batch + 1):
            if step == train_batch:
                bt = train_data_generator.data_size % batch_size
            else:
                bt = batch_size
            train_x, train_y_one_hot,train_y= train_data_generator.next_batch(bt,aug_data=True)
            _,train_loss,train_s_loss,train_t_loss,output,summary_train = sess.run([train_op,loss,s_loss,t_loss, softmax_output,merged_summary],
                                                                                        feed_dict={
                                                                                            X: train_x,
                                                                                            Y_one_hot:train_y_one_hot,
                                                                                            Y:train_y,
                                                                                            TRAINING: True,
                                                                                            keep_prob:0.5})
            voting = []
            for i in range(7):
                voting.append(np.argmax(output[i], 1))
            # print(voting)
            results = np.zeros((bt, class_num))
            # print(result)
            for i in range(len(results)):
                for l in range(len(voting)):
                    results[i][voting[l][i]] += 1
            voting_acc = Voting_acc(results, train_y_one_hot)
            mean_train_acc += voting_acc
            mean_train_loss += train_loss
            mean_s_loss += train_s_loss
            mean_t_loss += train_t_loss
        mean_train_loss /= train_batch
        mean_train_acc /= train_batch
        mean_s_loss /= train_batch
        mean_t_loss /= train_batch
        if mean_train_acc>max_train_acc:
            max_train_acc=mean_train_acc
        print("Train_loss: {:.4f}      train_t_loss: {:.4f}       Train_acc :{:.4f}".format(mean_train_loss,mean_t_loss,mean_train_acc))
        # print("train_loss:{:.4f}      train_s_loss: {:.4f}      train_t_loss: {:.4f}      train_acc :{:.4f}      max_train_acc:{:.4f}".format(mean_train_loss,mean_s_loss,mean_t_loss,mean_train_acc,max_train_acc))
        mean_val_acc = 0
        mean_val_loss = 0
        mean_val_s_loss = 0
        mean_val_t_loss = 0
        attention = []
        for i in range(1, val_batch + 1):
            val_x, val_y_one_hot,val_y= val_data_generator.next_batch(1,aug_data=False)

            val_s_loss,val_loss,outputs, summary_test = sess.run([s_loss,loss,softmax_output, merged_summary],
                                             feed_dict={X: val_x,
                                                        Y_one_hot: val_y_one_hot,
                                                        Y: val_y,
                                                        TRAINING: False,
                                                        keep_prob :1.
                                                        })
            voting = []
            for i in range(7):
                voting.append(np.argmax(outputs[i], 1))
            # print(voting)
            results = np.zeros((1, class_num))
            # print(result)
            for i in range(len(results)):
                for l in range(len(voting)):
                    results[i][voting[l][i]] += 1
            voting_acc = Voting_acc(results, val_y_one_hot)

            mean_val_acc+= voting_acc
            mean_val_loss += val_loss
            mean_val_s_loss += val_s_loss


        mean_val_acc /= val_batch
        mean_val_loss/=val_batch
        mean_val_s_loss /= val_batch
        print("val_loss:{:.4f}           val_acc :{:.4f}".format(mean_val_s_loss, mean_val_acc))
        if mean_val_acc >= max_acc:
             checkpoint_name = os.path.join(model_path, 'epoch' + str(epoch))
             model_saver.save(sess, checkpoint_name)
             max_acc = mean_val_acc
             follow_train_acc = mean_train_acc
             follow_epoch = epoch
             f = open(record_path, 'a')
             a = ('max_train_acc:%.4f , ' % max_train_acc)
             b = ('max_val_acc:%.4f , ' % max_acc)
             c = ('train_acc:%.4f , ' % follow_train_acc)
             d = ('epoch:%d' % follow_epoch)
             f.write(a)
             f.write(b)
             f.write(c)
             f.write(d)
             f.write('\n')

        # print("                       val_s_loss:{:.4f}                                  val_acc : {:.4f}       max_test._acc:{:.4f}      train_acc:{:.4f}      epoch:{:d}".format(
        #         mean_val_s_loss, mean_val_acc, max_acc, follow_train_acc, follow_epoch))
        print(max_acc)
        train_data_generator.reset_pointer()
        val_data_generator.reset_pointer()

sess.close()

print ('Training Done')

