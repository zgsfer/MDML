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
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, f1_score, precision_score
#创建一个解析器
parser=argparse.ArgumentParser()
parser.add_argument("--cuda_device",type=str,default='0')
parser.add_argument("--train_path",type=str,default='')
parser.add_argument("--val_path",type=str,default='')
parser.add_argument("--batch_size",type=int,default=128)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--class_num",type=int,default=7)
parser.add_argument("--train_name",type=str,default='0')
parser.add_argument("--l_r",type=float,default=5e-6)
parser.add_argument("--branch_num",type=int,default=7)
parser.add_argument("--branch_size",type=int,default=256)
parser.add_argument("--recordname",type=str,default='sfew')
parser.add_argument("--pretrain",type=str,default='')

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
#learning_rate=5e-6
epochs=config.epochs
batch_size=config.batch_size
class_num=config.class_num
branch_num = config.branch_num
branch_size=config.branch_size
weight=0.5/branch_num
train_name=config.train_name
recordname = config.recordname
train_path = config.train_path
val_path = config.val_path
pretrain_path = config.pretrain
model_path='./save'+'/%s'%recordname+'/%s'%train_name
record_path = './save'+'/%s'%recordname+'/%s'%train_name
if branch_num == 7:
    branch_margin = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
if branch_num == 13:
    branch_margin = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
if branch_num == 4:
    branch_margin = [0.15, 0.35, 0.55, 0.75]

if os.path.exists(record_path):
    shutil.rmtree(record_path)
os.makedirs(record_path)

record_path=record_path+'/%s.txt'%recordname
#进行批次处理
train_data_generator=Input_generator(train_path,class_num,shuffle=True)
if train_data_generator.data_size % batch_size == 0:
    train_batch = int(np.floor(train_data_generator.data_size/ batch_size))
else:
    train_batch = int(np.floor(train_data_generator.data_size / batch_size)) + 1

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

    return all_loss
def Accuracy(net_out, one_hot_label):
    with tf.name_scope('accuracy'):
        correct_predict = tf.equal(tf.argmax(net_out, 1), tf.argmax(one_hot_label, 1)) #记录每一行最大的元素的索引 判断是否正确
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

        pre_label = tf.argmax(net_out, 1)
    tf.summary.scalar('accuracy', accuracy)
    return accuracy,pre_label



# config=tf.ConfigProto()
# config.gpu_options.allow_growth=True  #动态申请显存


with tf.Session()as sess:
    X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])#[batch, height, width, channels]

    Y_one_hot=tf.placeholder(dtype=tf.float32, shape=[None,class_num])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])
    TRAINING = tf.placeholder(dtype=tf.bool)
    keep_prob = tf.placeholder(dtype=tf.float32)
    model= resnet_branch_net(X,class_num,TRAINING,branch_num,branch_size,keep_prob)

    softmax_output = model.softmax

    s_loss = Softmax_loss(softmax_output, Y_one_hot)
    acc,predict_label=Accuracy(softmax_output,Y_one_hot)

    t_loss = 0
    count = 0
    print(weight)
    for feature, margin in zip(model.features, branch_margin):
        count += 1
        embedding = tf.nn.l2_normalize(feature, 1, 1e-10)
        # embedding = feature
        triple_loss = New_triple_loss(embedding, Y, margin, count)
        t_loss += triple_loss
    loss = Sum_loss(s_loss,t_loss,weight)

    with tf.name_scope('train'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op=optimizer.minimize(loss)



    merged_summary = tf.summary.merge_all()
    var_restore = [var for var in tf.trainable_variables() if var.name.split('/')[0] in restore_layer]
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
    max_private_acc = 0
    mval = []
    max_train_acc = 0
    max_val_acc = 0
    follow_train_acc = 0
    follow_epoch = 0
    for epoch in range(1, epochs + 1):
        # if (epoch == 10 or epoch

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
        print("Train_loss: {:.4f}      train_t_loss: {:.4f}       Train_acc :{:.4f}".format(mean_train_loss,mean_t_loss,mean_train_acc))
        mean_val_acc = 0
        mean_val_loss = 0
        mean_val_s_loss = 0
        mean_val_t_loss = 0
        mean_val_prive_loss = 0
        mean_val_prive_acc = 0
        y_true=[]
        y_predict=[]
        for i in range(1, val_batch + 1):
            val_x, val_y_one_hot,val_y= val_data_generator.next_batch(1,aug_data=False)
            y_true.append(val_y)
            val_s_loss,pre_label,val_loss,val_acc, summary_test = sess.run([s_loss,predict_label,loss,acc, merged_summary],
                                             feed_dict={X: val_x,
                                                        Y_one_hot: val_y_one_hot,
                                                        Y: val_y,
                                                        TRAINING: False,
                                                        keep_prob :1.
                                                        })

            mean_val_acc+= val_acc
            mean_val_loss += val_loss
            mean_val_s_loss += val_s_loss
            y_predict.append(pre_label)

        mean_val_acc /= val_batch
        mean_val_loss/=val_batch
        mean_val_s_loss /= val_batch
        print("val_loss:{:.4f}           val_acc :{:.4f}".format(mean_val_s_loss, mean_val_acc))
        if mean_val_acc >= max_acc:
             max_acc = mean_val_acc
             checkpoint_name = os.path.join(model_path, 'epoch' + str(epoch))
             model_saver.save(sess, checkpoint_name)
             follow_train_acc = mean_train_acc
             follow_epoch = epoch
             f = open(record_path, 'a')
             # weighted_p = precision_score(y_true, y_predict, labels=[0,1, 2,3,4,5], average='weighted')
             # # micro-recall
             # weighted_r = recall_score(y_true, y_predict, labels=[0,1, 2,3,4,5], average='weighted')
             # # micro f1-score
             # weighted_f1 = f1_score(y_true, y_predict, labels=[0,1, 2,3,4,5], average='weighted')
             #
             # print(weighted_p, weighted_r, weighted_f1)  # 0.8 0.8 0.8000000000000002
             #
             # # macro-precision
             # macro_p = precision_score(y_true, y_predict, average='macro')
             #  # macro-recall
             # macro_r = recall_score(y_true, y_predict, average='macro')
             #  # macro f1-score
             # macro_f1 = f1_score(y_true, y_predict, average='macro')
             #
             # print(macro_p, macro_r, macro_f1)
             #
             #
             # f = open(record_path, 'a')
             a = ('max_train_acc:%.4f , ' % max_train_acc)
             b = ('max_val_acc:%.4f , ' % max_acc)
             c = ('train_acc:%.4f , ' % follow_train_acc)
             d = ('epoch:%d' % follow_epoch)
             # e = ('weighted_p:%.4f , ' % weighted_p)
             # q = ('weighted_r:%.4f , ' % weighted_r)
             # g = ('weighted_f1:%.4f , ' % weighted_f1)
             #
             # x = ('macro_p:%.4f , ' % macro_p)
             # y = ('macro_r:%.4f , ' % macro_r)
             # z = ('macro_f1:%.4f' % macro_f1)
             
             f.write(a)
             f.write(b)
             f.write(c)
             f.write(d)
             # f.write(e)
             # f.write(q)
             # f.write(g)
             # f.write(x)
             # f.write(y)
             # f.write(z)
             f.write('\n')

        # print(" val_s_loss:{:.4f}                                  val_acc : {:.4f}       max_test._acc:{:.4f}      train_acc:{:.4f}      epoch:{:d}".format(
        #         mean_val_s_loss, mean_val_acc, max_acc, follow_train_acc, follow_epoch))
        print('max_acc:',max_acc)
        train_data_generator.reset_pointer()
        val_data_generator.reset_pointer()
sess.close()

print ('Training Done')

