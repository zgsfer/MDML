# -*- coding:utf-8 -*-
import tensorflow as tf
import keras.backend as K

def _pairwise_distances(embeddings, squared=False):
    '''
           计算两两embedding的距离
           ------------------------------------------
           Args：
              embedding: 特征向量， 大小（batch_size, vector_size）
              squared:   是否距离的平方，即欧式距离 True->欧式距离的平方，False->欧氏距离

           Returns：
              distances: 两两embeddings的距离矩阵，大小 （batch_size, batch_size）
        '''
    # 矩阵相乘,得到（batch_size, batch_size），因为计算欧式距离|a-b|^2 = a^2 -2ab + b^2,
    # 其中 ab 可以用矩阵乘表示
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))#tf.transpose(embeddings)转置
    # print(dot_product)
    # dot_product对角线部分就是 每个embedding的平方
    # 'input' is [[1, 0, 0, 0]
    # [0, 2, 0, 0]
    # [0, 0, 3, 0]
    # [0, 0, 0, 4]]
    #
    # tf.diag_part(input) == > [1, 2, 3, 4]
    square_norm = tf.diag_part(dot_product)
    # |a-b|^2 = a^2 - 2ab + b^2
    # tf.expand_dims(square_norm, axis=1)是（batch_size, 1）大小的矩阵，减去 （batch_size, batch_size）大小的矩阵，相当于每一列操作
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)#计算| a - b | ^ 2
    distances = tf.maximum(distances, 0.0) # 小于0的距离置为0

    if not squared:      # 如果不平方，就开根号，但是注意有0元素，所以0的位置加上 1e*-16
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)#矫正距离

    return distances


def _get_anchor_positive_triple_mask(labels):
    '''
          得到一个3D的mask [a, p, n], 对应triplet（a, p, n）是valid的位置是True
          ----------------------------------
          Args:
             labels: 对应训练数据的labels, shape = (batch_size,)

          Returns:
             mask: 合法的positive mask, shape = (batch_size, batch_size)

       '''
    # 初始化一个二维矩阵，坐标(i, j)
    # 不相等置为1，得到indices_not_equal
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    # 因为最后得到一个3D的mask矩阵(i, j, k)，增加一个维度，则 i_not_equal_j 在第三个维度增加一个即，(batch_size, batch_size, 1), 其他同理
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_and(indices_not_equal, labels_equal) #逻辑与

    return mask


def _get_anchor_negative_triple_mask(labels):
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_not(labels_equal)  #逻辑非

    return mask


def batch_hard_triplet_loss(embeddings,labels, margin,summary_index,squared=False):
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    # print(pairwise_dist)
    mask_anchor_positive = _get_anchor_positive_triple_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)
    valid_mask = tf.cast(tf.reduce_max(mask_anchor_positive, axis=1,keep_dims=True), tf.bool) #按行求最大值

    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
    hardest_anchor_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keep_dims=True)
    hardest_positive_pos = tf.argmax(anchor_positive_dist, axis=1)
    tf.summary.scalar("hardest_anchor_positive_dist"+str(summary_index), tf.reduce_mean(hardest_anchor_positive_dist))

    mask_anchor_negative = _get_anchor_negative_triple_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keep_dims=True)
    anchor_negative_list = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_anchor_negative_dist = tf.reduce_min(anchor_negative_list, axis=1, keep_dims=True)
    hardest_negative_pos = tf.argmin(anchor_negative_list, axis=1)
    tf.summary.scalar("hardest_anchor_negative_dist"+str(summary_index), tf.reduce_mean(hardest_anchor_negative_dist))

    positive_negative_pos = tf.transpose(tf.concat([[hardest_positive_pos], [hardest_negative_pos]], axis=0))
    hardest_positive_negative_dis = tf.transpose([tf.gather_nd(pairwise_dist, positive_negative_pos)])
    tf.summary.scalar("hardest_positive_negative_dist"+str(summary_index), tf.reduce_mean(hardest_positive_negative_dis))

    triplet_loss = tf.divide((tf.maximum(hardest_anchor_positive_dist - hardest_anchor_negative_dist + margin, 0.0) + tf.maximum(hardest_anchor_positive_dist - hardest_positive_negative_dis + margin, 0.0)),2)
    # triplet_loss = tf.maximum(hardest_anchor_positive_dist - hardest_anchor_negative_dist + margin, 0.0)
    #triplet_loss = tf.divide((tf.maximum(hardest_anchor_positive_dist - hardest_anchor_negative_dist + margin, 0.0) +tf.maximum(hardest_anchor_negative_dist -hardest_anchor_positive_dist  -0.1- margin, 0.0)+ tf.maximum(hardest_anchor_positive_dist - hardest_positive_negative_dis + margin, 0.0)+tf.maximum(hardest_positive_negative_dis -hardest_anchor_positive_dist  -0.1- margin, 0.0)),4)
    

    triplet_loss=tf.boolean_mask(triplet_loss,valid_mask)

    triplet_loss = tf.reduce_mean(triplet_loss)


    return triplet_loss
def cal_similarity(features):
    sim_mat = features @ K.transpose(features)   # (N,N)
    abs_mat = K.sqrt(K.sum(K.square(features), axis=1, keepdims=True))
    abs_mat = abs_mat @ K.transpose(abs_mat)
    sim_mat = sim_mat / abs_mat
    return sim_mat
def circle_loss_(features, labels, scale=32, margin=0.25):
    # labels: (N,cls) one-hot label
    # features: (N,k) feature embedding
    sim_mat = cal_similarity(features)
    label_mat = K.cast(labels @ K.transpose(labels), tf.bool)
    sim_mat_p = tf.gather_nd(sim_mat, tf.where(label_mat))
    sim_mat_n = tf.gather_nd(sim_mat, tf.where(~label_mat))

    alpha_p = K.relu(1 + margin - sim_mat_p)
    alpha_n = K.relu(sim_mat_n + margin)

    delta_p = 1 - margin
    delta_n = margin

    circle_loss_n = K.mean(K.exp(scale * alpha_n * (sim_mat_n - delta_n)))
    circle_loss_p = K.mean(K.exp(-scale * alpha_p * (sim_mat_p - delta_p)))

    loss = K.log(1 + circle_loss_n * circle_loss_p)
    # loss = tf.Print(loss, [circle_loss_n, circle_loss_p], message='  circle_loss_n & circle_loss_p')

    return loss
# def get_circle_loss(embeddings,labels,gamma, margin):
#     pairwise_dist = _pairwise_distances(embeddings, squared=False)
#     mask_anchor_positive = _get_anchor_positive_triple_mask(labels)
#     mask_anchor_positive = tf.to_float(mask_anchor_positive)
#     # valid_mask = tf.cast(tf.reduce_max(mask_anchor_positive, axis=1,keep_dims=True), tf.bool) #按行求最大值
#
#     anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
#
#     mask_anchor_negative = _get_anchor_negative_triple_mask(labels)
#     mask_anchor_negative = tf.to_float(mask_anchor_negative)
#     anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
# def circle_loss(sp, sn, gamma=256, margin=0.25):
#     delta_p = 1 - margin
#     delta_n = margin
#
#     Op = 1 + margin
#     On = - margin
#
#     alpha_p = tf.nn.relu(Op - sp)
#     alpha_n = tf.nn.relu(sn - On)
#
#     sumexp_p = tf.math.reduce_logsumexp(-gamma * alpha_p * (sp - delta_p))
#     sumexp_n = tf.math.reduce_logsumexp(gamma * alpha_n * (sn - delta_n))
#
#     return tf.nn.softplus(sumexp_p + sumexp_n)