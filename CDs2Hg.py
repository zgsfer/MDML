import tensorflow as tf
import tensorflow.contrib.slim as slim
def pairwise_mahalanobis(X, means, log_vars,sbt,class_num):
    """
    Computes pairwise squared Mahalanobis distances between X (data points) and a set of distributions
    :param X: [N, F] where N is the batch size and F is the feature dimension
    :param means: [C, F] C is the number of classes
    :param log_vars: [C, F] C is the number of classes, we assume a diagonal covariance matrix
    :return: pairwise squared Mahalanobis distances... [N, C, F] matrix
    i.e., M_ij = (x_i-means_j)\top * inv_cov_j * (x_i - means_j)
    """
    # sz_batch = X.shape[0]
    # sz_batch = tf.convert_to_tensor(sz_batch)
    nb_classes = means.shape[0]

    new_X = tf.expand_dims(X, dim=1)  # [N, 1, F]
    new_X = tf.broadcast_to(new_X,[sbt, class_num, 1792])  # [N, C, F]

    # means = float(means)

    new_means = tf.expand_dims(means, dim=0)  # [1, C, F]
    new_means = tf.broadcast_to(new_means,[sbt, class_num, 1792])  # [N, C, F]

    # pairwise distances
    diff = new_X - new_means

    # convert log_var to covariance
    covs = tf.expand_dims(tf.exp(log_vars), dim=0)  # [1, C, F]

    # the squared Mahalanobis distances
    M = tf.reduce_sum(tf.div(tf.math.pow(diff,2), covs),axis=-1)  # [N, C]

    return M
def CDs2Hg(embds,Y,sbt,class_num, tau=32.0, alpha=0.9):
    with tf.variable_scope('CDHG', reuse=False):
        means = tf.get_variable(name='means', shape=[ class_num,embds.get_shape().as_list()[-1]], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(), regularizer=slim.l2_regularizer(5e-6), trainable=True)
        vars = tf.get_variable(name='var', shape=[ class_num,embds.get_shape().as_list()[-1]], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer(), regularizer=slim.l2_regularizer(5e-6),
                                trainable=True)
        var = tf.nn.relu6(vars)
        X =tf.nn.l2_normalize(embds)
        mu = tf.nn.l2_normalize(means)
        distance = pairwise_mahalanobis(X, mu, var,sbt,class_num)
        n = -1 * tau * distance
        mat = tf.nn.softmax(n, dim=1)
        loss = tf.reduce_sum(mat * Y, axis=1)
        non_zero = (loss != 0)
        # non_zero = tf.to_int32(non_zero)
        loss = -tf.log(loss*non_zero)


        return tf.reduce_mean(loss)