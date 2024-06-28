import tensorflow as tf


def circle_loss(sp, sn, gamma, margin):
    delta_p = 1 - margin
    delta_n = margin

    Op = 1 + margin
    On = - margin

    alpha_p = tf.nn.relu(Op - sp)
    alpha_n = tf.nn.relu(sn - On)

    sumexp_p = tf.math.reduce_logsumexp(-gamma * alpha_p * (sp - delta_p))
    sumexp_n = tf.math.reduce_logsumexp(gamma * alpha_n * (sn - delta_n))

    return tf.nn.softplus(sumexp_p + sumexp_n)