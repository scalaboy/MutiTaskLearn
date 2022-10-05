# -*- coding: utf-8 -*-
'''
  @author: jeffzhengye
  @contact: yezheng@scuec.edu.cn
  @file: loss.py
  @time: 2021/1/16 21:39
  @desc:
 '''
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.ops import math_ops
from tensorflow.keras.losses import binary_crossentropy, hinge
from tensorflow_addons.losses import sigmoid_focal_crossentropy, SigmoidFocalCrossEntropy
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike

binary_loss_weight = 0.000


def fake_loss(y_true, y_pred, from_logits=False):
    y_ctr_true = y_true[:, 1]
    y_cvr_true = y_true[:, 0:1]
    indices = tf.equal(y_ctr_true, 1)

    y_cvr_pred = y_pred[indices]
    y_cvr_true = y_cvr_true[indices]

    if True or tf.greater(tf.reduce_sum(y_cvr_true), 0):
        return direct_auc_loss(y_cvr_true, y_cvr_pred, from_logits=from_logits)
    else:
        return binary_loss_weight * binary_crossentropy(y_cvr_true, y_cvr_pred, from_logits=from_logits)

    # return direct_auc_loss(y_cvr_true, y_cvr_pred, from_logits=from_logits) + \
    #        binary_loss_weight * binary_crossentropy(
    #     y_cvr_true,
    #     y_cvr_pred,
    #     from_logits=from_logits)


def binary_crossentropy_cvr(y_true, y_pred, from_logits=False):
    y_ctr_true = y_true[:, 1]
    y_cvr_true = y_true[:, 0:1]
    indices = tf.equal(y_ctr_true, 1)

    y_cvr_pred = y_pred[indices]
    y_cvr_true = y_cvr_true[indices]

    loss= binary_crossentropy(y_cvr_true, y_cvr_pred, from_logits=from_logits)
    return loss


def ctnocvr_loss(y_true, y_pred, from_logits=False):
    y_ctr_true = y_true[:, 1]
    y_cvr_true = y_true[:, 0:1]
    indices = tf.equal(y_ctr_true, 1)

    y_cvr_pred = y_pred[indices]
    y_nocvr_pred = y_cvr_pred[:, 1:]
    y_cvr_true = y_cvr_true[indices]

    return binary_crossentropy(y_cvr_true, y_nocvr_pred, from_logits=from_logits)


def triplet_loss(y_true, y_pred, m=0.2, from_logits=False):
    if tf.greater(tf.reduce_sum(y_true), 0):
        indices_1 = tf.equal(y_true, 1)
        indices_0 = tf.equal(y_true, 0)
        triplet = tf.maximum(tf.reduce_mean(y_pred[indices_0]) - tf.reduce_mean(y_pred[indices_1]) + m, 0.)
        return binary_crossentropy(y_true, y_pred, from_logits=from_logits) * 0. + triplet
    else:
        return binary_crossentropy(y_true, y_pred, from_logits=from_logits) * 0.


def direct_auc_loss(y_true, y_pred, m=0.1, p=2, from_logits=False, pair_average=False):
    def matrix_build(y):
        # mm = tf.matmul(y, tf.transpose(tf.ones_like(y))) - tf.matmul(tf.ones_like(y), tf.transpose(y))
        mm = tf.repeat(y, [tf.shape(y)[0]], axis=-1) - tf.repeat(tf.transpose(y), [tf.shape(y)[0]], axis=0)
        return mm

    y_true1_count = tf.reduce_sum(y_true)
    y_true0_count = tf.reduce_sum(1 - y_true)
    total_pair_count = tf.cast(y_true1_count * y_true0_count, tf.float32)
    if tf.greater(total_pair_count, 0.):
        y_true_m = tf.maximum(matrix_build(y_true), 0)
        y_true_m = tf.cast(y_true_m, tf.float32)
        y_pred_m = matrix_build(y_pred)
        if True:
            y_pred_m = tf.exp(- p * y_pred_m)
        else:
            y_pred_m = tf.maximum(-y_pred_m + m, 0.)
            y_pred_m = tf.pow(y_pred_m, p)
        if pair_average:
            auc_loss = tf.reduce_sum(y_pred_m * y_true_m) / total_pair_count
        else:
            auc_loss = tf.reduce_mean(y_pred_m * y_true_m, axis=-1)
        return auc_loss
    else:
        return 0.
        # binary_loss = binary_crossentropy(y_true, y_pred, from_logits=from_logits)
        # return binary_loss_weight * binary_loss


def weighted_binary_crossentropy(class_weight=[1., 10.], all_negative_weight=1., hinge_w=0.):
    class_weight = ops.convert_to_tensor_v2(class_weight, dtype=K.floatx())
    all_negative_weight = tf.constant(all_negative_weight, dtype=K.floatx())

    def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
        # y_pred = ops.convert_to_tensor_v2(y_pred)
        cur_weight = tf.gather(class_weight, y_true)
        hinge_loss = tf.constant(0., dtype=K.floatx())
        if K.sum(y_true) <= 0:
            cur_weight = all_negative_weight * cur_weight
        if K.sum(y_true) > 0:
            indices = K.equal(y_true, 1)
            y_true_true = y_true[indices]
            y_pred_true = y_pred[indices]
            hinge_loss += hinge_w * hinge(y_true_true, y_pred_true)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        label_smoothing = ops.convert_to_tensor_v2(label_smoothing, dtype=K.floatx())

        def _smooth_labels():
            return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

        y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels, lambda: y_true)
        loss = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

        return K.mean(loss * cur_weight, axis=-1) + hinge_loss

    return binary_crossentropy


class DSigmoidFocalCrossEntropy(SigmoidFocalCrossEntropy):
    def __init__(
            self,
            sample_filter=False,
            from_logits: bool = False,
            alpha: FloatTensorLike = 0.25,
            gamma: FloatTensorLike = 2.0,
            reduction: str = tf.keras.losses.Reduction.NONE,
            name: str = "sigmoid_focal_crossentropy",
    ):
        # print(from_logits, gamma)
        self.sample_filter = sample_filter
        super(DSigmoidFocalCrossEntropy, self).__init__(from_logits=from_logits, alpha=alpha, gamma=gamma,
                                                        reduction=reduction)
        # super().__init__(
        #     sigmoid_focal_crossentropy,
        # name=name,
        # reduction=reduction,
        # from_logits=from_logits,
        # alpha=alpha,
        # gamma=gamma,
        # )

    def call(self, y_true, y_pred):
        if self.sample_filter:
            y_ctr_true = y_true[:, 1]
            y_cvr_true = y_true[:, 0:1]
            indices = tf.equal(y_ctr_true, 1)
            # tf.print('y_ctr_true', tf.reduce_sum(y_ctr_true))
            y_cvr_pred = y_pred[indices]
            y_cvr_true = y_cvr_true[indices]
            # tf.print('y_cvr_true', tf.reduce_sum(y_cvr_true))
            # if True or tf.greater(tf.reduce_sum(y_cvr_true), 0):
                # loss = super(DSigmoidFocalCrossEntropy, self).call(y_cvr_true, y_pred)
            loss = sigmoid_focal_crossentropy1(y_cvr_true, y_cvr_pred, **self._fn_kwargs)
            tf.print('loss', tf.shape(loss))
            return loss
            # else:
            #     return 0.
        # if True or tf.greater(tf.reduce_sum(y_true), 0):
        loss = sigmoid_focal_crossentropy1(y_true, y_pred, **self._fn_kwargs)
        # loss = super(DSigmoidFocalCrossEntropy, self).call(y_true, y_pred)
        tf.print('ctr or ctcvr loss', tf.shape(loss))
        return loss
        # else:
        #     return 0.


@tf.function
def sigmoid_focal_crossentropy1(
        y_true: TensorLike,
        y_pred: TensorLike,
        alpha: FloatTensorLike = 0.25,
        gamma: FloatTensorLike = 2.0,
        from_logits: bool = False,
) -> tf.Tensor:
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero")

    # y_pred = tf.convert_to_tensor(y_pred)
    # y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)

    y_true = tf.cast(y_true, y_pred.dtype)
    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    # v = alpha_factor * modulating_factor * ce
    # tf.print('v', tf.shape(v))
    # v1 = tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)
    # tf.print('v1', tf.shape(v1))
    return tf.reduce_mean(alpha_factor * modulating_factor * ce)
