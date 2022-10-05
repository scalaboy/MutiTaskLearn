# -*- coding: utf-8 -*-
'''
  @author: jeffzhengye
  @contact: yezheng@scuec.edu.cn
  @file: metrics.py
  @time: 2021/1/20 10:39
  @desc:
 '''

import tensorflow as tf
import numpy as np


class DropAUC(tf.keras.metrics.AUC):
    """
    reuse keras built-in AUC metric, but ignore invalid batch (all positive or negative)
    """
    def __init__(self, filter=False, *args, **kwargs):
        self._filter = filter
        super(DropAUC, self).__init__(*args, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        @tf.function
        def infer_weight(y):
            y_true1_count = tf.reduce_sum(y)
            y_true0_count = tf.reduce_sum(1 - y)
            total_pair_count = y_true1_count * y_true0_count
            if tf.less_equal(total_pair_count, 0):  # all positive or all negative
                # tf.print('invalid batch')
                return tf.zeros_like(y)
            else:
                return tf.ones_like(y)

        def _get_auc_op(yt, yp):
            if tf.greater(tf.reduce_sum(yt), 0):
                return super(DropAUC, self).update_state(yt, yp, sample_weight=infer_weight(yt))
            else:
                return tf.no_op()

        if self._filter:
            y_ctr_true = y_true[:, 1]
            y_cvr_true = y_true[:, 0:1]
            indices = tf.equal(y_ctr_true, 1)

            y_cvr_pred = y_pred[indices]
            y_cvr_true = y_cvr_true[indices]
            return _get_auc_op(y_cvr_true, y_cvr_pred)

            # if tf.greater(tf.reduce_sum(y_cvr_true), 0):
            #     return super(DropAUC, self).update_state(y_cvr_true, y_cvr_pred, sample_weight=infer_weight(y_cvr_true))
            # else:
            #     return tf.no_op

        return _get_auc_op(y_true, y_pred)


class DropAUCX(tf.keras.metrics.Metric):
    """
    reimplementation of x-deeplearning auc calculation.
    """
    def __init__(self, output_name='auc', softmax_out=False, filter=False, **kwargs):
        super(DropAUCX, self).__init__(**kwargs)
        # self._name = output_name
        self.output_name = output_name
        self.auc_value = self.add_weight(name='auc_value', initializer='zeros')
        self.valid_batch_count = self.add_weight(name='valid_batch_count', initializer='zeros')
        self.invalid_batch_count = self.add_weight(name='invalid_batch_count', initializer='zeros')
        self._filter = filter
        self.softmax_output = softmax_out

    @tf.function
    def _get_xy(self, y_true, y_pred):
        if self.softmax_output:
            y_pred = y_pred[:, 1]

        if self._filter:
            y_ctr_true = y_true[:, 1]
            y_cvr_true = y_true[:, 0:1]
            indices = tf.equal(y_ctr_true, 1)
            # if tf.reduce_any(tf.less(y_true[:, 1], y_true[:, 0])) :
            #     tf.print(y_true)

            y_cvr_pred = y_pred[indices]
            y_cvr_true = y_cvr_true[indices]
            preds = tf.reshape(y_cvr_pred, [-1])
            labels = tf.reshape(y_cvr_true, [-1])
        else:
            # print(self._name, len(y_pred), type(y_pred[0]))
            preds = tf.reshape(y_pred, [-1])
            labels = tf.reshape(y_true, [-1])
        return preds, labels

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        preds, labels = self._get_xy(y_true, y_pred)
        if tf.reduce_sum(labels) == 0:
            if self.name == 'ctr':
                tf.print('%s invalid batch' % self.name, self.invalid_batch_count)
            self.invalid_batch_count.assign_add(1)
        elif tf.reduce_sum(tf.shape(labels)) == 0:
            # tf.print('%s invalid batch due to cvr all 0' % self.name, self.invalid_batch_count)
            self.invalid_batch_count.assign_add(1)
        else:
            sorted_id = tf.argsort(preds, axis=-1, direction='ASCENDING', stable=True)
            record = tf.gather(labels, sorted_id)
            record = tf.cast(record, tf.float32)
            tp = tf.cumsum(record)
            tp1 = tf.concat([tf.constant([0.]), tp], axis=-1)[:-1]
            tp2_plus_tp1 = tp + tp1
            fp = tf.cumsum((1 - record))
            fp = tf.concat([tf.constant([0.]), fp], axis=-1)
            fp2_minus_fp1 = fp[1:] - fp[0:-1]

            threshold = tf.cast(tf.shape(preds)[0], tf.float32) - 1e-3
            if tf.greater(tp[-1], threshold) or tf.greater(fp[-1], threshold):
                # tf.print('%s invalid batch-inner' % self.name, self.invalid_batch_count)
                self.invalid_batch_count.assign_add(1)
                # tf.print(tp[-1], fp[-1])
                pass
            elif tf.greater(tp[-1] * fp[-1], 0.):
                auc = tf.reduce_sum(tp2_plus_tp1 * fp2_minus_fp1)
                res = (1.0 - auc / (2.0 * tp[-1] * fp[-1]))
                self.auc_value.assign_add(res)
                self.valid_batch_count.assign_add(1)
                # tf.print('%s_batch_auc=' % self.name, res)
            # if (self.invalid_batch_count + self.valid_batch_count) % 500 == 0:
            #     tf.print("\n", self.output_name, 'invalid, valid:', self.invalid_batch_count, self.valid_batch_count, self.result())

    @tf.function
    def result(self):
        if tf.greater(self.valid_batch_count, 0.):
            return self.auc_value / self.valid_batch_count
        else:
            return tf.constant(0.)

    def reset_states(self):
        self.auc_value.assign(0)
        self.valid_batch_count.assign(0)
        self.invalid_batch_count.assign(0)


class ADropAUCX(DropAUCX):
    def __init__(self, *args, **kwargs):
        super(ADropAUCX, self).__init__(*args, **kwargs)

    def _get_xy(self, y_true, y_pred):
        if self._filter:
            y_ctr_true = y_true[:, 1]
            y_cvr_true = y_true[:, 0:1]
            indices = tf.equal(y_ctr_true, 1)

            y_cvr_pred = y_pred[indices][:, 0:1]
            y_cvr_true = y_cvr_true[indices]
            preds = tf.reshape(y_cvr_pred, [-1])
            labels = tf.reshape(y_cvr_true, [-1])
        else:
            # print(self._name, len(y_pred), type(y_pred[0]))
            preds = tf.reshape(y_pred, [-1])
            labels = tf.reshape(y_true, [-1])
        return preds, labels

# print(tf.__version__)
# m = DropAUCX()
# # m = tf.metrics.AUC()
# # m.update_state([0, 1, 0, 0], [0, 1, 1, 0])
# print('Intermediate result:', float(m.result()))
#
# m.update_state([1, 0, 1, 1], [1, 0, 1, 1])
# print('Final result:', float(m.result()))
