# -*- coding: utf-8 -*-
'''
  @author: jeffzhengye
  @contact: yezheng@scuec.edu.cn
  @file: layers.py
  @time: 2021/2/2 21:52
  @desc:
 '''

from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import backend as K
from ..metrics.metrics import DropAUC, DropAUCX, ADropAUCX


class LinearCombination(Layer):
    def __init__(self, *args, **kwargs):
        super(LinearCombination, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', shape=(1,), initializer='ones', trainable=True,
                                     constraint=tf.keras.constraints.MinMaxNorm(
                                         min_value=0.0, max_value=1.0, rate=1.0))

    def call(self, ctnocvr_pred, ctcvr_pred):
        # x is a list of two tensors with shape=(batch_size, H, T)
        self.add_metric(self.alpha, name='alpha')
        return self.alpha * ctcvr_pred + (1 - self.alpha) * (1 - ctnocvr_pred)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class GateLayer(Layer):
    def __init__(self, output_dim, *args, **kwargs):
        super(GateLayer, self).__init__(*args, **kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.uf = tf.keras.layers.Dense(self.output_dim, use_bias=False)
        self.wf = tf.keras.layers.Dense(self.output_dim, use_bias=True)

    def call(self, x, h):
        return K.sigmoid(self.uf(h) + self.wf(x))


class CtnocvEndpoint(Layer):
    def __init__(self, name=None):
        super(CtnocvEndpoint, self).__init__(name=name)
        self.loss_ctr = tf.keras.losses.BinaryCrossentropy(name='ctr-loss')
        self.loss_cvr = tf.keras.losses.BinaryCrossentropy(name='cvr-loss')
        self.loss_ctcvr = tf.keras.losses.BinaryCrossentropy(name='ctcvr-loss')
        self.loss_ctnocvr = tf.keras.losses.BinaryCrossentropy(name='ctnocvr-loss')
        self.loss_ctcvr_linear = tf.keras.losses.BinaryCrossentropy(name='ctcvr_linear-loss')

        self.auc_ctr = DropAUCX(name='ctr-auc')
        self.auc_cvr = DropAUCX(name='cvr-auc', filter=True)
        self.auc_ctcvr = DropAUCX(name='ctcvr-auc')
        self.auc_ctcvr_nocvr = DropAUCX(name='ctcvr-nocvr-auc')
        self.auc_ctr_origin = tf.keras.metrics.AUC(name='ctr_origin-auc')
        self.linear_combine = LinearCombination(name="ctcvr_linear_pred1")

    def call(self, ctr_pred, cvr_pred, ct_nocvr_pred, ctr_targets, cvr_targets, sample_weights=None):
        ctcvr_pred = ctr_pred * cvr_pred

        ctcvr_targets = cvr_targets * ctr_targets

        ctcvr_linear_pred = self.linear_combine(ct_nocvr_pred, ctcvr_pred)

        ctr_loss = self.loss_ctr(ctr_targets, ctr_pred)
        ctcvr_loss = self.loss_ctcvr(ctcvr_targets, ctcvr_pred)

        ct_nocvr_targets = tf.maximum(ctr_targets - cvr_targets, 0)
        ct_nocvr_loss = self.loss_ctnocvr(ct_nocvr_targets, ct_nocvr_pred)

        ctcvr_linear_loss = self.loss_ctcvr_linear(ctcvr_targets, ctcvr_linear_pred)
        self.add_loss([ctr_loss, ctcvr_loss, 0. * ct_nocvr_loss, 0.2 * ctcvr_linear_loss])

        self.add_metric(ctr_loss, name='ctr_loss')
        self.add_metric(ctcvr_loss, name='ctcvr_loss')
        self.add_metric(ct_nocvr_loss, 'ct_nocvr_loss')

        cvr_auc_targets = tf.concat([ctcvr_targets, ctr_targets], axis=-1)
        self.add_metric(self.auc_ctr(ctr_targets, ctr_pred), name='ctr_auc')
        self.add_metric(self.auc_cvr(cvr_auc_targets, cvr_pred), name='cvr_auc')
        self.add_metric(self.auc_ctcvr(ctcvr_targets, ctcvr_pred), name='ctcvr_auc')
        self.add_metric(self.auc_ctr_origin(ctr_targets, ctr_pred), name='ctr_origin_auc')
        self.add_metric(self.auc_ctcvr_nocvr(ctcvr_targets, ctcvr_linear_pred), name='ctcvr_nocvr_auc')

        # Return the inference-time prediction tensor (for `.predict()`).
        return ctcvr_pred


class EsmmMLPLayer(Layer):
    def __init__(self, output_name, dense_shapes=[360, 200, 80], dropouts=[], activation='relu', batch_norm=False,
                 *args, **kwargs):
        super(EsmmMLPLayer, self).__init__(*args, **kwargs)
        self.output_name = output_name
        self.dense_shapes = dense_shapes
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropouts = dropouts

    def build(self, input_shape):
        # self.fcs = [Dense(out_dim, activation=self.activation,
        #                   bias_initializer=tf.keras.initializers.Constant(0.1),
        #                   ) for out_dim in self.dense_shapes]
        # self.bns = [] if not self.batch_norm else [layers.BatchNormalization() for i in range(len(self.dense_shapes))]
        # self.ctr_out = layers.Dense(1, activation='sigmoid', name=self.output_name)
        # self.dropout_layers = [layers.Dropout(rate) for rate in self.dropouts]

        self.fcs = []
        for out_dim in self.dense_shapes:
            initializer = tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', distribution='uniform')
            ds = Dense(out_dim, activation=self.activation,
                       kernel_initializer=initializer,
                       bias_initializer=tf.keras.initializers.Constant(0.1))
            self.fcs.append(ds)

        self.bns = [] if not self.batch_norm else [layers.BatchNormalization() for i in range(len(self.dense_shapes))]

        initializer = tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', distribution='uniform')
        bias_initializer = tf.keras.initializers.Constant(0.1)
        self.ctr_out = layers.Dense(1, activation='sigmoid', name=self.output_name,
                                    kernel_initializer=initializer, bias_initializer=bias_initializer)
        self.dropout_layers = [layers.Dropout(rate) for rate in self.dropouts]


    def call(self, x, training=None):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.dropout_layers):
                x = self.dropout_layers[i](x, training=training)
            if self.batch_norm:
                x = self.bns[i](x, training=training)
        output = self.ctr_out(x)
        self.add_metric(tf.reduce_mean(output), name=self.name + '_sigmoid')
        return output


class EsmmMLPSoftmaxLayer(Layer):
    def __init__(self, output_name, dense_shapes=[360, 200, 80], dropouts=[], activation='relu', batch_norm=False,
                 *args, **kwargs):
        super(EsmmMLPSoftmaxLayer, self).__init__(*args, **kwargs)
        self.output_name = output_name
        self.dense_shapes = dense_shapes
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropouts = dropouts

    def build(self, input_shape):
        self.fcs = []
        for out_dim in self.dense_shapes:
            initializer = tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', distribution='uniform')
            ds = Dense(out_dim, activation=self.activation,
                       kernel_initializer=initializer,
                       bias_initializer=tf.keras.initializers.Constant(0.1))
            self.fcs.append(ds)

        self.bns = [] if not self.batch_norm else [layers.BatchNormalization() for i in range(len(self.dense_shapes))]

        initializer = tf.keras.initializers.VarianceScaling(scale=1., mode='fan_in', distribution='uniform')
        bias_initializer = tf.keras.initializers.Constant(0.1)
        self.ctr_out = layers.Dense(2, activation='softmax', name=self.output_name,
                                    kernel_initializer=initializer, bias_initializer=bias_initializer)
        self.dropout_layers = [layers.Dropout(rate) for rate in self.dropouts]

    def build1(self, input_shape):
        self.fcs = [Dense(out_dim, activation=self.activation) for out_dim in self.dense_shapes]
        self.bns = [] if not self.batch_norm else [layers.BatchNormalization() for i in range(len(self.dense_shapes))]
        self.ctr_out = layers.Dense(2, activation='softmax', name=self.output_name)
        self.dropout_layers = [layers.Dropout(rate) for rate in self.dropouts]

    def call(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.dropout_layers):
                x = self.dropout_layers[i](x)
            if self.batch_norm:
                x = self.bns[i](x)
        output = self.ctr_out(x)
        self.add_metric(tf.reduce_mean(output[:, 0]), name=self.name + '_sigmoid0')
        self.add_metric(tf.reduce_mean(output[:, 1]), name=self.name + 'sigmoid1')
        return output
