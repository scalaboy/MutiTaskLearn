# -*- coding: utf-8 -*-
'''
  @author: jeffzhengye
  @contact: yezheng@scuec.edu.cn
  @file: train.py
  @time: 2021/2/17 17:44
  @desc:
 '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import copy
import tensorflow as tf

# whether to use gpu
tf.config.set_soft_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print('cannot set memory growth, use default')
        print(e)

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
from tensorflow.keras.metrics import AUC
from mctr.data.ali_ccp_datasets import get_dataset, get_dataset_ctnocvr,get_dataset_moebase
from mctr.metrics import DropAUCX, DropAUC
from mctr.modeling.layers import EsmmMLPLayer
from mctr.callbacks import BatchModelCheckpoint, CBatchModelCheckpoint
from mctr.modeling.mmoe import MMoE
from mctr.callbacks import BatchModelCheckpoint
from mctr.utils.args_utils import ArgsParser, load_config, merge_config
from mctr.utils.tf_utils import get_lr, get_layers, get_optimizer, get_loss, get_callbacks_from_config
from mctr.modeling.inputs_builder import build_din, build_inputs

FLAGS = ArgsParser().parse_args()
config = load_config(FLAGS.config)
merge_config(FLAGS.opt)

batch_size = config['Train']['dataset']['batch_size']
shuffle = config['Train']['dataset']['shuffle']

# model paras
model_name = config['Models']['active']
embed_size = config['Models'][model_name]['embed_size']
min_by = config['Models'][model_name]['min_by']
feature_dim = config['Models'][model_name]['feature_dim']  # used to hash features
share_embedding = config['Models'][model_name]['share_embedding']  # bool if share embedding.
mlp_ctr = config['Models'][model_name]['mlp_ctr']
mlp_cvr = config['Models'][model_name]['mlp_cvr']
activation = get_layers(config['Models'][model_name]['activation'])
batch_norm = config['Models'][model_name]['batch_norm']
dropouts = config['Models'][model_name]['dropouts']

# 'esmm_mmoe_add_loss' or 'esmm_mmoe'
if model_name in ['esmm_mmoe_add_loss', 'esmm_mmoe','mmoe_add_loss','mmoe_base']:
    units = config['Models'][model_name]['units']
    num_experts = config['Models'][model_name]['num_experts']
    num_tasks = config['Models'][model_name]['num_tasks']

save_freq = 'epoch'
# save_freq = 10

# Global
lr = get_lr(config)
epochs = config['Global']['epochs']
opt = get_optimizer(config)
name_tag = config['Global']['additional_tag']
# loss paras
loss_key = config['Loss'][model_name]['key']  # multi loss, each has an output name
loss_paras = config['Loss'][model_name]['paras']
# all loss function should be the same, use 'sample_filter': True for cvr in order to filter out ctr_label=0
loss_paras_dict = [dict(loss_paras, sample_filter=True) if k == 'cvr_output' else loss_paras for k in loss_key]
print(loss_paras_dict)
loss_value = [get_loss(l, paras=p) for l, p in zip(config['Loss'][model_name]['value'], loss_paras_dict)]  # loss type
weights = config['Loss'][model_name]['weights']
loss_weights = dict(zip(loss_key, weights))
loss_dict = dict(zip(loss_key, loss_value))


def get_prefix(paras=None):
    prefix_formatter = "{model}_lr{lr}_bn{bn}_shuffle{shuffle}_dp{dropout}_{loss_name}_{loss_weight}_share{share_embedding}_e{embedding_size}_{activation}_{optimizer}"
    if paras:
        f_dict = paras
    else:
        f_dict = {
            'model': model_name +  '-'+str(time.time().as_integer_ratio()[0])  + '-'.join(
                [str(d) for d in mlp_ctr]),
            'lr': lr,
            'bn': int(batch_norm),
            'dropout': dropouts[0] if dropouts else '0',
            'loss_name': type(loss_dict['ctr_output']).__name__,
            'loss_weight': '-'.join([str(w) for w in weights]),
            'share_embedding': int(share_embedding),
            'embedding_size': embed_size,
            'activation': activation if isinstance(activation, str) else type(activation).__name__,
            'optimizer': type(opt).__name__,
            'shuffle': int(shuffle)
        }
    if name_tag:
        prefix_formatter += "_" + name_tag
    return prefix_formatter.format_map(f_dict) + "_{epoch}"


prefix = get_prefix()

checkpoint_dir = os.path.join(config['Global']['checkpoint'], prefix)
result_dir = config['Global']['results_dir']
tensor_board_dir = os.path.join(result_dir, prefix + "_tensor_board")
csv_logger_dir = os.path.join(result_dir, prefix + '_csv_log.log')

inputs = build_inputs()
din_ctr = build_din(inputs, embed_size, feature_dim, min_by)

din = din_cvr = din_ctr
if not share_embedding:
    din_cvr = build_din(inputs)

# call back function
checkpoint = CBatchModelCheckpoint(checkpoint_dir, monitor='val_loss', verbose=1, save_best_only=False, mode='min',
                                   save_freq=save_freq)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=1, min_lr=0.0001, verbose=1)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1, mode='auto')
csv_logger = CSVLogger(csv_logger_dir)
callbacks = [reduce_lr, checkpoint, earlystopping, csv_logger]


def get_mmoe():
    mmoe_layers = MMoE(units=units, num_experts=num_experts, num_tasks=num_tasks)(din_ctr)
    ctr_pred = EsmmMLPLayer(name='ctr', output_name='ctr_output', dense_shapes=mlp_ctr, batch_norm=batch_norm,
                            dropouts=dropouts)(mmoe_layers[0])
    cvr_pred = EsmmMLPLayer(name='cvr', output_name='ctcvr_pred', dense_shapes=mlp_cvr)(mmoe_layers[1])
    #ctcvr_pred = layers.Lambda(lambda x: x[0] * x[1], name='ctcvr')([ctr_pred, cvr_pred])
    return ctr_pred,  cvr_pred


def get_esmm():
    ctr_pred = EsmmMLPLayer(name='ctr', output_name='ctr_output', dense_shapes=mlp_ctr, batch_norm=batch_norm,
                            dropouts=dropouts)(din_ctr)
    cvr_pred = EsmmMLPLayer(name='cvr', output_name='cvr_output', dense_shapes=mlp_cvr)(din_ctr)
    ctcvr_pred = layers.Lambda(lambda x: x[0] * x[1], name='ctcvr')([ctr_pred, cvr_pred])
    return ctr_pred, ctcvr_pred, cvr_pred


def get_mmoe_add_loss():
    mmoe_layers = MMoE(units=units, num_experts=num_experts, num_tasks=num_tasks)(din_ctr)
    ctr_pred = EsmmMLPLayer(name='ctr', output_name='ctr_output', dense_shapes=mlp_ctr, batch_norm=batch_norm,
                            dropouts=dropouts)(mmoe_layers[0])
    cvr_pred = EsmmMLPLayer(name='cvr', output_name='cvr_output', dense_shapes=mlp_cvr)(mmoe_layers[1])
    ctcvr_pred = layers.Lambda(lambda x: x[0] * x[1], name='ctcvr')([ctr_pred, cvr_pred])
    ct_nocvr_pred = EsmmMLPLayer(name='ct_nocvr', output_name='ct_nocvr_output', dense_shapes=mlp_cvr)(mmoe_layers[2])

    return ctr_pred, ctcvr_pred, cvr_pred, ct_nocvr_pred


def build_model(name):
    m_dict = {
        'esmm': get_esmm,
        'esmm_mmoe': get_mmoe,
        'esmm_mmoe_add_loss': get_mmoe_add_loss,
    }
    return m_dict[name]()


# model train
def train_graph1():
    model = Model(inputs=inputs,
                  outputs={'ctr_output': ctr_pred, 'ctcvr_pred': ctcvr_pred, 'cvr_output': cvr_pred})
    model.summary()
    model.compile(optimizer=opt, loss=loss_dict, loss_weights=loss_weights,
                  metrics={
                      'ctr_output': [DropAUCX(name='dauc', output_name='ctr'), DropAUC(name='tauc'), AUC()],
                      'ctcvr_pred': [DropAUCX(name='dauc', output_name='ctcvr'), DropAUC(name='tauc'), AUC()],
                      'cvr_output': [DropAUCX(name='dauc', output_name='cvr', filter=True),
                                     DropAUC(name='tauc', filter=True)]
                  },
                  )
    print('start fitting', prefix)
    train_dataset = get_dataset(batch_size=batch_size, train=True, config=config)
    valid_dataset = get_dataset(batch_size=batch_size, train=False, config=config)
    batch_eval_callback = BatchModelCheckpoint(valid_dataset)
    callbacks.append(batch_eval_callback)
    model.fit(train_dataset, validation_data=valid_dataset, callbacks=callbacks,
              verbose=1, workers=30, epochs=epochs, shuffle=True)


def train_graph():
    from mctr.modeling.custom_keras_model import BatchEvalModel
    outputs_list = build_model(model_name)
    outputs = dict(zip(loss_key, outputs_list))
    model = BatchEvalModel(batch_interval=1000, eval_batchs=[7000, 1300],
                           inputs=inputs, outputs=outputs
                           )
    model.summary()
    model.compile(optimizer=opt,
                  loss=loss_dict,
                  loss_weights=loss_weights,
                  metrics={
                      'ctr_output': [DropAUCX(name='dauc', output_name='ctr'), DropAUC(name='tauc'), AUC()],
                      'ctcvr_pred': [DropAUCX(name='dauc', output_name='ctcvr'), DropAUC(name='tauc'), AUC()],
                      'cvr_output': [DropAUCX(name='dauc', output_name='cvr', filter=True),
                                     DropAUC(name='tauc', filter=True)]
                  },
                  # steps_per_execution=10
                  )
    print('start fitting', prefix)
    if model_name == 'esmm_mmoe_add_loss':
        train_dataset = get_dataset_ctnocvr(batch_size=batch_size, train=True, config=config)
        valid_dataset = get_dataset_ctnocvr(batch_size=batch_size, train=False, config=config)
    else:
        train_dataset = get_dataset(batch_size=batch_size, train=True, config=config)
        valid_dataset = get_dataset(batch_size=batch_size, train=False, config=config)
    model.fit(train_dataset, validation_data=valid_dataset, callbacks=callbacks,
              # steps_per_epoch=8461, validation_steps=8604,
              verbose=1, workers=30, epochs=epochs, shuffle=shuffle)


def train_mmoe_add_loss():
    from mctr.modeling.custom_keras_model import BatchEvalModel
    ctr_pred, cvr_pred, ctcvr_pred, ct_nocvr_pred = get_mmoe_add_loss()
    model = BatchEvalModel(inputs=inputs,
                           outputs={'ctr_output': ctr_pred, 'ctcvr_pred': ctcvr_pred, 'cvr_output': cvr_pred,
                                    'ct_nocvr_pred': ct_nocvr_pred})

    model.summary()
    model.compile(optimizer=opt,
                  loss=loss_dict,
                  loss_weights=loss_weights,
                  metrics={
                      'ctr_output': [DropAUCX(name='dauc', output_name='ctr'), DropAUC(name='tauc'), AUC()],
                      'ctcvr_pred': [DropAUCX(name='dauc', output_name='ctcvr'), DropAUC(name='tauc'), AUC()],
                      'cvr_output': [DropAUCX(name='dauc', output_name='cvr', filter=True),
                                     DropAUC(name='tauc', filter=True)]
                  },
                  # steps_per_execution=10
                  )
    print('start fitting', prefix)
    train_dataset = get_dataset_ctnocvr(batch_size=batch_size, train=True, config=config)
    valid_dataset = get_dataset_ctnocvr(batch_size=batch_size, train=False, config=config)
    model.fit(train_dataset, validation_data=valid_dataset, callbacks=callbacks,
              # steps_per_epoch=8461, validation_steps=8604,
              verbose=1, workers=30, epochs=epochs, shuffle=shuffle)

def train_mmoe_base():
    from mctr.modeling.custom_keras_model import BatchEvalModel
    ctr_pred, ctcvr_pred = get_mmoe()
    model = BatchEvalModel(inputs=inputs,
                           outputs={'ctr_output': ctr_pred, 'ctcvr_pred': ctcvr_pred
                                    })

    model.summary()
    model.compile(optimizer=opt,
                  loss=loss_dict,
                  loss_weights=loss_weights,
                  metrics={
                      'ctr_output': [DropAUCX(name='dauc', output_name='ctr'), DropAUC(name='tauc'), AUC()],
                      'ctcvr_pred': [DropAUCX(name='dauc', output_name='cvr', filter=True),
                                     DropAUC(name='tauc', filter=True)]
                  },
                  # steps_per_execution=10
                  )
    print('start fitting', prefix)
    train_dataset = get_dataset_moebase(batch_size=batch_size, train=True, config=config)
    valid_dataset = get_dataset_moebase(batch_size=batch_size, train=False, config=config)
    model.fit(train_dataset, validation_data=valid_dataset, callbacks=callbacks,
              # steps_per_epoch=8461, validation_steps=8604,
              verbose=1, workers=30, epochs=epochs, shuffle=shuffle)





#train_graph()
if __name__ == '__main__':
    train_mmoe_base()
    #train_mmoe_add_loss()
