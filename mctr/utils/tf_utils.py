# -*- coding: utf-8 -*-
'''
  @author: jeffzhengye
  @contact: yezheng@scuec.edu.cn
  @file: model_utils.py
  @time: 2021/2/17 19:01
  @desc:
 '''

import tensorflow as tf
from copy import deepcopy
import mctr
import mctr.callbacks
import mctr.losses
import tensorflow.keras.losses
import importlib
import types
from tensorflow_addons import losses as tfa_losses


def get_optimizer(config):
    """
    from standard config dict to get the optimizer
    :param config:
    :return:
    """
    name = config['Optimizer']['active']
    opt_paras = deepcopy(config['Optimizer'][name])
    opt_paras = {
        'class_name': name,
        'config': opt_paras
    }
    return tf.keras.optimizers.get(opt_paras)


def get_lr(config):
    name = config['Optimizer']['active']
    return config['Optimizer'][name]['lr']


def get_layers(config):
    """
    :param config: layer config, contain 'name', and 'paras' (a dict of parameters)
    :return:
    """
    # module = __import__(tensorflow.keras.losses.get, globals(), locals())

    module = importlib.import_module(tf.keras.layers.__name__)

    # print('module', module, tf.keras.layers.__name__)
    name = config['name']
    paras = config['paras']
    c = getattr(module, name)
    return c(**paras)


loss_modules = [importlib.import_module(l.__name__) for l in [tf.keras.losses, mctr.losses, tfa_losses]]


def get_loss(name, paras={}):
    return get_fun_cls_from(loss_modules, name, paras=paras)


def get_fun_cls_from(modules: list, name, paras={}):
    for m in modules:
        if hasattr(m, name):
            att = getattr(m, name)
            if isinstance(att, types.MethodType) or isinstance(att, types.FunctionType):
                return att
            else:
                return att(**paras)
    raise RuntimeError('{0} is not defined in inclued loss modules'.format(name))


callbacks_modules = [importlib.import_module(l.__name__) for l in [tf.keras.callbacks, mctr.callbacks]]


def get_callbacks(name, paras={}):
    for m in callbacks_modules:
        if hasattr(m, name):
            att = getattr(m, name)
            return att(**paras)
    raise RuntimeError('{0} is not defined in inclued loss modules'.format(name))


def get_callbacks_from_config(config):
    callback_list = config['Callbacks']
    ret_list = []
    for callback in callback_list:
        for name, paras in callback.items():
            paras = {} if paras is None else paras
            c = get_fun_cls_from(callbacks_modules, name, paras=paras)
            ret_list.append(c)
    return ret_list
