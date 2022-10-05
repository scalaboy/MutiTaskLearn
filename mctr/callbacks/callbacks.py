# -*- coding: utf-8 -*-
'''
  @author: jeffzhengye
  @contact: yezheng@scuec.edu.cn
  @file: callbacks.py
  @time: 2021/2/16 14:32
  @desc:
 '''

from tensorflow.keras.callbacks import Callback, ModelCheckpoint

class BatchModelCheckpoint(Callback):
    def __init__(self, dataset, log_batch=None, *args, **kwargs):
        super(BatchModelCheckpoint, self).__init__(*args, **kwargs)
        self.dataset = dataset
        self.epoch = None
        self.total_batch = 0
        self.eval_batch = log_batch
        print('BatchModelCheckpoint init')

    def on_batch_end(self, batch, logs=None):
        self.total_batch += 1
        if self.eval_batch and self.total_batch in self.eval_batch:
            print('\nbatch end eval: ', self.total_batch)
            results = self.model.evaluate(self.dataset)
        elif self.total_batch % 1000 == 0:
            if 6000 <= self.total_batch <= 12000:
                print('\nbatch end eval: ', self.total_batch)
                results = self.model.evaluate(self.dataset)
                # print(', '.join(["%s: %s" % (a, b) for a, b in zip(self.model.metrics_names, results)]))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        print('BatchModelCheckpoint epoch', epoch)


class CBatchModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(CBatchModelCheckpoint, self).__init__(*args, **kwargs)

    def _should_save_on_batch(self, batch, logs=None):
        """Handles batch-level saving logic, supports steps_per_execution."""
        if logs and 'batch_evaled' in logs:
            print('batch eval begin', self._current_epoch, batch)
            return True
        return False

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch, logs):
            self._save_model(epoch=self._current_epoch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        super(CBatchModelCheckpoint, self).on_epoch_begin(epoch=epoch, logs=logs)
        print('BatchModelCheckpoint epoch', self._current_epoch)