# -*- coding: utf-8 -*-
'''
  @author: jeffzhengye
  @contact: yezheng@scuec.edu.cn
  @file: custom_keras_model.py
  @time: 2021/2/20 17:46
  @desc:
 '''

import copy

from tensorflow.keras import Model
from tensorflow.python.eager import context
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine.training import _disallow_inside_tf_function
from tensorflow.python.util import tf_inspect
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.profiler import trace

from tensorflow.python.eager import monitoring
#keras_api_gauge = monitoring.BoolGauge('/tensorflow/api/keras','keras-api-usage', 'method')

class BatchEvalModel(Model):
    def __init__(self, batch_interval=1000,  eval_batchs=[6000, 12000], *args, **kwargs):
        super(BatchEvalModel, self).__init__( *args, **kwargs)
        self.trained_steps = 0  # record how many steps we trained.
        self.eval_batchs = eval_batchs
        self.batch_interval = batch_interval

    def _should_eval_on_batch(self, epoch, total_batch):
        if total_batch > 0 and total_batch % self.batch_interval == 0:
            if self.eval_batchs and  self.eval_batchs[0] <= total_batch <= self.eval_batchs[1]:
                print('\n batch end eval: ', total_batch)
                return True
        return False

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):
        #base_layer.
        base_layer.keras_api_gauge.get_cell('fit').set(True)
        #keras_api_gauge.get_cell('fit').set(True)
        # Legacy graph support is contained in `training_v1.Model`.
        version_utils.disallow_legacy_graph('Model', 'fit')
        self._assert_compile_was_called()
        self._check_call_args('fit')
        _disallow_inside_tf_function('fit')

        if validation_split:
            # Create the validation data using the training data. Only supported for
            # `Tensor` and `NumPy` input.
            (x, y, sample_weight), validation_data = (
                data_adapter.train_validation_split(
                    (x, y, sample_weight), validation_split=validation_split))

        if validation_data:
            val_x, val_y, val_sample_weight = (
                data_adapter.unpack_x_y_sample_weight(validation_data))

        with self.distribute_strategy.scope(), \
             training_utils.RespectCompiledTrainableState(self):
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            data_handler = data_adapter.DataHandler(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                initial_epoch=initial_epoch,
                epochs=epochs,
                shuffle=shuffle,
                class_weight=class_weight,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self,
                steps_per_execution=self._steps_per_execution)

            # Container that configures and calls `tf.keras.Callback`s.
            if not isinstance(callbacks, callbacks_module.CallbackList):
                callbacks = callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=self,
                    verbose=verbose,
                    epochs=epochs,
                    steps=data_handler.inferred_steps)

            self.stop_training = False
            self.train_function = self.make_train_function()
            self._train_counter.assign(0)
            callbacks.on_train_begin()
            training_logs = None
            # Handle fault-tolerance for multi-worker.
            # TODO(omalleyt): Fix the ordering issues that mean this has to
            # happen after `callbacks.on_train_begin`.
            data_handler._initial_epoch = (  # pylint: disable=protected-access
                self._maybe_load_initial_epoch_from_ckpt(initial_epoch))
            logs = None
            for epoch, iterator in data_handler.enumerate_epochs():
                self.reset_metrics()
                callbacks.on_epoch_begin(epoch)
                with data_handler.catch_stop_iteration():
                    for step in data_handler.steps():
                        with trace.Trace(
                                'train',
                                epoch_num=epoch,
                                step_num=step,
                                batch_size=batch_size,
                                _r=1):
                            callbacks.on_train_batch_begin(step)
                            tmp_logs = self.train_function(iterator)
                            if data_handler.should_sync:
                                context.async_wait()
                            logs = tmp_logs  # No error, now safe to assign to logs.
                            end_step = step + data_handler.step_increment

                            ################ added to eval on batch
                            self.trained_steps += 1 + data_handler.step_increment
                            # print('steps', self.trained_steps, data_handler.step_increment)
                            # Run validation.
                            if validation_data and self._should_eval_on_batch(epoch, self.trained_steps):
                                # Create data_handler for evaluation and cache it.
                                if getattr(self, '_eval_data_handler', None) is None:
                                    self._fit_frame = tf_inspect.currentframe()
                                    self._eval_data_handler = data_adapter.DataHandler(
                                        x=val_x,
                                        y=val_y,
                                        sample_weight=val_sample_weight,
                                        batch_size=validation_batch_size or batch_size,
                                        steps_per_epoch=validation_steps,
                                        initial_epoch=0,
                                        epochs=1,
                                        max_queue_size=max_queue_size,
                                        workers=workers,
                                        use_multiprocessing=use_multiprocessing,
                                        model=self,
                                        steps_per_execution=self._steps_per_execution)
                                val_logs = self.evaluate(
                                    x=val_x,
                                    y=val_y,
                                    verbose=verbose,
                                    sample_weight=val_sample_weight,
                                    batch_size=validation_batch_size or batch_size,
                                    steps=validation_steps,
                                    callbacks=None,
                                    max_queue_size=max_queue_size,
                                    workers=workers,
                                    use_multiprocessing=use_multiprocessing,
                                    return_dict=True)
                                val_logs = {'val_' + name: val for name, val in val_logs.items()}
                                val_logs['batch_evaled'] = True
                                val_logs.update(logs)
                                logs = val_logs.update(val_logs)
                            #####################################

                            callbacks.on_train_batch_end(end_step, logs)
                            if self.stop_training:
                                break

                if logs is None:
                    raise ValueError('Expect x to be a non-empty array or dataset.')
                epoch_logs = copy.copy(logs)

                # Run validation.
                if validation_data and self._should_eval(epoch, validation_freq):
                    # Create data_handler for evaluation and cache it.
                    if getattr(self, '_eval_data_handler', None) is None:
                        self._fit_frame = tf_inspect.currentframe()
                        self._eval_data_handler = data_adapter.DataHandler(
                            x=val_x,
                            y=val_y,
                            sample_weight=val_sample_weight,
                            batch_size=validation_batch_size or batch_size,
                            steps_per_epoch=validation_steps,
                            initial_epoch=0,
                            epochs=1,
                            max_queue_size=max_queue_size,
                            workers=workers,
                            use_multiprocessing=use_multiprocessing,
                            model=self,
                            steps_per_execution=self._steps_per_execution)
                    val_logs = self.evaluate(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps=validation_steps,
                        callbacks=callbacks,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        return_dict=True)
                    val_logs = {'val_' + name: val for name, val in val_logs.items()}
                    epoch_logs.update(val_logs)

                callbacks.on_epoch_end(epoch, epoch_logs)
                training_logs = epoch_logs
                if self.stop_training:
                    break

            # If eval data_hanlder exists, delete it after all epochs are done.
            if getattr(self, '_eval_data_handler', None) is not None:
                del self._eval_data_handler
                del self._fit_frame
            callbacks.on_train_end(logs=training_logs)
            return self.history
