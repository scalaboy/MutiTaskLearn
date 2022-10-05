# coding: utf-8

import struct
import time

import numpy as np
import tensorflow as tf
from mctr.data.feaconf_pb2 import kSparse, kDense
from mctr.data.sample_pb2 import SampleGroup

user_fn = ['101', '109_14', '110_14', '127_14', '150_14', '121', '122', '124', '125', '126', '127', '128', '129']
# user_fn_type= [tf.int32]
ad_fn = ['205', '206', '207', '210', '216', '508', '509', '702', '853', '301']

user_ad_fn = user_fn + ad_fn
labels = ['ctr_output', 'ctcvr_pred']

user_fn_dict = dict([(k, tf.int32) for k in user_fn])
ad_fn_dict = dict([(k, tf.int32) for k in ad_fn])
label_type_dict = {'ctr_output': tf.int32, 'ctcvr_pred': tf.int32}
all_fn_dict = dict(user_fn_dict.items() | ad_fn_dict.items())
fn_id = dict(zip(user_ad_fn, range(len(user_ad_fn))))

output_shapes = (
    tuple([tf.TensorShape((None,))] * len(user_ad_fn)),
    tf.TensorShape((None,))
)

output_shapes = (
    tf.TensorShape((None,)),
    tf.TensorShape((None,))
)

merged_fn = user_fn + ad_fn
padded_shape = [2000] * len(merged_fn)
padded_shape_dict = dict(zip(merged_fn, padded_shape))
padded_shape_dict = {'101': 1, '109_14': 1001, '110_14': 1001, '127_14': 1001, '150_14': 335, '121': 1, '122': 1,
                     '124': 1, '125': 1, '126': 1, '127': 1, '128': 1, '129': 1, '205': 1, '206': 1, '207': 1,
                     '210': 38, '216': 1, '508': 1, '509': 1, '702': 1, '853': 25, '301': 1}
padded_shape_label_dict = {'ctr_output': 1, 'ctcvr_pred': 1}

fn_max = dict(zip(merged_fn, [1] * len(merged_fn)))

# 42300000 examples
distinct_keys = {'122': 13, '124': 2, '125': 7, '127': 3, '128': 2, '129': 4, '150_14': 99011, '127_14': 396179,
                 '110_14': 2545609, '121': 97, '109_14': 12204, '101': 294882, '210': 98714, '301': 3, '205': 3168671,
                 '206': 8601, '207': 610380, '853': 86895, '508': 7753, '509': 388805, '216': 209990, '126': 3,
                 '702': 143919}
fn_max = {'101': 1, '109_14': 1001, '110_14': 1001, '127_14': 1001, '150_14': 335, '121': 1, '122': 1, '124': 1,
          '125': 1, '126': 1, '127': 1, '128': 1, '129': 1, '205': 1, '206': 1, '207': 1, '210': 38, '216': 1, '508': 1,
          '509': 1, '702': 1, '853': 25, '301': 1}


def get_max(yield_dict):
    for k, v in yield_dict.items():
        len_d = len(yield_dict[k])
        if len_d > fn_max[k]:
            fn_max[k] = len_d


# print(len(all_fn_dict), len(merged_fn))

train_list = ['train.0']

train_list = ['../build/train.0', '../build/train.1', '../build/train.2', '../build/train.3', '../build/train.4']
test_list = ['../build/test.0', '../build/test.1', '../build/test.2', '../build/test.3', '../build/test.4']

batch_size = 5000
buffer_size = batch_size * 10


# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(v_list):
    # print(type(v_list), v_list)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v_list))


def serialize_example(x, y):
    """
  Creates a tf.train.Example message ready to be written to a file.
  """
    x_list = [(user_ad_fn[i], _int64_list_feature(x_i)) for i, x_i in enumerate(x)]  # fn_name: feature
    feature = dict(x_list)
    feature['ctr_output'] = _int64_feature(y[0])
    feature['ctcvr_pred'] = _int64_feature(y[1])
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def feature_line_to_(feature_line):
    keys, values = [], []  # keys:[name, name] values:[[v, ..v], [v..vv]]
    for feature in feature_line.features:
        if feature.type == kSparse:
            keys.append(feature.name)
            # values.append([value.key for value in feature.values])
            # values.append(np.array([value.key for value in feature.values]))
            nd_value = np.zeros(len(feature.values), np.int32)
            for i, value in enumerate(feature.values):
                nd_value[i] = value.key
            values.append(nd_value)
            # values = [value.key for value in feature.values]
            # ret_dict[feature.name] = np.array(values, dtype=np.int32)
        elif feature.type == kDense:
            raise NotImplemented('kDense encountered')
        else:
            print('error type,', feature.type)
            exit(1)
    return keys, values


all_fn_key_set = set(all_fn_dict)


def gen(file_name):
    with open(file_name, 'rb') as f:
        data = f.read()
    data_len = len(data)
    pos = 0
    counter = 0
    while pos < data_len:
        sample = SampleGroup()  # your message type
        len_bytes = data[pos:4 + pos]
        length = struct.unpack('<L', len_bytes)[0]
        sample.ParseFromString(data[pos + 4:pos + 4 + length])
        labels, feature_line, feature_line1 = sample.labels, sample.feature_tables[0].feature_lines, \
                                              sample.feature_tables[1].feature_lines
        user_keys, user_values = feature_line_to_(feature_line1[0])
        item_feature_iter = map(feature_line_to_, feature_line)

        for i in range(len(labels)):
            values = labels[i].values
            item_keys, item_values = next(item_feature_iter)
            yield_list = [np.zeros(1, np.int32)] * len(user_ad_fn)
            for k, v in zip(user_keys, user_values):
                yield_list[fn_id[k]] = v
            for k, v in zip(item_keys, item_values):
                yield_list[fn_id[k]] = v
            counter += 1
            yield serialize_example(yield_list, [int(values[0]), int(values[1])])

        pos += 4 + length


def build_one(input_filename):
    start_time = time.time()
    output_file = input_filename + '.tfrecord'
    d = tf.data.Dataset.from_generator(
        gen,
        tf.string,
        args=(input_filename,),
        # output_shapes=(tf.TensorShape((None,)),)
    )
    writer = tf.data.experimental.TFRecordWriter(output_file)
    writer.write(d)
    print("total time:", time.time() - start_time, output_file)
    return None


def build_tfrecord(file_list):
    from joblib import Parallel, delayed, parallel_backend
    with parallel_backend("multiprocessing", n_jobs=len(file_list)):
        Parallel()(delayed(build_one)(j) for j in file_list)


feature_des = dict([(k, tf.io.VarLenFeature(tf.int64)) for k in user_ad_fn + labels])


def get_dataset(batch_size, train=True, correct_label=False, config=None):
    # def read_tfrecord():
    if train:
        l = config['Train']['dataset']['names'] if config else train_list
        filenames = [name + ".tfrecord" for name in l]
    else:
        l = config['Eval']['dataset']['names'] if config else test_list
        filenames = [name + ".tfrecord" for name in l]

    raw_dataset = tf.data.TFRecordDataset(filenames)

    def _parse_example(example_proto):
        examples = tf.io.parse_example(example_proto, feature_des)
        y_dict = {}
        for label in labels:
            y_dict[label] = tf.sparse.to_dense(examples.pop(label))
        if correct_label and train:
            y_dict['ctcvr_pred'] = y_dict['ctcvr_pred'] * y_dict['ctr_output']
        y_dict['cvr_output'] = tf.concat([y_dict['ctcvr_pred'], y_dict['ctr_output']], -1)
        return examples, y_dict

    if train and config['Train']['dataset']['shuffle']:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    parsed_dataset = raw_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE) \
        .map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return parsed_dataset

def get_dataset_moebase(batch_size, train=True, correct_label=False, config=None):
    # def read_tfrecord():
    if train:
        l = config['Train']['dataset']['names'] if config else train_list
        filenames = [name + ".tfrecord" for name in l]
    else:
        l = config['Eval']['dataset']['names'] if config else test_list
        filenames = [name + ".tfrecord" for name in l]

    raw_dataset = tf.data.TFRecordDataset(filenames)

    def _parse_example(example_proto):
        examples = tf.io.parse_example(example_proto, feature_des)
        y_dict = {}
        for label in labels:
            y_dict[label] = tf.sparse.to_dense(examples.pop(label))
        if correct_label and train:
            y_dict['ctcvr_pred'] = y_dict['ctcvr_pred'] * y_dict['ctr_output']
        y_dict['ctcvr_pred'] = tf.concat([y_dict['ctcvr_pred'], y_dict['ctr_output']], -1)
        return examples, y_dict

    if train and config['Train']['dataset']['shuffle']:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    parsed_dataset = raw_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE) \
        .map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return parsed_dataset




def get_dataset_mof(batch_size, train=True, correct_label=False, config=None):
    # def read_tfrecord():
    if train:
        l = config['Train']['dataset']['names'] if config else train_list
        filenames = [name + ".tfrecord" for name in l]
    else:
        l = config['Eval']['dataset']['names'] if config else test_list
        filenames = [name + ".tfrecord" for name in l]

    raw_dataset = tf.data.TFRecordDataset(filenames)

    def _parse_example(example_proto):
        examples = tf.io.parse_example(example_proto, feature_des)
        y_dict = {}
        for label in labels:
            y_dict[label] = tf.sparse.to_dense(examples.pop(label))
        if correct_label and train:
            y_dict['ctcvr_pred'] = y_dict['ctcvr_pred'] * y_dict['ctr_output']
        y_dict['cvr_output'] = tf.concat([y_dict['ctcvr_pred'], y_dict['ctr_output']], -1)
        y_dict['ct_nocvr_pred'] = y_dict['ctr_output'] * (1 - y_dict['ctcvr_pred'])
        y_dict['ctcvr_pred'] = y_dict['cvr_output']
        return examples, y_dict

    if train and config['Train']['dataset']['shuffle']:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    parsed_dataset = raw_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE) \
        .map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return parsed_dataset




def get_dataset_ctnocvr(batch_size, train=True, correct_label=False, config=None):
    def _add_lables(x, y):
        # （点击&未转化=1；点击且转化=0; 不点击=0）
        y['ct_nocvr_pred'] = y['ctr_output'] * (1 - y['ctcvr_pred'])
        return x, y
    parsed_dataset = get_dataset(batch_size, train=train, correct_label=correct_label, config=config).map(_add_lables)
    return parsed_dataset



def get_dataset_addloss_ctnocvr(batch_size, train=True, correct_label=False, config=None):
    def _add_lables(x, y):
        # （点击&未转化=1；点击且转化=0; 不点击=0）
        y['ct_nocvr_pred'] = y['ctr_output'] * (1 - y['ctcvr_pred'])
        return x, y
    parsed_dataset = get_dataset_mof(batch_size, train=train, correct_label=correct_label, config=config).map(_add_lables)
    return parsed_dataset



def get_dataset_ctronly(batch_size, train=True, shuffle=True, max_test=-1):
    # def read_tfrecord():
    if train:
        filenames = [name + ".tfrecord" for name in train_list[:]]
    else:
        filenames = [name + ".tfrecord" for name in test_list[:1]]

    raw_dataset = tf.data.TFRecordDataset(filenames)

    def _parse_example(example_proto):
        examples = tf.io.parse_example(example_proto, feature_des)
        y_dict = {}
        for label in labels:
            if label == 'ctcvr_pred':
                examples.pop(label)
            else:
                # y_dict[label] = tf.sparse.to_dense(examples.pop(label))
                y_dict[label] = tf.sparse.to_dense(examples.pop(label))
        return examples, y_dict

    if train and shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    parsed_image_dataset = raw_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE) \
        .map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return parsed_image_dataset


def read_tfrecord():
    parsed_image_dataset = get_dataset(batch_size=batch_size)
    time_list = []
    start_time = time.time()
    for record in parsed_image_dataset.take(1000):
        time_list.append(time.time() - start_time)
        if len(time_list) > 3:
            print("average batch time:", sum(time_list[-3:]) / 3)
        start_time = time.time()


def get_stats():
    parsed_image_dataset = get_dataset(batch_size=batch_size)
    stats = [0, 0]
    for i, (x, y) in parsed_image_dataset.enumerate(0):
        stats[0] = stats[0] + tf.reduce_sum(y['ctr_output']).numpy()
        stats[1] = stats[1] + tf.reduce_sum(y['ctcvr_pred']).numpy()
        i = i.numpy()
        if i % 500 == 0:
            print(i, i * batch_size, stats)

    print(i, i * batch_size, stats)


def get_stats_from_origin():
    total = 0
    for f in train_list:
        cur_count = 0
        for record in gen(f):
            cur_count += 1
            total += 1
            if total % 5000 == 0:
                print(total, )
        print('total:', total, f, cur_count)


def get_dataset_v1(batch_size, train=True):
    # def read_tfrecord():
    if train:
        filenames = [name + ".tfrecord" for name in train_list]
    else:
        filenames = [name + ".tfrecord" for name in test_list]

    raw_dataset = tf.data.TFRecordDataset(filenames)

    def _parse_example(example_proto):
        examples = tf.io.parse_example(example_proto, feature_des)
        for label in labels:
            examples[label] = tf.sparse.to_dense(examples[label])
        return examples

    if train:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    parsed_image_dataset = raw_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE) \
        .map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return parsed_image_dataset


def get_count():
    dataset = get_dataset_ctronly(5000, train=True, shuffle=False)
    for i, (x, y) in dataset.enumerate(0):
        if i % 1000 == 0:
            print(i, )
    print('total batch count', i)


# get_stats()

# read_tfrecord()
# build_tfrecord(test_list)

if __name__ == "__main__":
    get_count()
    # build_tfrecord(train_list)
    # get_stats_from_origin()
