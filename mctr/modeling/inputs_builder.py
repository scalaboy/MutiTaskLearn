# -*- coding: utf-8 -*-
'''
  @author: jeffzhengye
  @contact: yezheng@scuec.edu.cn
  @file: inputs_builder.py
  @time: 2021/2/17 20:28
  @desc:
 '''

from mctr.data.ali_ccp_datasets import user_fn, ad_fn, merged_fn, padded_shape_dict, label_type_dict, get_dataset, \
    distinct_keys, get_dataset_ctronly
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import feature_column


# init keras input
def build_inputs():
    inputs = {}
    for k, v in padded_shape_dict.items():
        if k not in label_type_dict:
            inputs[k] = layers.Input(shape=(v,), name=k, dtype=tf.int32, sparse=True)
    return inputs


def build_input_shape():
    input_shape = {}
    for k, v in padded_shape_dict.items():
        if k not in label_type_dict:
            print(type(v))
            input_shape[k] = (None, v)
    print(input_shape)
    return input_shape


def build_din(inputs, embed_size, feature_dim, min_by):
    user_embeddings, item_embeddings = [], []

    def _get_embedding_column(key):
        hash_bucket_size = min(feature_dim, distinct_keys[key] * min_by)
        # hash_bucket_size = feature_dim
        bucket_k = feature_column.categorical_column_with_hash_bucket(key, hash_bucket_size, dtype=tf.int32)
        # 注意到嵌入列的输入是我们之前创建的类别列
        embedding_k = feature_column.embedding_column(bucket_k, dimension=embed_size, combiner='sum',
                                                      use_safe_embedding_lookup=True,
                                                      initializer=tf.initializers.truncated_normal(stddev=0.001)
                                                      )
        # feature_layer = layers.DenseFeatures(embedding_k)
        # embedding = feature_layer(inputs)
        # user_embeddings.append(embedding)
        return embedding_k

    for k in user_fn:
        user_embeddings.append(_get_embedding_column(k))
    for k in ad_fn:
        item_embeddings.append(_get_embedding_column(k))

    feature_layer = layers.DenseFeatures(user_embeddings + item_embeddings)
    din = feature_layer(inputs)
    return din
