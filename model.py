#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from cbp import compact_bilinear_pooling_layer
import numpy as np

IMAGE_SIZE = 256
BATCH_SIZE = 32


def no_act(x):
    return x


def max_pool(value, name='maxpool'):
    with tf.variable_scope(name) as scope:
        return tf.layers.max_pooling2d(value, [3, 3], [2, 2], padding = 'same',
                name=scope)


def unpool(x):

    # https://github.com/tensorflow/tensorflow/issues/2169

    out = tf.concat([x, tf.zeros_like(x)], 3)
    out = tf.concat([out, tf.zeros_like(out)], 2)

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        shv = tf.shape(x)
        ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2,
                         sh[3]]))
        return ret


def deconv(
    x,
    num,
    phase_train,
    name='deconv',
    ):
    with tf.variable_scope(name):
        
        # x = unpool(x)
        # layer1 = tf.layers.conv2d(
        #     x,
        #     num,
        #     [3, 3],
        #     padding='same',
        #     activation=tf.nn.relu,
        #     name='1',
        #     )
        # pool1 = max_pool(layer1)
        # layer2 = tf.layers.conv2d(
        #     pool1,
        #     num,
        #     [3, 3],
        #     padding='same',
        #     activation=tf.nn.relu,
        #     name='2',
        #     )
        # pool2 = max_pool(layer2)
        # layer3 = tf.layers.conv2d_transpose(
        #     x,
        #     num,
        #     [3, 3],
        #     padding='same',
        #     activation=tf.nn.relu,
        #     name='3',
        #     )
        # pool3 = unpool(layer3)
        layer4 = tf.layers.conv2d_transpose(
            x,
            num,
            [3, 3],
            padding='same',
            activation=tf.nn.relu,
            name='4',
            )
        pool4 = unpool(layer4)
        layer5 = tf.layers.conv2d_transpose(
            pool4,
            num,
            [3, 3],
            padding='same',
            activation=tf.nn.relu,
            name='5',
            )
        pool5 = unpool(layer5)
        layer6 = tf.layers.conv2d_transpose(
            pool5,
            num,
            [3, 3],
            padding='same',
            activation=tf.nn.relu,
            name='6',
            )
        pool6 = unpool(layer6)
        layer7 = tf.layers.conv2d_transpose(
            pool6,
            num,
            [3, 3],
            padding='same',
            activation=tf.nn.relu,
            name='7',
            )
        pool7 = unpool(layer7)
        layer8 = tf.layers.conv2d_transpose(
            pool7,
            num,
            [3, 3],
            padding='same',
            activation=tf.nn.relu,
            name='8',
            )
        pool8 = unpool(layer8)
        layer9 = tf.layers.conv2d_transpose(
            pool8,
            3,
            [3, 3],
            padding='same',
            activation=tf.nn.relu,
            name='9',
            )
        return layer9


def cosine_similarity(u, v):
    return tf.reduce_sum(u * v, -1, keepdims=True) / (abs(u) * abs(v))


def mlp_similarity(
    u,
    v,
    num1,
    phase_train,
    name='mlp_similarity',
    ):

    with tf.variable_scope(name):
        x = tf.concat([u, v], -1)
        layer1 = tf.layers.dense(x, num1, activation=tf.nn.relu,
                                 name='1')
        layer1 = tf.layers.dropout(layer1, training=phase_train)
        layer2 = tf.layers.dense(layer1, 1, name='2')
        return layer2


def abs(x):
    return tf.sqrt(tf.reduce_sum(x ** 2, -1, keepdims=True))


def softmax(x, axis=-1):
    return tf.exp(x) / tf.reduce_sum(tf.exp(x), axis, keepdims=True)


def mlp_output(
    x,
    num1,
    num2,
    phase_train,
    name='mlp_output',
    ):

    with tf.variable_scope(name):
        layer1 = tf.layers.dense(x, num1, activation=tf.nn.relu,
                                 name='1')
        layer2 = tf.layers.dense(layer1, num1, activation=tf.nn.relu,
                                 name='2')
        layer3 = tf.layers.dense(layer2, num2, name='3')
        return layer3


def mlp2(
    x,
    num1,
    num2,
    phase_train,
    name='mlp',
    ):

    with tf.variable_scope(name):
        layer1 = tf.layers.dense(x, num1, activation=tf.nn.relu,
                                 name='1')
        # layer1 = tf.layers.batch_normalization(layer1, -1,
        #         training=phase_train)
        layer2 = tf.layers.dense(layer1, num2, activation=tf.nn.relu,
                                 name='2')
        # layer2 = tf.layers.batch_normalization(layer2, -1,
        #         training=phase_train)
        return layer2


def mlp3(
    x,
    num1,
    num2,
    num3,
    phase_train,
    name='mlp',
    ):

    with tf.variable_scope(name):
        layer1 = tf.layers.dense(x, num1, activation=tf.nn.relu,
                                 name='1')
        # layer1 = tf.layers.batch_normalization(layer1, -1,
        #         training=phase_train)
        layer2 = tf.layers.dense(layer1, num2, activation=tf.nn.relu,
                                 name='2')
        # layer2 = tf.layers.batch_normalization(layer2, -1,
        #         training=phase_train)
        layer3 = tf.layers.dense(layer2, num3, activation=tf.nn.relu,
                                 name='3')
        # layer3 = tf.layers.batch_normalization(layer3, -1,
        #         training=phase_train)
        return layer3


def res_block(
    x,
    num,
    phase_train,
    name='res_block',
    ):

    with tf.variable_scope(name):
        layer1 = tf.layers.separable_conv2d(tf.nn.relu(x), num, [3, 3],
                                  padding='same', name='1')
        layer1 = tf.layers.batch_normalization(layer1, -1,
                training=phase_train)
        layer1 = tf.nn.relu(layer1)

        layer2 = tf.layers.separable_conv2d(layer1, num, [3, 3], padding='same',
                                  name='2')
        layer2 = tf.layers.batch_normalization(layer2, -1,
                training=phase_train)
        layer2 = x + layer2
        return layer2


def conv_block(
    x,
    num,
    phase_train,
    with_pool=True,
    name='conv_block',
    act=tf.nn.relu,
    ):

    with tf.variable_scope(name):
        layer0 = tf.layers.conv2d(x, num, [1, 1], padding='same',
                                  name='0')
        layer0 = tf.layers.batch_normalization(layer0, -1,
                training=phase_train)

        block1 = res_block(layer0, num, phase_train, '1')
        block2 = res_block(block1, num, phase_train, '2')
        block3 = res_block(block2, num, phase_train, '3')
        block3 = act(block3)
        if with_pool:
            pool = max_pool(block3)
            return pool
        else:
            return block3


def conv(
    x,
    num,
    phase_train,
    name='conv',
    ):

    with tf.variable_scope(name):
        layer1 = tf.layers.separable_conv2d(x, num, [7, 7], strides = (2, 2), padding='same',
                                  name='1')
        layer1 = tf.layers.batch_normalization(layer1, -1,
                training=phase_train)
        layer1 = max_pool(tf.nn.relu(layer1))
        layer2 = conv_block(layer1, num , phase_train, True, '2')
        layer3 = conv_block(layer2, num * 2, phase_train, True, '3')
        layer4 = conv_block(layer3, num * 4, phase_train, False, '4')
        layer5 = conv_block(layer4, num * 4, phase_train, True, '5')
        layer6 = conv_block(layer5, num * 8, phase_train, False, '6')

        return layer6


def embedding(
    inputs,
    vocabulary_size,
    embedding_size,
    name='embedding',
    ):

    with tf.variable_scope(name):
        embeddings = tf.get_variable(name='embeddings',
                initializer=tf.random_uniform([vocabulary_size,
                embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, inputs)
        bow = tf.reduce_sum(embed, 1)
        return embed, bow


def multi_seq_conv(inputs, num_filters, name='multiseqconv'):
    with tf.variable_scope(name):
        return tf.concat([tf.layers.conv1d(inputs, num_filters, i, activation=tf.nn.relu,
                         padding='same', name='%d' % i) for i in [1, 2,
                         3, 4]], -1)

def multi_seq_conv_transpose(inputs, num_filters, name='multiseqdeconv'):
    with tf.variable_scope(name):
        inputs = tf.expand_dims(inputs, 1)
        return tf.squeeze(tf.concat([tf.layers.conv2d_transpose(inputs, num_filters, [1, i],
                         activation=tf.nn.relu, padding='same', name='%d' % i) for i in [1, 2,
                         3, 4]], -1), 1)


def rnn(
    inputs,
    phase_train,
    num_layers=4,
    num_hidden=2048,
    name='rnn',
    ):

    keep_prob = 1.0

    # keep_prob = tf.cond(phase_train, lambda: 0.5, lambda: 1.0)

    with tf.variable_scope(name):
        cell = tf.contrib.rnn.BasicLSTMCell
        cells = [tf.nn.rnn_cell.DropoutWrapper(cell(num_hidden,
                 name=str(i)), output_keep_prob=keep_prob) for i in
                 range(num_layers)]
        rnn_cell = tf.contrib.rnn.MultiRNNCell(cells,
                state_is_tuple=True)
        (_outputs, _) = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=inputs,
                dtype=tf.float32)
        last = _outputs[:, -1, :]
        return last


def encoding_conv(
    inputs,
    num_filters,
    phase_train,
    name='encoding_conv',
    ):

    with tf.variable_scope(name):
        layer1 = multi_seq_conv(inputs, num_filters=num_filters,
                                name='conv1')
        layer1 = tf.layers.batch_normalization(layer1, -1,
                training=phase_train)
        layer2 = multi_seq_conv(layer1, num_filters=num_filters,
                                name='conv2')
        layer2 = tf.layers.batch_normalization(layer2, -1,
                training=phase_train)
        layer3 = multi_seq_conv(layer2, num_filters=num_filters,
                                name='conv3')
        layer3 = tf.layers.batch_normalization(layer3, -1,
                training=phase_train)

        layer3_reduce = tf.reduce_max(layer3, 1)

        layer4 = tf.layers.dense(layer3_reduce, num_filters,
                                 activation=tf.nn.relu, name='4')

        return layer4, layer3


def decoding_conv(
    inputs,
    num_filters,
    phase_train,
    name='decoding_conv',
    ):

    with tf.variable_scope(name):
        layer1 = multi_seq_conv_transpose(inputs, num_filters=num_filters,
                                name='conv1')
        layer1 = tf.layers.batch_normalization(layer1, -1,
                training=phase_train)
        layer2 = multi_seq_conv_transpose(layer1, num_filters=num_filters,
                                name='conv2')
        layer2 = tf.layers.batch_normalization(layer2, -1,
                training=phase_train)
        layer3 = multi_seq_conv_transpose(layer2, num_filters=num_filters,
                                name='conv3')
        layer3 = tf.layers.batch_normalization(layer3, -1,
                training=phase_train)

        return tf.layers.conv1d(layer3, 256, 1)


def encoding_rnn(
    inputs,
    num_filters,
    phase_train,
    name='encoding_rnn',
    ):

    with tf.variable_scope(name):
        layer1 = rnn(inputs, phase_train, num_hidden=num_filters)
        layer2 = tf.layers.dense(layer1, 64 * 8,
                                 activation=tf.nn.relu, name='2')
        return layer2


def model(
    images,
    questions,
    num_output,
    vocabulary_size,
    embedding_size,
    phase_train,
    name='model',
    ):    
    
    with tf.variable_scope(name):

        # Processing Images

        image_feature = conv(images, 64, phase_train)
        image_index = image_feature
        image_feature_1 = tf.reduce_mean(image_index, [1, 2])

        # Processing Questions

        question_embedded, question_bow = embedding(questions, vocabulary_size,
                embedding_size)
        question_conv, question_conv_origional = encoding_conv(question_embedded, 64 * 8,
                phase_train)

        # with tf.device('/cpu:0'):
        #     question_bow = tf.contrib.layers.bow_encoder(questions,
        #             vocabulary_size, 64 * 8, scope='bow')

        question_feature = tf.concat([question_bow, question_conv], -1)

        question_key = mlp3(
            question_feature,
            2048,
            2048,
            64 * 8,
            phase_train,
            'mlp_key',
            )

        # Attention Using Key Matching

        alpha = tf.Variable(1.0, name = 'alpha')
        beta = tf.Variable(1.0, name = 'beta')
        question_key_expanded = \
            tf.expand_dims(tf.expand_dims(question_key, 1), 1)
        similarity = cosine_similarity(image_index,
                question_key_expanded)
        hotmap = softmax(alpha * similarity, [1, 2])

        hotmap = softmax(similarity, [1, 2])

        image_feature_2 = tf.reduce_mean(image_feature * hotmap, [1, 2])

        # Compact Bilinear Pooling

        # with tf.device('/cpu:0'):
        #     feature = compact_bilinear_pooling_layer(image_feature_concated,
        #             question_feature, 4096)
        feature = tf.concat([image_feature_2, question_feature], -1)
        output = mlp_output(feature, 2048, num_output, phase_train,
                            'output')

        # autoencoder

        rebulid_image = deconv(image_index, 32, phase_train)
        rebulid_question = decoding_conv(question_conv_origional, 64 * 8, phase_train)
        mi_loss = tf.reduce_sum(tf.square(rebulid_image - images))\
                        + tf.reduce_sum(tf.square(rebulid_question - question_embedded))
        # mi_loss = 0.
        return (output, mi_loss)