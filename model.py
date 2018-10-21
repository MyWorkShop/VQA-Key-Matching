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
        return tf.layers.max_pooling2d(value, [2, 2], [2, 2],
                name=scope)


def cosine_similarity(u, v):
    return tf.reduce_sum(u * v, -1, keepdims = True) / (abs(u) * abs(v))


def mlp_similarity(
    u,
    v,
    num1,
    phase_train,
    name='mlp_similarity',
    ):

    with tf.variable_scope(name):
        x = tf.concat([u, v], -1)
        layer1 = tf.layers.dense(x, num1, activation=tf.nn.relu, name='1'
                                 )
        layer1 = tf.layers.dropout(layer1, training=phase_train)
        layer2 = tf.layers.dense(layer1, 1, name='2')
        return layer2


def abs(x):
    return tf.sqrt(tf.reduce_sum(x ** 2, -1, keepdims = True))

def softmax(x, axis = -1):
    return tf.exp(x) / tf.reduce_sum(tf.exp(x), axis, keepdims = True)


def mlp_output(
    x,
    num1,
    num2,
    phase_train,
    name='mlp_output',
    ):

    with tf.variable_scope(name):
        layer1 = tf.layers.dense(x, num1, activation=tf.nn.relu, name='1'
                                 )
        layer1 = tf.layers.dropout(layer1, training=phase_train)
        layer2 = tf.layers.dense(layer1, num2, name='2')
        return layer2


def mlp2(
    x,
    num1,
    num2,
    phase_train,
    name='mlp',
    ):

    with tf.variable_scope(name):
        layer1 = tf.layers.dense(x, num1, activation=tf.nn.relu, name='1'
                                 )
        layer1 = tf.layers.dropout(layer1, training=phase_train)
        layer2 = tf.layers.dense(layer1, num2, activation=tf.nn.relu, name='2'
                                 )
        # layer2 = tf.layers.dropout(layer2, training=phase_train)
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
        layer1 = tf.layers.dense(x, num1, activation=tf.nn.relu, name='1'
                                 )
        layer1 = tf.layers.dropout(layer1, training=phase_train)
        layer2 = tf.layers.dense(layer1, num2, activation=tf.nn.relu, name='2'
                                 )
        layer2 = tf.layers.dropout(layer2, training=phase_train)
        layer3 = tf.layers.dense(layer2, num3, activation=tf.nn.relu, name='3')
        # layer3 = tf.layers.dropout(layer3, training=phase_train)
        return layer3


def res_block(x, num, phase_train, name = 'res_block'):
    with tf.variable_scope(name):
        layer1 = tf.layers.conv2d(
            tf.nn.relu(x),
            num,
            [3, 3],
            padding='same',
            name='1',
            )
        layer1 = tf.layers.batch_normalization(layer1, -1, training = phase_train)
        layer1 = tf.nn.relu(layer1)
        # layer1 = tf.layers.dropout(layer1, training=phase_train)

        layer2 = tf.layers.conv2d(
            layer1,
            num,
            [3, 3],
            padding='same',
            name='2',
            )
        layer2 = tf.layers.batch_normalization(layer2, -1, training = phase_train)
        layer2 = tf.nn.relu(x + layer2)
        # layer2 = tf.layers.dropout(layer2, training=phase_train)
        return layer2


def conv_block(
    x,
    num,
    phase_train,
    with_pool = True,
    name='conv_block',
    act = tf.nn.relu
    ):

    with tf.variable_scope(name):
        layer0 = tf.layers.conv2d(
            x,
            num,
            [1, 1],
            padding='same',
            name='0',
            )
        layer0 = tf.layers.batch_normalization(layer0, -1, training = phase_train)

        block1 = res_block(layer0, num, phase_train, '1')
        block2 = res_block(layer0, num, phase_train, '2')
        block2 = act(block2)
        if with_pool:
                pool = max_pool(block2)
                return pool
        else:
                return block2


def conv(
    x,
    num,
    phase_train,
    name='conv',
    ):

    with tf.variable_scope(name):
        layer1 = tf.layers.conv2d(
            x,
            num,
            [7, 7],
            padding='same',
            name='1',
            )
        layer1 = tf.layers.batch_normalization(layer1, -1, training = phase_train)
        layer1 = max_pool(tf.nn.relu(layer1))
        layer2 = conv_block(layer1, num, phase_train, True, '2')
        layer3 = conv_block(layer2, num * 2, phase_train, True, '3')
        layer4 = conv_block(layer3, num * 4, phase_train, True, '4', no_act)

        layer_attention1 = conv_block(layer3, num, phase_train, True, 'a1')
        layer_attention2 = conv_block(layer_attention1, num, phase_train, False, 'a2')
        mask = tf.layers.conv2d(
            layer_attention2,
            1,
            [1, 1],
            activation = tf.nn.sigmoid, 
            padding='same',
            name='m',
            )
        mask_bias = tf.Variable(0.5, name="mask_bias")
        return tf.nn.relu((mask + mask_bias) * layer4)


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
        return embed


def multi_seq_conv(inputs, num_filters, name='multiseqconv'):
    with tf.variable_scope(name):
        return tf.concat([tf.layers.conv1d(inputs, num_filters, i,
                         padding='same', name='%d' % i) for i in [
            1,
            2,
            3,
            4,
            ]], -1)


def rnn(
    inputs,
    phase_train,
    num_layers=4,
    num_hidden=2048,
    name='rnn',
    ):
    keep_prob = tf.cond(phase_train, lambda: 0.5, lambda: 1.0)
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
        # layer1 = tf.layers.dropout(layer1, training=phase_train)
        layer2 = multi_seq_conv(layer1, num_filters=num_filters,
                                name='conv2')
        # layer2 = tf.layers.dropout(layer2, training=phase_train)
        layer3 = multi_seq_conv(layer2, num_filters=num_filters,
                                name='conv3')
        layer3 = tf.reduce_max(layer3, 1)
        # layer3 = tf.layers.dropout(layer3, training=phase_train)

        layer4 = tf.layers.dense(layer3, num_filters, activation=tf.nn.relu, name='4')

        return layer4

def encoding_rnn(
    inputs,
    num_filters,
    phase_train,
    name='encoding_rnn',
    ):

    with tf.variable_scope(name):
        layer1 = rnn(inputs, phase_train, num_hidden = num_filters)
        question_key = mlp3(layer1, 64, phase_train, 'mlp_key')
        question_feature = mlp3(layer1, 64 * 4, phase_train, 'mlp_feature')
        return question_key, question_feature


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

        #Processing Images

        image_feature = conv(images, 64, phase_train)

        #Processing Questions

        question_embedded = embedding(questions, vocabulary_size,
                embedding_size)
        question_feature = encoding_conv(question_embedded, 256,
                phase_train)

        with tf.device('/cpu:0'):
                question_bow = tf.contrib.layers.bow_encoder(questions, vocabulary_size, embedding_size, scope = 'bow')
        # question_key = question_bow
        question_key = mlp3(question_bow, 1024, 1024, 64 * 4, phase_train, 'mlp_key')
        # index, _ = tf.split(image_feature, [64 * 4, 64 * 4], -1)
        index = image_feature

        question_key_expanded = tf.expand_dims(tf.expand_dims(question_key, 1), 1)
        similarity = cosine_similarity(index, question_key_expanded)
        hotmap = softmax(similarity, [1, 2])
        image_feature1d = tf.reduce_mean(image_feature * hotmap, [1, 2])

        # feature = cosine_similarity(image_feature1d, question_feature)
        # feature = tf.concat([image_feature1d, question_feature], -1)
        with tf.device('/cpu:0'):
            feature = compact_bilinear_pooling_layer(image_feature1d, question_feature, 4096)
        output = mlp_output(feature, 2048, num_output, phase_train, 'output')

        # mi_loss = 0.0001 * tf.reduce_mean(tf.reduce_sum(tf.log(index + 1e-8) * question_key_expanded, -1))
        mi_loss = 0
        return output, mi_loss