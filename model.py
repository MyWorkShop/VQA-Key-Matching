#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from indrnn import IndRNNCell
import numpy as np

IMAGE_SIZE = 256
BATCH_SIZE = 32


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
    num3,
    phase_train,
    name='mlp_output',
    ):

    with tf.variable_scope(name):
        layer1 = tf.layers.dense(x, num1, activation=tf.nn.relu, name='1'
                                 )
        layer1 = tf.layers.dropout(layer1, training=phase_train)
        layer2 = tf.layers.dense(layer1, num2, activation=tf.nn.relu, name='2'
                                 )
        layer2 = tf.layers.dropout(layer2, training=phase_train)
        layer3 = tf.layers.dense(layer2, num3, name='3')
        return layer3


def mlp(
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
        layer2 = tf.layers.dense(layer1, num2, activation=tf.nn.relu,
                                 name='2')
        return layer2


def conv_block(
    x,
    num,
    phase_train,
    name='conv_block',
    ):

    with tf.variable_scope(name):
        layer0 = tf.layers.conv2d(
            x,
            num,
            [1, 1],
            padding='same',
            name='0',
            )
        layer0 = tf.layers.batch_normalization(layer0, -1)

        layer1 = tf.layers.conv2d(
            x,
            num,
            [3, 3],
            padding='same',
            name='1',
            )
        layer1 = tf.layers.batch_normalization(layer1, -1)
        layer1 = tf.nn.relu(layer1)

        layer2 = tf.layers.conv2d(
            layer1,
            num,
            [3, 3],
            padding='same',
            name='2',
            )
        layer2 = tf.layers.batch_normalization(layer2, -1)
        layer0 = layer0 + layer2
        layer2 = tf.nn.relu(layer0)

        layer3 = tf.layers.conv2d(
            layer2,
            num,
            [3, 3],
            padding='same',
            name='3',
            )
        layer3 = tf.layers.batch_normalization(layer3, -1)
        layer3 = tf.nn.relu(layer1)

        layer4 = tf.layers.conv2d(
            layer3,
            num,
            [3, 3],
            padding='same',
            name='4',
            )
        layer4 = tf.layers.batch_normalization(layer4, -1)
        layer4 = tf.nn.relu(layer0 + layer4)

        pool = max_pool(layer3)
    return pool


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
            activation=tf.nn.relu,
            name='1',
            )
        layer1 = tf.layers.batch_normalization(layer1, -1)
        layer1 = max_pool(layer1)
        layer2 = conv_block(layer1, num, phase_train, '2')
        layer3 = conv_block(layer2, num * 2, phase_train, '3')
        layer_feature = conv_block(layer3, num * 4, phase_train, 'feature')
        layer_key = conv_block(layer3, num, phase_train, 'key')
        return (layer_key, layer_feature)
        # return layer_feature


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
    num_layers=2,
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
        # layer2 = multi_seq_conv(layer1, num_filters=num_filters,
        #                         name='conv2')
        # layer2 = tf.layers.dropout(layer2, training=phase_train)
        # layer3 = multi_seq_conv(layer2, num_filters=num_filters,
        #                         name='conv3')
        layer3 = tf.reduce_max(layer1, 1)
        # layer3 = tf.layers.dropout(layer3, training=phase_train)

        question_key = mlp(layer3, 2048, 64, phase_train, 'mlp_key')
        question_feature = mlp(layer3, 2048, 64 * 4, phase_train, 'mlp_feature')
        
        return question_key, question_feature

def encoding_rnn(
    inputs,
    num_filters,
    phase_train,
    name='encoding_rnn',
    ):

    with tf.variable_scope(name):
        layer1 = rnn(inputs, phase_train, num_hidden = num_filters)
        question_key = mlp(layer1, 2048, 64, phase_train, 'mlp_key')
        question_feature = mlp(layer1, 2048, 64 * 4, phase_train, 'mlp_feature')
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

        index, image_feature3d = conv(images, 64, phase_train)

        #Processing Questions

        question_embedded = embedding(questions, vocabulary_size,
                embedding_size)
        question_key, question_feature = encoding_conv(question_embedded, 256,
                phase_train)

        queston_key = tf.tile(tf.expand_dims(tf.expand_dims(question_key, 1), 1), [1, 16, 16, 1])
        similarity = cosine_similarity(index, queston_key)
        hotmap = softmax(similarity, [1, 2])
        image_feature1d = tf.reduce_mean(image_feature3d * hotmap, [1, 2])

        feature = tf.concat([image_feature1d, question_feature], -1)
        output = mlp_output(feature, 2048, 2048, num_output, phase_train, 'output')
        return output
