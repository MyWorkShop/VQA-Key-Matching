#!/usr/bin/python
# -*- coding: utf-8 -*-

from tr_read import data
from model import model, embedding
import tensorflow as tf
import numpy as np

# Config of the data

RESIZE_SIZE = 256
BATCH_SIZE = 128
NUM_GPU = 4


def get_model(
    image,
    question,
    answer,
    phase_train,
    ):

    (output, mi_loss) = model(
        image,
        question,
        data.fixed_num + 1,
        data.fixed_num + 1,
        128,
        phase_train,
        tf.get_variable_scope(),
        )
    one_hot_label = tf.one_hot(answer, data.fixed_num + 1)
    cross_entropy = \
        tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=one_hot_label,
                       logits=output))  
    correct_prediction = tf.equal(tf.argmax(output, 1), answer)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    return (cross_entropy, mi_loss, correct_prediction)


def make_parallel(
    num_gpus,
    images,
    questions,
    answers,
    phase_train,
    ):

    with tf.device('/cpu:0'):
        image = tf.split(images, num_gpus)
        answer = tf.split(answers, num_gpus)

        # question = tf.split(questions, num_gpus)

        question = tf.split(tf.reverse(questions, [-1]), num_gpus)
    loss_split = []
    mi_loss_split = []
    accuracy_split = []
    with tf.variable_scope(tf.get_variable_scope(),
                           reuse=tf.AUTO_REUSE):
        for i in range(num_gpus):
            with tf.name_scope('Tower_%d' % i):
                with tf.device(tf.DeviceSpec(device_type='GPU',
                               device_index=i)):
                    (cross_entropy, mi_loss, correct_prediction) = \
                        get_model(image[i], question[i], answer[i],
                                  phase_train)
                loss_split.append(cross_entropy)
                mi_loss_split.append(mi_loss)
                accuracy_split.append(correct_prediction)
    with tf.device('/cpu:0'):
        mean_loss = tf.reduce_mean(loss_split)
        mean_mi_loss = tf.reduce_mean(mi_loss_split)
        mean_accuracy = tf.reduce_mean(accuracy_split)
    return (mean_loss, mean_mi_loss, mean_accuracy)


with tf.device('/cpu:0'):
    images = tf.placeholder(tf.float32, [BATCH_SIZE, RESIZE_SIZE,
                            RESIZE_SIZE, 3], name='images')
    questions = tf.placeholder(tf.int64, [BATCH_SIZE,
                               data.max_len_question], name='questions')
    answers = tf.placeholder(tf.int64, [BATCH_SIZE], name='questions')

    images_op = data.op_images
    (questions_op, answers_op, _) = tf.py_func(data.get_batch,
            [data.op_imgids], [tf.int64, tf.int64, tf.float64])

    is_training = tf.placeholder(tf.bool, name='train')

    lr = 1e-3
    step_rate = 3000
    decay = 0.3

    global_step = tf.Variable(0, trainable=False)
    increment_global_step = tf.assign(global_step, global_step + 1)
    learning_rate = tf.train.exponential_decay(lr, global_step,
            step_rate, decay, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    (loss, mi_loss, accuracy) = make_parallel(NUM_GPU, images, questions,
            answers, is_training)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss + mi_loss,
                colocate_gradients_with_ops=True)
        pretrain_step = optimizer.minimize(mi_loss,
                colocate_gradients_with_ops=True)

# for x in tf.global_variables():
#     print x.name

with tf.Session() as sess:

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    (images_batches, questions_batches, answers_batches) = \
        sess.run([images_op, questions_op, answers_op])
    print('[pretrain]')
    for i in range(1000):
        (_, _, images_batches, questions_batches, answers_batches, ml) = \
            sess.run([pretrain_step, increment_global_step, images_op,
                     questions_op, answers_op, mi_loss], feed_dict={
            images: images_batches,
            questions: questions_batches,
            answers: answers_batches,
            is_training: True,
            })
        if i % 10 == 0 and i != 0:
            print(str(i / 10) + ',' + str(ml))
    print('[train]')
    for i in range(100000):
        (_, _, images_batches, questions_batches, answers_batches) = \
            sess.run([train_step, increment_global_step, images_op,
                     questions_op, answers_op], feed_dict={
            images: images_batches,
            questions: questions_batches,
            answers: answers_batches,
            is_training: True,
            })
        if i % 100 == 0 and i != 0:

            (a, l, images_batches, questions_batches,
             answers_batches) = sess.run([accuracy, loss, images_op,
                    questions_op, answers_op], feed_dict={
                images: images_batches,
                questions: questions_batches,
                answers: answers_batches,
                is_training: False,
                })
            print(str(i / 100) + ',' + str(a * 100) + ',' + str(l))
