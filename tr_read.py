#!/usr/bin/python
# -*- coding: utf-8 -*-

from vqaTools.vqa import VQA
from collections import Counter
from stemmer import stem
import random
import os
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np

# Config of the dir

dataDir = '.'
versionType = 'v2_'
taskType = 'OpenEnded'
dataType = 'mscoco'
dataSubType = 'train2014'  # Replace it to 'val2014' for validation
annFile = '%s/Annotations/%s%s_%s_annotations.json' % (dataDir,
        versionType, dataType, dataSubType)
quesFile = '%s/Questions/%s%s_%s_%s_questions.json' % (dataDir,
        versionType, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType)
gtDir = '%s/QuestionTypes/abstract_v002_question_types.txt' % dataDir
trDir = '%s/TR/%s_im.tfrecord' % (dataDir, dataSubType)

# Config of the data

RESIZE_SIZE = 256
BATCH_SIZE = 128
NUM_THREADS = 32
FIXED_NUM = 2048


class data_vqa:

    """ Data class of VQA dataset. """

    def __init__(
        self,
        resize_size=RESIZE_SIZE,
        batch_size=BATCH_SIZE,
        num_threads=NUM_THREADS,
        fixed_num=FIXED_NUM,
        ):
        """ Initlization """

        print '[__init__]'

        self.fixed_num = fixed_num

        # Ininlize the offical json processing api

        self.data = VQA(annFile, quesFile)
        self.data_ids = self.data.getQuesIds()
        self.data_len = len(self.data_ids)
        print(self.data_len)
        self.copy_data()
        del self.data
        del self.data_ids
        self.question_processed = self.process_question(self.questions,
                self.max_len_question)

        # self.test_question()

        del self.questions
        self.build_dict_question()
        self.build_dict_answer()

        # Build the reader of the tfrecord file
        # The tfrecord file is generated by tr.write.py

        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'imgid': tf.FixedLenFeature([], tf.int64)}
        filename_queue = tf.train.string_input_producer([trDir])
        reader = tf.TFRecordReader()
        (_, serialized_example) = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                features=feature)
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.cast(image, tf.float32)
        image = image / 255.
        imgid = tf.cast(features['imgid'], tf.int32)
        image = tf.reshape(image, [resize_size, resize_size, 3])
        (self.op_images, self.op_imgids) = \
            tf.train.shuffle_batch([image, imgid],
                                   batch_size=batch_size,
                                   capacity=20480,
                                   num_threads=num_threads,
                                   min_after_dequeue=10240)

    def copy_data(self):
        """ Copy the data from the official json api """

        print '    [copy_data]'
        self.answers = [[self.data.qa[data_id]['answers'][i]['answer'
                        ].encode('ascii', 'ignore').lower() for i in
                        range(10)] for data_id in self.data_ids]
        self.confidence = [[(lambda x: (1. if x == 'yes'
                            else 0.5))(self.data.qa[data_id]['answers'
                           ][i]['answer_confidence'].encode('ascii',
                           'ignore')) for i in range(10)]
                           for data_id in self.data_ids]
        self.imgids = [self.data.qa[data_id]['image_id'] for data_id in
                       self.data_ids]
        self.questions = \
            [self.preprocessing(self.data.qqa[ques_id]['question'])
             for ques_id in self.data_ids]
        self.max_len_question = max([len(question.split())
                                    for question in self.questions])
        print self.max_len_question

    def build_dict_question(self):
        """ Build the mapping from image's imgid to index of image's questions index """

        print '    [build_dict_question]'
        self.imgid_dict = {}
        imgid_set = list(set(self.imgids))
        for imgid in imgid_set:
            self.imgid_dict[imgid] = []
        for i in range(self.data_len):
            imgid = self.imgids[i]
            self.imgid_dict[imgid].append(i)

    def test_question(self):
        print '    [test_question]'
        chars = set()
        for question in self.questions:
            chars.update(question)
        char_list = list(chars)
        print len(char_list)

    def build_dict_answer(self):
        """ Build the mapping from answer's char set to id """

        print '    [build_dict_answer]'
        answer_list = []
        for answers in self.answers:
            for answer in answers:
                answer_list.append(answer)
        counts = Counter(answer_list)
        top_n = counts.most_common(self.fixed_num)
        fixed_list = [elem[0] for elem in top_n]

        # print(fixed_list)

        total = 0
        for elem in top_n:
            total += elem[1]
        print top_n[self.fixed_num - 1][1]
        print total
        print len(answer_list)

        self.answer_dict = dict((c, i) for (i, c) in
                                enumerate(fixed_list))

    def preprocessing(self, text):
        """ Replace the unusual character in the text """

        to_replace = [
            '!',
            '#',
            '%',
            '$',
            "'",
            '&',
            ')',
            '(',
            '+',
            '*',
            '-',
            ',',
            '/',
            '.',
            '1',
            '0',
            '3',
            '2',
            '5',
            '4',
            '7',
            '6',
            '9',
            '8',
            ';',
            ':',
            '?',
            '_',
            '^',
            ]
        lowered = text.encode('ascii', 'ignore').lower()
        replacing = lowered
        for char_to_replace in to_replace:
            replacing = replacing.replace(char_to_replace, ' '
                    + char_to_replace + ' ')
        stemming = ' '
        splited = replacing.split()
        return stemming.join([stem(item) for item in splited])

    def tokenization(self, stentance, preprocess=True):
        """ Split the stentance into words """

        if preprocess == True:
            stentance = self.preprocessing(stentance)
        splited = stentance.split()
        return splited

    def process_question(self, sentences, max_len_question):
        """ Preprocessing the question data """

        print '    [process_question]'
        question_list = []
        for sentence in sentences:
            splited = sentence.split()
            for word in splited:
                question_list.append(word)
        counts = Counter(question_list)
        top_n = counts.most_common(self.fixed_num)
        fixed_list = [elem[0] for elem in top_n]

        # print(fixed_list)

        total = 0
        for elem in top_n:
            total += elem[1]
        print top_n[self.fixed_num - 1][1]
        print total
        print len(question_list)

        self.question_dict = dict((c, i) for (i, c) in
                                  enumerate(fixed_list))

        processed_question = []
        for sentence in sentences:
            splited = sentence.split()
            processed_sentence = []
            for word in splited:
                processed_sentence.append(self.question_dict.get(word,
                        self.fixed_num))
            processed_sentence = processed_sentence + [self.fixed_num] \
                * (max_len_question - len(splited))
            processed_question.append(processed_sentence)

        return processed_question

    def get_batch(self, imgids):
        """ Get the next batch of data """

        questions = []
        answers = []
        confidences = []
        # (images, imgids) = sess.run([self.op_images, self.op_imgids])
        for imgid in imgids:
            index = random.choice(self.imgid_dict[imgid])
            questions.append(self.question_processed[index])
            answer_to_choice = random.choice(range(10))
            confidences.append(self.confidence[index][answer_to_choice])
            answer = self.answers[index][answer_to_choice]
            answers.append(self.answer_dict.get(answer, self.fixed_num))
        return (np.array(questions), np.array(answers),
                np.array(confidences))


data = data_vqa()
