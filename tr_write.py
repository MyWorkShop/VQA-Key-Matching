#!/usr/bin/python
# -*- coding: utf-8 -*-

from vqaTools.vqa import VQA
import random
import scipy.misc as misc
import os
import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


dataDir = '.'
versionType = 'v2_'
taskType = 'OpenEnded'
dataType = 'mscoco'
dataSubType = 'train2014'

# dataSubType ='val2014'

annFile = '%s/Annotations/%s%s_%s_annotations.json' % (dataDir,
        versionType, dataType, dataSubType)
quesFile = '%s/Questions/%s%s_%s_%s_questions.json' % (dataDir,
        versionType, taskType, dataType, dataSubType)
imgDir = '%s/Images/%s/%s/' % (dataDir, dataType, dataSubType)
gtDir = '%s/QuestionTypes/abstract_v002_question_types.txt' % dataDir

resize_size = 256

vqa = VQA(annFile, quesFile)
imgids = vqa.getImgIds()
print len(imgids)
imgids = list(set(imgids))
print len(imgids)
writer = tf.python_io.TFRecordWriter('%s/TR/%s_im.tfrecord' % (dataDir,
        dataSubType))
idx = 0
for imgid in imgids:
    imgFilename = 'COCO_' + dataSubType + '_' + str(imgid).zfill(12) \
        + '.jpg'
    if os.path.isfile(imgDir + imgFilename):
        image = misc.imread(imgDir + imgFilename)
        if len(image.shape) < 3:
            image = np.array([image for i in range(3)])
        image = misc.imresize(image, [resize_size, resize_size],
                              interp='nearest')
        image = image.astype(np.uint8)
    else:
        print 'error'
    feature = {'imgid': _int64_feature(imgid),
               'image': _bytes_feature(tf.compat.as_bytes(image.tostring()))}
    example = \
        tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
    idx = idx + 1
    if idx % 10 == 0:
        print idx / 10
writer.close()
