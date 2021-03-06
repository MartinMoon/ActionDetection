from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from inception.inception import inception_resnet_v2
from inception.inception import inception_v3
from inception.inception import inception_v2

slim = tf.contrib.slim

def feature_net(images, is_training=True):

    # Do it by yourself!!!
    #images = preprocess_image(image, 299, 299)

    #with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(batch_norm_decay=1)):
    #    _, endpoints = inception_resnet_v2.inception_resnet_v2(images, is_training=is_training)

    #with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    #    _, endpoints = inception_v3.inception_v3(images, is_training=is_training)

    #with slim.arg_scope(inception_v2.inception_v2_arg_scope(use_batch_norm=False)):
    with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
        _, endpoints = inception_v2.inception_v2(images, is_training=is_training)

    # 56 * 56 * 64
    feature_1 = endpoints['MaxPool_2a_3x3']

    # 28 * 28 * 192
    feature_2 = endpoints['Mixed_4e']

    # 28 * 28 * 256
    feature_3 = endpoints['Mixed_5a']

    # 14 * 14 * 576
    feature_4 = endpoints['Mixed_5b']

    # 7 * 7 * 1024
    feature_5 = endpoints['Mixed_5c']

    #pool_feature = tf.nn.avg_pool(conv_feature,
    #        ksize=[1, 7, 7, 1],
    #        strides=[1, 1, 1, 1],
    #        padding='VALID',
    #        name='inception_feature')

    return [feature_1, feature_2, feature_3, feature_4, feature_5]
    #return pool_feature
    #return endpoints['PreLogitsFlatten']
    #return endpoints['AuxLogits']


