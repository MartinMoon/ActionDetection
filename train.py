from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import data_input
import ActionNetAttentionHier as ActionNet
import cPickle
import numpy as np

FLAGS = tf.app.flags.FLAGS

def load_pre_trained_vars(checkpoint_file):

    var_list = tf.contrib.framework.list_variables(checkpoint_file)

    assignment_map = {}
    for key, shape in var_list:
        if shape==[]:
            continue
        else:
            assignment_map[key] = 'feature_net/'+key

    tf.contrib.framework.init_from_checkpoint(checkpoint_file, assignment_map)

def train():
    "Train UCF dataset"

    videos_path = '/mnt/local_disk/video/UCF101/UCF-101-Train/train/'
    video_names = [os.path.join(videos_path, name) for name in os.listdir(videos_path)]
    video_nums = len(video_names)

    batch_size = 4
    start_learning_rate = 0.00001

    with tf.Graph().as_default():

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

	learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 4000, 0.96, staircase=True)

        opt = tf.train.AdamOptimizer(learning_rate)

        # Get videos and labels from UCF101 dataset.
        videos, labels = data_input.inputs(videos_path, batch_size, is_training=True)

        # Build the inference graph.
        logits = ActionNet.inference(videos, batch_size)

        # Build the loss graph.
        _ = ActionNet.loss(logits, labels)

        # Assemble all the losses for the current tower only
        losses = tf.get_collection('losses')

        # Calculate the total loss
        total_loss = tf.add_n(losses, name='total_loss')

        tf.summary.scalar('loss', total_loss)

        train_op = opt.minimize(total_loss)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops.append(train_op)
        update_op = tf.group(*update_ops)

        train_tensor = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')

        merged = tf.summary.merge_all()

        gpu_option = tf.GPUOptions(visible_device_list="1")
        config = tf.ConfigProto(gpu_options=gpu_option, allow_soft_placement=True)

        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:

            checkpoint_file = '/home/cypress/Documents/mm/ActionDetection/model/pretrain/inception/inception_v2.ckpt'
            load_pre_trained_vars(checkpoint_file)
            sess.run(tf.global_variables_initializer())

            train_writer = tf.summary.FileWriter('./summary', sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            iter_num = 0
            try:
                while not coord.should_stop():
                    loss_, summary = sess.run([train_tensor, merged])

                    print("Loss at {}-th iteration: {}".format(iter_num, loss_))

                    if iter_num%10==0:
                        summary = sess.run(merged)
                        train_writer.add_summary(summary, iter_num)

                    if iter_num%4000==0:
                        saver.save(sess, './checkpoint/model', global_step=iter_num)
                    iter_num += 1

            except tf.errors.OutOfRangeError:
                print('Done.')

            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)



def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()







