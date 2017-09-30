from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import tensorflow as tf
import data_input
import ActionNetAttentionHier as ActionNet
#import ActionNet
import cPickle
import numpy as np

FLAGS = tf.app.flags.FLAGS

def evaluate():
    
    videos_path = '/mnt/local_disk/video/UCF101/UCF-101-Train/test/' 
    #videos_path = '/mnt/local_disk/video/UCF101/UCF-101-Train/train/' 
    #videos_path = '/home/cypress/martin/triplet-loss/multi_gpu/test' 
    video_names = [os.path.join(videos_path, name) for name in os.listdir(videos_path)]
    video_nums = len(video_names) 
    print(video_nums)    
    with tf.Graph().as_default():
        
        videos_queue = tf.train.string_input_producer(video_names, num_epochs=1)
        # Get videos and labels from UCF101 dataset.
        video, label = data_input.read_video(videos_queue, is_training=False)
        
        # Build the inference graph.
        logits = ActionNet.inference(video, is_training=False)
        
        gpu_option = tf.GPUOptions(visible_device_list="0")
        config = tf.ConfigProto(gpu_options=gpu_option, allow_soft_placement=True)
        
        #config = tf.ConfigProto(device_count = {'GPU': 0})
        saver = tf.train.Saver()

        #variable_averages = tf.train.ExponentialMovingAverage(0.9999)
        #variables_to_restore = variable_averages.variables_to_restore()
        #saver = tf.train.Saver(variables_to_restore)
       
        with tf.Session(config=config) as sess:

            #sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, './checkpoint/model-21000')
            #saver.restore(sess, './checkpoint/model-wd-90000')

            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            num = 0
            true_count = 0
            try:
                while num < video_nums and not coord.should_stop():
                
                    real_label, pred_logits = sess.run([label, logits])
                    labels = np.argmax(pred_logits, axis=1)
                    pred_label = np.argmax(np.bincount(labels))
                    
                    if real_label == pred_label:
                        true_count += 1
                    num += 1
                    if num%100 == 0:
                        print('Video nums: {}'.format(num))
        
                # Compute precision @ 1.
                precision = true_count / video_nums

                print('precision @ 1 = %.3f' % (precision))
           
            except tf.errors.OutOfRangeError:
                print('Done.')
            
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)



def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()







