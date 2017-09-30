from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import config

def read_video(videoname_queue, is_training):
    """Reads and parses examples from UCF101 data files.

    Returns:
        An object representing a single video,
    """
    width=0; height=0; channels=0; label=0

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(videoname_queue)

    features = tf.parse_single_example(
            serialized_example,
            features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'frames': tf.FixedLenFeature([], tf.string)
            })

    # Here featuers['height'] is a tensor, need to cast to a scalar
    height = tf.cast(features['height'], tf.int32)
    width  = tf.cast(features['width'], tf.int32)
    channels = tf.cast(features['channels'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    frames = tf.decode_raw(features['frames'], tf.uint8)
    frames_tensor = tf.reshape(frames, (-1, height, width, channels))

    # Pre-process the frame tensor
    #frames_tensor = tf.cast(frames_tensor, tf.float32)

    #frames_tensor = tf.image.resize_images(frames_tensor, [config.height, config.width])
    frames_tensor = tf.image.convert_image_dtype(frames_tensor, tf.float32)

    if is_training:
        
        frames_tensor = tf.image.resize_images(frames_tensor, [299, 299])
        n_frames = tf.shape(frames_tensor)[0]
        frames_tensor = tf.random_crop(frames_tensor, [n_frames, 244, 244, 3])
        frames_tensor = tf.image.resize_images(frames_tensor, [config.height, config.width])

    else:

        #frames_tensor = tf.map_fn(lambda img: tf.image.central_crop(img, 0.9), frames_tensor)
        frames_tensor = tf.image.resize_images(frames_tensor, [299, 299])

        frames_tensor_1 = tf.map_fn(lambda img: tf.slice(img, [0,  0,  0], [224, 224, 3]), frames_tensor)
        frames_tensor_1 = tf.image.resize_images(frames_tensor_1, [config.height, config.width])
        
        frames_tensor_2 = tf.map_fn(lambda img: tf.slice(img, [0,  55, 0], [224, 224, 3]), frames_tensor)
        frames_tensor_2 = tf.image.resize_images(frames_tensor_2, [config.height, config.width])
        
        frames_tensor_3 = tf.map_fn(lambda img: tf.slice(img, [55, 0,  0], [224, 224, 3]), frames_tensor)
        frames_tensor_3 = tf.image.resize_images(frames_tensor_3, [config.height, config.width])
        
        frames_tensor_4 = tf.map_fn(lambda img: tf.slice(img, [55, 55, 0], [224, 224, 3]), frames_tensor)
        frames_tensor_4 = tf.image.resize_images(frames_tensor_4, [config.height, config.width])
        
        frames_tensor_5 = tf.map_fn(lambda img: tf.slice(img, [27, 27, 0], [224, 224, 3]), frames_tensor)
        frames_tensor_5 = tf.image.resize_images(frames_tensor_5, [config.height, config.width])
        
        frames_tensor = tf.concat([frames_tensor_1,
                                    frames_tensor_2,
                                    frames_tensor_3,
                                    frames_tensor_4,
                                    frames_tensor_5], axis=0)
    #resize after convert into float type seems will bring some problem, because will cast into int
    #frames_tensor = tf.image.resize_images(frames_tensor, [config.height, config.width])

    frames_tensor = tf.subtract(frames_tensor, 0.5)
    frames_tensor = tf.multiply(frames_tensor, 2)

    return frames_tensor, label

def _generate_video_and_label_batch(image, label, min_queue_examples,
        batch_size, shuffle):
    """Construct a queued batch of videos and labels."""
    pass

def inputs(videos_path, batch_size, is_training=True):
    """Construct inputs for UCF101 training."""

    video_names = [os.path.join(videos_path, name) for name in os.listdir(videos_path)]

    # Create a queue that produces the videonames to read
    # Notice here the queue can be fetched many times, since I do not set the num_epoch parameter.
    videoname_queue = tf.train.string_input_producer(video_names)

    # Read videos from files in the videoname queue.
    video, label = read_video(videoname_queue, is_training)
    
    if is_training == True:
        n_seqs = 25
    else:
        n_seqs = 125

    videos, labels = tf.train.shuffle_batch(
            [video, label],
            batch_size=batch_size,
            num_threads=4,
            capacity=100 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=batch_size,
            shapes = [(n_seqs, config.height, config.width, 3), ()],
            name = 'train_op')

    return videos, labels

if __name__ == '__main__':
    import sys


    with tf.Graph().as_default():

        videos_path = sys.argv[1]
        #video_names = [os.path.join(videos_path, name) for name in os.listdir(videos_path)]

        videos, labels = inputs(videos_path, 4)
        train_op = labels + 1

        gpu_option = tf.GPUOptions(visible_device_list="1")
        config = tf.ConfigProto(gpu_options=gpu_option, allow_soft_placement=True)

        with tf.Session(config=config) as sess:
        # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not coord.should_stop():
                    label = sess.run([train_op])
                    print(label)

            except tf.errors.OutOfRangeError:
                print('Done.')

            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)



