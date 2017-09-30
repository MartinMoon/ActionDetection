from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import config
from inception.inception_net_hier import feature_net

height = config.height
width = config.width
channels = config.channels
max_length = config.max_length
n_features = config.n_features
lstm_size = config.lstm_size
n_classes = config.n_classes

def _variable_init(name, initial, dtype=tf.float32, wd=None):
    """Create a variable on CPU, for efficiency of multi-GPU training
       A weight decay is added if specified
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, initializer=initial, dtype=dtype)

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var

def _batch_norm_wrapper(inputs, is_training, decay=0.999):

    epsilon = tf.constant(1e-3)

    with tf.device('/cpu:0'):
        scale = tf.get_variable('scale', initializer=tf.ones([inputs.get_shape()[-1]]))
        beta =  tf.get_variable('beta', initializer=tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.get_variable('pop_mean', initializer=tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var  = tf.get_variable('pop_var',  initializer=tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if len(inputs.get_shape().as_list()) == 4:
            batch_mean, batch_var = tf.nn.moments(inputs,[0, 1, 2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])

        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_var)

        return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)

    else:

        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

def f_init_cell(feature_frames, n_features, is_training=True):
    """Feature frames: Batch_size*max_length*n_feature """

    with tf.variable_scope('INIT_CELL_MLP_1') as scope:

        kernel_init = tf.truncated_normal(shape=[n_features, 1024], stddev=0.01)
        #kernel_init = tf.truncated_normal(shape=[n_features, 1024])/tf.sqrt(n_features/2)

        weights = _variable_init('kernel', initial=kernel_init, wd=0.00001)
        biases = _variable_init('bias', initial=tf.constant(0.001, shape=[1024]))
        bias = tf.matmul(feature_frames, weights) + biases
        bn = _batch_norm_wrapper(bias, is_training)
        fc_1 = tf.nn.relu(bn, name=scope.name)

    with tf.variable_scope('INIT_CELL_MLP_2') as scope:

        kernel_init = tf.truncated_normal(shape=[1024, lstm_size], stddev=0.01)
        #kernel_init = tf.truncated_normal(shape=[1024, lstm_size])/tf.sqrt(1024/2)
        weights = _variable_init('kernel', initial=kernel_init, wd=0.00001)
        biases = _variable_init('bias', initial=tf.constant(0.001, shape=[lstm_size]))

        bias = tf.matmul(fc_1, weights) + biases
        bn = _batch_norm_wrapper(bias, is_training)
        init_cell = tf.nn.tanh(bn, name=scope.name)

    return init_cell

def f_init_state(feature_frames, n_features, is_training=True):
    """Feature frames: Batch_size*max_length*n_feature """

    with tf.variable_scope('INIT_STATE_MLP_1') as scope:

        kernel_init = tf.truncated_normal(shape=[n_features, 1024], stddev=0.01)
        #kernel_init = tf.truncated_normal(shape=[n_features, 1024])/tf.sqrt(n_features/2)
        weights = _variable_init('kernel', initial=kernel_init, wd=0.00001)
        biases = _variable_init('bias', initial=tf.constant(0.001, shape=[1024]))
        bias = tf.matmul(feature_frames, weights) + biases
        bn = _batch_norm_wrapper(bias, is_training)
        fc_1 = tf.nn.relu(bn, name=scope.name)

    with tf.variable_scope('INIT_STATE_MLP_2') as scope:

        kernel_init = tf.truncated_normal(shape=[1024, lstm_size], stddev=0.01)
        #kernel_init = tf.truncated_normal(shape=[1024, lstm_size])/tf.sqrt(1024/2)
        weights = _variable_init('kernel', initial=kernel_init, wd=0.00001)
        biases = _variable_init('bias', initial=tf.constant(0.001, shape=[lstm_size]))
        bias = tf.matmul(fc_1, weights) + biases
        bn = _batch_norm_wrapper(bias, is_training)
        init_state = tf.add(tf.matmul(bn, weights), biases)

    return init_state


def f_att(hidden_state, feature_input, height, width, n_features):
    """Generate the attention map
       Here is the visual attention, which determines which part to look
       given one input image (more precisely, feature image)
        
       Input:
        hidden_state:  n_batch * lstm_size
        feature_input: n_batch * n_features

       Output:
        att_map is a normalized vector with size height*width
    """
    with tf.variable_scope('Attention_function') as scope:
        
        kernel_init_1 = tf.truncated_normal(shape=[n_features, 512], stddev=0.01)
        #kernel_init_1 = tf.truncated_normal(shape=[n_features, 512])/tf.sqrt(n_features/2)
        weights_1 = _variable_init('kernel_1', initial=kernel_init_1, wd=0.00001)
        
        kernel_init_2 = tf.truncated_normal(shape=[lstm_size, 512], stddev=0.01)
        #kernel_init_2 = tf.truncated_normal(shape=[lstm_size, 512])/tf.sqrt(lstm_size/2)
        weights_2 = _variable_init('kernel_2', initial=kernel_init_2, wd=0.00001)
        biases_12 = _variable_init('bias_12', initial=tf.constant(0.001, shape=[512])) 

        kernel_init_3 = tf.truncated_normal(shape=[512, height*width], stddev=0.01)
        #kernel_init_3 = tf.truncated_normal(shape=[512, height*width])/tf.sqrt(512/2)
        weights_3 = _variable_init('kernel_3', initial=kernel_init_3, wd=0.00001)
        biases_3 = _variable_init('bias_3', initial=tf.constant(0.001, shape=[height*width])) 
        
        f_att_input  = tf.matmul(feature_input,  weights_1)
        f_att_output = tf.matmul(hidden_state, weights_2)

        att_map = tf.matmul(tf.nn.tanh(tf.add(f_att_input, f_att_output) + biases_12), weights_3) + biases_3
        att_map = tf.nn.softmax(att_map, name='spatial_attention_map')
        
        """
        #kernel_init = tf.truncated_normal(shape=[lstm_size+n_features, height*width], stddev=0.01)
        kernel_init = tf.truncated_normal(shape=[lstm_size+n_features, height*width])/tf.sqrt((lstm_size+n_features)/2)
        weights = _variable_init('kernel', initial=kernel_init, wd=0.0001)
        biases = _variable_init('bias', initial=tf.constant(0.001, shape=[height*width])) 
        
        feature_attention = tf.concat([hidden_state, feature_input], axis=1) 
        feature_attention = tf.matmul(feature_attention, weights) + biases

        att_map = tf.nn.softmax(feature_attention)
        """
    return att_map

def batch_matmul(x, y):
    
    batch_x, seq_x, fea_x = x.get_shape().as_list()
    w_y = y.get_shape().as_list()[-1]
    
    x = tf.reshape(x, [-1, fea_x])
    res = tf.matmul(x, y)
    res = tf.reshape(res, [-1, seq_x, w_y])
    
    return res

def attention_wrapper(input_frames, output_frames, n_features):
    """Generate the output with attention

       input_frames: batch_size * n_sequence * n_features
       output_frames: batch_size * n_sequence * lstm_size
    """
    
    with tf.variable_scope('Sequence_attention_function') as seq_att_scope:

        kernel_init_1 = tf.truncated_normal(shape=[n_features, 512], stddev=0.01)
        #kernel_init_1 = tf.truncated_normal(shape=[n_features, 512])/tf.sqrt(n_features/2)
        weights_1 = _variable_init('kernel_1', initial=kernel_init_1, wd=0.00001)
        
        kernel_init_2 = tf.truncated_normal(shape=[lstm_size, 512], stddev=0.01)
        #kernel_init_2 = tf.truncated_normal(shape=[lstm_size, 512])/tf.sqrt(lstm_size/2)
        weights_2 = _variable_init('kernel_2', initial=kernel_init_2, wd=0.00001)
        
        kernel_init_3 = tf.truncated_normal(shape=[512, 1], stddev=0.01)
        #kernel_init_3 = tf.truncated_normal(shape=[512, 1])/tf.sqrt(512/2)
        weights_3 = _variable_init('kernel_3', initial=kernel_init_3, wd=0.00001)
        
        #f_att_input  = tf.scan(lambda a, x: tf.matmul(x, weights_1), input_frames)
        #f_att_output = tf.scan(lambda a, x: tf.matmul(x, weights_2), output_frames)
        f_att_input  = batch_matmul(input_frames, weights_1)
        f_att_output = batch_matmul(output_frames, weights_2)

        att_map = batch_matmul(tf.nn.relu(tf.add(f_att_input, f_att_output)), weights_3)
        att_map = tf.reshape(att_map, [-1, max_length])
        att_map = tf.nn.softmax(att_map, name='sequence_attention_map')
        
        feature_attention = tf.transpose(output_frames, [2, 0, 1])
        feature_attention = tf.multiply(feature_attention, att_map)
        feature_attention = tf.reduce_sum(feature_attention, 2)
        feature_attention = tf.transpose(feature_attention, [1, 0])
    
    return feature_attention


def featureNet(video_frames, is_training=True):
    """Infer the frame features from input video

    Parameters:
        video_frames:
            None * max_length * height * width * channels

    Return:
        feature_frames:
            None * max_length * n_features
    """
    with tf.variable_scope('feature_net') as scope:

        # (max_length*batch_size, height, width, channels)
        input_frames = tf.reshape(video_frames, [-1, height, width, channels])
        feature_frames = feature_net(input_frames, is_training=is_training)
        #print(feature_frames.get_shape())
        #feature_frames = tf.reshape(feature_frames, [-1, max_length, 7, 7, n_features])

    return feature_frames

def sequenceNet(feature_frames, batch_size, is_training=True):
    """Infer the motion class based on feature frames

    Parameters:
        feature_frames:
            Feature tensor with shape
            None * max_length * 7 * 7 * n_features
    """
    _, h, w, n_features = feature_frames.get_shape().as_list()

    feature_frames_pool = tf.nn.avg_pool(feature_frames,
            ksize=[1, h, w, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='pool_feature')

    feature_frames_pool = tf.reshape(feature_frames_pool, [-1, max_length, n_features])
    feature_frames_pool = tf.reduce_mean(feature_frames_pool, 1)

    feature_frames = tf.reshape(feature_frames, [-1, max_length, h, w, n_features])
    
    feature_frames_flatten = tf.reshape(feature_frames, [-1, max_length, h*w, n_features])
    feature_frames_flatten_mean = tf.reduce_mean(feature_frames_flatten, 2)
     
    with tf.variable_scope('Sequence') as scope:

        with tf.variable_scope('LSTM_cell') as lstm_scope:

            # Define a 3-layer LSTM
            #basic_cell_1 = rnn.BasicLSTMCell(lstm_size)
            basic_cell_1 = rnn.GRUCell(lstm_size)
            #cell_1 = rnn.DropoutWrapper(basic_cell_1, output_keep_prob=0.9)
            basic_cell_2 = rnn.GRUCell(lstm_size)
            #cell_2 = rnn.DropoutWrapper(basic_cell_2, output_keep_prob=0.9)
            basic_cell_3 = rnn.GRUCell(lstm_size)
            #cell_3 = rnn.DropoutWrapper(basic_cell_3, output_keep_prob=0.9)
            
            cell_list = [basic_cell_1, basic_cell_2, basic_cell_3]
            #cell_list = [cell_1, cell_2, cell_3]
            cell = tf.contrib.rnn.MultiRNNCell(cell_list)

            # Get the initial state of the LSTM from an auxilary MLP
	    # init_layer_1
            with tf.variable_scope('Init_layer_1'):
                init_cell_1 = f_init_cell(feature_frames_pool, n_features, is_training)
	        init_state_1 = f_init_state(feature_frames_pool, n_features, is_training)
            # init_layer_2
            with tf.variable_scope('Init_layer_2'):
                init_cell_2 = f_init_cell(feature_frames_pool, n_features, is_training)
	        init_state_2 = f_init_state(feature_frames_pool, n_features, is_training)
            # init_layer_3
            with tf.variable_scope('Init_layer_3'):
                init_cell_3 = f_init_cell(feature_frames_pool, n_features, is_training)
	        init_state_3 = f_init_state(feature_frames_pool, n_features, is_training)

            #state = (tf.contrib.rnn.LSTMStateTuple(init_cell_1, init_state_1),
            #        tf.contrib.rnn.LSTMStateTuple(init_cell_2, init_state_2),
            #        tf.contrib.rnn.LSTMStateTuple(init_cell_3, init_state_3))

            state = (init_state_1, init_state_2, init_state_3)
            
            inputs = []; outputs = []
            for i in range(max_length):

                if i > 0:
                    lstm_scope.reuse_variables()
                
                #feature_attention_input = tf.concat([state[2], feature_frames_flatten_mean[:, i, :]], axis=1)
                attention_map = f_att(state[2], feature_frames_flatten_mean[:, i, :], h, w, n_features)

                feature_attention = tf.transpose(feature_frames_flatten[:, i, :, :], [2, 0, 1])
                feature_attention = tf.multiply(feature_attention, attention_map)
                feature_attention = tf.reduce_sum(feature_attention, 2)
                feature_attention = tf.transpose(feature_attention, [1, 0])

                output, state = cell(feature_attention, state)
                
                inputs.append(feature_attention)
                outputs.append(output)
            
        inputs  = tf.transpose(inputs,  [1, 0, 2])
        outputs = tf.transpose(outputs, [1, 0, 2])
        
        output_end = attention_wrapper(inputs, outputs, n_features)
    
    return output_end

def fusionNet(feature_frames, batch_size=1, is_training=True):

    feature_frames_1, feature_frames_2, feature_frames_3, feature_frames_4, feature_frames_5 = feature_frames

    num_fusion_channels = len(feature_frames)

    with tf.variable_scope('fusion_net'):
        
        """
        with tf.variable_scope('sequence_1'):
            feature_1 = sequenceNet(feature_frames_1, batch_size, is_training)
                 
        with tf.variable_scope('sequence_2'):
            feature_2 = sequenceNet(feature_frames_2, batch_size, is_training)
        
        with tf.variable_scope('sequence_3'):
            feature_3 = sequenceNet(feature_frames_3, batch_size, is_training)
        """
        with tf.variable_scope('sequence_4'):
            feature_4 = sequenceNet(feature_frames_4, batch_size, is_training)
        """
        with tf.variable_scope('sequence_5'):
            feature_5 = sequenceNet(feature_frames_5, batch_size, is_training)
        """
        #feature_combine = tf.concat([feature_1, feature_2, feature_3], axis=1)
        """
        with tf.variable_scope('LSTM_fc1') as lstm_fc1_scope:

            kernel_init = tf.truncated_normal(shape=[lstm_size, 512], stddev=0.01)
            weights = _variable_init('kernel', initial=kernel_init, wd=0.0001)
            biases = _variable_init('bias', initial=tf.constant(0.001, shape=[512]))

            bias = tf.matmul(feature_1, weights) + biases
            bn = _batch_norm_wrapper(bias, is_training)
            fc_1 = tf.nn.relu(bn, name=lstm_fc1_scope.name)
                
        with tf.variable_scope('LSTM_fc2') as lstm_fc2_scope:

            kernel_init = tf.truncated_normal(shape=[lstm_size, 512], stddev=0.01)
            weights = _variable_init('kernel', initial=kernel_init, wd=0.00001)
            biases = _variable_init('bias', initial=tf.constant(0.001, shape=[512]))

            bias = tf.matmul(feature_2, weights) + biases
            bn = _batch_norm_wrapper(bias, is_training)
            fc_2 = tf.nn.relu(bn, name=lstm_fc2_scope.name)
         
        with tf.variable_scope('LSTM_fc3') as lstm_fc3_scope:

            kernel_init = tf.truncated_normal(shape=[lstm_size, 512], stddev=0.01)
            #kernel_init = tf.truncated_normal(shape=[lstm_size, 512])/tf.sqrt(lstm_size/2)
            
            weights = _variable_init('kernel', initial=kernel_init, wd=0.00001)
            biases = _variable_init('bias', initial=tf.constant(0.001, shape=[512]))

            bias = tf.matmul(feature_3, weights) + biases
            bn = _batch_norm_wrapper(bias, is_training)
            fc_3 = tf.nn.relu(bn, name=lstm_fc3_scope.name)
        """

        with tf.variable_scope('LSTM_fc4') as lstm_fc4_scope:

            kernel_init = tf.truncated_normal(shape=[lstm_size, 512], stddev=0.01)
            #kernel_init = tf.truncated_normal(shape=[lstm_size, 512])/tf.sqrt(lstm_size/2)
            weights = _variable_init('kernel', initial=kernel_init, wd=0.00001)
            biases = _variable_init('bias', initial=tf.constant(0.001, shape=[512]))

            bias = tf.matmul(feature_4, weights) + biases
            bn = _batch_norm_wrapper(bias, is_training)
            fc_4 = tf.nn.relu(bn, name=lstm_fc4_scope.name)
        """
        with tf.variable_scope('LSTM_fc5') as lstm_fc5_scope:

            kernel_init = tf.truncated_normal(shape=[lstm_size, 512], stddev=0.01)
            #kernel_init = tf.truncated_normal(shape=[lstm_size, 512])/tf.sqrt(lstm_size/2)
            weights = _variable_init('kernel', initial=kernel_init, wd=0.00001)
            biases = _variable_init('bias', initial=tf.constant(0.001, shape=[512]))

            bias = tf.matmul(feature_5, weights) + biases
            bn = _batch_norm_wrapper(bias, is_training)
            fc_5 = tf.nn.relu(bn, name=lstm_fc5_scope.name)
        """
        #feature_combine = tf.concat([fc_2, fc_3, fc_4, fc_5], axis=1)
        feature_combine = tf.concat([fc_4], axis=1)

        with tf.variable_scope('LSTM_combine_fc1') as lstm_combine_fc1_scope:

            #kernel_init = tf.truncated_normal(shape=[512*num_fusion_channels, 1024], stddev=0.01)
            kernel_init = tf.truncated_normal(shape=[512, 1024], stddev=0.01)
            
            #kernel_init = tf.truncated_normal(shape=[512*3, 1024])/tf.sqrt(512*3/2)
            weights = _variable_init('kernel', initial=kernel_init, wd=0.00001)
            biases = _variable_init('bias', initial=tf.constant(0.001, shape=[1024]))

            bias = tf.matmul(feature_combine, weights) + biases
            bn = _batch_norm_wrapper(bias, is_training)
            fc_combine = tf.nn.relu(bn, name=lstm_combine_fc1_scope.name)
            
            #if drop_out:
            #fc_combine = tf.nn.dropout(fc_combine, keep_prob=0.9)

        with tf.variable_scope('LSTM_softmax_output') as lstm_softmax_scope:

            kernel_init = tf.truncated_normal(shape=[1024, n_classes], stddev=0.01)
            #kernel_init = tf.truncated_normal(shape=[1024, n_classes])/tf.sqrt(1024/2)
            weights = _variable_init('kernel', initial=kernel_init, wd=0.00001)
            biases = _variable_init('bias', initial=tf.constant(0.001, shape=[n_classes]))

            logits = tf.add(tf.matmul(fc_combine, weights), biases)

    return logits


def inference(video_frames, batch_size=1, is_training=True, visual_mode=False):

    """Have to set is_training as False to frozen the BN layers!!!
    """
    if is_training == True:
        feature_frames = featureNet(video_frames, is_training)
        #logits = sequenceNet(feature_frames, batch_size, is_training)
        logits = fusionNet(feature_frames, batch_size, is_training)
    else:
        if visual_mode==True:
            video_frames = tf.reshape(video_frames, [batch_size, max_length, height, width, channels])
        else:
            video_frames = tf.reshape(video_frames, [batch_size*5, max_length, height, width, channels])
        
        feature_frames = featureNet(video_frames, is_training)
        logits = fusionNet(feature_frames, batch_size, is_training)

    return logits

def loss(logits, labels):
  """Set the loss function of the model
     total_loss = cross_entropy + reguarlization_loss
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')
  #return cross_entropy_mean

if __name__ == '__main__':

    with tf.Graph().as_default():

        video_frames = tf.placeholder(tf.float32, shape=[None, max_length, height, width, channels], name='input_videos')
        batch_size = tf.placeholder(tf.int32, name='batch_size')

        logits = inference(video_frames, batch_size)

        gpu_option = tf.GPUOptions(visible_device_list="1")
        config = tf.ConfigProto(gpu_options=gpu_option)

        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())

            sequence = np.ones((4, 25, height, width, 3))
            output_ = sess.run(logits, feed_dict={video_frames:sequence, batch_size:4})
            print(output_.shape)





