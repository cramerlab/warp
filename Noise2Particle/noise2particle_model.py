
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def leaky_relu_norm(inputs, is_training):

  inputs = tf.layers.batch_normalization(inputs=inputs, 
                                         axis=1,
                                         momentum=_BATCH_NORM_DECAY, 
                                         epsilon=_BATCH_NORM_EPSILON, 
                                         center=True, 
                                         scale=True, 
                                         training=is_training, 
                                         fused=True)

  inputs = tf.nn.leaky_relu(inputs, alpha=0.1)
  return inputs

def leaky_relu(inputs):

  inputs = tf.nn.leaky_relu(inputs, alpha=0.1)
  return inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, dilation):

  return tf.layers.conv2d(inputs=inputs, 
                          filters=filters, 
                          kernel_size=kernel_size, 
                          strides=strides,
                          dilation_rate=dilation,
                          padding='SAME', 
                          use_bias=False,
                          data_format='channels_first',
                          kernel_initializer=tf.variance_scaling_initializer())


def deconv2d_fixed_padding(inputs, filters, kernel_size, strides):

  return tf.layers.conv2d_transpose(inputs=inputs, 
                                    filters=filters, 
                                    kernel_size=kernel_size, 
                                    strides=strides,
                                    padding='SAME', 
                                    use_bias=False,
                                    data_format='channels_first',
                                    kernel_initializer=tf.variance_scaling_initializer())


def maxpool2d(inputs):

  return tf.layers.max_pooling2d(inputs, 2, 2, padding='SAME', data_format='channels_first')
  
  
def upsample(inputs):

  return tf.keras.layers.UpSampling2D(size=[2, 2], data_format='channels_first')(inputs)



def noise2particle_generator():

  def model(inputs, is_training):
  
    training_batchnorm = True
    concat_dim = 1
    
    channels_encode = 32
    channels_decode = 64
  
    # input is channels_last, but we need channels_first
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    
    encode0 = leaky_relu_norm(conv2d_fixed_padding(inputs, channels_encode, 3, 1, 1), is_training)               # 80
    encode1 = leaky_relu_norm(conv2d_fixed_padding(encode0, channels_encode, 3, 1, 1), is_training)              # 80
    pool1 = maxpool2d(encode1)    # 128
    
    encode2 = leaky_relu_norm(conv2d_fixed_padding(pool1, channels_encode, 3, 1, 1), is_training)              # 40
    pool2 = maxpool2d(encode2)    # 64
    
    encode3 = leaky_relu_norm(conv2d_fixed_padding(pool2, channels_encode, 3, 1, 1), is_training)              # 20
    pool3 = maxpool2d(encode3)    # 32
    
    encode4 = leaky_relu_norm(conv2d_fixed_padding(pool3, channels_encode, 3, 1, 1), is_training)              # 10
    pool4 = maxpool2d(encode4)    # 16
    
    encode5 = leaky_relu_norm(conv2d_fixed_padding(pool4, channels_encode, 3, 1, 1), is_training)              # 5
    
    deconv4 = upsample(encode5)    # 10
    concat4 = tf.concat([deconv4, pool3], concat_dim)                  # 10
    decode4a = leaky_relu_norm(conv2d_fixed_padding(concat4, channels_decode, 3, 1, 1), is_training)     # 10
    decode4b = leaky_relu_norm(conv2d_fixed_padding(decode4a, channels_decode, 3, 1, 1), is_training)    # 10
    
    deconv3 = upsample(decode4b)    # 20
    concat3 = tf.concat([deconv3, pool2], concat_dim)                  # 20
    decode3a = leaky_relu_norm(conv2d_fixed_padding(concat3, channels_decode, 3, 1, 1), is_training)     # 20
    decode3b = leaky_relu_norm(conv2d_fixed_padding(decode3a, channels_decode, 3, 1, 1), is_training)    # 20
    
    deconv2 = upsample(decode3b)    # 40
    concat2 = tf.concat([deconv2, pool1], concat_dim)                  # 40
    decode2a = leaky_relu_norm(conv2d_fixed_padding(concat2, channels_decode, 3, 1, 1), is_training)     # 40
    decode2b = leaky_relu_norm(conv2d_fixed_padding(decode2a, channels_decode, 3, 1, 1), is_training)    # 40
    
    deconv1 = upsample(decode2b)    # 80
    concat1 = tf.concat([deconv1, inputs], concat_dim)                  # 80
    decode1a = leaky_relu(conv2d_fixed_padding(concat1, 48, 3, 1, 1))     # 80
    decode1b = leaky_relu(conv2d_fixed_padding(decode1a, 24, 3, 1, 1))    # 80
    
    decode1c = conv2d_fixed_padding(decode1b, 1, 3, 1, 1)    # 80
    
    # and back to channels_last
    inputs = tf.transpose(decode1c, [0, 2, 3, 1])
    
    return inputs

  return model