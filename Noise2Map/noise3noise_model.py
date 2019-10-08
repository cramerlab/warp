
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def leaky_relu_norm(inputs, is_training):

  """inputs = tf.layers.batch_normalization(inputs=inputs, 
                                         axis=1,
                                         momentum=_BATCH_NORM_DECAY, 
                                         epsilon=_BATCH_NORM_EPSILON, 
                                         center=True, 
                                         scale=True, 
                                         training=is_training, 
                                         fused=True)"""

  inputs = tf.nn.leaky_relu(inputs, alpha=0.2)
  return inputs
  

def leaky_relu(inputs):
  inputs = tf.nn.leaky_relu(inputs, alpha=0.2)
  return inputs


def conv3d_fixed_padding(inputs, filters, kernel_size, strides, dilation):

  return tf.layers.conv3d(inputs=inputs, 
                          filters=filters, 
                          kernel_size=kernel_size, 
                          strides=strides,
                          dilation_rate=dilation,
                          padding='SAME', 
                          use_bias=False,
                          data_format='channels_first',
                          kernel_initializer=tf.variance_scaling_initializer())


def deconv3d_fixed_padding(inputs, filters, kernel_size, strides):

  return tf.layers.conv3d_transpose(inputs=inputs, 
                                    filters=filters, 
                                    kernel_size=kernel_size, 
                                    strides=strides,
                                    padding='SAME', 
                                    use_bias=False,
                                    data_format='channels_first',
                                    kernel_initializer=tf.variance_scaling_initializer())


def maxpool3d(inputs):

  return tf.layers.max_pooling3d(inputs, 2, 2, padding='SAME', data_format='channels_first')
  
  
def upsample(inputs):

  return tf.keras.layers.UpSampling3D(size=[2, 2, 2], data_format='channels_first')(inputs)



def noise3noise_generator():

  def model(inputs, is_training):
  
    training_batchnorm = True
    concat_axis = 1
    encoderkernels = 64
    decoderkernels = 128
  
    # input is channels_last, but we need channels_first
    inputs = tf.transpose(inputs, [0, 4, 1, 2, 3])
    
    encode0 = leaky_relu_norm(conv3d_fixed_padding(inputs, encoderkernels, 3, 1, 1), is_training)               # 64
    encode1 = leaky_relu_norm(conv3d_fixed_padding(encode0, encoderkernels, 3, 1, 1), is_training)              # 64
    pool1 = maxpool3d(encode1)    # 32
    print(pool1.shape)
    
    encode2 = leaky_relu_norm(conv3d_fixed_padding(pool1, encoderkernels, 3, 1, 1), is_training)              # 32
    pool2 = maxpool3d(encode2)    # 16
    print(pool2.shape)
    
    encode3 = leaky_relu_norm(conv3d_fixed_padding(pool2, encoderkernels, 3, 1, 1), is_training)              # 16
    pool3 = maxpool3d(encode3)    # 8
    print(pool3.shape)
    
    encode4 = leaky_relu_norm(conv3d_fixed_padding(pool3, encoderkernels, 3, 1, 1), is_training)              # 8
    pool4 = maxpool3d(encode4)    # 4
    print(pool4.shape)
    
    encode5 = leaky_relu_norm(conv3d_fixed_padding(pool4, encoderkernels, 3, 1, 1), is_training)              # 4
    
    deconv4 = upsample(encode5)    # 8
    concat4 = tf.concat([deconv4, pool3], concat_axis)                  # 8
    decode4a = leaky_relu_norm(conv3d_fixed_padding(concat4, decoderkernels, 3, 1, 1), is_training)     # 8
    decode4b = leaky_relu_norm(conv3d_fixed_padding(decode4a, decoderkernels, 3, 1, 1), is_training)    # 8
    print(decode4b.shape)
    
    deconv3 = upsample(decode4b)    # 16
    concat3 = tf.concat([deconv3, pool2], concat_axis)                  # 16
    decode3a = leaky_relu_norm(conv3d_fixed_padding(concat3, decoderkernels, 3, 1, 1), is_training)     # 16
    decode3b = leaky_relu_norm(conv3d_fixed_padding(decode3a, decoderkernels, 3, 1, 1), is_training)    # 16
    print(decode3b.shape)
    
    deconv2 = upsample(decode3b)    # 32
    concat2 = tf.concat([deconv2, pool1], concat_axis)                  # 32
    decode2a = leaky_relu_norm(conv3d_fixed_padding(concat2, decoderkernels, 3, 1, 1), is_training)     # 32
    decode2b = leaky_relu_norm(conv3d_fixed_padding(decode2a, decoderkernels, 3, 1, 1), is_training)    # 32
    print(decode2b.shape)
    
    deconv1 = upsample(decode2b)    # 64
    concat1 = tf.concat([deconv1, inputs], concat_axis)                  # 64
    decode1a = leaky_relu(conv3d_fixed_padding(concat1, (decoderkernels*2)//3, 3, 1, 1))     # 64
    print(decode1a.shape)
    decode1b = leaky_relu(conv3d_fixed_padding(decode1a, decoderkernels//3, 3, 1, 1))    # 64
    print(decode1b.shape)
    
    decode1c = conv3d_fixed_padding(decode1b, 1, 3, 1, 1)    # 64
    
    # and back to channels_last
    inputs = tf.transpose(decode1c, [0, 2, 3, 4, 1])
    #inputs = decode1c
    
    return inputs

  return model