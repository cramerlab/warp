
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import shutil

import tensorflow as tf
import numpy as np

import noise2particle_model
from estimator_v2 import EstimatorV2

parser = argparse.ArgumentParser()


_HEIGHT = 80
_WIDTH = 80

_BATCHSIZE = 128

_WEIGHT_DECAY = 1e-4
_MOMENTUM = 0.9

def input_fn(batch_size):
  
  volume_source = tf.random_normal([batch_size, _WIDTH, _HEIGHT, 1])
  volume_target = tf.random_normal([batch_size, _WIDTH, _HEIGHT, 1])
  volume_weight = tf.random_normal([batch_size, _WIDTH, _HEIGHT, 1])
  learning_rate = tf.constant(1e-8, shape=[batch_size])

  return {'volume_source' : volume_source, 
          'volume_target' : volume_target, 
          'volume_weight' : volume_weight, 
          'training_learning_rate' : learning_rate}, volume_source


def noise2particle_model_fn(features, labels, mode, params):

  network = noise2particle_model.noise2particle_generator()

  inputs = features["volume_source"]
  input_target = features["volume_target"]
  input_weight = features["volume_weight"]
  
  prediction = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)
  prediction = tf.identity(prediction, name='volume_predict')
  print(prediction.shape)
  

  predictions = {
      'volume_predict': prediction
  }
  
  export_outputs = {
      'prediction': tf.estimator.export.PredictOutput(predictions)
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, export_outputs=export_outputs, predictions=predictions)
  
  predictionodd = tf.layers.flatten(prediction)[0::2,:]
  predictioneven = tf.layers.flatten(prediction)[1::2,:]
  print(predictionodd.shape)
  
  allbutself = 1 - tf.to_float(tf.constant(np.eye(_BATCHSIZE // 2), shape=[_BATCHSIZE // 2, _BATCHSIZE // 2]))
  print(allbutself.shape)
  
  simodd = tf.matmul(predictionodd, tf.transpose(predictionodd, [1, 0])) / (_WIDTH * _HEIGHT) * allbutself
  simeven = tf.matmul(predictioneven, tf.transpose(predictioneven, [1, 0])) / (_WIDTH * _HEIGHT) * allbutself
  
  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  l2_loss = tf.losses.mean_squared_error(labels=input_target, predictions=prediction)
  l2_loss += tf.losses.mean_squared_error(labels=predictionodd, predictions=predictioneven) * 15
  l2_loss += (tf.reduce_mean(tf.square(simodd)) + tf.reduce_mean(tf.square(simeven))) * 6400
  tf.identity(l2_loss, name='l2_loss')
  print(predictionodd.shape)
  
  # Add weight decay to the loss.
  loss = l2_loss + _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  global_step = tf.train.get_or_create_global_step()
  
  learning_rate = features["training_learning_rate"][0]

  #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=_MOMENTUM)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.99)

  # Batch norm requires update ops to be added as a dependency to the train_op
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, global_step, name='train_momentum')


  trainings = {
      'loss': loss
  }
  
  export_outputs = {
      'training': tf.estimator.export.PredictOutput(trainings)
  }

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      export_outputs=export_outputs)


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  
  if os.path.isdir('noise2particle_model'):
    shutil.rmtree('noise2particle_model')

  # Set up a RunConfig to only save checkpoints once per training cycle.
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=999999).replace(session_config=config)
  
  supernet = EstimatorV2(model_fn=noise2particle_model_fn, model_dir='noise2particle_model', config=run_config,
                         params=
                         {
                             'batch_size': _BATCHSIZE,
                         })

  supernet.train_one_step(input_fn=lambda: input_fn(_BATCHSIZE))
	  
  feature_spec = {'volume_source': tf.placeholder(tf.float32, [None, _WIDTH, _HEIGHT, 1], name="volume_source"),
                  'volume_target': tf.placeholder(tf.float32, [None, _WIDTH, _HEIGHT, 1], name="volume_target"),
                  'volume_weight': tf.placeholder(tf.float32, [None, _WIDTH, _HEIGHT, 1], name="volume_weight"),
                  'training_learning_rate': tf.placeholder(tf.float32, [1], name="training_learning_rate")}
	  
  supernet.export_savedmodel(export_dir_base='noise2particle_model_export', 
                             serving_input_receiver_fn=tf.estimator.export.build_raw_serving_input_receiver_fn(features=feature_spec),
                             export_name='Noise2Particle_80',
                             as_text=False)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
