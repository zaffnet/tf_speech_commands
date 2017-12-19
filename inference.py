from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
import input_data
import models
from tensorflow.python.framework import graph_util

FLAGS = None

class InferenceProcessor(object):
  
  def __init__(self, data_dir, model_settings):
    self.data_dir = data_dir
    self.prepare_data_index()
    self.prepare_processing_graph(model_settings)

  def prepare_data_index(self):
    self.data_index = []
    search_path = os.path.join(self.data_dir, '*.wav')
    for wav_path in gfile.Glob(search_path):
      self.data_index.append(wav_path)
    self.data_index = np.array(self.data_index)

  def prepare_processing_graph(self, model_settings):
    desired_samples = model_settings['desired_samples']
    self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(self.wav_filename_placeholder_)

    wav_decoder = contrib_audio.decode_wav(
        wav_loader, desired_channels=1, desired_samples=desired_samples)
    self.wav_decoder_ = wav_decoder

    spectrogram = contrib_audio.audio_spectrogram(
        wav_decoder.audio,
        window_size=model_settings['window_size_samples'],
        stride=model_settings['window_stride_samples'],
        magnitude_squared=True)
    self.mfcc_ = contrib_audio.mfcc(
        spectrogram,
        wav_decoder.sample_rate,
        dct_coefficient_count=model_settings['dct_coefficient_count'])

  def get_data(self, how_many, offset, model_settings, sess):
    candidates = self.data_index
    sample_count = max(0, min(how_many, len(candidates) - offset))
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    desired_samples = model_settings['desired_samples']

    for i in xrange(offset, offset + sample_count):
      sample_index = i
      sample = candidates[sample_index]
      input_dict = {self.wav_filename_placeholder_: sample}
      data[i - offset, :] = sess.run(self.mfcc_, feed_dict=input_dict).flatten()

    return data

  def get_1d_data(self, how_many, offset, model_settings, sess):
    candidates = self.data_index
    sample_count = max(0, min(how_many, len(candidates) - offset))
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    desired_samples = model_settings['desired_samples']

    for i in xrange(offset, offset + sample_count):
      sample_index = i
      sample = candidates[sample_index]
      input_dict = {self.wav_filename_placeholder_: sample}
      data[i - offset, :] = sess.run(self.wav_decoder_, 
                                     feed_dict=input_dict).flatten()

    return data

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    words_list = input_data.prepare_words_list(FLAGS.wanted_words.split(','))
    model_settings = models.prepare_model_settings(
      len(words_list), FLAGS.sample_rate, FLAGS.clip_duration_ms, 
      FLAGS.window_size_ms, FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)

    inference_processor = InferenceProcessor(FLAGS.data_dir, model_settings)

    # Load labels
    labels_list = [line.rstrip() for line in tf.gfile.GFile(FLAGS.labels)]

    num_samples = len(inference_processor.data_index)
    tf.logging.info("Inference of %d samples", num_samples)

    labels = []
    softmaxes = []
    with tf.Session() as sess:
      fingerprint_size = model_settings['fingerprint_size']
      fingerprint_input = tf.placeholder(
          tf.float32, [None, fingerprint_size], name='fingerprint_input')
      logits, dropout_prob, mode_placeholder = models.create_model(
              fingerprint_input, model_settings, 
              FLAGS.model_architecture, is_training=True)
      softmax = tf.nn.softmax(logits, name='predicted_softmax')
      predicted_indices = tf.argmax(softmax, 1)
      predicted_softmax = tf.reduce_max(softmax, 1)

      sess.run(tf.global_variables_initializer())
      models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)

      for i in xrange(0, num_samples, FLAGS.batch_size):
        fingerprints = inference_processor.get_data(
            FLAGS.batch_size, i, model_settings, sess
        )
        num_fingerprints = fingerprints.shape[0]

        predictions, softmax_predictions = sess.run(
            [predicted_indices, predicted_softmax],
            feed_dict={
                fingerprint_input: fingerprints,
                dropout_prob: 1.0,
                mode_placeholder: False
            })
        print('predicted')
        for j in xrange(num_fingerprints):
          labels.append(labels_list[predictions[j]])
          softmaxes.append(softmax_predictions[j])
        print(i)

    with open(FLAGS.output_file, 'w') as f:
      f.write('fname,label\n')
      for i in xrange(num_samples):
        f.write('%s,%s,%f,\n' % (  os.path.basename(inference_processor.data_index[i]),
                                  labels[i],
                                  softmaxes[i]
                              ))
      
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--labels',
      type=str,
      default='',
      help='Path to file containing labels.',)
  parser.add_argument(
      '--data_dir',
      type=str,
      default='',
      help='Directory to data.',)
  parser.add_argument(
      '--output_file',
      type=str,
      default='',
      help='Output File.',)
  parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='Batch Size.',)
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--clip_stride_ms',
      type=int,
      default=30,
      help='How often to run recognition. Useful for models with cache.',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How long the stride is between spectrogram timeslices',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
