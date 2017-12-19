import tensorflow as tf



#################################### FROZEN ####################################
###                         DO NOT CHANGE THIS FILE 
################################################################################
'''
  python train.py --data_url= --data_dir=/home/zafar/speech_dataset/ \
  --how_many_training_step=4000 --batch_size=50 --model_architecture=conv_batchnorm  \
  --learning_rate=0.0001 --summaries_dir=/home/zafar/Desktop/logs5/ --eval_step_interval=100 \
  --start_checkpoint=/tmp/speech_commands_train/conv_batchnorm.ckpt-2000

'''
# Trained with Adam 0.0001, LB socre 0.83


def create_conv_batchnorm_model(fingerprint_input, model_settings, is_training):
  

  ### Dropout placeholder
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  ### Mode placeholder
  mode_placeholder = tf.placeholder(tf.bool, name="mode_placeholder")

  # Input Layer
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']
  net = tf.reshape(
                          fingerprint_input,
                          [-1, input_time_size, input_frequency_size, 1],
                          name="reshape")
  net = tf.layers.batch_normalization(
                          net, 
                          training=mode_placeholder,
                          name='bn_0')

  ### First Convolution
  net = tf.layers.conv2d(
                          inputs=net,
                          filters=64,
                          kernel_size=[10, 4],
                          padding='same',
                          name="conv_1")
  net = tf.layers.batch_normalization(
                          net,
                          training=mode_placeholder,
                          name='bn_1')
  net = tf.nn.relu(net)
  net = tf.layers.max_pooling2d(
                          net, [2, 2], [2, 2], 'SAME', 
                          name="maxpool_1")

  ### Second Convolution
  net = tf.layers.conv2d(
                          inputs=net,
                          filters=128,
                          kernel_size=[10, 4],
                          padding='same',
                          name="conv_2")
  # net = tf.layers.batch_normalization(
  #                         net, 
  #                         training=mode_placeholder,
  #                         name='bn_2')
  net = tf.nn.relu(net)
  net = tf.layers.max_pooling2d(
                          net, [2, 2], [2, 2], 'SAME', 
                          name='maxpool_2')

  ### Third Convolution
  net = tf.layers.conv2d(
                          inputs=net,
                          filters=256,
                          kernel_size=[10, 4],
                          padding='same',
                          name="conv_3")
  # net = tf.layers.batch_normalization(
  #                         net, 
  #                         training=mode_placeholder,
  #                         name='bn_3')
  net = tf.nn.relu(net)
  net = tf.layers.max_pooling2d(
                          net, [2, 2], [2, 2], 'SAME', 
                          name='maxpool_3')

  ### Fourth Convolution
  net = tf.layers.conv2d(
                          inputs=net,
                          filters=512,
                          kernel_size=[4, 4],
                          padding='same',
                          name="conv_4")
  # net = tf.layers.batch_normalization(
  #                         net, 
  #                         training=mode_placeholder,
  #                         name='bn_4')
  net = tf.nn.relu(net)
  net = tf.layers.max_pooling2d(
                          net, [2, 2], [2, 2], 'SAME', 
                          name='maxpool_4')

  ### Fifth Convolution
  net_shape = net.get_shape().as_list()
  net_height = net_shape[1]
  net_width = net_shape[2]
  net = tf.layers.conv2d(
                  inputs=net,
                  filters=1024,
                  kernel_size=[net_height, net_width],
                  strides=(net_height, net_width),
                  padding='same',
                  name="conv_5"
  )
  # net = tf.layers.batch_normalization(
  #                 net, 
  #                 training=mode_placeholder,
  #                 name='bn_5')
  net = tf.nn.relu(net)

  ## Sixth Convolution
  net = tf.layers.conv2d(
                  inputs=net,
                  filters=model_settings['label_count'],
                  kernel_size=[1, 1],
                  padding='same',
                  name="conv_6"
  )

  ### Squeeze
  squeezed = tf.squeeze(net, axis=[1, 2], name="squeezed")

  if is_training:
    return squeezed, dropout_prob, mode_placeholder
  else:
    return squeezed, mode_placeholder