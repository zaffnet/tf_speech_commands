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

#conv_batchnorm block
def conv_bn_2d(x, out_channels, k_size=3, stride=1, is_bn=True, is_training=True):
  x = tf.layers.conv2d(
                        x,
                        filters=out_channels, 
                        kernel_size=[k_size,k_size], 
                        strides=[stride,stride],
                        padding='same')
  
  # >>>>>I have not understood the purpose of mode_placeholder, if you need to keep track 
  # of train/test for batchnorm, please make changes to is_bn paramater.
  if is_bn:
    x = tf.layers.batch_normalization(x, training=is_training)
    
  return x

#se_scale block
def se_scale(x, reduction=16):
  _,H,W,C = x.get_shape().as_list()

  x = tf.layers.average_pooling2d(x, pool_size=[H,W])
  x = tf.layers.conv2d(
                      x,
                      filters=reduction,
                      kernel_size=[1,1],
                      padding='same')
  x = tf.nn.relu(x)
  x = tf.layers.conv2d(
                      x,
                      filters=C,
                      kernel_size=[1,1],
                      padding='same')
  x = tf.nn.sigmoid(x)

  ## >>>>>>may need to be squeezed for getting scalar array shape
  ## [None, 1, 1, C] --> [None, C]
  return x

#res block
def res_block(x, out_planes, reduction=16):
  _,H,W,C = x.get_shape().as_list()
  assert(C==out_planes)

  z = conv_bn_2d(x, out_planes, k_size=3, stride=1)
  z = tf.nn.relu(z)
  z = conv_bn_2d(z, out_planes, k_size=3, stride=1)
  z = se_scale(z,reduction)*z + x
  z = tf.nn.relu(z)

  return z




def create_resnet_model(fingerprint_input, model_settings, is_training):
  

  ### Dropout placeholder
  #>>>>> I have used multiple constant dropout rates.
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  ### Mode placeholder
  mode_placeholder = tf.placeholder(tf.bool, name="mode_placeholder")

  # Input Layer
  input_frequency_size = model_settings['dct_coefficient_count']
  input_time_size = model_settings['spectrogram_length']

  # Label variable
  num_classes = model_settings['label_count']


  net = tf.reshape(
                          fingerprint_input,
                          [-1, input_time_size, input_frequency_size, 1],
                          name="reshape")
  

  net = tf.nn.relu(conv_bn_2d(net, 16, is_bn=True, is_training=is_training))
  net = res_block(net, 16)
  net = tf.nn.max_pool(net, ksize=[2,2], stride=[2,2])

  # >>>>>pytorch parameter is drop-probability,hence the larger value.
  # >>>>>seems like I was right about dropout all this time, I was just talking in different terms :(

  net = tf.nn.dropout(net, keep_prob=0.9)
  net = tf.nn.relu(conv_bn_2d(net, 32, is_bn=True, is_training=is_training))
  net = res_block(net, 32)
  net = res_block(net, 32)
  net = tf.nn.max_pool(net, ksize=[2,2], stride=[2,2])
  
  # 3a/3b/3c
  net = tf.nn.dropout(net, keep_prob=0.8)
  net = tf.nn.relu(conv_bn_2d(net, 64, is_bn=True, is_training=is_training))
  net = res_block(net, 64)
  net = res_block(net, 64)
  net = tf.nn.max_pool(net, ksize=[2,2], stride=[2,2])
  
  # 4a/4b
  net = tf.nn.dropout(net, keep_prob=0.8)
  net = tf.nn.relu(conv_bn_2d(net, 128, is_bn=True, is_training=is_training))
  net = res_block(net, 128)
  net = res_block(net, 128)

  # 5a/5b
  net = tf.nn.dropout(net, keep_prob=0.8)
  net = tf.nn.relu(conv_bn_2d(net, 256, is_bn=True, is_training=is_training))
  _,H,W,C = net.get_shape().as_list()
  net = tf.layers.average_pooling2d(net, pool_size=[H,W])
  net = tf.squeeze(net, axis=[1,2])
  
  #fc 1
  w1 = tf.get_variable('w1', 
                      shape = [256, 256],
                      dtype=tf.float32,
                      initializer=tf.contrib.layers.xavier_initializer())
  b1 = tf.get_variable('b1',
                      shape = [256],
                      dtype=tf.float32,
                      initializer=tf.zeros_initializer())
  net = tf.nn.relu(tf.matmul(net, w1) + b)
  net = tf.nn.dropout(net, keep_prob=0.8)

  #fc2
  w2 = tf.get_variable('w2', 
                      shape = [256, num_classes],
                      dtype=tf.float32,
                      initializer=tf.contrib.layers.xavier_initializer())
  b2 = tf.get_variable('b2',
                      shape = [num_classes],
                      dtype=tf.float32,
                      initializer=tf.zeros_initializer())

  net = tf.matmul(net, w2) + b2


  # neither dropout_prob nor mode_placeholder make any sense in this case
  if is_training:
    return net, dropout_prob, mode_placeholder
  else:
    return net, mode_placeholder