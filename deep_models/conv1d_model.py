import tensorflow as tf

def create_conv1d_model(fingerprint_input, model_settings, is_training):

  ### Dropout placeholder
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

  ### Mode placeholder
  mode_placeholder = tf.placeholder(tf.bool, name="mode_placeholder")

  input_time_size = model_settings["desired_samples"]
  net = tf.reshape(tensor=fingerprint_input, 
                              shape=[-1, input_time_size, 1],
                              name="reshape")
  net = tf.layers.batch_normalization(net, name="bn_in") 

  ##############################################################################

  for i in range(7):
    net = tf.layers.conv1d(inputs=net, 
                            filters=8*(2**i), 
                            kernel_size=4,
                            padding="same",
                            name="conv_%d"%(i))
    net = tf.layers.batch_normalization(net, name="bn_%d"%i    )
    net = tf.nn.relu(net, name="relu_%d"%i)
    net = tf.layers.max_pooling1d(net, 
                                  pool_size=2, 
                                  strides=2, 
                                  padding="same",
                                  name="maxpool_%d"%i)

  ##############################################################################

  # ### Eigth Convolution
  # net = tf.layers.conv1d(
  #                         inputs=net,
  #                         filters=model_settings["label_count"],
  #                         kernel_size=1,
  #                         name="label_conv")
  # net = tf.layers.batch_normalization(net, name="bn_out") 

  net_shape = net.get_shape().as_list()
  net = tf.reshape(net, [-1, net_shape[1]*net_shape[2]])

  # ### Squeeze
  # with tf.name_scope("Squeeze"):
  #   net = tf.squeeze(net, axis=[1], name="squeezed")

  net = tf.layers.dense(net, 1024)
  net = tf.nn.relu(net)
  if is_training:
    net = tf.layers.dropout(net, dropout_prob)
  else:
    pass
  net = tf.layers.dense(net, model_settings["label_count"])

  if is_training:
    return net, dropout_prob, mode_placeholder
  else:
    return net, mode_placeholder

  
  


  


  
  


  
    
  