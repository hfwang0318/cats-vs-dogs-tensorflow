from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from tensorflow.contrib.slim import nets
import os


def full_net(x, is_training=True):
	pretrained_model_dir = 'pretrained_model'
	if not os.path.exists(pretrained_model_dir):
		os.mkdir(pretrained_model_dir)
	pretrained_model_path = pretrained_model_dir + '/vgg_16.ckpt'
	nets.vgg.vgg_16(x, num_classes=2, is_training=is_training)

	x = tf.get_default_graph().get_tensor_by_name('vgg_16/pool5/MaxPool:0')

	with tf.variable_scope('trainable'):
		x = slim.flatten(x)
		x = slim.fully_connected(x, 1024, scope='fc1')
		x = slim.fully_connected(x, 2, activation_fn=None, scope='fc2')

	total_vars = tf.trainable_variables()
	init_vars = []
	for v in total_vars:
		if 'fc' not in v.name:
			init_vars.append(v)
	try:
		var_init_op, feed_dict = slim.assign_from_checkpoint(pretrained_model_path, init_vars, ignore_missing_vars=True)
	except:
		print('downloading vgg16 pretrained model......')
		os.chdir(pretrained_model_dir)
		os.system('wget -c http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz')
		os.system('tar -xzf vgg_16_2016_08_28.tar.gz')
		os.chdir('..')
		var_init_op, feed_dict = slim.assign_from_checkpoint(pretrained_model_path, init_vars, ignore_missing_vars=True)
	return x, var_init_op, feed_dict