# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:50:44 2019

@author: LiXiaoGang
"""

import parameters as para
import tensorflow as tf

def darknet(clip_X,mode):
    '''
    Implementation of Darknet-19.
    Architecture: https://arxiv.org/abs/1612.08242
    '''
    
    with tf.variable_scope('Darknet-19'):
        
        # ---------------------
        net = tf.layers.conv2d(clip_X,filters=32,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-1')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-1')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-1')
        net = tf.layers.max_pooling2d(net,pool_size=(2,2),strides=(2,2),padding='valid',name='max_pool-1')

        # ---------------------
        net = tf.layers.conv2d(net,filters=64,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-2')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-2')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-2')
        net = tf.layers.max_pooling2d(net,pool_size=(2,2),strides=(2,2),padding='valid',name='max_pool-2')
        
        # ---------------------        
        net = tf.layers.conv2d(net,filters=128,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-3')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-3')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-3')
        
        net = tf.layers.conv2d(net,filters=64,kernel_size=(1,1),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-4')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-4')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-4')
        
        net = tf.layers.conv2d(net,filters=128,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-5')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-5')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-5')
        net = tf.layers.max_pooling2d(net,pool_size=(2,2),strides=(2,2),padding='valid',name='max_pool-3')
        
        # --------------------
        net = tf.layers.conv2d(net,filters=256,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-6')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-6')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-6')
        
        net = tf.layers.conv2d(net,filters=128,kernel_size=(1,1),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-7')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-7')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-7')
        
        net = tf.layers.conv2d(net,filters=128,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-8')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-8')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-8')
        net = tf.layers.max_pooling2d(net,pool_size=(2,2),strides=(2,2),padding='valid',name='max_pool-4')
        
        # --------------------
        net = tf.layers.conv2d(net,filters=512,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-9')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-9')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-9')
        
        net = tf.layers.conv2d(net,filters=256,kernel_size=(1,1),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-10')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-10')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-10')
        
        net = tf.layers.conv2d(net,filters=512,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-11')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-11')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-11')
        
        net = tf.layers.conv2d(net,filters=256,kernel_size=(1,1),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-12')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-12')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-12')
        
        net = tf.layers.conv2d(net,filters=512,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-13')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-13')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-13')
        net = tf.layers.max_pooling2d(net,pool_size=(2,2),strides=(2,2),padding='valid',name='max_pool-5')
        
        # --------------------
        net = tf.layers.conv2d(net,filters=1024,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-14')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-14')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-14')
        
        net = tf.layers.conv2d(net,filters=512,kernel_size=(1,1),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-15')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-15')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-15')
        
        net = tf.layers.conv2d(net,filters=1024,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-16')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-16')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-16')
        
        net = tf.layers.conv2d(net,filters=512,kernel_size=(1,1),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-17')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-17')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-17')
        
        net = tf.layers.conv2d(net,filters=1024,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-18')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-18')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-18')
        
        
        # ========== add convolutional layers for detection ===================
        net = tf.layers.conv2d(net,filters=1024,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-19')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-19')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-19')
        
        net = tf.layers.conv2d(net,filters=1024,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-20')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-20')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-20')
        
        net = tf.layers.conv2d(net,filters=1024,kernel_size=(3,3),strides=(1,1),
                               padding='same',activation=None,use_bias=True,name='conv-21')
        net = tf.layers.batch_normalization(net,training=mode,name='batch_norm-21')
        net = tf.nn.leaky_relu(net,name = 'leaky_relu-21')
        
        logits = tf.layers.conv2d(net,filters=para.OUYPUT_CHANNELS,
                                  kernel_size=(1,1),
                                  strides=(1,1),
                                  padding='same',
                                  activation=None,
                                  use_bias=True,
                                  name='Output')
        return logits
        