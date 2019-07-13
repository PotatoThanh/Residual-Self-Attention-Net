from __future__ import print_function
import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Lambda, Layer
from keras.layers import AveragePooling2D, Input, Flatten, MaxPool2D, Reshape, Add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
from resnet_layer import resnet_layer

class Attention_Layer(Layer):
    gamma = K.variable(0.0, name='gamma') # class variable sharing for all attention layers.

    def __init__(self, strides, num_filters, **kwargs):
        self.strides = strides
        self.num_filters = num_filters
        self.att_map = None
        self.att_feature = None
        super(Attention_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.trainable_weights.append(Attention_Layer.gamma)

        super(Attention_Layer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.attention(x, strides = self.strides, num_filters=self.num_filters)

    def compute_output_shape(self, input_shape):
        output_shape =(None, input_shape[1]//self.strides, input_shape[2]//self.strides, self.num_filters)
        return output_shape
    
    def get_att_map(self):
        return self.att_map
    
    def get_att_feature(self):
        return self.att_feature

    def attention(self, x,
                    num_filters=16,
                    kernel_size=3,
                    strides=1,
                    max_pooling=False):
        """2D Convolution-Batch Normalization-Activation for self-attention map
        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            h_feature (tensor): input tensor from feature map
            name (str): indicate name to value of attention easier
            num_filters (int): Conv2D number of filters output
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            max_pooling (bool): if True, apply max pooling 2D for inputs, f, g
        # Returns
            x (tensor): tensor as attention map
        """
        # get key, query and value
        h = resnet_layer(inputs=x,
                        num_filters=num_filters,
                        strides=strides)
        h = resnet_layer(inputs=h,
                        num_filters=num_filters,
                        activation=None,
                        batch_normalization=False)
        h = resnet_layer(inputs=h,
                        num_filters=num_filters,
                        kernel_size=1,
                        activation=None,
                        batch_normalization=False)  # linear layer [bs, h, w, c]
        if strides == 2:
            x = MaxPool2D()(x)
            
        f = resnet_layer(inputs=x,
                        num_filters=num_filters,
                        kernel_size=1,
                        activation=None,
                        batch_normalization=False)  # linear layer [bs, h, w, c]

        g = resnet_layer(inputs=x,
                        num_filters=num_filters,
                        kernel_size=1,
                        activation=None,
                        batch_normalization=False)  # linear layer [bs, h, w, c]

        # get output shape
        _, height, width, num_filters = K.int_shape(h)

        # flatten h and w
        f = Reshape((height*width, num_filters))(f)
        g = Reshape((height*width, num_filters))(g)
        h = Reshape((height*width, num_filters))(h)    
        
        # N = h * w
        s = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([g, f])  # [bs, N, N]

        att_map = Activation('softmax')(s) # attention map [0, 1] 
        
        att_feature = Lambda(lambda x: tf.matmul(x[0], x[1]))([att_map, h]) # apply attention map
        att_feature = Lambda(lambda x: Attention_Layer.gamma*x[0] + x[1])([att_feature, h])
        att_feature = Reshape((height, width, num_filters))(att_feature)

        att_feature = resnet_layer(inputs=att_feature,
                                    num_filters=num_filters,
                                    kernel_size=1,
                                    activation=None,
                                    batch_normalization=False)  # linear layer [bs, h, w, c]
        
        return att_feature