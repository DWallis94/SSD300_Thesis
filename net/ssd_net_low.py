# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import quantization as q

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5
_USE_FUSED_BN = True
_DROPOUT_RATE = 0.5

# vgg_16/conv2/conv2_1/biases
# vgg_16/conv4/conv4_3/biases
# vgg_16/conv1/conv1_1/biases
# vgg_16/fc6/weights
# vgg_16/conv3/conv3_2/biases
# vgg_16/conv5/conv5_3/biases
# vgg_16/conv3/conv3_1/weights
# vgg_16/conv4/conv4_2/weights
# vgg_16/conv1/conv1_1/weights
# vgg_16/conv5/conv5_3/weights
# vgg_16/conv4/conv4_1/weights
# vgg_16/conv3/conv3_3/weights
# vgg_16/conv5/conv5_2/biases
# vgg_16/conv3/conv3_2/weights
# vgg_16/conv4/conv4_2/biases
# vgg_16/conv5/conv5_2/weights
# vgg_16/conv3/conv3_1/biases
# vgg_16/conv2/conv2_2/weights
# vgg_16/fc7/weights
# vgg_16/conv5/conv5_1/biases
# vgg_16/conv1/conv1_2/biases
# vgg_16/conv2/conv2_2/biases
# vgg_16/conv4/conv4_1/biases
# vgg_16/fc7/biases
# vgg_16/fc6/biases
# vgg_16/conv4/conv4_3/weights
# vgg_16/conv2/conv2_1/weights
# vgg_16/conv5/conv5_1/weights
# vgg_16/conv3/conv3_3/biases
# vgg_16/conv1/conv1_2/weights

class ReLuLayer(tf.layers.Layer):
    def __init__(self, name, **kwargs):
        super(ReLuLayer, self).__init__(name=name, trainable=trainable, **kwargs)
        self._name = name
    def build(self, input_shape):
        self._relu = lambda x : tf.nn.relu(x, name=self._name)
        self.built = True

    def call(self, inputs):
        return self._relu(inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

def forward_module(m, inputs, training=False):
    if isinstance(m, tf.layers.BatchNormalization) or isinstance(m, tf.layers.Dropout):
        return m.apply(inputs, training=training)
    return m.apply(inputs)

class VGG16Backbone(object):
    def __init__(self, data_format='channels_first'):
        super(VGG16Backbone, self).__init__()
        self._data_format = data_format
        self._bn_axis = -1 if data_format == 'channels_last' else 1
        #initializer = tf.glorot_uniform_initializer  glorot_normal_initializer
        self._conv_initializer = tf.glorot_uniform_initializer
        self._conv_bn_initializer = tf.glorot_uniform_initializer#lambda : tf.truncated_normal_initializer(mean=0.0, stddev=0.005)
        # VGG layers
        self._pool1 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool1')
        self._pool2 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool2')
        self._pool3 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool3')
        self._pool4 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool4')
        self._pool5 = tf.layers.MaxPooling2D(3, 1, padding='same', data_format=self._data_format, name='pool5')

    def l2_normalize(self, x, name):
        with tf.name_scope(name, "l2_normalize", [x]) as name:
            axis = -1 if self._data_format == 'channels_last' else 1
            square_sum = tf.reduce_sum(tf.square(x), axis, keep_dims=True)
            x_inv_norm = tf.rsqrt(tf.maximum(square_sum, 1e-10))
            return tf.multiply(x, x_inv_norm, name=name)

    def forward(self, inputs, feature_scale=1.0, training=False):
        # inputs should in BGR
        feature_layers = []
        # forward vgg layers
        with tf.variable_scope('conv1') as scope:
            inputs = self.conv_block(inputs, 64, 3, (1, 1, 1, 1), 'conv1_1')
            inputs = self.conv_block_low(inputs, 64, 3, (1, 1, 1, 1), 'conv1_2', feature_scale)
        inputs = self._pool1.apply(inputs)
        with tf.variable_scope('conv2') as scope:
            inputs = self.conv_block_low(inputs, 128, 3, (1, 1, 1, 1), 'conv2_1', feature_scale)
            inputs = self.conv_block_low(inputs, 128, 3, (1, 1, 1, 1), 'conv2_2', feature_scale)
        inputs = self._pool2.apply(inputs)
        with tf.variable_scope('conv3') as scope:
            inputs = self.conv_block_low(inputs, 256, 3, (1, 1, 1, 1), 'conv3_1', feature_scale)
            inputs = self.conv_block_low(inputs, 256, 3, (1, 1, 1, 1), 'conv3_2', feature_scale)
            inputs = self.conv_block_low(inputs, 256, 3, (1, 1, 1, 1), 'conv3_3', feature_scale)
        inputs = self._pool3.apply(inputs)
        with tf.variable_scope('conv4') as scope:
            inputs = self.conv_block_low(inputs, 512, 3, (1, 1, 1, 1), 'conv4_1', feature_scale)
            inputs = self.conv_block_low(inputs, 512, 3, (1, 1, 1, 1), 'conv4_2', feature_scale)
            inputs = self.conv_block_low(inputs, 512, 3, (1, 1, 1, 1), 'conv4_3', feature_scale)
        # conv4_3
        with tf.variable_scope('conv4_3_scale') as scope:
            weight_scale = tf.Variable([20.] * int(512*feature_scale), trainable=training, name='weights')
            if self._data_format == 'channels_last':
                weight_scale = tf.reshape(weight_scale, [1, 1, 1, -1], name='reshape')
            else:
                weight_scale = tf.reshape(weight_scale, [1, -1, 1, 1], name='reshape')

            feature_layers.append(tf.multiply(weight_scale, self.l2_normalize(inputs, name='norm'), name='rescale')
                                )
        inputs = self._pool4.apply(inputs)
        with tf.variable_scope('conv5') as scope:
            inputs = self.conv_block_low(inputs, 512, 3, (1, 1, 1, 1), 'conv5_1', feature_scale)
            inputs = self.conv_block_low(inputs, 512, 3, (1, 1, 1, 1), 'conv5_2', feature_scale)
            inputs = self.conv_block_low(inputs, 512, 3, (1, 1, 1, 1), 'conv5_3', feature_scale)
        inputs = self._pool5.apply(inputs)
        # forward fc layers
        dilation = [1, 6, 6, 6]
        dilation[self._bn_axis] = 1
        inputs = self.conv_block_low(inputs, filters=1024, feature_scale=feature_scale, kernel_size=3, strides=[1,1,1,1], padding='SAME', dilations=dilation,
                                activation=tf.nn.relu, batch_norm=False, use_bias=True, name='fc6', reuse=None)
        inputs = self.conv_block_low(inputs, filters=1024, feature_scale=feature_scale, kernel_size=1, strides=[1,1,1,1], padding='SAME',
                                activation=tf.nn.relu, batch_norm=False, use_bias=True, name='fc7', reuse=None)
        # fc7
        feature_layers.append(inputs)
        
        # forward ssd layers
        with tf.variable_scope('additional_layers') as scope:
            with tf.variable_scope('conv8') as scope:
                stride = [1, 2, 2, 2]
                stride[self._bn_axis] = 1
                inputs = self.conv_block_low(inputs=inputs, filters=256, feature_scale=feature_scale, kernel_size=1, strides=(1, 1, 1, 1), use_bias=True, name='conv8_1')
                inputs = self.conv_block_low(inputs=inputs, filters=512, feature_scale=feature_scale, kernel_size=3, strides=stride, use_bias=True, name='conv8_2')
            # conv8
            feature_layers.append(inputs)
            with tf.variable_scope('conv9') as scope:
                stride = [1, 2, 2, 2]
                stride[self._bn_axis] = 1
                inputs = self.conv_block_low(inputs=inputs, filters=128, feature_scale=feature_scale, kernel_size=1, strides=(1, 1, 1, 1), use_bias=True, name='conv9_1')
                inputs = self.conv_block_low(inputs=inputs, filters=256, feature_scale=feature_scale, kernel_size=3, strides=stride, use_bias=True, name='conv9_2')
            # conv9
            feature_layers.append(inputs)
            with tf.variable_scope('conv10') as scope:
                inputs = self.conv_block_low(inputs=inputs, filters=128, feature_scale=feature_scale, kernel_size=1, strides=(1, 1, 1, 1), use_bias=True, name='conv10_1', padding='VALID')
                inputs = self.conv_block_low(inputs=inputs, filters=256, feature_scale=feature_scale, kernel_size=3, strides=(1, 1, 1, 1), use_bias=True, name='conv10_2', padding='VALID')
            # conv10
            feature_layers.append(inputs)
            with tf.variable_scope('conv11') as scope:
                inputs = self.conv_block_low(inputs=inputs, filters=128, feature_scale=feature_scale, kernel_size=1, strides=(1, 1, 1, 1), use_bias=True, name='conv11_1', padding='VALID')
                inputs = self.conv_block_low(inputs=inputs, filters=256, feature_scale=feature_scale, kernel_size=3, strides=(1, 1, 1, 1), use_bias=True, name='conv11_2', padding='VALID')
            # conv11
            feature_layers.append(inputs)

        return feature_layers

## A block which generates the low precision weights and convolves them with the input
    def conv_block_low(self, inputs, filters, kernel_size, strides, name, feature_scale=1.0, padding='SAME', dilations=[1, 1, 1, 1],
                    activation=tf.nn.relu, batch_norm=True, use_bias=True, reuse=None):
        with tf.variable_scope(name):
            data_format = "NHWC" if self._data_format == 'channels_last' else "NCHW"
            filter_shape = [ kernel_size, kernel_size, inputs.shape[self._bn_axis], filters ]
            conv_filter = tf.get_variable( 'kernel', filter_shape )
            tf.summary.histogram( "weights_r", conv_filter )
            with tf.variable_scope("quantize_weights"):
                weights_q = q.quantize_weights(conv_filter, 8) #quantize the weights
            tf.summary.histogram( "weights_q", weights_q )
            bias = tf.get_variable('bias', filters)
            conv = tf.nn.conv2d(input=inputs,filter=weights_q,strides=strides,padding=padding,use_cudnn_on_gpu=True,
                                data_format=data_format,dilations=dilations,name=name)
            tf.summary.histogram( "act", conv )
            conv = tf.nn.bias_add(conv, bias, data_format=data_format)
            if batch_norm:
                conv = tf.layers.batch_normalization(conv,axis=self._bn_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, fused=_USE_FUSED_BN,
                        reuse=None)
            else:
                conv = tf.layers.dropout(conv, rate=_DROPOUT_RATE)
            tf.summary.histogram( "act_bn", conv )
            #conv = tf.nn.relu(conv)
            with tf.variable_scope("quantize_activations"):
                conv = q.quantize_activations( conv, 8 )
            tf.summary.histogram( "act_bn_q", conv )
            conv = tf.nn.relu(conv)
            tf.summary.histogram( "act_bn_q_r", conv )
            return conv

    def conv_block(self, inputs, filters, kernel_size, strides, name, padding='SAME', dilations=[1, 1, 1, 1],
                    activation=tf.nn.relu, batch_norm=True, use_bias=True, reuse=None):
        with tf.variable_scope(name):
            data_format = "NHWC" if self._data_format == 'channels_last' else "NCHW"
            filter_shape = [ kernel_size, kernel_size, inputs.shape[self._bn_axis], filters ]
            conv_filter = tf.get_variable( 'kernel', filter_shape )
            tf.summary.histogram( "weights_r", conv_filter )
            bias = tf.get_variable('bias', filters)
            conv = tf.nn.conv2d(input=inputs,filter=conv_filter,strides=strides,padding=padding,use_cudnn_on_gpu=True,
                                data_format=data_format,dilations=dilations,name=name)
            tf.summary.histogram( "act", conv )
            conv = tf.nn.bias_add(conv, bias, data_format=data_format)
            if batch_norm:
                conv = tf.layers.batch_normalization(conv,axis=self._bn_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, fused=_USE_FUSED_BN,
                        reuse=None)
            else:
                conv = tf.layers.dropout(conv, rate=_DROPOUT_RATE)
            tf.summary.histogram( "act_bn", conv )
            conv = tf.nn.relu(conv)
            tf.summary.histogram( "act_bn_q_r", conv )
            return conv

    def ssd_conv_bn_block(self, filters, strides, name, reuse=None):
        #apply conv_block to this too
        with tf.variable_scope(name):
            conv_bn_blocks = []
            conv_bn_blocks.append(
                    tf.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same',
                        data_format=self._data_format, activation=None, use_bias=False,
                        kernel_initializer=self._conv_bn_initializer(),
                        bias_initializer=None,
                        name='{}_1'.format(name), _scope='{}_1'.format(name), _reuse=None)
                )
            conv_bn_blocks.append(
                    tf.layers.BatchNormalization(axis=self._bn_axis, momentum=BN_MOMENTUM, epsilon=BN_EPSILON, fused=USE_FUSED_BN,
                        name='{}_bn1'.format(name), _scope='{}_bn1'.format(name), _reuse=None)
                )
            conv_bn_blocks.append(
                    ReLuLayer('{}_relu1'.format(name), _scope='{}_relu1'.format(name), _reuse=None)
                )
            conv_bn_blocks.append(
                    tf.layers.Conv2D(filters=filters * 2, kernel_size=3, strides=strides, padding='same',
                        data_format=self._data_format, activation=None, use_bias=False,
                        kernel_initializer=self._conv_bn_initializer(),
                        bias_initializer=None,
                        name='{}_2'.format(name), _scope='{}_2'.format(name), _reuse=None)
                )
            conv_bn_blocks.append(
                    tf.layers.BatchNormalization(axis=self._bn_axis, momentum=BN_MOMENTUM, epsilon=BN_EPSILON, fused=USE_FUSED_BN,
                        name='{}_bn2'.format(name), _scope='{}_bn2'.format(name), _reuse=None)
                )
            conv_bn_blocks.append(
                    ReLuLayer('{}_relu2'.format(name), _scope='{}_relu2'.format(name), _reuse=None)
                )
            return conv_bn_blocks



def multibox_head(feature_layers, num_classes, num_anchors_depth_per_layer, data_format='channels_first'):
    #apply conv_block to this too
    with tf.variable_scope('multibox_head'):
        cls_preds = []
        loc_preds = []
        for ind, feat in enumerate(feature_layers):
            loc_preds.append(tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * 4, (3, 3), use_bias=True,
                        name='loc_{}'.format(ind), strides=(1, 1),
                        padding='same', data_format=data_format, activation=None,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        bias_initializer=tf.zeros_initializer()))
            cls_preds.append(tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * num_classes, (3, 3), use_bias=True,
                        name='cls_{}'.format(ind), strides=(1, 1),
                        padding='same', data_format=data_format, activation=None,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        bias_initializer=tf.zeros_initializer()))

        return loc_preds, cls_preds


