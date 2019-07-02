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

class Conv2D_layer(tf.keras.layers.Layer):

    def __init__(self, kernel_size, axis, filters, strides, padding, use_cudnn_on_gpu,
                        data_format, dilations, name, **kwargs):
        #self.output_dim = output_dim
        self._kernel_size = kernel_size
        self._axis = axis
        self._filters = filters
        self._filter = filter
        self._strides = strides
        self._padding = padding
        self._use_cudnn_on_gpu = use_cudnn_on_gpu
        self._data_format = data_format
        self._dilations = dilations
        self._name = name
        super(Conv2D_layer, self).__init__(**kwargs)

    def lambda_conv(x):
        filter_shape = [ self._kernel_size, self._kernel_size, x.shape[self._axis], self._filters ]
        conv_filter = tf.get_variable( 'kernel', filter_shape )
        tf.summary.histogram( "weights", conv_filter )
        conv = tf.nn.conv2d(input=x, filter=self._filter, strides=self._strides, padding=self._padding, use_cudnn_on_gpu=self._use_cudnn_on_gpu,
                            data_format=self._data_format, dilations=self._dilations, name=self._name)
        tf.summary.histogram( "act", conv )
        return conv

    def build(self, input_shape):
        self._Conv = tf.keras.layers.Lambda(self.lambda_conv)
        super(Conv2D_layer, self).build(input_shape)

    def call(self, inputs, mask=None):
        return self._Conv(inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

class Bias_add_layer(tf.keras.layers.Layer):

    def __init__(self, bias, data_format, **kwargs):
        #self.output_dim = output_dim
        self._data_format = data_format
        self._bias = bias
        super(Bias_add_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self._BiasAdd = lambda x : tf.nn.bias_add(value=x, bias=self._bias, data_format=self._data_format)
        super(Bias_add_layer, self).build(input_shape)

    def call(self, inputs, mask=None):
        return self._BiasAdd(inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)


class L2Normalization(tf.keras.layers.Layer):
    '''
    Performs L2 normalization on the input tensor with a learnable scaling parameter
    as described in the paper "Parsenet: Looking Wider to See Better" (see references)
    and as used in the original SSD model.
    Arguments:
        gamma_init (int): The initial scaling parameter. Defaults to 20 following the
            SSD paper.
    Input shape:
        4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
        or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.
    Returns:
        The scaled tensor. Same shape as the input tensor.
    References:
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    '''
    '''
    def l2_normalize(self, x, name):
        with tf.name_scope(name, "l2_normalize", [x]) as name:
            axis = -1 if self._data_format == 'channels_last' else 1
            square_sum = tf.reduce_sum(tf.square(x), axis, keep_dims=True)
            x_inv_norm = tf.rsqrt(tf.maximum(square_sum, 1e-10))
            return tf.multiply(x, x_inv_norm, name=name)
    '''

    def __init__(self, name, feature_scale, training, data_format, **kwargs):
        #self.output_dim = output_dim
        self._data_format = data_format
        self._name = name
        self._axis = -1 if self._data_format == 'channels_last' else 1
        self._weight_scale = tf.Variable([20.] * int(512*feature_scale), trainable=training, name='weights')
        if self._data_format == 'channels_last':
            self._weight_scale = tf.reshape(self._weight_scale, [1, 1, 1, -1], name='reshape')
        else:
            self._weight_scale = tf.reshape(self._weight_scale, [1, -1, 1, 1], name='reshape')
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self._L2Norm = lambda x : tf.multiply(self._weight_scale , tf.multiply(x, tf.rsqrt(tf.maximum(tf.reduce_sum(tf.square(x), self._axis, keep_dims=True), 1e-10)), name=self._name))
        super(L2Normalization, self).build(input_shape)

    def call(self, inputs, mask=None):
        return self._L2Norm(inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)


class ReLuLayer(tf.keras.layers.Layer):
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
    def __init__(self, feature_scale=1.0, training=False, data_format='channels_first'):
        super(VGG16Backbone, self).__init__()
        self._data_format = data_format
        self._bn_axis = -1 if data_format == 'channels_last' else 1
        #initializer = tf.glorot_uniform_initializer  glorot_normal_initializer
        self._conv_initializer = tf.glorot_uniform_initializer
        self._conv_bn_initializer = tf.glorot_uniform_initializer#lambda : tf.truncated_normal_initializer(mean=0.0, stddev=0.005)
        # VGG layers
        self._pool1 = tf.keras.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format)
        self._pool5 = tf.keras.layers.MaxPooling2D(3, 1, padding='same', data_format=self._data_format)

        self._dilation6 = [1, 6, 6, 6]
        self._dilation6[self._bn_axis] = 1

        self._stride8 = [1, 2, 2, 2]
        self._stride8[self._bn_axis] = 1

        self._stride9 = [1, 2, 2, 2]
        self._stride9[self._bn_axis] = 1

        self._weight_scale = tf.Variable([20.] * int(512*feature_scale), trainable=training, name='weights')
        if self._data_format == 'channels_last':
            self._weight_scale = tf.reshape(self._weight_scale, [1, 1, 1, -1], name='reshape')
        else:
            self._weight_scale = tf.reshape(self._weight_scale, [1, -1, 1, 1], name='reshape')

        self._model = tf.keras.Sequential()
        with tf.variable_scope("conv1"):
            for op in self.conv_block(64, 3, (1, 1, 1, 1), 'conv1_1', feature_scale):
                self._model.add(op)
            for op in self.conv_block(64, 3, (1, 1, 1, 1), 'conv1_2', feature_scale):
                self._model.add(op)
        self._model.add(self._pool1)
        with tf.variable_scope("conv2"):
            for op in self.conv_block(128, 3, (1, 1, 1, 1), 'conv2_1', feature_scale):
                self._model.add(op)
            for op in self.conv_block(128, 3, (1, 1, 1, 1), 'conv2_2', feature_scale):
                self._model.add(op)
        self._model.add(self._pool1)
        with tf.variable_scope("conv3"):
            for op in self.conv_block(256, 3, (1, 1, 1, 1), 'conv3_1', feature_scale):
                self._model.add(op)
            for op in self.conv_block(256, 3, (1, 1, 1, 1), 'conv3_2', feature_scale):
                self._model.add(op)
            for op in self.conv_block(256, 3, (1, 1, 1, 1), 'conv3_3', feature_scale):
                self._model.add(op)
            self._model.add(self._pool1)
        with tf.variable_scope("conv4"):
            for op in self.conv_block(512, 3, (1, 1, 1, 1), 'conv4_1', feature_scale):
                self._model.add(op)
            for op in self.conv_block(512, 3, (1, 1, 1, 1), 'conv4_2', feature_scale):
                self._model.add(op)
            for op in self.conv_block(512, 3, (1, 1, 1, 1), 'conv4_3', feature_scale):
                self._model.add(op)
            # conv4_3
        with tf.variable_scope("conv4_3_scale"):
            self._model.add(L2Normalization(name='norm', feature_scale=feature_scale, training=training, data_format=self._data_format))
        self._model.add(self._pool1)
        with tf.variable_scope("conv5"):
            for op in self.conv_block(512, 3, (1, 1, 1, 1), 'conv5_1', feature_scale):
                self._model.add(op)
            for op in self.conv_block(512, 3, (1, 1, 1, 1), 'conv5_2', feature_scale):
                self._model.add(op)
            for op in self.conv_block(512, 3, (1, 1, 1, 1), 'conv5_3', feature_scale):
                self._model.add(op)
        self._model.add(self._pool5)
            # forward fc layers
        with tf.variable_scope("conv6"):
            for op in self.conv_block(filters=1024, feature_scale=feature_scale, kernel_size=3, strides=[1,1,1,1], padding='SAME', dilations=self._dilation6,
                                    activation=tf.nn.relu, batch_norm=False, use_bias=True, name='fc6', reuse=None):
                                    self._model.add(op)
            for op in self.conv_block(filters=1024, feature_scale=feature_scale, kernel_size=1, strides=[1,1,1,1], padding='SAME',
                                    activation=tf.nn.relu, batch_norm=False, use_bias=True, name='fc7', reuse=None):
                                    self._model.add(op)

            # forward ssd layers
        with tf.variable_scope("additional_layers"):
            with tf.variable_scope("conv8"):
                for op in self.conv_block(filters=256, feature_scale=feature_scale, kernel_size=1, strides=(1, 1, 1, 1), use_bias=True, name='conv8_1'):
                    self._model.add(op)
                for op in self.conv_block(filters=512, feature_scale=feature_scale, kernel_size=3, strides=self._stride8, use_bias=True, name='conv8_2'):
                    self._model.add(op)
            with tf.variable_scope("conv9"):
            # conv8
                for op in self.conv_block(filters=128, feature_scale=feature_scale, kernel_size=1, strides=(1, 1, 1, 1), use_bias=True, name='conv9_1'):
                    self._model.add(op)
                for op in self.conv_block(filters=256, feature_scale=feature_scale, kernel_size=3, strides=self._stride9, use_bias=True, name='conv9_2'):
                    self._model.add(op)
            with tf.variable_scope("conv10"):
            # conv9
                for op in self.conv_block(filters=128, feature_scale=feature_scale, kernel_size=1, strides=(1, 1, 1, 1), use_bias=True, name='conv10_1', padding='VALID'):
                    self._model.add(op)
                for op in self.conv_block(filters=256, feature_scale=feature_scale, kernel_size=3, strides=(1, 1, 1, 1), use_bias=True, name='conv10_2', padding='VALID'):
                    self._model.add(op)
            with tf.variable_scope("conv11"):
            # conv10
                for op in self.conv_block(filters=128, feature_scale=feature_scale, kernel_size=1, strides=(1, 1, 1, 1), use_bias=True, name='conv11_1', padding='VALID'):
                    self._model.add(op)
                for op in self.conv_block(filters=256, feature_scale=feature_scale, kernel_size=3, strides=(1, 1, 1, 1), use_bias=True, name='conv11_2', padding='VALID'):
                    self._model.add(op)
            # conv11

    def forward(self):
        return self._model

    '''
    def forward(self, inputs, feature_scale=1.0, training=False):
        # inputs should in BGR
        feature_layers = []
        # forward vgg layers
        with tf.variable_scope('conv1') as scope:
            inputs = self.conv_block(inputs, 64, 3, (1, 1, 1, 1), 'conv1_1', feature_scale)
            inputs = self.conv_block(inputs, 64, 3, (1, 1, 1, 1), 'conv1_2', feature_scale)
        inputs = self._pool1.apply(inputs)
        with tf.variable_scope('conv2') as scope:
            inputs = self.conv_block(inputs, 128, 3, (1, 1, 1, 1), 'conv2_1', feature_scale)
            inputs = self.conv_block(inputs, 128, 3, (1, 1, 1, 1), 'conv2_2', feature_scale)
        inputs = self._pool2.apply(inputs)
        with tf.variable_scope('conv3') as scope:
            inputs = self.conv_block(inputs, 256, 3, (1, 1, 1, 1), 'conv3_1', feature_scale)
            inputs = self.conv_block(inputs, 256, 3, (1, 1, 1, 1), 'conv3_2', feature_scale)
            inputs = self.conv_block(inputs, 256, 3, (1, 1, 1, 1), 'conv3_3', feature_scale)
        inputs = self._pool3.apply(inputs)
        with tf.variable_scope('conv4') as scope:
            inputs = self.conv_block(inputs, 512, 3, (1, 1, 1, 1), 'conv4_1', feature_scale)
            inputs = self.conv_block(inputs, 512, 3, (1, 1, 1, 1), 'conv4_2', feature_scale)
            inputs = self.conv_block(inputs, 512, 3, (1, 1, 1, 1), 'conv4_3', feature_scale)
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
            inputs = self.conv_block(inputs, 512, 3, (1, 1, 1, 1), 'conv5_1', feature_scale)
            inputs = self.conv_block(inputs, 512, 3, (1, 1, 1, 1), 'conv5_2', feature_scale)
            inputs = self.conv_block(inputs, 512, 3, (1, 1, 1, 1), 'conv5_3', feature_scale)
        inputs = self._pool5.apply(inputs)
        # forward fc layers
        dilation = [1, 6, 6, 6]
        dilation[self._bn_axis] = 1
        inputs = self.conv_block(inputs, filters=1024, feature_scale=feature_scale, kernel_size=3, strides=[1,1,1,1], padding='SAME', dilations=dilation,
                                activation=tf.nn.relu, batch_norm=False, use_bias=True, name='fc6', reuse=None)
        inputs = self.conv_block(inputs, filters=1024, feature_scale=feature_scale, kernel_size=1, strides=[1,1,1,1], padding='SAME',
                                activation=tf.nn.relu, batch_norm=False, use_bias=True, name='fc7', reuse=None)
        # fc7
        feature_layers.append(inputs)

        # forward ssd layers
        with tf.variable_scope('additional_layers') as scope:
            with tf.variable_scope('conv8') as scope:
                stride = [1, 2, 2, 2]
                stride[self._bn_axis] = 1
                inputs = self.conv_block(inputs=inputs, filters=256, feature_scale=feature_scale, kernel_size=1, strides=(1, 1, 1, 1), use_bias=True, name='conv8_1')
                inputs = self.conv_block(inputs=inputs, filters=512, feature_scale=feature_scale, kernel_size=3, strides=stride, use_bias=True, name='conv8_2')
            # conv8
            feature_layers.append(inputs)
            with tf.variable_scope('conv9') as scope:
                stride = [1, 2, 2, 2]
                stride[self._bn_axis] = 1
                inputs = self.conv_block(inputs=inputs, filters=128, feature_scale=feature_scale, kernel_size=1, strides=(1, 1, 1, 1), use_bias=True, name='conv9_1')
                inputs = self.conv_block(inputs=inputs, filters=256, feature_scale=feature_scale, kernel_size=3, strides=stride, use_bias=True, name='conv9_2')
            # conv9
            feature_layers.append(inputs)
            with tf.variable_scope('conv10') as scope:
                inputs = self.conv_block(inputs=inputs, filters=128, feature_scale=feature_scale, kernel_size=1, strides=(1, 1, 1, 1), use_bias=True, name='conv10_1', padding='VALID')
                inputs = self.conv_block(inputs=inputs, filters=256, feature_scale=feature_scale, kernel_size=3, strides=(1, 1, 1, 1), use_bias=True, name='conv10_2', padding='VALID')
            # conv10
            feature_layers.append(inputs)
            with tf.variable_scope('conv11') as scope:
                inputs = self.conv_block(inputs=inputs, filters=128, feature_scale=feature_scale, kernel_size=1, strides=(1, 1, 1, 1), use_bias=True, name='conv11_1', padding='VALID')
                inputs = self.conv_block(inputs=inputs, filters=256, feature_scale=feature_scale, kernel_size=3, strides=(1, 1, 1, 1), use_bias=True, name='conv11_2', padding='VALID')
            # conv11
            feature_layers.append(inputs)

        return feature_layers
        '''

    def conv_block(self, filters, kernel_size, strides, name, feature_scale=1.0, padding='SAME', dilations=[1, 1, 1, 1],
                activation=tf.nn.relu, batch_norm=True, use_bias=True, reuse=None):
        with tf.variable_scope(name):
            conv_ops = []
            data_format = "NHWC" if self._data_format == 'channels_last' else "NCHW"
            bias = tf.get_variable('bias', filters)
            conv_ops.append(Conv2D_layer(kernel_size=kernel_size, axis=self._bn_axis, filters=filters, strides=strides, padding=padding, use_cudnn_on_gpu=True,
                                data_format=data_format, dilations=dilations,name=name))
            conv_ops.append(Bias_add_layer(bias=bias, data_format=data_format))
            if batch_norm:
                conv_ops.append(tf.keras.layers.BatchNormalization(axis=self._bn_axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, fused=_USE_FUSED_BN))
            else:
                conv_ops.append(tf.keras.layers.Dropout(rate=_DROPOUT_RATE))
            #tf.summary.histogram( "act_bn", conv_ops[-1] )
            conv_ops.append(tf.keras.layers.ReLU())
            #tf.summary.histogram( "act_bn_r", conv_ops[-1] )
            return conv_ops

    '''
    def conv_block(self, inputs, filters, kernel_size, strides, name, feature_scale=1.0, padding='SAME', dilations=[1, 1, 1, 1],
                    activation=tf.nn.relu, batch_norm=True, use_bias=True, reuse=None):
        with tf.variable_scope(name):
            data_format = "NHWC" if self._data_format == 'channels_last' else "NCHW"
            filter_shape = [ kernel_size, kernel_size, inputs.shape[self._bn_axis], filters ]
            filter_shape_scaled = [ kernel_size, kernel_size, inputs.shape[self._bn_axis], filters * feature_scale ]
            # new stuff in testing
            #conv_filter_scaled = tf.get_variable( 'kernel', filter_shape_scaled)
            #print(conv_filter_scaled.get_shape())
            conv_filter = tf.get_variable( 'kernel', filter_shape )
            #print(conv_filter.get_shape())
            tf.summary.histogram( "weights", conv_filter )
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
            tf.summary.histogram( "act_bn_r", conv )
            return conv
    '''

    def ssd_conv_block(self, filters, strides, name, padding='same', reuse=None):
        with tf.variable_scope(name):
            conv_blocks = []
            conv_blocks.append(
                    tf.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding=padding,
                        data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                        kernel_initializer=self._conv_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name='{}_1'.format(name), _scope='{}_1'.format(name), _reuse=None)
                )
            conv_blocks.append(
                    tf.layers.Conv2D(filters=filters * 2, kernel_size=3, strides=strides, padding=padding,
                        data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                        kernel_initializer=self._conv_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name='{}_2'.format(name), _scope='{}_2'.format(name), _reuse=None)
                )
            return conv_blocks

    def ssd_conv_bn_block(self, filters, strides, name, reuse=None):
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
    with tf.variable_scope('multibox_head'):
        cls_preds = []
        loc_preds = []
        for ind, feat in enumerate(feature_layers.layers):
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
