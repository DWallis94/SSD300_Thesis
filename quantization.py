#! /usr/bin/python3

'''
Methods to train:

A) Two stage training - train Q(weights) then Q(act)
B) Slowly lowering precision
C) Teacher student
'''

import tensorflow as tf
import numpy as np


def stochastic_round(x):
    prob_1 = 1 - (x - np.floor(x))
    prob_2 = x - np.floor(x)
    x_rounded = np.random.choice(a=[0, 1], p=[prob_1, prob_2])
    return x_rounded

# tf.where( in_our_region, tf.zeros(), tf.ones() ) returns a tensor whose elements come from A or B depending on the condition evaluation at each position
# tf.py_func(np_func, inputs) converts a np function to a tensorflow operation


def quantize_and_prune(x, k, quant_range, begin_pruning, end_pruning, pruning_frequency):
    global_step = tf.train.get_global_step()
    begin_pruning = tf.constant(begin_pruning, dtype=tf.int64)
    end_pruning = tf.constant(end_pruning, dtype=tf.int64)
    pruning_frequency = tf.constant(pruning_frequency, dtype=tf.int64)

    min = quant_range[0]
    max = quant_range[1]
    step_size = (max - min) / 2**k
    ## Define quant region, zero elsewhere
    x_quant = tf.where(tf.logical_and(tf.greater_equal(x, min), tf.less_equal(x, max)), x, tf.zeros(shape=tf.shape(x)))
    x_pruned = tf.where(tf.logical_or(tf.less(x, min), tf.greater(x, max)), x, tf.zeros(shape=tf.shape(x)))

    ## Perform quantization in quant region
    x_quant = step_size * (tf.floor(x_quant / step_size) + .5)

    ## Perform pruning in pruning region
    # If within pruning window, prune
    cond_A = tf.reduce_all(tf.logical_and(tf.logical_and(tf.reduce_all(tf.greater_equal(global_step, begin_pruning)), tf.reduce_all(tf.less(global_step, end_pruning))), tf.reduce_all(tf.equal(tf.mod(global_step, pruning_frequency), tf.zeros(shape=tf.shape(global_step), dtype=tf.int64)))))
    cond_B = tf.reduce_all(tf.greater_equal(global_step, end_pruning))

    stochastic_round_vect = np.vectorize(stochastic_round)
    def prune_stochastic(x): return tf.py_func(stochastic_round_vect, [x], tf.float32)
    def prune_absolute(x): return tf.zeros(shape=tf.shape(x))
    def dont_prune(x): return x

    x_pruned = tf.cond(cond_A, lambda: prune_stochastic(x_pruned), lambda: tf.cond(cond_B, lambda: prune_absolute(x_pruned), lambda: dont_prune(x_pruned)))

    ## x_pruned and x_quant do not overlap (by design). So add them to get the pruned and quantized output
    return tf.add(x_pruned, x_quant)


def stop_grad(real, quant):
    return real + tf.stop_gradient(quant - real)


def quantize_and_prune_weights(w, k, thresh, begin_pruning, end_pruning, pruning_frequency):
    w_clipped = tf.clip_by_value(w, -1, 1)
    w_quant_pos = quantize_and_prune(
        w_clipped, np.ceil((k - 1) / 2), [abs(thresh), 1], begin_pruning, end_pruning, pruning_frequency)
    w_quant_neg = quantize_and_prune(
        w_clipped, np.floor((k - 1) / 2), [-1, -abs(thresh)], begin_pruning, end_pruning, pruning_frequency)
    w_quant = tf.add(w_quant_pos, w_quant_neg)
    return tf.reshape(stop_grad(w, w_quant), shape=tf.shape(w))


def quantize_and_prune_activations(a, k, thresh, begin_pruning, end_pruning, pruning_frequency):
    a_clipped = tf.clip_by_value(a, 0, 1)
    a_quant = quantize_and_prune(a_clipped, k - 1, [abs(thresh), 1], begin_pruning, end_pruning, pruning_frequency)
    return tf.reshape(stop_grad(a, a_quant), shape=tf.shape(a))


def quantize(zr, k):
    scaling = tf.cast(tf.pow(2.0, k) - 1, tf.float32)
    return tf.round(scaling * zr) / scaling


def quantize_weights(w, k):
    # normalize first
    # zr = tf.tanh( w )/( tf.reduce_max( tf.abs( tf.tanh( w ) ) ) )
    zr = tf.clip_by_value(w, -1, 1)
    quant = quantize(zr, k)
    return stop_grad(w, quant)


def quantize_activations(xr, k):
    clipped = tf.clip_by_value(xr, 0, 1)
    quant = quantize(clipped, k)
    return stop_grad(xr, quant)


def shaped_relu(x, a=1.0):
    # return tf.nn.relu( x )
    b = 0.5  # ( 1 - a )/ 2
    act = tf.clip_by_value(a * x + b, 0, 1)
    quant = tf.round(act)
    return act + tf.stop_gradient(quant - act)


def trinarize(x, nu=1.0):
    clip_val = x  # tf.clip_by_value( x, -1, 1 )
    x_shape = x.get_shape()
    thres = nu * tf.reduce_mean(tf.abs(clip_val))
    unmasked = tf.where(
        tf.logical_and(
            tf.greater(clip_val, -thres),
            tf.less(clip_val, thres)
        ),
        tf.constant(0.0, shape=x_shape),
        clip_val)
    eta = tf.reduce_mean(tf.abs(unmasked))
    #unmasked = tf.multiply( unmasked, block_mask )
    t_x = tf.where(tf.less_equal(unmasked, -thres),
                   tf.multiply(tf.constant(-1.0, shape=x_shape), eta),
                   unmasked)
    t_x = tf.where(tf.greater_equal(unmasked, thres),
                   tf.multiply(tf.constant(1.0, shape=x_shape), eta),
                   t_x)
    return x + tf.stop_gradient(t_x - x)
