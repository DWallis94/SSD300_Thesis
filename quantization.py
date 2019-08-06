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
    prob_0 = 1 - (x - np.floor(x))
    prob_1 = x - np.floor(x)
    x_rounded = np.random.choice(a=[0, 1], p=[prob_0, prob_1])
    return x_rounded

# tf.where( in_our_region, tf.zeros(), tf.ones() ) returns a tensor whose elements come from A or B depending on the condition evaluation at each position
# tf.py_func(np_func, inputs) converts a np function to a tensorflow operation


def quantize_and_prune_by_sparsity(x, k, target_sparsity, quant_range, begin_pruning, end_pruning, pruning_frequency):

    # Define constant parameters as tensors for computations
    target_sparsity = tf.constant(target_sparsity, dtype=tf.float32)
    begin_pruning = tf.constant(begin_pruning, dtype=tf.int32)
    end_pruning = tf.constant(end_pruning, dtype=tf.int32)
    pruning_frequency = tf.constant(pruning_frequency, dtype=tf.int32)

    # Get the current global step
    global_step = tf.cast(tf.train.get_global_step(), tf.int32)

    # Calculate some quantization parameters
    min = quant_range[0]
    max = quant_range[1]
    step_size = (max - min) / 2**k

    # Calculate current tensor sparsity
    sparsity = tf.nn.zero_fraction(x)

    # For all values between min and max, quantize
    x_quant = tf.where(tf.logical_and(tf.greater_equal(
        x, min), tf.less_equal(x, max)), x, tf.zeros(shape=tf.shape(x)))
    x_quant = min + step_size * (tf.floor((x_quant - min) / step_size) + 0.5)

    # Conditions for pruning based on global step and sparsity
    cond_A = tf.reduce_all(tf.logical_and(tf.logical_and(tf.reduce_all(tf.greater_equal(global_step, begin_pruning)), tf.reduce_all(tf.less(
        global_step, end_pruning))), tf.reduce_all(tf.equal(tf.mod(global_step, pruning_frequency), tf.zeros(shape=tf.shape(global_step), dtype=tf.int32)))))
    cond_B = tf.reduce_all(tf.greater_equal(global_step, end_pruning))
    cond_C = tf.less(sparsity, target_sparsity)

    # Some function definitions needed for feeding to pruning tensor ops
    stochastic_round_vect = np.vectorize(stochastic_round)
    def prune_stochastic(x): return tf.py_func(stochastic_round_vect, [x], tf.float32)*(min + step_size/2)
    def prune_absolute(x): return tf.zeros(shape=tf.shape(x))
    def dont_prune(x): return x

    # For all values outside the [min, max] region, prune
    x_pruned = tf.where(tf.logical_or(tf.less(x, min), tf.greater(
        x, max)), x, tf.zeros(shape=tf.shape(x)))
    x_pruned = tf.case(pred_fn_pairs=[(tf.logical_and(cond_A, cond_C), lambda: prune_stochastic(
        x_pruned)), (tf.logical_and(cond_B, cond_C), lambda: prune_absolute(x_pruned))], default=lambda: dont_prune(x_pruned), exclusive=True)

    # Return the combined tensor containing quantized and pruned regions as appropriate
    return tf.where(tf.logical_and(tf.greater_equal(x, min), tf.less_equal(x, max)), x_quant, x_pruned)


def stop_grad(real, quant):
    return real + tf.stop_gradient(quant - real)


def quantize_and_prune_weights(w, k, thresh, begin_pruning, end_pruning, pruning_frequency, target_sparsity):
    #w_clipped = tf.clip_by_value(w, -1, 1)
    w_norm = rescale(w, [-1, 1])
    w_quant_pos = quantize_and_prune_by_sparsity(
        w_norm, np.ceil((k - 1) / 2), target_sparsity, [abs(thresh), 1], begin_pruning, end_pruning, pruning_frequency)
    w_quant_neg = quantize_and_prune_by_sparsity(
        w_norm, np.floor((k - 1) / 2), target_sparsity, [-1, -abs(thresh)], begin_pruning, end_pruning, pruning_frequency)
    w_quant = tf.where(tf.greater_equal(w_norm, 0),
                       w_quant_pos, w_quant_neg)
    return stop_grad(w_norm, w_quant)


def quantize_and_prune_activations(a, k, thresh, begin_pruning, end_pruning, pruning_frequency, target_sparsity):
    #a_clipped = tf.clip_by_value(a, 0, 1)
    a_norm = rescale(a, [0, 1])
    a_quant = quantize_and_prune_by_sparsity(
        a_norm, k - 1, target_sparsity, [abs(thresh), 1], begin_pruning, end_pruning, pruning_frequency)
    return stop_grad(a_norm, a_quant)


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

def rescale(x, range):
    l = range[0]
    u = range[1]
    min = tf.reduce_min(x)
    max = tf.reduce_max(x)
    return l + tf.divide(tf.subtract(x,min),tf.subtract(max,min))*(u-l)


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
    # unmasked = tf.multiply( unmasked, block_mask )
    t_x = tf.where(tf.less_equal(unmasked, -thres),
                   tf.multiply(tf.constant(-1.0, shape=x_shape), eta),
                   unmasked)
    t_x = tf.where(tf.greater_equal(unmasked, thres),
                   tf.multiply(tf.constant(1.0, shape=x_shape), eta),
                   t_x)
    return x + tf.stop_gradient(t_x - x)
