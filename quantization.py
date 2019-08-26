#! /usr/bin/python3

'''
Methods to train:

A) Two stage training - train Q(weights) then Q(act)
B) Slowly lowering precision
C) Teacher student
'''

import tensorflow as tf
import numpy as np


def quantize_and_prune_weights(w, k, thresh, begin_pruning, end_pruning, pruning_frequency, target_sparsity):
    """
    Implements quantization and pruning as appropriate for weights.
    """
    #w_norm = rescale(w, [-1, 1])
    #w_norm = tf.clip_by_value(w, -1, 1)
    w_norm = w
    if thresh == 0 and k == 32:
        ## Don't quantize or prune
        return w_norm
    elif thresh == 0:
        ## Quantize
        w_quant = quantize_midtread_unbounded(w_norm, k) # quantize entire weight range
        return stop_grad(w, w_quant)
    elif k == 32:
        ## Prune
        w_range = tf.reduce_max(w_norm) - tf.reduce_min(w_norm)
        thresh_dynamic = (abs(thresh)/2)*w_range
        thresh_dynamic = tf.Print(thresh_dynamic, [thresh_dynamic])
        w_prune = prune_simplest(w_norm, [-thresh_dynamic, thresh_dynamic]) # Set all weights within specified region to zero
        w_prune = prune_simple_stochastic(w_norm, [-thresh_dynamic, thresh_dynamic]) # Round all weights within specified region stochastically
        #w_prune = prune_region(w_norm, [-abs(thresh), abs(thresh)], target_sparsity, begin_pruning, end_pruning, pruning_frequency) # Ideal pruning (but extremely memory inefficient)
        return stop_grad(w, w_prune)
    else:
        ## Quantize and Prune
        #quant_force_val_pos = abs(thresh) + (1 - abs(thresh))/2**np.ceil((k - 1) / 2)
        #quant_force_val_neg = abs(thresh) + (1 - abs(thresh))/2**np.floor((k - 1) / 2)
        w_Q_pos = quantize_region_midtread_unbounded_pos(w_norm, np.ceil((k - 1) / 2), abs(thresh)) # Positive quant region
        w_Q_pos_neg = quantize_region_midtread_unbounded_neg(w_Q_pos, np.floor((k - 1) / 2), -abs(thresh)) # Negative quant region
        #w_Q_P_pos_neg = prune_region(w_Q_pos_neg, [-abs(thresh), abs(thresh)], target_sparsity, begin_pruning, end_pruning, pruning_frequency, quant_force_val_pos, quant_force_val_neg) # Prune region close to zero
        w_Q_P_pos_neg = prune_simplest(w_Q_pos_neg, [-abs(thresh), abs(thresh)])
        return stop_grad(w, w_Q_P_pos_neg)


def quantize_and_prune_activations(a, k, thresh, begin_pruning, end_pruning, pruning_frequency, target_sparsity):
    """
    Implements quantization and pruning as appropriate for activations.
    """
    #a_norm = rescale(a, [0, 1])
    #a_norm = tf.clip_by_value(a, 0, 1)
    a_norm = a
    if thresh == 0 and k == 32:
        ## Don't quantize or prune
        return a_norm
    elif thresh == 0:
        ## Quantize
        a_quant = quantize_region_midtread_unbounded_pos(a_norm, k, 0) # quantize positive activation range
        return stop_grad(a, a_quant)
    elif k == 32:
        ## Prune
        #a_prune = prune_simplest(a_norm, [0, abs(thresh)])
        a_prune = prune_simple_stochastic(a_norm, [0, abs(thresh)])
        #a_prune = prune_region(a_norm, [0, abs(thresh)], target_sparsity, begin_pruning, end_pruning, pruning_frequency)
        return stop_grad(a, a_prune)
    else:
        ## Quantize and Prune
        #quant_force_val = abs(thresh) + (1 - abs(thresh))/2**(k-1)
        a_Q = quantize_region_midtread_unbounded_pos(a_norm, k - 1, abs(thresh)) # quantize positive activation region
        #a_Q_P = prune_region(a_Q, [0, abs(thresh)], target_sparsity, begin_pruning, end_pruning, pruning_frequency, quant_force_val, 0) # Prune region close to zero
        a_Q_P = prune_simplest(a_Q, [0, abs(thresh)])
        return stop_grad(a, a_Q_P)


def quantize_region_midrise(zr, k, quant_range, epsilon=1e-12):
    """
    Given a tensor `zr`, number of bits `k`, and quantization range `quant_range`, quantizes all values within this region to the preset number of bits, leaving values in non-specified regions un-quantized.
    """
    min_val = quant_range[0]
    max_val = quant_range[1]
    levels = 2**k
    step_size = (max_val - min_val) / levels
    max_quant_step = max_val - step_size / 2

    zr_quant = min_val + step_size / 2 + step_size * tf.floor((zr - min_val) / step_size)
    zr_quant = tf.minimum(zr_quant, max_quant_step)

    return tf.where(tf.logical_and(tf.greater_equal(zr, min_val - epsilon), tf.less_equal(zr, max_val + epsilon)), zr_quant, zr)

def quantize_region_midtread(zr, k, quant_range, epsilon=1e-12):
    """
    Given a tensor `zr`, number of bits `k`, and quantization range `quant_range`, quantizes all values within this region to the preset number of bits, leaving values in non-specified regions un-quantized.
    """
    min_val = quant_range[0]
    max_val = quant_range[1]
    levels = 2**k
    step_size = (max_val - min_val) / (levels - 1)

    zr_quant = min_val + step_size * tf.round((zr - min_val) / step_size)

    return tf.where(tf.logical_and(tf.greater_equal(zr, min_val - epsilon), tf.less_equal(zr, max_val + epsilon)), zr_quant, zr)

def quantize_region_midtread_unbounded_pos(zr, k, quant_thresh, epsilon=1e-12):
    """
    Given a tensor `zr`, number of bits `k`, and quantization range `quant_range`, quantizes all values within this region to the preset number of bits, leaving values in non-specified regions un-quantized.
    """
    min_bound = quant_thresh
    max_bound = tf.reduce_max(zr)

    levels = 2**k
    step_size = (max_bound - min_bound) / (levels - 1)
    max_quant_step = max_bound - step_size / 2

    zr_quant = min_bound + step_size * tf.round((zr - min_bound) / step_size)

    return tf.where(tf.greater_equal(zr, min_bound - epsilon), zr_quant, zr)

def quantize_region_midtread_unbounded_neg(zr, k, quant_thresh, epsilon=1e-12):
    """
    Given a tensor `zr`, number of bits `k`, and quantization range `quant_range`, quantizes all values within this region to the preset number of bits, leaving values in non-specified regions un-quantized.
    """
    min_bound = tf.reduce_min(zr)
    max_bound = quant_thresh

    levels = 2**k
    step_size = (max_bound - min_bound) / (levels - 1)
    max_quant_step = max_bound - step_size / 2

    zr_quant = min_bound + step_size * tf.round((zr - min_bound) / step_size)

    return tf.where(tf.less_equal(zr, max_bound + epsilon), zr_quant, zr)

def quantize_midtread_unbounded(zr, k, epsilon=1e-12):
    """
    Given a tensor `zr`, number of bits `k`, and quantization range `quant_range`, quantizes all values within this region to the preset number of bits, leaving values in non-specified regions un-quantized.
    """
    min_bound = tf.reduce_min(zr)
    max_bound = tf.reduce_max(zr)

    levels = 2**k
    step_size = (max_bound - min_bound) / (levels - 1)
    max_quant_step = max_bound - step_size / 2

    zr_quant = min_bound + step_size * tf.round((zr - min_bound) / step_size)

    return zr_quant

def prune_region(zr, prune_range, target_sparsity, begin_pruning, end_pruning, pruning_frequency, quant_force_val_pos=0, quant_force_val_neg=0, epsilon=1e-12):
    """
    Prunes a given tensor within the range specified, returns the other regions unpruned.
    """

    min_val = prune_range[0]
    max_val = prune_range[1]

    # Get the current global step
    global_step = tf.cast(tf.train.get_global_step(), tf.int32)

    # Calculate current tensor sparsity
    sparsity = tf.nn.zero_fraction(zr)

    # Conditions for pruning based on global step and sparsity
    cond_A = tf.logical_and(tf.logical_and(tf.greater_equal(global_step, begin_pruning), tf.less(global_step, end_pruning)), tf.equal(tf.mod(global_step, pruning_frequency), tf.zeros(shape=tf.shape(global_step), dtype=tf.int32))) # Are we in the soft-prune phase?
    cond_B = tf.greater_equal(global_step, end_pruning) # Are we in the hard-prune yet?
    cond_C = tf.less(sparsity, target_sparsity) # Have we reached the target sparsity yet?

    def prune_stochastic(zr): return zr * stochastic_round_tensor(zr)
    def prune_stoch_final(zr):
        stoch_tensor = stochastic_round_tensor(zr)
        if quant_force_val_pos != 0 or quant_force_val_neg != 0:
            mapped_tensor = tf.sign(zr) * tf.where(tf.greater(zr, 0), abs(quant_force_val_pos)*tf.ones(shape=tf.shape(zr)), abs(quant_force_val_neg)*tf.ones(shape=tf.shape(zr)))
            mapped_stoch_tensor = stoch_tensor * mapped_tensor
            return mapped_stoch_tensor
        else:
            return zr * stoch_tensor
    def dont_prune(zr): return zr

    zr_pruned = tf.case(pred_fn_pairs=[(tf.logical_and(cond_A, cond_C), lambda: prune_stochastic(
        zr)), (cond_B, lambda: prune_stoch_final(zr))], default=lambda: dont_prune(zr), exclusive=True)

    return tf.where(tf.logical_and(tf.greater_equal(zr, min_val - epsilon), tf.less_equal(zr, max_val + epsilon)), zr_pruned, zr)

def prune_simplest(zr, prune_range, epsilon=1e-12):
    """
    Prunes a given tensor within the range specified, returns the other regions unpruned.
    """
    min_val = prune_range[0]
    max_val = prune_range[1]
    return tf.where(tf.logical_and(tf.greater_equal(zr, min_val - epsilon), tf.less_equal(zr, max_val + epsilon)), tf.zeros(shape=tf.shape(zr)), zr)

def prune_simple_stochastic(zr, prune_range, quant_force_val_pos=0, quant_force_val_neg=0, epsilon=1e-12):
    """
    Prunes a given tensor within the range specified, returns the other regions unpruned.
    """
    min_val = prune_range[0]
    max_val = prune_range[1]
    stoch_tensor = stochastic_round_tensor(zr)
    mapped_tensor = tf.sign(zr) * tf.where(tf.greater(zr, 0), abs(quant_force_val_pos)*tf.ones(shape=tf.shape(zr)), abs(quant_force_val_neg)*tf.ones(shape=tf.shape(zr)))
    mapped_stoch_tensor = stoch_tensor * mapped_tensor
    return tf.where(tf.logical_and(tf.greater_equal(zr, min_val - epsilon), tf.less_equal(zr, max_val + epsilon)), mapped_stoch_tensor, zr)

def rescale(x, rescale_range, epsilon=1e-12):
    """
    Given an input tensor `x`, a target range `rescale_range`, and epsilon value, rescales the tensor such that the minimum value is mapped to rescale_range[0] and maximum value to rescale_range[1].
    NB: Epsilon prevents 0/0 error.
    """
    l = rescale_range[0]
    u = rescale_range[1]
    min_val = tf.reduce_min(x)
    max_val = tf.reduce_max(x)
    #return l + ((x - min_val) / (max_val - min_val)) * (u - l)
    return l + ((x - min_val + epsilon) / tf.maximum((max_val - min_val), 2*epsilon)) * (u - l)

def stop_grad(real, quant):
    """
    Needed for backpropogation.
    """
    return real + tf.stop_gradient(quant - real)

def stochastic_round_tensor(x):
    """
    Given a tensor x, stochastically rounds between x and zero depending on magnitude.
    """
    x_abs = tf.math.abs(x)
    rand = tf.random_uniform(shape=tf.shape(x), maxval=1)
    #prob_0 = 1 - (x_abs - tf.floor(x_abs)) # Useful to know, but not used in this implementation. So save memory by commenting it out!
    prob_1 = x - tf.floor(x_abs)
    x_rounded = tf.where(tf.less(rand, prob_1), tf.ones(shape=tf.shape(x)), tf.zeros(shape=tf.shape(x)))
    return x_rounded

def quantize_old( zr, k ):
    """
    Deprecated. Use quantize instead.
    """
    scaling = tf.cast( tf.pow( 2.0, k ) - 1, tf.float32 )
    return tf.round( scaling * zr )/scaling

def quantize_weights(w, k):
    """
    Deprecated. Use quantize_and_prune_weights instead.
    """
    # normalize first
    # zr = tf.tanh( w )/( tf.reduce_max( tf.abs( tf.tanh( w ) ) ) )
    zr = tf.clip_by_value(w, -1, 1)
    quant = quantize_old(zr, k)
    return stop_grad(w, quant)


def quantize_activations(xr, k):
    """
    Deprecated. Use quantize_and_prune_activations instead.
    """
    clipped = tf.clip_by_value(xr, 0, 1)
    quant = quantize_old(clipped, k)
    return stop_grad(xr, quant)

def stochastic_round(x):
    """
    Deprecated. Use stochastic_round_tensor instead.
    """
    prob_0 = 1 - (x - np.floor(x))
    prob_1 = x - np.floor(x)
    x_rounded = np.random.choice(a=[0, 1], p=[prob_0, prob_1])
    return x_rounded

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
