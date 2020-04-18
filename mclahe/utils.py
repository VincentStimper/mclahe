#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def batch_gather(params, indices, axis):
    """
    Gather slices from `params` according to `indices` with leading batch dims.
    This operation assumes that the leading dimensions of `indices` are dense,
    and the gathers on the axis corresponding to the last dimension of `indices`.
    More concretely it computes:
    `result[i1, ..., in, j1, ..., jm, k1, ...., kl] = params[i1, ..., in, indices[i1, ..., in, j1, ..., jm], k1, ..., kl]`
    Therefore `params` should be a Tensor of shape [A1, ..., AN, C0, B1, ..., BM],
    `indices` should be a Tensor of shape [A1, ..., AN, C1, ..., CK] and `result` will be
    a Tensor of size `[A1, ..., AN, C1, ..., CK, B1, ..., BM]`.
    Args:
      params: The array from which to gather values.
      indices: Must be one of the following types: int32, int64. Index
          array. Must be in range `[0, params.shape[axis]`, where `axis` is the
          last dimension of `indices` itself.
      axis: Must be one of the following types: int32, int64. The axis
            in `params` to gather `indices` from.
    Returns:
      An array. Has the same type as `params`.
    """

    indices_shape = indices.shape
    params_shape = params.shape
    ndim = indices.ndim

    # Adapt indices to act on respective batch
    accum_dim_value = 1
    for dim in range(axis, 0, -1):
        dim_value = params_shape[dim - 1]
        accum_dim_value *= params_shape[dim]
        dim_indices = np.arange(dim_value)
        dim_indices *= accum_dim_value
        dim_shape = [1] * (dim - 1) + [dim_value] + [1] * (ndim - dim)
        indices += dim_indices.reshape(*dim_shape)

    flat_inner_shape_indices = np.prod(indices_shape[:(axis + 1)])
    flat_indices = indices.reshape(*((flat_inner_shape_indices,) + indices_shape[(axis + 1):]))
    outer_shape = params_shape[(axis + 1):]
    flat_inner_shape_params = np.prod(params_shape[:(axis + 1)])

    flat_params = params.reshape(*((flat_inner_shape_params,) + outer_shape))
    flat_result = flat_params[flat_indices,...]
    result = flat_result.reshape(*(indices_shape + outer_shape))
    return result


def tf_batch_histogram(values, value_range, axis, nbins=100, dtype=tf.int32, use_map=True):
    """
    Computes histogram with fixed width considering batch dimensions
    :param values: Numeric `Tensor` containing the values for histogram computation.
    :param value_range: Shape [2] `Tensor` of same `dtype` as `values`. values <= value_range[0] will be mapped to
    hist[0], values >= value_range[1] will be mapped to hist[-1].
    :param axis: Number of batch dimensions. First axis to apply histogram computation to.
    :param nbins: Scalar `int32 Tensor`. Number of histogram bins.
    :param dtype: dtype for returned histogram, can be either tf.int32 or tf.int64.
    :return: histogram with batch dimensions.
    """

    # Get shape
    values_shape = tf.shape(values)
    batch_dim = values_shape[:axis]
    rest_dim = values_shape[axis:]
    num_batch = tf.reduce_prod(batch_dim)

    if use_map:
        values_reshaped = tf.reshape(values, tf.concat([[num_batch], rest_dim], 0))
        hist = tf.map_fn(lambda x: tf.histogram_fixed_width(x, value_range, nbins=nbins, dtype=dtype), values_reshaped,
                         dtype=dtype, parallel_iterations=64)
    else:
        # Normalize
        values_float = tf.cast(values, tf.float32)
        value_range_float = tf.cast(value_range, tf.float32)

        # Clip values
        values_norm = (values_float - value_range_float[0]) / (value_range_float[1] - value_range_float[0])
        values_clip1 = tf.maximum(values_norm, 0.5 / tf.cast(nbins, tf.float32))
        values_clip2 = tf.minimum(values_clip1, 1.0 - 0.5 / tf.cast(nbins, tf.float32))

        # Shift values
        values_shift = values_clip2 + tf.reshape(tf.range(tf.cast(num_batch, tf.float32), dtype=tf.float32),
                                                 tf.concat([batch_dim, tf.ones(tf.size(rest_dim), tf.int32)], 0))

        # Get histogram
        hist = tf.histogram_fixed_width(values_shift, [0., tf.cast(num_batch, tf.float32)], nbins=num_batch * nbins,
                                        dtype=dtype)

    return tf.reshape(hist, tf.concat([batch_dim, [nbins]], 0))