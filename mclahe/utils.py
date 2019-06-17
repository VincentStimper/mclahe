#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops


def tf_batch_gather(params, indices, axis, name=None):
    """
    Extension of the batch_gather function in tensorflow
    (see https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/array_ops.py
    or https://www.tensorflow.org/api_docs/python/tf/batch_gather)
    Gather slices from `params` according to `indices` with leading batch dims.
    This operation assumes that the leading dimensions of `indices` are dense,
    and the gathers on the axis corresponding to the last dimension of `indices`.
    More concretely it computes:
    `result[i1, ..., in, j1, ..., jm, k1, ...., kl] = params[i1, ..., in, indices[i1, ..., in, j1, ..., jm], k1, ..., kl]`
    Therefore `params` should be a Tensor of shape [A1, ..., AN, C0, B1, ..., BM],
    `indices` should be a Tensor of shape [A1, ..., AN, C1, ..., CK] and `result` will be
    a Tensor of size `[A1, ..., AN, C1, ..., CK, B1, ..., BM]`.
    In the case in which indices is a 1D tensor, this operation is equivalent to
    `tf.gather`.
    See also `tf.gather` and `tf.gather_nd`.
    Args:
      params: A `Tensor`. The tensor from which to gather values.
      indices: A `Tensor`. Must be one of the following types: int32, int64. Index
          tensor. Must be in range `[0, params.shape[axis]`, where `axis` is the
          last dimension of `indices` itself.
      axis: A `Tensor`. Must be one of the following types: int32, int64. The axis
            in `params` to gather `indices` from.
      name: A name for the operation (optional).
    Returns:
      A Tensor. Has the same type as `params`.
    Raises:
      ValueError: if `indices` has an unknown shape.
    """

    with ops.name_scope(name):
        indices = ops.convert_to_tensor(indices, name="indices")
        params = ops.convert_to_tensor(params, name="params")
        indices_shape = tf.shape(indices)
        params_shape = tf.shape(params)

        ndims = indices.shape.ndims
        if ndims is None:
            raise ValueError("batch_gather does not allow indices with unknown "
                             "shape.")
        batch_indices = indices
        indices_dtype = indices.dtype.base_dtype
        accum_dim_value = tf.ones((), dtype=indices_dtype)
        # Use correct type for offset index computation
        casted_params_shape = gen_math_ops.cast(params_shape, indices_dtype)
        for dim in range(axis, 0, -1):
            dim_value = casted_params_shape[dim - 1]
            accum_dim_value *= casted_params_shape[dim]
            start = tf.zeros((), dtype=indices_dtype)
            step = tf.ones((), dtype=indices_dtype)
            dim_indices = gen_math_ops._range(start, dim_value, step)
            dim_indices *= accum_dim_value
            dim_shape = tf.stack([1] * (dim - 1) + [dim_value] + [1] * (ndims - dim),
                                 axis=0)
            batch_indices += tf.reshape(dim_indices, dim_shape)

        flat_inner_shape_indices = gen_math_ops.prod(indices_shape[:(axis + 1)], [0], False)
        flat_indices = tf.reshape(batch_indices, tf.concat([[flat_inner_shape_indices],
                                                            indices_shape[(axis + 1):]], axis=0))
        outer_shape = params_shape[(axis + 1):]
        flat_inner_shape_params = gen_math_ops.prod(params_shape[:(axis + 1)], [0], False)

        flat_params = tf.reshape(params, tf.concat([[flat_inner_shape_params], outer_shape], axis=0))
        flat_result = tf.gather(flat_params, flat_indices)
        result = tf.reshape(flat_result, tf.concat([indices_shape, outer_shape], axis=0))
        final_shape = indices.get_shape()[:axis].merge_with(
                params.get_shape()[:axis])
        final_shape = final_shape.concatenate(indices.get_shape()[axis:])
        final_shape = final_shape.concatenate(params.get_shape()[(axis + 1):])
        result.set_shape(final_shape)
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