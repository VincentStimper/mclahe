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


def batch_histogram(values, value_range, axis, nbins=100, use_map=False):
    """
    Computes histogram with fixed width considering batch dimensions
    :param values: Array containing the values for histogram computation.
    :param value_range: Shape [2] iterable. values <= value_range[0] will be mapped to
    hist[0], values >= value_range[1] will be mapped to hist[-1].
    :param axis: Number of batch dimensions. First axis to apply histogram computation to.
    :param nbins: Scalar. Number of histogram bins.
    :param use_map: Flag indicating whether map function is used
    :return: histogram with batch dimensions.
    """

    # Get shape
    values_shape = values.shape
    batch_dim = values_shape[:axis]
    outer_dim = values_shape[axis:]
    num_batch = np.prod(batch_dim)

    if use_map:
        values_reshaped = values.reshape(*((num_batch,) + outer_dim))
        hist = np.array(list(map(lambda x: np.histogram(x, range=value_range, bins=nbins)[0], values_reshaped)))
    else:
        # Normalize
        values_double = values.astype('double')
        value_range_double = np.array(value_range).astype('double')

        # Clip values
        values_norm = (values_double - value_range_double[0]) / (value_range_double[1] - value_range_double[0])
        values_clip1 = np.maximum(values_norm, 0.5 / nbins)
        values_clip2 = np.minimum(values_clip1, 1.0 - 0.5 / nbins)

        # Shift values
        values_shift = values_clip2 + np.arange(num_batch).reshape(*(batch_dim + len(outer_dim) * (1,)))

        # Get histogram
        hist = np.histogram(values_shift, range=[0, num_batch], bins=num_batch * nbins)[0]

    return hist.reshape(*(batch_dim + (nbins,)))