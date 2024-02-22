# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of custom initializers for NN parameters."""

from typing import Any, Optional, Sequence

import gin
import haiku as hk
import jax
import numpy as np


# Registering default initializers.
Constant = gin.external_configurable(hk.initializers.Constant)
VarianceScaling = gin.external_configurable(hk.initializers.VarianceScaling)
Orthogonal = gin.external_configurable(hk.initializers.Orthogonal)


def _compute_fans(
    shape: Sequence[int],
    fan_in_axes: Optional[Sequence[int]] = None,
) -> tuple[int, int]:
  """Computes the number of input and output units for a weight shape."""
  # adapted from dm-haiku/_src/initializers.py
  if len(shape) < 1:
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in, fan_out = shape
  else:
    if fan_in_axes is not None:
      # Compute fan-in using user-specified fan-in axes.
      fan_in = np.prod([shape[i] for i in fan_in_axes])
      fan_out = np.prod([s for i, s in enumerate(shape)
                         if i not in fan_in_axes])
    else:
      # If no axes specified, assume convolution kernels (2D, 3D, or more.)
      # kernel_shape: (..., input_depth, depth)
      receptive_field_size = np.prod(shape[:-2])
      fan_in = shape[-2] * receptive_field_size
      fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out


@gin.register
class ReducingVarianceScaling(hk.initializers.Initializer):
  """Initializer that result in variance that reduces as width increases.

  Initializes weights that result in features with expected variance of
  `scale / n`, where `n` corresponds to the width of the layer. This initializer
  can be used in the output layer to achieve µ parameterization [1].

  References:
    [1]: https://arxiv.org/abs/2203.03466
  """

  def __init__(
      self,
      scale=1.0,
      mode='fan_in',
      distribution='truncated_normal',
      fan_in_axes=None,
  ):
    """Constructs `ReducingVarianceScaling` initializer.

    Args:
      scale: Variance scale for a width == 1 initialization.
      mode: One of ``fan_in``, ``fan_out``, ``fan_avg``
      distribution: Random distribution to use. One of ``truncated_normal``,
        ``normal`` or ``uniform``.
      fan_in_axes: Optional sequence of int specifying which axes of the shape
        are part of the fan-in. If none provided, then the weight is assumed
        to be like a convolution kernel, where all leading dimensions are part
        of the fan-in, and only the trailing dimension is part of the fan-out.
        Useful if instantiating multi-headed attention weights.
    """
    if scale < 0.0:
      raise ValueError('`scale` must be a positive float.')
    if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
      raise ValueError('Invalid `mode` argument:', mode)
    distribution = distribution.lower()
    if distribution not in {'normal', 'truncated_normal', 'uniform'}:
      raise ValueError('Invalid `distribution` argument:', distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.fan_in_axes = fan_in_axes

  def __call__(self, shape: Sequence[int], dtype: Any) -> jax.Array:
    scale = self.scale
    fan_in, fan_out = _compute_fans(shape, self.fan_in_axes)
    if self.mode == 'fan_in':
      scale /= max(1.0, fan_in) ** 2
    elif self.mode == 'fan_out':
      scale /= max(1.0, fan_out) ** 2
    else:
      scale /= max(1.0, (fan_in + fan_out) / 2.0) ** 2

    if self.distribution == 'truncated_normal':
      stddev = np.sqrt(scale)
      # Adjust stddev for truncation.
      # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      distribution_stddev = np.asarray(.87962566103423978, dtype=dtype)
      stddev = stddev / distribution_stddev
      return hk.initializers.TruncatedNormal(stddev=stddev)(shape, dtype)
    elif self.distribution == 'normal':
      stddev = np.sqrt(scale)
      return hk.initializers.RandomNormal(stddev=stddev)(shape, dtype)
    else:
      limit = np.sqrt(3.0 * scale)
      uniform_init = hk.initializers.RandomUniform(minval=-limit, maxval=limit)
      return uniform_init(shape, dtype)
