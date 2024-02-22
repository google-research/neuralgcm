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
"""Tests for initializers.py."""

from typing import Tuple
from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
from neuralgcm import initializers
import numpy as np


class InitializaersTest(parameterized.TestCase):
  """Tests initializer primitives."""

  @parameterized.parameters(
      dict(shape=(100, 100), scale=0.1),
      dict(shape=(200, 500), scale=10.5),
      dict(shape=(3000, 2000), scale=1.0),
  )
  def test_reducing_variance_initializer_values(
      self,
      shape: Tuple[int, ...],
      scale: float,
  ):
    """Tests ReducingVarianceScaling initialization."""

    @hk.transform
    def get_init_params(shape, dtype):
      init = initializers.ReducingVarianceScaling(scale)
      p = hk.get_parameter('p', shape, dtype, init=init)
      return p

    params = get_init_params.init(jax.random.PRNGKey(42), shape, np.float32)
    params = jax.tree_util.tree_leaves(params)[0]
    actual_scaled_variance = shape[0] * np.std(params.sum(0))**2
    expected_scaled_variance = scale
    np.testing.assert_allclose(
        actual_scaled_variance, expected_scaled_variance, rtol=0.1)


if __name__ == '__main__':
  absltest.main()
