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
"""Tests for standard_layers.py."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental import standard_layers
import numpy as np


class StandardLayersTest(parameterized.TestCase):
  """Tests standard layers primitives."""

  @parameterized.parameters(
      dict(
          input_size=11,
          output_size=3,
          num_hidden_units=6,
          num_hidden_layers=1,
          use_bias=True,
      ),
      dict(
          input_size=2,
          output_size=6,
          num_hidden_units=6,
          num_hidden_layers=0,
          use_bias=True,
      ),
      dict(
          input_size=8,
          output_size=5,
          num_hidden_units=13,
          num_hidden_layers=4,
          use_bias=False,
      ),
  )
  def test_mlp_uniform(
      self,
      input_size: int,
      output_size: int,
      num_hidden_units: int,
      num_hidden_layers: int,
      use_bias: bool,
  ):
    """Tests output_shape, number and initialization of params in MlpUniform."""
    mlp = standard_layers.MlpUniform(
        input_size=input_size,
        output_size=output_size,
        hidden_size=num_hidden_units,
        n_hidden_layers=num_hidden_layers,
        use_bias=use_bias,
        rngs=nnx.Rngs(0),
    )
    inputs = np.random.uniform(size=input_size)
    params = nnx.state(mlp, nnx.Param)
    out = mlp(inputs)
    with self.subTest('output_shape'):
      self.assertEqual(out.shape, (output_size,))
    with self.subTest('total_params_count'):
      expected = count_mlp_uniform_params(
          input_size=input_size,
          num_hidden_units=num_hidden_units,
          num_hidden_layers=num_hidden_layers,
          output_size=output_size,
          use_bias=use_bias,
      )
      actual = sum([np.prod(x.shape) for x in jax.tree.leaves(params)])
      self.assertEqual(actual, expected)

    if num_hidden_layers > 1:
      with self.subTest('independent_params'):
        for previous_layer, layer in zip(mlp.layers[1:-3], mlp.layers[2:-2]):
          kernel_diff = previous_layer.kernel.value - layer.kernel.value
          self.assertGreater(jnp.linalg.norm(kernel_diff), 1e-1)

  @parameterized.parameters(
      dict(
          input_size=8,
          output_size=2,
          kernel_size=3,
      ),
      dict(
          input_size=9,
          output_size=4,
          kernel_size=5,
          use_bias=False,
      ),
      dict(
          input_size=5,
          output_size=11,
          kernel_size=3,
          use_bias=False,
      ),
      dict(
          input_size=2,
          output_size=4,
          kernel_size=1,
      ),
  )
  def test_conv_level(
      self,
      input_size: int,
      output_size: int,
      kernel_size: int,
      use_bias: bool = True,
  ):
    """Tests output_shape and number of params in ConvLevel."""
    level_size = 24
    input_shape = (input_size, level_size)
    inputs = jax.random.uniform(jax.random.PRNGKey(42), shape=input_shape)
    conv_level_layer = standard_layers.ConvLevel(
        input_size=input_size,
        output_size=output_size,
        kernel_size=kernel_size,
        use_bias=use_bias,
        rngs=nnx.Rngs(0),
    )
    params = nnx.state(conv_level_layer, nnx.Param)
    out = conv_level_layer(inputs)
    with self.subTest('output_shape'):
      self.assertEqual(out.shape, (output_size, level_size))
    with self.subTest('total_params_count'):
      expected = count_conv_params(
          input_size=input_size,
          kernel_size=kernel_size,
          output_size=output_size,
          use_bias=use_bias,
      )
      actual = sum([np.prod(x.shape) for x in jax.tree.leaves(params)])
      self.assertEqual(actual, expected)

  @parameterized.parameters(
      dict(
          input_size=8,
          output_size=2,
          channels=(6,),
          kernel_sizes=6,
      ),
      dict(
          input_size=9,
          output_size=4,
          channels=(4, 5),
          kernel_sizes=(3, 3, 3),
          use_bias=False,
      ),
      dict(
          input_size=2,
          output_size=4,
          channels=(7, 3),
          kernel_sizes=(5, 3, 6),
          kernel_dilations=(3, 1, 1),
      ),
  )
  def test_cnn_level_shape(
      self,
      input_size: int,
      output_size: int,
      channels: tuple[int, ...],
      kernel_sizes: int | tuple[int, ...],
      kernel_dilations: int | tuple[int, ...] = 1,
      use_bias: bool = True,
  ):
    """Tests that VerticalConvNet outputs and params have expected shapes."""
    n_levels = 13
    input_shape = (input_size, n_levels)
    inputs = jax.random.uniform(jax.random.PRNGKey(42), shape=input_shape)
    cnn_level_layer = standard_layers.CnnLevel(
        input_size=input_size,
        output_size=output_size,
        channels=channels,
        kernel_sizes=kernel_sizes,
        use_bias=use_bias,
        kernel_dilations=kernel_dilations,
        rngs=nnx.Rngs(0),
    )
    params = nnx.state(cnn_level_layer, nnx.Param)
    out = cnn_level_layer(inputs)
    with self.subTest('output_shape'):
      actual_shape = out.shape
      expected_shape = (output_size, n_levels)
      self.assertEqual(actual_shape, expected_shape)

    with self.subTest('total_params_count'):
      if isinstance(kernel_sizes, int):
        kernel_sizes = (kernel_sizes,) * (len(channels) + 1)
      in_out_and_kernels = zip(
          (input_size,) + channels,
          channels + (output_size,),
          kernel_sizes,
      )
      expected_by_part = [
          count_conv_params(din, kernel_size, dout, use_bias)
          for din, dout, kernel_size in in_out_and_kernels
      ]
      expected = sum(expected_by_part)
      actual = sum([np.prod(x.shape) for x in jax.tree.leaves(params)])
      self.assertEqual(actual, expected)


def count_mlp_uniform_params(
    input_size: int,
    num_hidden_units: int,
    num_hidden_layers: int,
    output_size: int,
    use_bias: bool,
):
  """Returns the number of parameters in MlpUniform based on settings."""
  n_input = input_size * num_hidden_units + num_hidden_units * use_bias
  n_hidden = num_hidden_units * num_hidden_units + num_hidden_units * use_bias
  n_output = num_hidden_units * output_size + output_size * use_bias
  return n_input + n_hidden * (num_hidden_layers - 1) + n_output


def count_conv_params(
    input_size: int, kernel_size: int, output_size: int, use_bias: bool
):
  """Returns the number of parameters in ConvLonLat based on settings."""
  n_w_params = np.prod(kernel_size) * input_size * output_size
  n_b_params = output_size
  return n_w_params + n_b_params * use_bias


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
