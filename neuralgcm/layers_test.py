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
"""Tests for layers.py."""

from typing import Optional, Sequence, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import gin
import haiku as hk
import jax
from neuralgcm import layers  # pylint: disable=unused-import
import numpy as np


class LayersTest(parameterized.TestCase):
  """Tests layers primitives."""

  @parameterized.parameters(
      dict(
          output_size=3,
          num_hidden_units=6,
          num_hidden_layers=7,
          with_bias=True,),
      dict(
          output_size=5,
          num_hidden_units=13,
          num_hidden_layers=4,
          with_bias=False,),
  )
  def test_mlp_uniform_shape(
      self,
      output_size: int,
      num_hidden_units: int,
      num_hidden_layers: int,
      with_bias: bool,
  ):
    """Tests that EpdTower outputs and params have expected shapes."""
    gin.enter_interactive_mode()

    @gin.configurable
    def layer_forward(inputs, layer):  # defines a forward function
      return layer(output_size)(inputs)

    config = '\n'.join([
        'layer_forward.layer = @MlpUniform',
        f'MlpUniform.num_hidden_units = {num_hidden_units}',
        f'MlpUniform.num_hidden_layers = {num_hidden_layers}',
        f'MlpUniform.with_bias = {with_bias}',
        'ConvLonLat.name = "mlp_uniform"',
    ])

    gin.clear_config()
    gin.parse_config(config)

    n_features = 8
    input_size = (n_features,)
    inputs = np.random.uniform(size=input_size)
    model = hk.without_apply_rng(hk.transform(layer_forward))
    params = model.init(jax.random.PRNGKey(42), inputs)
    out = model.apply(params, inputs)
    with self.subTest('output_shape'):
      self.assertEqual(out.shape, (output_size,))
    with self.subTest('total_params_count'):
      expected = count_mlp_uniform_params(
          input_size=n_features,
          num_hidden_units=num_hidden_units,
          num_hidden_layers=num_hidden_layers,
          output_size=output_size,
          with_bias=with_bias,)
      actual = sum([np.prod(x.shape)
                    for x in jax.tree_util.tree_leaves(params)])
      self.assertEqual(actual, expected)
    with self.subTest('parameter_keys'):
      n_layers = num_hidden_layers + 1
      self.assertEqual(
          tuple(params.keys()),
          tuple([f'mlp_uniform/~/linear_{i}' for i in range(n_layers)]))

  @parameterized.parameters(
      dict(
          output_size=2,
          kernel_shape=(3, 3),
          with_bias=False,),
      dict(
          output_size=4,
          kernel_shape=(5, 5),
          with_bias=True,),
  )
  def test_conv_lon_lat_shape(
      self,
      output_size: int,
      kernel_shape: int,
      with_bias: bool,
  ):
    """Tests that ConvLonLat outputs and params have expected shapes."""
    gin.enter_interactive_mode()

    @gin.configurable
    def layer_forward(inputs, layer):  # defines a forward function
      return layer(output_size)(inputs)

    config = '\n'.join([
        'layer_forward.layer = @ConvLonLat',
        f'ConvLonLat.kernel_shape = {kernel_shape}',
        f'ConvLonLat.with_bias = {with_bias}',
        'ConvLonLat.name = "conv_lon_lat"',
    ])

    gin.clear_config()
    gin.parse_config(config)

    n_features = 8
    input_size = (n_features, 10, 20)
    inputs = np.random.uniform(size=input_size)
    model = hk.without_apply_rng(hk.transform(layer_forward))
    params = model.init(jax.random.PRNGKey(42), inputs)
    out = model.apply(params, inputs)
    with self.subTest('output_shape'):
      self.assertEqual(out.shape, (output_size, 10, 20))
    with self.subTest('total_params_count'):
      expected = count_conv_lon_lat_params(
          input_size=n_features,
          kernel_shape=kernel_shape,
          output_size=output_size,
          with_bias=with_bias)
      actual = sum([np.prod(x.shape)
                    for x in jax.tree_util.tree_leaves(params)])
      self.assertEqual(actual, expected)

  @parameterized.parameters(
      dict(
          output_channels=2,
          kernel_shape=3,
          ),
      dict(
          output_channels=4,
          kernel_shape=5,
          ),
    )
  def test_conv1d_shape(
      self,
      output_channels: int,
      kernel_shape: int,
  ):
    """Tests that ConvLonLat outputs and params have expected shapes."""
    gin.enter_interactive_mode()

    @gin.configurable
    def layer_forward(inputs, layer):  # defines a forward function
      return layer()(inputs)

    config = '\n'.join([
        'layer_forward.layer = @ConvLevel',
        f'ConvLevel.kernel_shape = {kernel_shape}',
        f'ConvLevel.output_channels = {output_channels}',
        'ConvLevel.name = "conv_sigma"',
    ])

    gin.clear_config()
    gin.parse_config(config)

    in_channels = 8
    input_shape = (in_channels, 24)
    key = jax.random.PRNGKey(42)
    inputs = jax.random.uniform(key, shape=input_shape)
    model = hk.without_apply_rng(hk.transform(layer_forward))
    params = model.init(key, inputs)
    out = model.apply(params, inputs)
    with self.subTest('output_shape'):
      self.assertEqual(out.shape, (output_channels, 24))
    with self.subTest('total_params_count'):
      expected = count_conv_lon_lat_params(
          input_size=in_channels,
          kernel_shape=kernel_shape,
          output_size=output_channels,
          with_bias=True)
      actual = sum([np.prod(x.shape)
                    for x in jax.tree_util.tree_leaves(params)])
      self.assertEqual(actual, expected)

  @parameterized.parameters(
      dict(
          output_channels=2,
          channels=[6],
          kernel_shapes=[3, 3],
          dilation_rates=[1, 1],
      ),
      dict(
          output_channels=4,
          channels=[7],
          kernel_shapes=[5, 3],
          dilation_rates=[3, 1],
      ),
  )
  def test_cnn_level_shape(
      self,
      output_channels: int,
      channels: Sequence[int],
      kernel_shapes: Sequence[int],
      dilation_rates: Sequence[int],
  ):
    """Tests that VerticalConvNet outputs and params have expected shapes."""
    gin.enter_interactive_mode()

    @gin.configurable
    def layer_forward(inputs, layer):  # defines a forward function
      return layer(output_channels)(inputs)

    config = '\n'.join([
        'layer_forward.layer = @VerticalConvNet',
        f'VerticalConvNet.channels = {channels}',
        f'VerticalConvNet.kernel_shapes = {kernel_shapes}',
        f'VerticalConvNet.dilation_rates = {dilation_rates}',
    ])

    gin.clear_config()
    gin.parse_config(config)

    in_channels = 8
    n_levels = 12
    input_shape = (in_channels, n_levels)
    key = jax.random.PRNGKey(42)
    inputs = jax.random.uniform(key, shape=input_shape)
    model = hk.without_apply_rng(hk.transform(layer_forward))
    params = model.init(key, inputs)
    out = model.apply(params, inputs)
    with self.subTest('output_shape'):
      actual_shape = out.shape
      expected_shape = (output_channels, n_levels)
      self.assertEqual(actual_shape, expected_shape)

  @parameterized.parameters(
      dict(
          inputs_shape=(3, 12),  # self-attention on d=3 inputs, returning 4.
          latents_shape=None,
          pos_encoding_shape=None,
          expected_output_shape=(4, 12),
          ),
      dict(
          inputs_shape=(3, 7),  # attending to latents on 12 levels.
          latents_shape=(2, 12),
          pos_encoding_shape=None,
          expected_output_shape=(5, 7),  # results on the same levels as inputs.
          ),
      dict(
          inputs_shape=(3, 8),  # attending to latents on 7 levels.
          latents_shape=(2, 7),
          pos_encoding_shape=(1, 8),
          expected_output_shape=(4, 8),  # results on the same levels as inputs.
          ),
      dict(
          inputs_shape=(3, 6),
          latents_shape=(2, 7),
          pos_encoding_shape=None,
          expected_output_shape=(8, 6),
          # can skip projection because output_size == latent_size.
          extra_config_str='LevelTransformer.skip_final_projection = True',
          ),
      dict(
          inputs_shape=(8, 6),
          latents_shape=(2, 7),
          pos_encoding_shape=None,
          expected_output_shape=(1, 6),
          # can skip input projection because input_shape[0] == latent_size.
          extra_config_str='LevelTransformer.input_projection_net = None',
          ),
    )
  def test_level_transformer_shape(
      self,
      inputs_shape: Tuple[int, int],
      latents_shape: Optional[Tuple[int, int]],
      pos_encoding_shape: Optional[Tuple[int, int]],
      expected_output_shape: Tuple[int, int],
      extra_config_str: str = '',
  ):
    """Tests that LevelTransformer outputs have expected shapes."""
    gin.enter_interactive_mode()
    output_channels, _ = expected_output_shape

    @gin.configurable
    def layer_forward(inputs, latents, pos_encoding, layer):
      return layer(output_channels)(inputs, latents, pos_encoding)

    config = '\n'.join([
        'layer_forward.layer = @LevelTransformer',
        'LevelTransformer.n_layers = 1',
        'LevelTransformer.latent_size = 8',
        'LevelTransformer.num_heads = 2',
        'LevelTransformer.key_size = 4',
    ])
    gin.clear_config()
    gin.parse_config(config)
    gin.parse_config(extra_config_str)

    rng_seq = hk.PRNGSequence(42)
    inputs = jax.random.uniform(next(rng_seq), shape=inputs_shape)
    if latents_shape is not None:
      latents = jax.random.uniform(next(rng_seq), shape=latents_shape)
    else:
      latents = None
    if pos_encoding_shape is not None:
      pos_encoding = jax.random.uniform(next(rng_seq), shape=pos_encoding_shape)
    else:
      pos_encoding = None

    model = hk.without_apply_rng(hk.transform(layer_forward))
    params = model.init(next(rng_seq), inputs, latents, pos_encoding)
    out = model.apply(params, inputs, latents, pos_encoding)
    with self.subTest('output_shape'):
      self.assertEqual(out.shape, expected_output_shape)


def count_mlp_uniform_params(
    input_size: int,
    num_hidden_units: int,
    num_hidden_layers: int,
    output_size: int,
    with_bias: bool):
  """Returns the number of parameters in MlpUniform based on settings."""
  n_input = input_size * num_hidden_units + num_hidden_units * with_bias
  n_hidden = num_hidden_units * num_hidden_units + num_hidden_units * with_bias
  n_output = num_hidden_units * output_size + output_size * with_bias
  return n_input + n_hidden * (num_hidden_layers - 1) + n_output


def count_conv_lon_lat_params(
    input_size: int,
    kernel_shape: int,
    output_size: int,
    with_bias: bool):
  """Returns the number of parameters in ConvLonLat based on settings."""
  n_w_params = np.prod(kernel_shape) * input_size * output_size
  n_b_params = output_size
  return n_w_params + n_b_params * with_bias

if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
