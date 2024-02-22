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
"""Tests for towers.py."""

from absl.testing import absltest
from absl.testing import parameterized
import gin
import haiku as hk
import jax
from neuralgcm import towers  # pylint: disable=unused-import
import numpy as np


class ColumnTowerTest(parameterized.TestCase):
  """Tests Column tower."""

  @parameterized.parameters(
      dict(
          output_size=2,
          num_hidden_units=3,
          num_hidden_layers=4,
          with_bias=True),
      dict(
          output_size=4,
          num_hidden_units=5,
          num_hidden_layers=6,
          with_bias=False),
      )
  def test_column_tower_mlp_shapes(
      self,
      output_size: int,
      num_hidden_units: int,
      num_hidden_layers: int,
      with_bias: bool,
  ):
    gin.enter_interactive_mode()

    @gin.configurable
    def tower_forward(x, *, tower):  # defines a forward function
      return tower(output_size)(x)

    config = '\n'.join([
        'tower_forward.tower = @ColumnTower',
        'ColumnTower.column_net_factory = @MlpUniform',
        'ColumnTower.name = "column_tower"',
        f'MlpUniform.num_hidden_units = {num_hidden_units}',
        f'MlpUniform.num_hidden_layers = {num_hidden_layers}',
        f'MlpUniform.with_bias = {with_bias}',
    ])

    gin.clear_config()
    gin.parse_config(config)

    n_features = 8
    input_size = (n_features, 10, 20)
    inputs = np.random.uniform(size=input_size)
    model = hk.without_apply_rng(hk.transform(tower_forward))
    params = model.init(jax.random.PRNGKey(42), inputs)
    out = model.apply(params, inputs)
    with self.subTest('output_shape'):
      self.assertEqual(out.shape, (output_size, 10, 20))
    with self.subTest('total_params_count'):
      expected = count_mlp_uniform_params(
          input_size=n_features,
          num_hidden_units=num_hidden_units,
          num_hidden_layers=num_hidden_layers,
          output_size=output_size,
          with_bias=with_bias)
      actual = sum([np.prod(x.shape)
                    for x in jax.tree_util.tree_leaves(params)])
      self.assertEqual(actual, expected)


class VerticalConvTowerTest(parameterized.TestCase):
  """Tests VerticalConvTower tower."""

  @parameterized.parameters(
      dict(
          output_size=7,
          channels=[4],
          kernel_shape=3,
      ),
      dict(
          output_size=20,
          channels=[4, 8, 10],
          kernel_shape=5,
      ),
  )
  def test_conv_tower_shapes(
      self,
      output_size: int,
      channels: list[int],
      kernel_shape: int,
  ):
    """Tests that VerticalConvTower outputs and params have expected shapes."""
    gin.enter_interactive_mode()

    @gin.configurable
    def tower_forward(x, tower):  # defines a forward function
      return tower(output_size)(x)

    config = '\n'.join([
        'tower_forward.tower = @VerticalConvTower',
        f'VerticalConvTower.channels = {channels}',
        f'VerticalConvTower.kernel_shape = {kernel_shape}',
        'VerticalConvTower.name = "conv1D_tower"',
    ])

    gin.clear_config()
    gin.parse_config(config)
    n_levels = 24
    in_channels = 8
    input_shape = (in_channels, n_levels, 10, 20)
    key = jax.random.PRNGKey(42)
    inputs = jax.random.uniform(key, shape=input_shape)
    model = hk.without_apply_rng(hk.transform(tower_forward))
    params = model.init(key, inputs)
    out = model.apply(params, inputs)
    tot_channels = channels
    tot_channels.append(output_size)
    with self.subTest('output_shape'):
      self.assertEqual(out.shape, (output_size, n_levels, 10, 20))
    with self.subTest('total_params_count'):
      expected = count_conv1d_tower_params(
          in_channels=in_channels,
          kernel_shape=kernel_shape,
          channels=tot_channels,
          )
      actual = sum([np.prod(x.shape)
                    for x in jax.tree_util.tree_leaves(params)])
      self.assertEqual(actual, expected)


class Conv2DTowerTest(parameterized.TestCase):
  """Tests ConvNet tower."""

  @parameterized.parameters(
      dict(
          output_size=2,
          num_hidden_units=3,
          num_hidden_layers=4,
          kernel_shape=(3, 3),
          with_bias=True,),
      dict(
          output_size=4,
          num_hidden_units=5,
          num_hidden_layers=6,
          kernel_shape=(5, 5),
          with_bias=False,),
      )
  def test_conv_tower_shapes(
      self,
      output_size: int,
      num_hidden_units: int,
      num_hidden_layers: int,
      kernel_shape: tuple[int, int],
      with_bias: bool,
  ):
    """Tests that Conv2DTower outputs and params have expected shapes."""
    gin.enter_interactive_mode()

    @gin.configurable
    def tower_forward(x, tower):  # defines a forward function
      return tower(output_size)(x)

    config = '\n'.join([
        'tower_forward.tower = @Conv2DTower',
        f'Conv2DTower.num_hidden_units = {num_hidden_units}',
        f'Conv2DTower.num_hidden_layers = {num_hidden_layers}',
        f'Conv2DTower.kernel_shape = {kernel_shape}',
        f'Conv2DTower.with_bias = {with_bias}',
        'Conv2DTower.name = "conv_tower"',
    ])

    gin.clear_config()
    gin.parse_config(config)

    n_features = 8
    input_size = (n_features, 10, 20)
    inputs = np.random.uniform(size=input_size)
    model = hk.without_apply_rng(hk.transform(tower_forward))
    params = model.init(jax.random.PRNGKey(42), inputs)
    out = model.apply(params, inputs)
    with self.subTest('output_shape'):
      self.assertEqual(out.shape, (output_size, 10, 20))
    with self.subTest('total_params_count'):
      expected = count_conv_tower_params(
          input_size=n_features,
          kernel_shape=kernel_shape,
          num_hidden_units=num_hidden_units,
          num_hidden_layers=num_hidden_layers,
          output_size=output_size,
          with_bias=with_bias)
      actual = sum([np.prod(x.shape)
                    for x in jax.tree_util.tree_leaves(params)])
      self.assertEqual(actual, expected)


class EpdTowerTest(parameterized.TestCase):
  """Tests encode/process/decode tower."""

  @parameterized.parameters(
      dict(
          output_size=3,
          latent_size=6,
          layer_size=7,
          num_process_layers=2,
          num_process_blocks=2,),
      dict(
          output_size=5,
          latent_size=13,
          layer_size=4,
          num_process_layers=3,
          num_process_blocks=1,),
  )
  def test_epd_tower_mlp_shapes(
      self,
      output_size: int,
      latent_size: int,
      layer_size: int,
      num_process_layers: int,
      num_process_blocks: int,
  ):
    """Tests that EpdTower outputs and params have expected shapes."""
    gin.enter_interactive_mode()

    @gin.configurable
    def tower_forward(x, tower):  # defines a forward function
      return tower(output_size)(x)

    config = '\n'.join([
        'tower_forward.tower = @EpdTower',
        f'EpdTower.latent_size = {latent_size}',
        f'EpdTower.num_process_blocks = {num_process_blocks}',
        'EpdTower.encode_tower_factory = @encode/ColumnTower',
        'EpdTower.process_tower_factory = @process/ColumnTower',
        'EpdTower.decode_tower_factory = @decode/ColumnTower',
        'encode/ColumnTower.column_net_factory = @encode/MlpUniform',
        'encode/ColumnTower.name = "encode_tower"',
        f'encode/MlpUniform.num_hidden_units = {layer_size}',
        'encode/MlpUniform.num_hidden_layers = 1',
        'encode/MlpUniform.name = "encode_layer"',
        'process/ColumnTower.column_net_factory = @process/MlpUniform',
        'process/ColumnTower.name = "process_tower"',
        f'process/MlpUniform.num_hidden_units = {layer_size}',
        f'process/MlpUniform.num_hidden_layers = {num_process_layers}',
        'process/MlpUniform.name = "process_layer"',
        'decode/ColumnTower.column_net_factory = @decode/MlpUniform',
        'decode/ColumnTower.name = "decode_tower"',
        f'decode/MlpUniform.num_hidden_units = {layer_size}',
        'decode/MlpUniform.num_hidden_layers = 2',
        'decode/MlpUniform.name = "decode_layer"',
    ])

    gin.clear_config()
    gin.parse_config(config)

    n_features = 8
    input_size = (n_features, 10, 20)
    inputs = np.random.uniform(size=input_size)
    model = hk.without_apply_rng(hk.transform(tower_forward))
    params = model.init(jax.random.PRNGKey(42), inputs)
    out = model.apply(params, inputs)
    with self.subTest('output_shape'):
      self.assertEqual(out.shape, (output_size, 10, 20))
    with self.subTest('total_params_count'):
      num_encode_params = count_mlp_uniform_params(
          input_size=n_features,
          num_hidden_units=layer_size,
          num_hidden_layers=1,
          output_size=latent_size,
          with_bias=True)
      num_process_params = count_mlp_uniform_params(
          input_size=latent_size,
          num_hidden_units=layer_size,
          num_hidden_layers=num_process_layers,
          output_size=latent_size,
          with_bias=True)
      num_decode_params = count_mlp_uniform_params(
          input_size=latent_size,
          num_hidden_units=layer_size,
          num_hidden_layers=2,
          output_size=output_size,
          with_bias=True)
      expected = (
          num_encode_params + num_process_blocks * num_process_params +
          num_decode_params)
      actual = sum([np.prod(x.shape)
                    for x in jax.tree_util.tree_leaves(params)])
      self.assertEqual(actual, expected)


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


def count_conv_tower_params(
    input_size: int,
    kernel_shape: int,
    num_hidden_units: int,
    num_hidden_layers: int,
    output_size: int,
    with_bias: bool):
  """Returns the number of parameters in Conv2DTower based on settings."""
  n_input = count_conv_lon_lat_params(
      input_size=input_size,
      kernel_shape=kernel_shape,
      output_size=num_hidden_units,
      with_bias=with_bias)
  n_hidden = count_conv_lon_lat_params(
      input_size=num_hidden_units,
      kernel_shape=kernel_shape,
      output_size=num_hidden_units,
      with_bias=with_bias)
  n_output = count_conv_lon_lat_params(
      input_size=num_hidden_units,
      kernel_shape=kernel_shape,
      output_size=output_size,
      with_bias=with_bias)
  return n_input + n_hidden * (num_hidden_layers - 1) + n_output


def count_conv1d_tower_params(
    in_channels: int,
    kernel_shape: int,
    channels: list[int],
    with_bias: bool = True,
    ) -> int:
  """Returns the number of parameters in Conv1D."""
  n_params = []
  input_current = in_channels
  for channel in channels:
    n_params.append(count_conv_lon_lat_params(
        input_size=input_current,
        kernel_shape=kernel_shape,
        output_size=channel,
        with_bias=with_bias))
    input_current = channel
  return sum(n_params)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
