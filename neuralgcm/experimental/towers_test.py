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
import functools
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from neuralgcm.experimental import standard_layers
from neuralgcm.experimental import towers
import numpy as np


class TowersTest(parameterized.TestCase):
  """Tests Tower primitives."""

  @parameterized.named_parameters(
      dict(
          testcase_name='_mlp_uniform',
          nn_input_shape=(12,),
          spatial_shape=(10, 20),
          output_size=4,
          nn_factory=functools.partial(
              standard_layers.MlpUniform,
              hidden_size=3,
              n_hidden_layers=4,
              use_bias=True,
          ),
      ),
      dict(
          testcase_name='_mlp_uniform_with_remat',
          nn_input_shape=(3,),
          spatial_shape=(16, 7),
          output_size=3,
          nn_factory=functools.partial(
              standard_layers.MlpUniform,
              hidden_size=1,
              n_hidden_layers=13,
              use_bias=False,
          ),
          apply_remat=True,
      ),
      dict(
          testcase_name='_cnn',
          nn_input_shape=(4, 12),  # here 12 corresponds to level axis.
          spatial_shape=(6, 6),
          output_size=2,
          nn_factory=functools.partial(
              standard_layers.CnnLevel,
              channels=(3, 5, 7),
              kernel_sizes=3,
              use_bias=False,
          ),
      )
  )
  def test_column_tower(
      self,
      nn_input_shape: tuple[int, ...],
      spatial_shape: tuple[int, ...],
      output_size: int,
      nn_factory: standard_layers.UnaryLayerFactory,
      apply_remat: bool = False,
  ):
    input_size = nn_input_shape[0]
    tower = towers.ColumnTower(
        input_size,
        output_size,
        column_net_factory=nn_factory,
        rngs=nnx.Rngs(0),
        apply_remat=apply_remat,
    )
    inputs = jax.random.uniform(
        jax.random.PRNGKey(42), nn_input_shape + spatial_shape
    )
    outputs = tower(inputs)
    with self.subTest('output_shape'):
      expected_shape = (output_size,) + nn_input_shape[1:] + spatial_shape
      self.assertEqual(outputs.shape, expected_shape)
    with self.subTest('same_params_count_as_column'):
      net = nn_factory(input_size, output_size, rngs=nnx.Rngs(1))
      expected = sum([
          np.prod(x.shape) for x in jax.tree.leaves(nnx.state(net, nnx.Param))
      ])
      actual = sum([
          np.prod(x.shape) for x in jax.tree.leaves(nnx.state(tower, nnx.Param))
      ])
      self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='_mlp_uniform',
          nn_input_shape=(12,),
          spatial_shape=(10, 20),
          output_size=4,
          tower_factory=functools.partial(
              towers.ColumnTower,
              column_net_factory=functools.partial(
                  standard_layers.MlpUniform, hidden_size=3, n_hidden_layers=4
              ),
          ),
      ),
      dict(
          testcase_name='_cnn',
          nn_input_shape=(4, 12),  # here 12 corresponds to level axis.
          spatial_shape=(6, 6),
          output_size=2,
          tower_factory=functools.partial(
              towers.ColumnTower,
              column_net_factory=functools.partial(
                  standard_layers.CnnLevel,
                  channels=(3, 5, 7),
                  kernel_sizes=3,
                  use_bias=False,
              ),
          ),
      ),
  )
  def test_epd_tower(
      self,
      nn_input_shape: tuple[int, ...],
      spatial_shape: tuple[int, ...],
      output_size: int,
      tower_factory: towers.UnaryTowerFactory,
  ):
    input_size = nn_input_shape[0]
    tower = towers.EpdTower(
        input_size,
        output_size,
        latent_size=8,
        num_process_blocks=2,
        encode_tower_factory=tower_factory,
        process_tower_factory=tower_factory,
        decode_tower_factory=tower_factory,
        rngs=nnx.Rngs(0),
    )
    inputs = jax.random.uniform(
        jax.random.PRNGKey(42), nn_input_shape + spatial_shape
    )
    outputs = tower(inputs)
    with self.subTest('output_shape'):
      expected_shape = (output_size,) + nn_input_shape[1:] + spatial_shape
      self.assertEqual(outputs.shape, expected_shape)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
