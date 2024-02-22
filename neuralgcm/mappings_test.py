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
"""Tests for mappings.py."""

from absl.testing import absltest
from absl.testing import parameterized
import gin
import haiku as hk
import jax
import jax.numpy as jnp

from neuralgcm import mappings  # pylint: disable=unused-import
from neuralgcm import transforms  # pylint: disable=unused-import
import numpy as np


@gin.register
class MappingsTowerMock(hk.Module):
  """Tower that returns random outputs with same nodal shape and ndim as inputs."""

  def __init__(self, output_size: int):
    super().__init__(name=None)
    self.output_size = output_size

  def __call__(self, inputs, latents=None, position_encoding=None):
    nodal_shape = inputs.shape[1:]
    return jax.random.uniform(
        jax.random.PRNGKey(42), shape=(self.output_size,) + nodal_shape
    )


class MappingsTest(parameterized.TestCase):
  """Tests mappings modules."""

  # Sharp bits: output shape values need to be arrays, otherwise mapping over
  # pytree leaves will return the integers of the shape instead.
  # Using `input_shapes` to set inputs to make sponge link more readable
  @parameterized.parameters(
      dict(
          mapping_name='NodalMapping',
          input_shapes={
              'u': np.asarray((13, 64, 32)),
              'v': np.asarray((13, 64, 32)),
              'cos_latitude': np.asarray((1, 64, 32)),
              'learned_feature': np.asarray((4, 64, 32)),
          },
          # the tower output_size will be 5 (sum of dim=0 sizes)
          output_shapes={
              'prediction_1': np.asarray((4, 64, 32)),
              'prediction_2': np.asarray((4, 64, 32)),
              'nesting_key': {
                  'nested_prediction': np.asarray((1, 64, 32)),
              },
          },
          value_error=None,
      ),
      dict(
          mapping_name='NodalVolumeMapping',
          input_shapes={
              'u': np.asarray((13, 64, 32)),
              'v': np.asarray((13, 64, 32)),
          },
          # the tower output_size will be 3 (number of keys)
          output_shapes={
              'prediction_1': np.asarray((13, 64, 32)),
              'prediction_2': np.asarray((13, 64, 32)),
              'nesting_key': {
                  'nested_prediction_1': np.asarray((13, 64, 32)),
                  'nested_prediction_2': np.asarray((13, 64, 32)),
              },
          },
          value_error=None,
      ),
      dict(
          mapping_name='NodalVolumeMapping',
          input_shapes={
              'u': np.asarray((1, 64, 32)),
              'v': np.asarray((4, 64, 32)),  # can't stack with u
          },
          # the tower output_size will be 1 (number of keys)
          output_shapes={'prediction': np.asarray((1, 64, 32))},
          value_error='All input arrays must have the same shape.',
      ),
      dict(
          mapping_name='NodalVolumeTransformerMapping',
          input_shapes={
              'u': np.asarray((1, 64, 32)),
              'v': np.asarray((4, 64, 32)),  # can't stack with u
          },
          # the tower output_size will be 1 (number of keys)
          output_shapes={'prediction': np.asarray((1, 64, 32))},
          value_error='All input arrays must have the same shape.',
      ),
  )
  def test_nodal_mapping_shapes(
      self, mapping_name, input_shapes, output_shapes, value_error,
  ):
    """Tests that mapping produces expected output structures."""
    gin.enter_interactive_mode()
    gin.clear_config()

    @gin.configurable
    def mapping_forward(x, mapping):  # defines a forward function
      return mapping(output_shapes)(x)

    long_name = 'NodalVolumeTransformerMapping'
    config = '\n'.join([
        f'mapping_forward.mapping = @{mapping_name}',
        'NodalMapping.tower_factory = @MappingsTowerMock',
        'NodalVolumeMapping.tower_factory = @MappingsTowerMock',
        f'{long_name}.latent_size = 5',
        f'{long_name}.encoder_transformer_tower_factory = @MappingsTowerMock',
        f'{long_name}.decoder_transformer_tower_factory = @MappingsTowerMock',
        f'{long_name}.encoder_inputs_selection_module = @FeatureSelector',
        'FeatureSelector.regex_patterns = ".*"'  # selects all features.
    ])
    gin.parse_config(config)

    inputs = jax.tree_util.tree_map(jnp.ones, input_shapes)
    model = hk.without_apply_rng(hk.transform(mapping_forward))
    if value_error is not None:
      with self.assertRaisesRegex(ValueError, value_error):
        params = model.init(jax.random.PRNGKey(42), inputs)
    else:
      params = model.init(jax.random.PRNGKey(42), inputs)
      output = model.apply(params, inputs)
      for output_item, expected_shape in zip(
          jax.tree_util.tree_leaves(output),
          jax.tree_util.tree_leaves(output_shapes)):
        self.assertEqual(output_item.shape, tuple(expected_shape.tolist()))


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
