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
"""Tests for transforms."""

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from neuralgcm import transforms  # pylint: disable=unused-import
import numpy as np

NLAYER = 10
COORDS = coordinate_systems.CoordinateSystem(
    spherical_harmonic.Grid.T21(),
    sigma_coordinates.SigmaCoordinates.equidistant(NLAYER)
    )


class TransformsTest(parameterized.TestCase):
  """Tests transforms."""

  @parameterized.parameters(
      dict(),
  )
  def test_nondimensionalization_transforms(
      self,
  ):
    """Tests that mapping produces expected output structures."""
    gin.enter_interactive_mode()
    gin.clear_config()
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(6)
    )
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    dt = physics_specs.nondimensionalize(10 * scales.units.minute)
    aux_features = {}

    @gin.configurable
    def transform_forward(x, transform):  # defines a forward function
      return transform(coords, dt, physics_specs, aux_features, None)(x)

    units_map = {
        'divergence': '1 / second',
        'vorticity': '1 / second',
        'temperature_variation': 'kelvin',
        'log_surface_pressure': 'dimensionless',
        'tracers': {'specific_humidity': 'dimensionless', 'unused_val': 'm**2'},
    }
    matching_units_map = units_map | {'diagnostics': {}}
    matching_units_map['tracers'].pop('unused_val')
    config = '\n'.join([
        f'NondimensionalizeTransform.inputs_to_units_mapping = {units_map}',
        f'RedimensionalizeTransform.inputs_to_units_mapping = {units_map}',
        'nondim/transform_forward.transform = @NondimensionalizeTransform',
        'redim/transform_forward.transform = @RedimensionalizeTransform'
    ])
    gin.parse_config(config)

    inputs = {
        'diagnostics': {},
        'vorticity': jnp.ones(coords.nodal_shape),
        'divergence': jnp.ones(coords.nodal_shape),
        'temperature_variation': jnp.ones(coords.nodal_shape),
        'log_surface_pressure': jnp.ones(coords.surface_nodal_shape),
        'tracers': {
            'specific_humidity': jnp.ones(coords.nodal_shape)},
    }
    with self.subTest('nondimensionalization'):
      expected = jax.tree_util.tree_map(
          lambda x, y: physics_specs.nondimensionalize(x * scales.Quantity(y)),
          inputs, matching_units_map)
      with gin.config_scope('nondim'):
        transform_model = hk.without_apply_rng(hk.transform(transform_forward))
        actual = transform_model.apply(None, inputs)
        for x, y in zip(jax.tree_util.tree_leaves(expected),
                        jax.tree_util.tree_leaves(actual)):
          np.testing.assert_allclose(x, y)
    nondimensional_inputs = actual

    with self.subTest('redimensionalization'):
      expected = inputs
      with gin.config_scope('redim'):
        transform_model = hk.without_apply_rng(hk.transform(transform_forward))
        actual = transform_model.apply(None, nondimensional_inputs)
        for x, y in zip(jax.tree_util.tree_leaves(expected),
                        jax.tree_util.tree_leaves(actual)):
          np.testing.assert_allclose(x, y)

  def test_level_scale_transform(self):
    """Tests that level scaling works as expected."""
    gin.enter_interactive_mode()
    gin.clear_config()
    n_layers = 6
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(n_layers)
    )
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    dt = physics_specs.nondimensionalize(10 * scales.units.minute)
    aux_features = {}

    @gin.configurable
    def transform_forward(x, transform):  # defines a forward function
      return transform(coords, dt, physics_specs, aux_features)(x)

    level_scales = [1., 1., 1., 10., 100., 3.]
    config = '\n'.join([
        'scale/transform_forward.transform = @LevelScale',
        'inverse_scale/transform_forward.transform = @InverseLevelScale',
        f'LevelScale.scales = {level_scales}',
        'LevelScale.keys_to_scale = ("x", "z", "specific_humidity")',
        f'InverseLevelScale.scales = {level_scales}',
        'InverseLevelScale.keys_to_scale = ("x", "z", "specific_humidity")',
    ])
    gin.parse_config(config)

    vertical_values = np.arange(n_layers)[:, np.newaxis, np.newaxis]
    inputs = {
        'vorticity': np.ones(coords.nodal_shape) * vertical_values,
        'x': np.ones(coords.nodal_shape) * vertical_values,
        'tracers': {
            'specific_humidity': np.ones(coords.nodal_shape) * vertical_values,
            'y': np.ones(coords.nodal_shape) * vertical_values,
        },
    }
    transform_model = hk.without_apply_rng(hk.transform(transform_forward))

    with gin.config_scope('scale'):
      scaled = transform_model.apply(None, inputs)

    with self.subTest('where_scaling_is_applied'):
      expected_level_scale = np.asarray(level_scales)[:, np.newaxis, np.newaxis]
      np.testing.assert_allclose(
          scaled['tracers']['specific_humidity'],
          inputs['tracers']['specific_humidity'] * expected_level_scale)
      np.testing.assert_allclose(
          scaled['x'], inputs['x'] * expected_level_scale)
    with self.subTest('where_scaling_is_omitted'):
      np.testing.assert_allclose(scaled['vorticity'], inputs['vorticity'])
      np.testing.assert_allclose(scaled['tracers']['y'], inputs['tracers']['y'])

    with self.subTest('level_scale_round_trip'):
      with gin.config_scope('inverse_scale'):
        round_trip = transform_model.apply(None, scaled)
      jax.tree_util.tree_map(np.testing.assert_allclose, inputs, round_trip)

  @parameterized.named_parameters(
      dict(
          testcase_name='test_1_specific_humidity',
          inputs={
              'vorticity': np.ones(COORDS.nodal_shape),
              'divergence': np.ones(COORDS.nodal_shape),
              'temperature_variation': np.ones(COORDS.nodal_shape),
              'log_surface_pressure': np.ones(COORDS.surface_nodal_shape),
              'tracers': {
                  'specific_humidity': np.broadcast_to(
                      np.arange(NLAYER)[:, np.newaxis, np.newaxis],
                      (NLAYER,) + COORDS.horizontal.nodal_shape,
                  )
              },
          },
          expected_outputs={
              'vorticity': np.ones(COORDS.nodal_shape),
              'divergence': np.ones(COORDS.nodal_shape),
              'temperature_variation': np.ones(COORDS.nodal_shape),
              'log_surface_pressure': np.ones(COORDS.surface_nodal_shape),
              'tracers': {
                  'specific_humidity': np.broadcast_to(
                      np.arange(NLAYER)[2:8, np.newaxis, np.newaxis],
                      (6,) + COORDS.horizontal.nodal_shape,
                  )
              },
          },
          sigma_ranges={'specific_humidity': (0.2, 0.8)},
      ),
      dict(
          testcase_name='test_2_fields',
          inputs={
              'vorticity': np.ones(COORDS.nodal_shape),
              'divergence': np.ones(COORDS.nodal_shape),
              'temperature_variation': np.ones(COORDS.nodal_shape),
              'log_surface_pressure': np.ones(COORDS.surface_nodal_shape),
              'tracers': {
                  'specific_humidity': np.broadcast_to(
                      np.arange(NLAYER)[:, np.newaxis, np.newaxis],
                      (NLAYER,) + COORDS.horizontal.nodal_shape,
                  )
              },
          },
          expected_outputs={
              'vorticity': np.ones(COORDS.nodal_shape)[2:8, :, :],
              'divergence': np.ones(COORDS.nodal_shape),
              'temperature_variation': np.ones(COORDS.nodal_shape),
              'log_surface_pressure': np.ones(COORDS.surface_nodal_shape),
              'tracers': {
                  'specific_humidity': np.broadcast_to(
                      np.arange(NLAYER)[2:8, np.newaxis, np.newaxis],
                      (6,) + COORDS.horizontal.nodal_shape,
                  )
              },
          },
          sigma_ranges={
              'specific_humidity': (0.2, 0.8),
              'vorticity': (0.2, 0.8),
          },
      ),
  )
  def test_truncate_sigma_levels_transform(
      self, inputs, expected_outputs, sigma_ranges):
    """Tests TruncateSigmaLevels."""
    coords = COORDS
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    dt = physics_specs.nondimensionalize(10 * scales.units.minute)
    aux_features = {}

    def truncate_transform_fwd(x):
      return transforms.TruncateSigmaLevels(
          coords, dt, physics_specs, aux_features, sigma_ranges=sigma_ranges)(x)
    transform_model = hk.without_apply_rng(hk.transform(truncate_transform_fwd))
    actual_outputs = transform_model.apply(None, inputs)
    with self.subTest('dict'):
      actual_outputs = transform_model.apply(None, inputs)
      jax.tree_util.tree_map(
          np.testing.assert_allclose, expected_outputs, actual_outputs
      )
    with self.subTest('state'):
      state_inputs = primitive_equations.State(**inputs)
      expected_state_outputs = primitive_equations.State(**expected_outputs)
      actual_state_outputs = transform_model.apply(None, state_inputs)
      jax.tree_util.tree_map(
          np.testing.assert_allclose,
          expected_state_outputs,
          actual_state_outputs,
      )

  def test_take_surface_adjacent_sigma_level(self):
    """Tests TakeSurfaceAdjacentSigmaLevel."""
    coords = COORDS
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    dt = physics_specs.nondimensionalize(10 * scales.units.minute)
    aux_features = {}

    # define data that has value 1 only on the index=-1 sigma level
    volume_data = np.zeros(COORDS.nodal_shape)
    volume_data[-1] += 1
    surface_data = np.ones(COORDS.surface_nodal_shape)
    inputs = {
        'vorticity': volume_data,
        'divergence': volume_data,
        'temperature_variation': volume_data,
        'log_surface_pressure': surface_data,
        'tracers': {'specific_humidity': volume_data},
    }
    # expect sliced data to be shape (1, lon, lat) and have value 1
    expected_outputs = {
        'vorticity': surface_data,
        'divergence': surface_data,
        'temperature_variation': surface_data,
        'log_surface_pressure': surface_data,
        'tracers': {'specific_humidity': surface_data},
    }

    def transform_fwd(x):
      return transforms.TakeSurfaceAdjacentSigmaLevel(
          coords, dt, physics_specs, aux_features)(x)
    transform_model = hk.without_apply_rng(hk.transform(transform_fwd))
    actual_outputs = transform_model.apply(None, inputs)

    with self.subTest('dict'):
      actual_outputs = transform_model.apply(None, inputs)
      jax.tree_util.tree_map(
          np.testing.assert_allclose, expected_outputs, actual_outputs
      )

    with self.subTest('state'):
      state_inputs = primitive_equations.State(**inputs)
      expected_state_outputs = primitive_equations.State(**expected_outputs)
      actual_state_outputs = transform_model.apply(None, state_inputs)
      jax.tree_util.tree_map(
          np.testing.assert_allclose,
          expected_state_outputs,
          actual_state_outputs,
      )

  @parameterized.parameters(
      dict(transform_module=transforms.SoftClip, max_value=10.0),
      dict(transform_module=transforms.HardClip, max_value=15.5),
      dict(transform_module=transforms.SoftClip, max_value=50.0),
  )
  def test_clip_transforms(self, transform_module, max_value):
    """Tests that clipping transforms work as expected."""
    def transform_forward(x):  # defines a forward function
      return transform_module(None, None, None, None, max_value=max_value)(x)

    inputs = {
        'positive': np.arange(20),
        'negative': -np.arange(30),
        'both': np.arange(-30, 30),
        '~same': np.linspace(-max_value / 4, max_value / 4, 50),  # ~same.
        'tracers': {
            'c': np.ones(3),
            'y': np.arange(-30, 30),
        },
    }
    transform_model = hk.without_apply_rng(hk.transform(transform_forward))
    clipped = transform_model.apply(None, inputs)

    with self.subTest('valid_range'):
      jax.tree_util.tree_map(
          lambda x: self.assertGreaterEqual(max_value, x.max()), clipped)
      jax.tree_util.tree_map(
          lambda x: self.assertGreaterEqual(x.min(), -max_value), clipped)
    with self.subTest('near_identity_in_range'):
      jax.tree_util.tree_map(
          lambda x, y: np.testing.assert_allclose(x, y, atol=1e-3),
          inputs['~same'], clipped['~same'])

  @parameterized.parameters(
      dict(
          input_keys=('a', 'b'),
          regex_patterns='a',
          expected_out_keys=('a',)
      ),
      dict(
          input_keys=('a_1', 'b', 'a_2', 'a_3', 'b_1', 'ab_1', 't_a_1'),
          regex_patterns='a_.*',
          expected_out_keys=('a_1', 'a_2', 'a_3'),
      ),
      dict(
          input_keys=('a_1', 'b', 'a_2', 'a_3', 'b_1', 'ab_1', 't_a_1'),
          regex_patterns='a_.|ab.*',
          expected_out_keys=('a_1', 'a_2', 'a_3', 'ab_1'),
      ),
      dict(
          input_keys=('q1rpop', 'e1g', 'y2q', 'a_1e', 'b21t'),
          regex_patterns='.1.*',
          expected_out_keys=('q1rpop', 'e1g'),
      ),
  )
  def test_feature_selector(
      self,
      input_keys,
      regex_patterns,
      expected_out_keys,
  ):
    """Tests that FeatureSelector work as expected."""
    selector_fn = transforms.FeatureSelector(regex_patterns)
    values = np.arange(20)  # values to grab for inputs.
    inputs = {k: values[i] for i, k in enumerate(input_keys)}
    actual = selector_fn(inputs)
    expected = {
        k: values[i]
        for i, k in enumerate(input_keys) if k in expected_out_keys}
    self.assertDictEqual(expected, actual)

  def test_sigma_squash_transform(self):
    """Tests that SquashLevelsTransform preserves shape."""
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    transform_fn = transforms.SquashLevelsTransform(
        COORDS, 0.1, physics_specs, {},
        low_cutoffs=(0.05, 0.1),
        # Make the high cutoff > 1 so ground level variables will not be zero'd.
        high_cutoffs=(0.86, 1.1),
    )
    # COORDS.nodal_shape ~ (levels, lon, lat)
    values = np.ones(COORDS.nodal_shape)  # values to grab for inputs.
    inputs = {k: values for k in ('a', 'b', 'c', 'd')}
    inputs['sim_time'] = np.ones(shape=())
    inputs['tracers'] = {'sh': values}

    # log_surface_pressure is defined only at ground level.
    inputs['log_surface_pressure'] = values[:1]  # (1, lon, lat)

    # Our high_cutoffs ensured we do not squash at ground level.
    self.assertGreater(inputs['log_surface_pressure'].min(), 0.5)

    out = transform_fn(inputs)

    for x, y in zip(jax.tree_util.tree_leaves(inputs),
                    jax.tree_util.tree_leaves(out)):
      self.assertEqual(x.shape, y.shape)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
