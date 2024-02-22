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
"""Tests for features.py."""

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import coordinate_systems
from dinosaur import primitive_equations_states
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import gin
import haiku as hk
import jax
from neuralgcm import embeddings  # pylint: disable=unused-import
from neuralgcm import features  # pylint: disable=unused-import
from neuralgcm import mappings  # pylint: disable=unused-import
from neuralgcm import physics_specifications
import numpy as np


units = scales.units


class FeaturesTest(parameterized.TestCase):
  """Tests feature modules."""

  @parameterized.parameters(
      ('VelocityAndPrognostics', 'modal'),
      ('NodalInputVelocityAndPrognostics', 'nodal'),
      ('RadiationFeatures', 'modal'),
      ('RadiationFeatures', 'nodal'),
      ('OrbitalTimeFeatures', 'modal'),
      ('OrbitalTimeFeatures', 'nodal'),
      ('ForcingFeatures', 'modal'),
      ('ForcingFeatures', 'nodal'),
      ('LatitudeFeatures', 'modal'),
      ('LatitudeFeatures', 'nodal'),
      ('OrographyFeatures', 'modal'),
      ('OrographyFeatures', 'nodal'),
      ('OneHotAuxFeatures', 'modal'),
      ('OneHotAuxFeatures', 'nodal'),
      ('LearnedPositionalFeatures', 'modal'),
      ('LearnedPositionalFeatures', 'nodal'),
      ('CombinedFeatures', 'modal'),
      ('CombinedFeatures', 'nodal'),
      ('MemoryVelocityAndValues', 'modal'),
      ('RandomnessFeatures', 'modal'),
      ('RandomnessFeatures', 'modal', True),
  )
  def test_feature_module_shapes(
      self,
      features_name,
      input_type,
      dict_randomness=False,
  ):
    """Tests that the returned features have expected nodal shape."""
    gin.enter_interactive_mode()
    gin.clear_config()
    physics_gin_config = '\n'.join([
        'get_physics_specs.construct_fn = @primitive_eq_specs_constructor',
    ])
    gin.parse_config(physics_gin_config)
    physics_specs = physics_specifications.get_physics_specs()
    dt = physics_specs.nondimensionalize(100 * units.s)
    # setting up coordinates and initial state.
    grid = spherical_harmonic.Grid.T21()
    vertical_grid = sigma_coordinates.SigmaCoordinates.equidistant(4)
    coords = coordinate_systems.CoordinateSystem(grid, vertical_grid)
    initial_state_fn, aux_features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    # handling float and integer covariates
    fake_land_sea_mask = np.random.random(grid.nodal_shape)  # float [0, 1)
    aux_features['land_sea_mask'] = fake_land_sea_mask
    fake_soil_type = np.random.randint(0, 8, grid.nodal_shape)
    aux_features['soil_type'] = fake_soil_type
    reference_datetime = np.datetime64('1970-01-01T00:00:00')
    aux_features[xarray_utils.REFERENCE_DATETIME_KEY] = reference_datetime
    inputs = initial_state_fn(jax.random.PRNGKey(42)).asdict()
    inputs['tracers'] = {'specific_humidity': inputs['divergence']}
    inputs['sim_time'] = 0.5
    if input_type == 'nodal':
      inputs = coords.horizontal.to_nodal(inputs)
    forcing = {
        'total_cloud_cover': np.random.random(coords.surface_nodal_shape),
        'sim_time': 0.5,
    }

    @gin.configurable
    def compute_feature_fwd(
        x, memory, diagnostics, randomness, forcing, feature_module
    ):
      feature_fn = feature_module(coords, dt, physics_specs, aux_features)
      return feature_fn(x, memory, diagnostics, randomness, forcing)

    gin_config = '\n'.join([
        f'compute_feature_fwd.feature_module = @{features_name}',
        'OneHotAuxFeatures.covariate_keys = ("land_sea_mask", "soil_type")',
        'OneHotAuxFeatures.convert_float_to_int = True',
        'LearnedPositionalFeatures.latent_size = 4',
        'ForcingFeatures.forcing_to_include = ("total_cloud_cover",)',
        'CombinedFeatures.feature_modules ='
        + ' (@OneHotAuxFeatures, @RadiationFeatures)',
    ])
    gin.parse_config(gin_config)
    model = hk.without_apply_rng(hk.transform(compute_feature_fwd))
    rng = jax.random.PRNGKey(42)
    # in this test we assume that `memory` has the same structure as state.
    memory = inputs
    diagnostics = None
    if dict_randomness:
      randomness = {
          f'random_field_{i:01}': jax.random.uniform(rng, grid.nodal_shape)
          for i in range(3)
      }
    else:
      randomness = jax.random.uniform(rng, grid.nodal_shape)
    params = model.init(rng, inputs, memory, diagnostics, randomness, forcing)
    feature_values = model.apply(
        params, inputs, memory, diagnostics, randomness, forcing
    )
    for _, v in feature_values.items():
      self.assertEqual(v.shape[-2:], grid.nodal_shape)
      self.assertEqual(v.ndim, 3)


@gin.register
class TowerMock(hk.Module):
  """Tower that returns random outputs with shape (output_size,) + nodal_shape."""

  def __init__(
      self,
      output_size: int,  # size of dim 0
      nodal_shape: tuple[int, ...] = gin.REQUIRED,  # sizes of remaining dims
      **unused_kwargs,
  ):
    super().__init__(name=None)
    self.output_size = output_size
    self.nodal_shape = nodal_shape

  def __call__(self, inputs):
    del inputs  # unused.
    return jax.random.uniform(
        jax.random.PRNGKey(42), shape=(self.output_size,) + self.nodal_shape
    )


class EmbeddingFeaturesTest(parameterized.TestCase):
  """Tests embedding feature modules."""

  @parameterized.parameters(
      dict(
          features_name='EmbeddingSurfaceFeatures',
          embedding_name='ModalToNodalEmbedding',
          mapping_name='NodalMapping',
          tower_nodal_shape=(64, 32),
          input_type='modal',
      ),
      dict(
          features_name='EmbeddingVolumeFeatures',
          embedding_name='ModalToNodalEmbedding',
          mapping_name='NodalVolumeMapping',
          tower_nodal_shape=(4, 64, 32),
          input_type='modal',
      ),
  )
  def test_embedding_feature_module_shapes(
      self,
      features_name,
      embedding_name,
      mapping_name,
      tower_nodal_shape,
      input_type,
  ):
    """Tests that the returned features have expected nodal shape."""
    gin.enter_interactive_mode()
    gin.clear_config()
    physics_gin_config = '\n'.join([
        'get_physics_specs.construct_fn = @primitive_eq_specs_constructor',
    ])
    gin.parse_config(physics_gin_config)
    physics_specs = physics_specifications.get_physics_specs()
    dt = physics_specs.nondimensionalize(100 * units.s)
    # setting up coordinates and initial state.
    grid = spherical_harmonic.Grid.T21()
    vertical_grid = sigma_coordinates.SigmaCoordinates.equidistant(4)
    coords = coordinate_systems.CoordinateSystem(grid, vertical_grid)
    initial_state_fn, aux_features = primitive_equations_states.steady_state_jw(
        coords, physics_specs
    )
    inputs = initial_state_fn(jax.random.PRNGKey(42)).asdict()
    inputs['sim_time'] = 0.5
    if input_type == 'nodal':
      inputs = coords.horizontal.to_nodal(inputs)
    forcing = {}

    @gin.configurable
    def compute_feature_fwd(
        x,
        memory,
        diagnostics,
        randomness,
        forcing,
        feature_module,
    ):
      feature_fn = feature_module(coords, dt, physics_specs, aux_features)
      return feature_fn(x, memory, diagnostics, randomness, forcing)

    gin_config = '\n'.join([
        f'compute_feature_fwd.feature_module = @{features_name}',
        # Config for EmbeddingSurfaceFeatures
        f'{features_name}.feature_name = "embedding_features"',
        f'{features_name}.output_size = 8',
        f'{features_name}.embedding_module = @{embedding_name}',
        f'{embedding_name}.modal_to_nodal_features_module ='
        + ' @VelocityAndPrognostics',
        'VelocityAndPrognostics.fields_to_include = ("u", "v")',
        f'{embedding_name}.nodal_mapping_module = @{mapping_name}',
        f'{mapping_name}.tower_factory = @TowerMock',
        f'TowerMock.nodal_shape = {tower_nodal_shape}',
    ])
    gin.parse_config(gin_config)
    model = hk.without_apply_rng(hk.transform(compute_feature_fwd))
    rng = jax.random.PRNGKey(42)
    # in this test we assume that `memory` has the same structure as state.
    memory = inputs
    diagnostics = None
    randomness = None
    params = model.init(rng, inputs, memory, diagnostics, randomness, forcing)
    feature_values = model.apply(
        params, inputs, memory, diagnostics, randomness, forcing
    )
    for _, v in feature_values.items():
      self.assertEqual(v.shape[-2:], grid.nodal_shape)
      self.assertEqual(v.ndim, 3)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
