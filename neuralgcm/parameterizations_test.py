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
"""Tests for parameterization modules in gcm.ml.parameterizations.py."""

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import coordinate_systems
from dinosaur import primitive_equations_states
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
import gin
import haiku as hk
import jax

from neuralgcm import features  # pylint: disable=unused-import
from neuralgcm import parameterizations  # pylint: disable=unused-import
from neuralgcm import physics_specifications
from neuralgcm import transforms  # pylint: disable=unused-import

SCALE = scales.DEFAULT_SCALE
units = scales.units


@gin.register
class ParameterizationTowerMock(hk.Module):
  """Tower that returns random outputs of expected shape."""

  def __init__(self, output_size, *args):
    super().__init__(name=None)
    self.output_size = output_size

  def __call__(self, inputs):
    del inputs  # unused.
    return jax.random.uniform(
        jax.random.PRNGKey(42),
        shape=[self.output_size, 1, 1])  # include 2d spatial dimensions


class ParameterizationsTest(parameterized.TestCase):
  """Tests parameterization modules."""

  @parameterized.parameters(
      dict(
          module_name='DirectNeuralParameterization',
          features_name='PrimitiveEquationsDiagnosticState',),
      dict(
          module_name='DivCurlNeuralParameterization',
          features_name='PrimitiveEquationsDiagnosticState',),
  )
  def test_parameterization_output_shapes(self, module_name, features_name):
    """Tests that parameterizations produce outputs with expected shapes."""
    gin.enter_interactive_mode()
    gin.clear_config()
    physics_gin_config = '\n'.join([
        'get_physics_specs.construct_fn = @primitive_eq_specs_constructor',
    ])
    gin.parse_config(physics_gin_config)
    physics_specs = physics_specifications.get_physics_specs()
    dt = physics_specs.nondimensionalize(100 * units.s)
    # setting up coordinates and initial state.
    grid = spherical_harmonic.Grid.with_wavenumbers(32)
    vertical_grid = sigma_coordinates.SigmaCoordinates.equidistant(4)
    coords = coordinate_systems.CoordinateSystem(grid, vertical_grid)
    initial_state_fn, aux_features = (
        primitive_equations_states.isothermal_rest_atmosphere(
            coords, physics_specs))

    @gin.configurable
    def parameterization_fwd(x, parameterization_cls):
      return parameterization_cls(coords, dt, physics_specs, aux_features)(x)

    parameterization_config = '\n'.join([
        f'parameterization_fwd.parameterization_cls = @{module_name}',
        f'{module_name}.modal_to_nodal_features_module = @{features_name}',
        f'{features_name}.features_transform_module = @IdentityTransform',
        f'{module_name}.nodal_mapping_module = @NodalMapping',
        f'{module_name}.tendency_transform_module = @IdentityTransform',
        'NodalMapping.tower_factory = @ParameterizationTowerMock'
    ])
    gin.parse_config(parameterization_config)
    inputs = initial_state_fn(jax.random.PRNGKey(42))
    model = hk.without_apply_rng(hk.transform(parameterization_fwd))
    params = model.init(jax.random.PRNGKey(42), inputs)
    predicted_tendencies = model.apply(params, inputs)
    for actual, expected in zip(jax.tree_util.tree_leaves(predicted_tendencies),
                                jax.tree_util.tree_leaves(inputs)):
      self.assertEqual(actual.shape, expected.shape)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
