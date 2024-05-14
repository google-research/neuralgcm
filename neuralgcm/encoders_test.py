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
"""Tests for encoders.py."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import typing
from dinosaur import vertical_interpolation
from dinosaur import xarray_utils
import haiku as hk
import jax
from neuralgcm import encoders
from neuralgcm import mappings
from neuralgcm import towers
from neuralgcm import transforms
import numpy as np


ModelState = typing.ModelState
units = scales.units


class TowerMock(hk.Module):
  """Vertical tower mock that that returns random outputs of expected shape."""

  def __init__(self, output_size, *args):
    super().__init__(name=None)
    self.output_size = output_size

  def __call__(self, inputs):
    del inputs  # unused.
    return jax.random.uniform(jax.random.PRNGKey(42), shape=[self.output_size])


class FeaturesMock(hk.Module):
  """Mock for features module that returns random outputs of expected shape."""

  def __init__(self, coords, *args):
    super().__init__(name=None)
    self.feature_shape = coords.nodal_shape
    self.key = jax.random.PRNGKey(42)

  def __call__(self, inputs, next_inputs=None, forcing=None, randomness=None):
    del inputs, next_inputs, forcing, randomness  # unused.
    return {'rng_field': jax.random.uniform(self.key, shape=self.feature_shape)}


class EncodersTest(parameterized.TestCase):
  """Tests encoder modules."""

  @parameterized.named_parameters(
      dict(
          testcase_name='WeatherbenchToPrimitiveEncoder',
          encoder_module=encoders.WeatherbenchToPrimitiveEncoder,
          encoder_kwargs={}),
      dict(
          testcase_name='LearnedWeatherbenchToPrimitiveEncoder',
          encoder_module=encoders.LearnedWeatherbenchToPrimitiveEncoder,
          encoder_kwargs={
              'modal_to_nodal_model_features_module': FeaturesMock,
              'modal_to_nodal_data_features_module': FeaturesMock,
              'nodal_mapping_module': functools.partial(
                  mappings.NodalMapping, tower_factory=functools.partial(
                      towers.ColumnTower, column_net_factory=TowerMock)),
              'correction_transform_module': transforms.IdentityTransform,
              'prediction_mask':
                  {'divergence': True, 'vorticity': True,
                   'log_surface_pressure': True,
                   'temperature_variation': True, 'sim_time': False,
                   'tracers': {}},
          },
      ),
  )
  def test_weatherbench_to_primitive_encoder(
      self,
      encoder_module,
      encoder_kwargs,
  ):
    """Tests that encoder produces expected output structures."""
    n_sigma_layers = 18
    pressure_levels = [50, 100, 500, 700, 1000]
    input_coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        vertical_interpolation.PressureCoordinates(pressure_levels),)
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(n_sigma_layers)
    )
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    dt = physics_specs.nondimensionalize(10 * units.minute)
    aux_features = {
        xarray_utils.OROGRAPHY: np.ones(input_coords.horizontal.nodal_shape),
        xarray_utils.REF_TEMP_KEY: np.ones(coords.vertical.layers) * 288,
    }

    def encode_fwd(inputs):
      encoder = encoder_module(
          coords, dt, physics_specs, aux_features, input_coords,
          **encoder_kwargs)
      return encoder(inputs, forcing={})
    encoder_model = hk.transform(encode_fwd)

    inputs = {
        'u': np.ones(input_coords.modal_shape),
        'v': np.ones(input_coords.modal_shape),
        'z': np.ones(input_coords.modal_shape),
        't': np.ones(input_coords.modal_shape),
        'sim_time': np.asarray(0.0),
        'tracers': {},
    }
    inputs = jax.tree.map(lambda x: x[np.newaxis, ...], inputs)
    rng = jax.random.PRNGKey(42)
    params = encoder_model.init(rng, inputs)
    encoded_state = (encoder_model.apply(params, rng, inputs).state).asdict()

    get_shape_fn = lambda tree: jax.tree.map(np.shape, tree)
    expected_shapes = primitive_equations.StateWithTime(
        divergence=coords.modal_shape,
        vorticity=coords.modal_shape,
        temperature_variation=coords.modal_shape,
        log_surface_pressure=coords.surface_modal_shape,
        tracers=jax.tree.map(
            lambda x: coords.modal_shape,
            inputs['tracers']
        ),
        sim_time=tuple(),
    ).asdict()
    self.assertSameStructure(get_shape_fn(encoded_state), expected_shapes)


if __name__ == '__main__':
  jax.config.parse_flags_with_absl()
  absltest.main()
