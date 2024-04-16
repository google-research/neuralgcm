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
"""Tests for forcings.py."""

import functools
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import haiku as hk
import jax
from neuralgcm import forcings
import numpy as np


units = scales.units


class ForcingsTest(parameterized.TestCase):
  """Tests corrector modules."""

  def setUp(self):
    super().setUp()
    n_sigma_layers = 3
    self.coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(n_sigma_layers)
    )
    self.physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    self.dt = self.physics_specs.nondimensionalize(10 * units.minute)
    ref_datetime = np.datetime64('1970-01-01T00:00:00')
    self.aux_features = {xarray_utils.REFERENCE_DATETIME_KEY: ref_datetime}

  def test_no_forcing(self):

    def forcing_fwd(forcing_data, sim_time):
      forcing_fn = forcings.NoForcing(self.coords, self.dt, self.physics_specs,
                                      self.aux_features)
      return forcing_fn(forcing_data, sim_time)

    forcing_model = hk.without_apply_rng(hk.transform(forcing_fwd))
    forcing_data = {}
    sim_time = None
    params = forcing_model.init(jax.random.PRNGKey(42), forcing_data, sim_time)
    forcing = forcing_model.apply(params, forcing_data, sim_time)

    expected_forcing = {}
    self.assertEqual(forcing, expected_forcing)

  def test_check_errors(self):
    # The mock in test_dynamic_data_forcing checks that
    # _FORCING_ERRORS.append was indeed called the correct number of times.

    with self.assertRaisesRegex(forcings.ForcingDataError, 'forcing_sim_time'):
      forcings._check_errors(err_list=['forcing_sim_time was bad'])

  @mock.patch('neuralgcm.forcings._FORCING_ERRORS', new=mock.MagicMock(list))
  def test_dynamic_data_forcing(self):
    forcings._check_errors()  # Should not raise

    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    one_hour_nondim = physics_specs.nondimensionalize(units.hour)
    n_times = 120
    sim_time_data = one_hour_nondim * np.arange(n_times)
    surface_shape = (n_times,) + self.coords.surface_nodal_shape
    volume_shape = (n_times,) + self.coords.nodal_shape
    # specify time-dependent data
    snow_temperature = 260 * np.ones(surface_shape)
    snow_depth = np.arange(n_times).reshape((n_times, 1, 1, 1)) * np.ones(
        surface_shape)
    cloud_cover = np.zeros(volume_shape)
    forcing_data = {'cloud_cover': cloud_cover,
                    'snow_temperature': snow_temperature,
                    'snow_depth': snow_depth,
                    'sim_time': sim_time_data}
    inputs_to_units_mapping = {'cloud_cover': 'dimensionless',
                               'snow_temperature': 'K',
                               'snow_depth': 'meter',
                               'sim_time': 'dimensionless'}

    def forcing_fwd(forcing_data, sim_time):
      forcing_fn = forcings.DynamicDataForcing(
          self.coords, self.dt, self.physics_specs, self.aux_features,
          check_sim_time_errors=True,
          inputs_to_units_mapping=inputs_to_units_mapping)
      return forcing_fn(forcing_data, sim_time)

    forcing_model = hk.without_apply_rng(hk.transform(forcing_fwd))
    params = forcing_model.init(jax.random.PRNGKey(42), forcing_data, 0)
    forcing_fn = jax.jit(forcing_model.apply)

    with self.subTest('forcing shapes'):
      sim_time = 0
      forcing = forcing_fn(params, forcing_data, sim_time)
      self.assertSetEqual(
          set(forcing.keys()),
          {'cloud_cover', 'snow_temperature', 'snow_depth', 'sim_time'})
      self.assertEqual(forcing['cloud_cover'].shape, (3, 64, 32))
      self.assertEqual(forcing['snow_temperature'].shape, (1, 64, 32))
      self.assertEqual(forcing['snow_depth'].shape, (1, 64, 32))
      self.assertEqual(forcing['sim_time'].shape, ())

    with self.subTest('exact sim_time'):
      for idx, sim_time in enumerate(sim_time_data):
        forcing = forcing_fn(params, forcing_data, sim_time)
        self.assertFalse(np.any(np.isnan(forcing['sim_time'])))
        np.testing.assert_allclose(forcing['sim_time'], sim_time)
        np.testing.assert_allclose(forcing['snow_temperature'], 260)
        nondim_snow_depth = self.physics_specs.nondimensionalize(
            idx * units.meter)
        np.testing.assert_allclose(forcing['snow_depth'],
                                   nondim_snow_depth, atol=1e-6)

    with self.subTest('nearby sim_time'):
      for idx, sim_time in enumerate(sim_time_data):
        nearby_sim_time = sim_time + one_hour_nondim / 12  # +5 minutes
        forcing = forcing_fn(params, forcing_data, nearby_sim_time)
        self.assertFalse(np.any(np.isnan(forcing['sim_time'])))
        np.testing.assert_allclose(forcing['sim_time'], sim_time)
        nondim_snow_depth = self.physics_specs.nondimensionalize(
            idx * units.meter)
        np.testing.assert_allclose(forcing['snow_depth'],
                                   nondim_snow_depth, atol=1e-6)

    with self.subTest('in-between sim_time'):
      sim_time = 3.5 * one_hour_nondim
      forcing = forcing_fn(params, forcing_data, sim_time)
      np.testing.assert_allclose(forcing['sim_time'], sim_time_data[4])
      nondim_snow_depth = self.physics_specs.nondimensionalize(
          forcing_data['snow_depth'][4] * units.meter)
      np.testing.assert_allclose(forcing['snow_depth'],
                                 nondim_snow_depth, atol=1e-6)

    with self.subTest('thinned sim_time'):
      for idx, sim_time in list(enumerate(sim_time_data))[::10]:
        thinned_forcing_data = {k: v[::10] for k, v in forcing_data.items()}
        forcing = forcing_fn(params, thinned_forcing_data, sim_time)
        self.assertFalse(np.any(np.isnan(forcing['sim_time'])))
        np.testing.assert_allclose(forcing['sim_time'], sim_time)
        np.testing.assert_allclose(forcing['snow_temperature'], 260)
        nondim_snow_depth = self.physics_specs.nondimensionalize(
            idx * units.meter)
        np.testing.assert_allclose(forcing['snow_depth'],
                                   nondim_snow_depth, atol=1e-6)

    with self.subTest('sim_time too small'):
      sim_time = -1 * one_hour_nondim
      jax.block_until_ready(forcing_fn(params, forcing_data, sim_time))
      self.assertEqual(1, forcings._FORCING_ERRORS.append.call_count)

    with self.subTest('sim_time too big'):
      sim_time = (n_times + 1) * one_hour_nondim
      jax.block_until_ready(forcing_fn(params, forcing_data, sim_time))
      self.assertEqual(2, forcings._FORCING_ERRORS.append.call_count)

    with self.subTest('sim_time is None'):
      sim_time = None
      with self.assertRaises(TypeError):
        forcing_fn(params, forcing_data, sim_time)

    with self.subTest('sim_time is nan'):
      # jax casts None to an array with nan
      sim_time = jax.numpy.asarray(None)  # = DeviceArray(nan, dtype=float32)
      # inside forcing_fn, sim_time gets cast to int32
      # this is unsafe, since it turns nan into an actual integer,
      # idx=2147483647.
      jax.block_until_ready(forcing_fn(params, forcing_data, sim_time))
      self.assertEqual(3, forcings._FORCING_ERRORS.append.call_count)

    with self.subTest('vmap sim_time'):
      sim_time = sim_time_data[2:4]

      def forcing_fwd_vmap(forcing_data, sim_time):
        forcing_fn = forcings.DynamicDataForcing(
            self.coords, self.dt, self.physics_specs, self.aux_features,
            check_sim_time_errors=True,
            inputs_to_units_mapping=inputs_to_units_mapping)
        forcing_fn = jax.vmap(forcing_fn, in_axes=(None, 0))
        return forcing_fn(forcing_data, sim_time)

      forcing_vmap_model = hk.without_apply_rng(hk.transform(forcing_fwd_vmap))
      params = forcing_vmap_model.init(
          jax.random.PRNGKey(42), forcing_data, sim_time)

      forcing = forcing_vmap_model.apply(params, forcing_data, sim_time)
      self.assertSetEqual(
          set(forcing.keys()),
          {'cloud_cover', 'snow_temperature', 'snow_depth', 'sim_time'})
      self.assertEqual(forcing['cloud_cover'].shape, (2, 3, 64, 32))
      self.assertEqual(forcing['snow_temperature'].shape, (2, 1, 64, 32))
      self.assertEqual(forcing['snow_depth'].shape, (2, 1, 64, 32))
      self.assertEqual(forcing['sim_time'].shape, (2,))
      np.testing.assert_allclose(forcing['sim_time'], sim_time_data[2:4])

    with self.subTest('single sim_time forcing_data'):
      single_forcing_data = {k: v[2:3] for k, v in forcing_data.items()}
      forcing = forcing_fn(params, single_forcing_data, sim_time_data[2])
      self.assertEqual(forcing['cloud_cover'].shape, (3, 64, 32))
      self.assertEqual(forcing['snow_temperature'].shape, (1, 64, 32))
      self.assertEqual(forcing['snow_depth'].shape, (1, 64, 32))
      self.assertEqual(forcing['sim_time'].shape, ())
      np.testing.assert_allclose(forcing['sim_time'], sim_time_data[2])

  def test_persistence_data_forcing(self):
    sim_time_data = 1.0 * np.arange(5)
    surface_shape = (len(sim_time_data),) + self.coords.surface_nodal_shape
    volume_shape = (len(sim_time_data),) + self.coords.nodal_shape
    # specify time-dependent data
    snow_temperature = 260 * np.ones(surface_shape)
    snow_depth = np.arange(5).reshape((5, 1, 1, 1)) * np.ones(surface_shape)
    cloud_cover = np.zeros(volume_shape)
    forcing_data = {'cloud_cover': cloud_cover,
                    'snow_temperature': snow_temperature,
                    'snow_depth': snow_depth,
                    'sim_time': sim_time_data}
    inputs_to_units_mapping = {'cloud_cover': 'dimensionless',
                               'snow_temperature': 'K',
                               'snow_depth': 'meter',
                               'sim_time': 'dimensionless'}

    def forcing_fwd(forcing_data, sim_time):
      forcing_fn = forcings.PersistenceDataForcing(
          self.coords, self.dt, self.physics_specs, self.aux_features,
          inputs_to_units_mapping=inputs_to_units_mapping)
      return forcing_fn(forcing_data, sim_time)

    forcing_model = hk.without_apply_rng(hk.transform(forcing_fwd))
    params = forcing_model.init(jax.random.PRNGKey(42), forcing_data, 0)
    forcing_fn = jax.jit(forcing_model.apply)

    with self.subTest('forcing shapes'):
      sim_time = 0
      forcing = forcing_fn(params, forcing_data, sim_time)
      self.assertSetEqual(
          set(forcing.keys()),
          {'cloud_cover', 'snow_temperature', 'snow_depth', 'sim_time'})
      self.assertEqual(forcing['cloud_cover'].shape, (3, 64, 32))
      self.assertEqual(forcing['snow_temperature'].shape, (1, 64, 32))
      self.assertEqual(forcing['snow_depth'].shape, (1, 64, 32))
      self.assertEqual(forcing['sim_time'].shape, ())

    with self.subTest('numeric sim_time'):
      for sim_time in [-1.0, 0, 1.2, 3.5]:
        forcing = forcing_fn(params, forcing_data, sim_time)
        expected_forcing = forcing_fn(params, forcing_data, sim_time_data[0])
        for v, v_expected in zip(forcing.values(), expected_forcing.values()):
          np.testing.assert_allclose(v, v_expected)

    with self.subTest('sim_time is None'):
      sim_time = None
      forcing = forcing_fn(params, forcing_data, sim_time)
      np.testing.assert_allclose(forcing['sim_time'], sim_time_data[0])

    with self.subTest('sim_time is nan'):
      # jax casts None to an array with nan
      sim_time = jax.numpy.asarray(None)  # = DeviceArray(nan, dtype=float32)
      forcing = forcing_fn(params, forcing_data, sim_time)
      np.testing.assert_allclose(forcing['sim_time'], sim_time_data[0])

    with self.subTest('vmap sim_time'):
      sim_time = np.array([2., 3.])

      def forcing_fwd_vmap(forcing_data, sim_time):
        forcing_fn = forcings.PersistenceDataForcing(
            self.coords, self.dt, self.physics_specs, self.aux_features,
            inputs_to_units_mapping=inputs_to_units_mapping)
        forcing_fn = jax.vmap(forcing_fn, in_axes=(None, 0))
        return forcing_fn(forcing_data, sim_time)

      forcing_vmap_model = hk.without_apply_rng(hk.transform(forcing_fwd_vmap))
      params = forcing_vmap_model.init(
          jax.random.PRNGKey(42), forcing_data, sim_time)

      forcing = forcing_vmap_model.apply(params, forcing_data, sim_time)
      self.assertSetEqual(
          set(forcing.keys()),
          {'cloud_cover', 'snow_temperature', 'snow_depth', 'sim_time'})
      self.assertEqual(forcing['cloud_cover'].shape, (2, 3, 64, 32))
      self.assertEqual(forcing['snow_temperature'].shape, (2, 1, 64, 32))
      self.assertEqual(forcing['snow_depth'].shape, (2, 1, 64, 32))
      self.assertEqual(forcing['sim_time'].shape, (2,))
      np.testing.assert_allclose(forcing['sim_time'], [0., 0.])

    with self.subTest('single sim_time forcing_data'):
      single_forcing_data = {k: v[2:3] for k, v in forcing_data.items()}
      forcing = forcing_fn(params, single_forcing_data, sim_time=4.0)
      self.assertEqual(forcing['cloud_cover'].shape, (3, 64, 32))
      self.assertEqual(forcing['snow_temperature'].shape, (1, 64, 32))
      self.assertEqual(forcing['snow_depth'].shape, (1, 64, 32))
      self.assertEqual(forcing['sim_time'].shape, ())
      np.testing.assert_allclose(forcing['sim_time'], 2.)  # 0 index value

  @parameterized.parameters(
      dict(forcing_module=forcings.DynamicDataForcing),
      dict(forcing_module=forcings.PersistenceDataForcing),
  )
  def test_forcing_transforms(self, forcing_module):
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    one_hour_nondim = physics_specs.nondimensionalize(units.hour)
    n_times = 120
    sim_time_data = one_hour_nondim * np.arange(n_times)
    surface_shape = (n_times,) + self.coords.surface_nodal_shape
    # specify time-dependent data
    sea_surface_temperature = 334 * np.ones(surface_shape)

    forcing_data = {'sea_surface_temperature': sea_surface_temperature,
                    'sim_time': sim_time_data}
    inputs_to_units_mapping = {'sea_surface_temperature': 'K',
                               'sim_time': 'dimensionless'}

    with self.subTest('IncrementSSTForcingTransform'):
      forcing_transform = functools.partial(
          forcings.IncrementSSTForcingTransform,
          temperature_change='2.5 K',
          key='sea_surface_temperature',
      )

      def forcing_fwd(forcing_data, sim_time):
        forcing_fn = forcing_module(
            self.coords, self.dt, self.physics_specs, self.aux_features,
            inputs_to_units_mapping=inputs_to_units_mapping,
            forcing_transform=forcing_transform)
        return forcing_fn(forcing_data, sim_time)

      forcing_model = hk.without_apply_rng(hk.transform(forcing_fwd))
      params = forcing_model.init(jax.random.PRNGKey(42), forcing_data, 0)
      forcing_fn = jax.jit(forcing_model.apply)

      sim_time = 0
      forcing = forcing_fn(params, forcing_data, sim_time)
      self.assertEqual(forcing['sea_surface_temperature'].shape, (1, 64, 32))
      np.testing.assert_allclose(forcing['sea_surface_temperature'], 334 + 2.5)


if __name__ == '__main__':
  absltest.main()
