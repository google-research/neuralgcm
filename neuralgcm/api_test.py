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
import importlib.resources
import pickle

from absl.testing import absltest
from dinosaur import horizontal_interpolation
from dinosaur import pytree_utils
from dinosaur import spherical_harmonic
import jax
import neuralgcm
from neuralgcm import api
import numpy as np
import xarray


def horizontal_regrid(
    regridder: horizontal_interpolation.Regridder, dataset: xarray.Dataset
) -> xarray.Dataset:
  """Horizontally regrid an xarray Dataset."""
  # TODO(shoyer): consider moving to public API
  regridded = xarray.apply_ufunc(
      regridder,
      dataset,
      input_core_dims=[['longitude', 'latitude']],
      output_core_dims=[['longitude', 'latitude']],
      exclude_dims={'longitude', 'latitude'},
      vectorize=True,  # loops over level, for lower memory usage
  )
  regridded.coords['longitude'] = np.rad2deg(regridder.target_grid.longitudes)
  regridded.coords['latitude'] = np.rad2deg(regridder.target_grid.latitudes)
  return regridded


def _assert_allclose(actual, desired, *, err_msg=None, range_rtol=1e-5):
  span = desired.max() - desired.min()
  np.testing.assert_allclose(
      actual / span, desired / span, rtol=0, atol=range_rtol, err_msg=err_msg
  )


def load_tl63_stochastic_model() -> api.PressureLevelModel:
  package = importlib.resources.files(neuralgcm)
  file = package.joinpath('data/tl63_stochastic_mini.pkl')
  ckpt = pickle.loads(file.read_bytes())
  return api.PressureLevelModel.from_checkpoint(ckpt)


def load_tl63_data() -> xarray.Dataset:
  package = importlib.resources.files(neuralgcm)
  with package.joinpath('data/era5_tl31_19590102T00.nc').open('rb') as f:
    ds = xarray.load_dataset(f).expand_dims('time')
  regridder = horizontal_interpolation.ConservativeRegridder(
      spherical_harmonic.Grid.TL31(), spherical_harmonic.Grid.TL63()
  )
  return horizontal_regrid(regridder, ds)


class APITest(absltest.TestCase):

  def assertDictEqual(self, actual, desired, *, err_msg=None):
    self.assertEqual(sorted(actual.keys()), sorted(desired.keys()))
    for key in actual:
      x = actual[key]
      y = desired[key]
      err_msg2 = key if err_msg is None else f'{err_msg}/{key}'
      if isinstance(y, dict):
        self.assertDictAllclose(x, y, err_msg=err_msg2)
      else:
        np.testing.assert_array_equal(x, y, err_msg=err_msg2)

  def assertDictAllclose(
      self, actual, desired, *, err_msg=None, range_rtol=1e-5
  ):
    self.assertEqual(sorted(actual.keys()), sorted(desired.keys()))
    for key in actual:
      x = actual[key]
      y = desired[key]
      err_msg2 = key if err_msg is None else f'{err_msg}/{key}'
      if isinstance(y, dict):
        self.assertDictAllclose(x, y, err_msg=err_msg2, range_rtol=range_rtol)
      else:
        _assert_allclose(x, y, err_msg=err_msg2, range_rtol=range_rtol)

  def test_to_and_from_nondim_units(self):
    model = load_tl63_stochastic_model()

    nondim = model.to_nondim_units(1.0, 'meters')
    self.assertAlmostEqual(nondim, 1 / 6.37122e6)  # Earth radius units

    roundtripped = model.from_nondim_units(nondim, 'meters')
    self.assertAlmostEqual(roundtripped, 1.0)

  def test_sim_time_utilities(self):
    model = load_tl63_stochastic_model()
    origin = model.datetime64_to_sim_time(np.datetime64('1979-01-01T00:00:00'))
    self.assertEqual(origin, 0.0)

    original = np.datetime64('1959-01-01T00:00:00')
    roundtripped = model.sim_time_to_datetime64(
        model.datetime64_to_sim_time(original)
    )
    self.assertEqual(original, roundtripped)

  def test_from_xarray(self):
    model = load_tl63_stochastic_model()
    ds = load_tl63_data()

    state_variables = [
        'u_component_of_wind',
        'v_component_of_wind',
        'geopotential',
        'temperature',
        'specific_humidity',
        'specific_cloud_liquid_water_content',
        'specific_cloud_ice_water_content',
    ]
    forcing_variables = ['sea_surface_temperature', 'sea_ice_cover']

    expected_data = {k: ds[k].values for k in state_variables}
    expected_data['sim_time'] = np.array([-92034.607104])

    expected_forcings = {k: ds[k].values for k in forcing_variables}
    expected_forcings['sim_time'] = np.array([-92034.607104])

    data, forcings = model.data_from_xarray(ds)
    self.assertDictEqual(data, expected_data)
    self.assertDictEqual(forcings, expected_forcings)

    forcings2 = model.forcings_from_xarray(ds)
    self.assertDictEqual(forcings2, expected_forcings)

  def test_stochastic_model_basics(self):
    timesteps = 3
    dt = np.timedelta64(1, 'h')

    model = load_tl63_stochastic_model()
    ds_in = load_tl63_data()

    data, forcings = model.data_from_xarray(ds_in)
    data_in, forcings_in = pytree_utils.slice_along_axis(
        (data, forcings), axis=0, idx=0
    )

    # run model
    encoded = model.encode(data_in, forcings_in, rng_key=jax.random.key(0))
    _, data_out = model.unroll(
        encoded, forcings, steps=timesteps, timedelta=dt, start_with_input=True
    )

    # convert to xarray
    t0 = ds_in.time.values[0]
    times = np.arange(t0, t0 + timesteps * dt, dt)
    ds_out = model.data_to_xarray(data_out, times=times)

    # validate
    actual = ds_out.head(time=1)

    sim_time = model.datetime64_to_sim_time(ds_in.time.data)
    expected = ds_in.drop_vars(
        ['sea_surface_temperature', 'sea_ice_cover']
    ).assign(sim_time=('time', sim_time))

    # check matching variable shapes
    xarray.testing.assert_allclose(actual, expected, atol=1e6)

    # check that round-tripping the initial condition is approximately correct
    typical_relative_error = abs(actual - expected).median() / expected.std()
    tolerance = xarray.Dataset({
        'u_component_of_wind': 0.04,
        'v_component_of_wind': 0.08,
        'temperature': 0.02,
        'geopotential': 0.0005,
        'specific_humidity': 0.003,
        'specific_cloud_liquid_water_content': 0.12,
        'specific_cloud_ice_water_content': 0.15,
    })
    self.assertTrue(
        (typical_relative_error < tolerance).to_array().values.all(),
        msg=f'typical relative error is too large:\n{typical_relative_error}',
    )

    # test decode()
    decoded = model.decode(encoded, forcings_in)
    expected = jax.tree.map(lambda x: x[0, ...], data_out)
    self.assertDictAllclose(decoded, expected)

    # test advance()
    self.assertEqual(model.timestep, dt)
    decoded2 = model.decode(model.advance(encoded, forcings_in), forcings_in)
    expected2 = jax.tree.map(lambda x: x[1, ...], data_out)
    self.assertDictAllclose(decoded2, expected2)

    # TODO(shoyer): verify RNG key works correctly
    # TODO(shoyer): verify RNG key is optional for deterministic models


if __name__ == '__main__':
  absltest.main()
