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

from dinosaur import coordinate_systems
from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
import neuralgcm
import numpy as np
import xarray


def _horizontal_regrid(
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


def load_checkpoint_tl63_stochastic():
  """Load a checkpoint for a toy TL63 stochastic model."""
  package = importlib.resources.files(neuralgcm)
  file = package.joinpath('data/tl63_stochastic_mini.pkl')
  return pickle.loads(file.read_bytes())


def load_data(coords: coordinate_systems.CoordinateSystem) -> xarray.Dataset:
  """Load demo data for the given coordinate system."""
  if coords.vertical.layers != 37:
    raise ValueError('can only load demo data for 37 pressure levels')
  package = importlib.resources.files(neuralgcm)
  with package.joinpath('data/era5_tl31_19590102T00.nc').open('rb') as f:
    ds = xarray.load_dataset(f).expand_dims('time')
  regridder = horizontal_interpolation.ConservativeRegridder(
      spherical_harmonic.Grid.TL31(), coords.horizontal
  )
  return _horizontal_regrid(regridder, ds)
