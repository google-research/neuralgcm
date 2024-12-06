# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules that hold orographic data."""

from flax import nnx
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import coordinates
from neuralgcm.experimental import interpolators
from neuralgcm.experimental import spatial_filters
from neuralgcm.experimental import typing
from neuralgcm.experimental import units
from neuralgcm.experimental import xarray_utils
import xarray


class OrographyVariable(nnx.Variable):
  """Variable class for orography data."""


class ModalOrography(nnx.Module):
  """Orogrphay module that provoides elevation in modal representation."""

  def __init__(
      self,
      *,
      grid: coordinates.SphericalHarmonicGrid,
      initializer: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    self.grid = grid
    self.orography = OrographyVariable(
        initializer(rngs, grid.ylm_grid.modal_shape))

  @property
  def nodal_orography(self) -> typing.Array:
    return self.grid.ylm_grid.to_nodal(self.orography.value)

  @property
  def modal_orography(self) -> typing.Array:
    """Returns orography converted to modal representation with filtering."""
    return self.orography.value

  def update_orography_from_data(
      self,
      dataset: xarray.Dataset,
      sim_units: units.SimUnits,
      spatial_filter=None,
  ):
    """Updates ``self.orography`` with filtered orography from dataset."""
    # TODO(dkochkov) use units attr on dataset with default to `meter` here.
    if spatial_filter is None:
      spatial_filter = lambda x: x
    nodal_orography = xarray_utils.nodal_orography_from_ds(dataset)
    nodal_orography = xarray_utils.xarray_nondimensionalize(
        nodal_orography, sim_units
    ).values
    data_coords = xarray_utils.coordinates_from_dataset(dataset)
    assert isinstance(data_coords, coordinates.DinosaurCoordinates)
    if not isinstance(spatial_filter, spatial_filters.ModalSpatialFilter):
      nodal_orography = spatial_filter(nodal_orography)
    data_grid = data_coords.horizontal
    if isinstance(data_grid, coordinates.SphericalHarmonicGrid):
      data_modal_grid = data_grid
    elif isinstance(data_grid, coordinates.LonLatGrid):
      data_modal_grid = data_grid.to_spherical_harmonic_grid()
    else:
      raise ValueError(f'Unsupported data grid {data_grid=}')
    modal_orography = data_modal_grid.ylm_grid.to_modal(nodal_orography)
    interpolator = interpolators.SpectralRegridder(self.grid)
    modal_orography = interpolator(cx.wrap(modal_orography, data_modal_grid))
    modal_orography = modal_orography.unwrap(self.grid)
    if isinstance(spatial_filter, spatial_filters.ModalSpatialFilter):
      modal_orography = spatial_filter.filter_modal(modal_orography)
    self.orography.value = modal_orography


class Orography(nnx.Module):
  """Orography module that provides elevation in real space."""

  def __init__(
      self,
      *,
      grid: coordinates.LonLatGrid,
      initializer: nnx.initializers.Initializer = nnx.initializers.zeros_init(),
      rngs: nnx.Rngs,
  ):
    self.grid = grid
    self.orography = OrographyVariable(
        initializer(rngs, grid.ylm_grid.nodal_shape)
    )

  @property
  def nodal_orography(self) -> typing.Array:
    return self.orography.value

  def update_orography_from_data(
      self,
      dataset: xarray.Dataset,
      sim_units: units.SimUnits,
      spatial_filter=None,
  ):
    """Updates ``self.orography`` with filtered orography from dataset."""
    # TODO(dkochkov) use units attr on dataset with default to `meter` here.
    if spatial_filter is None:
      spatial_filter = lambda x: x
    nodal_orography = xarray_utils.nodal_orography_from_ds(dataset)
    nodal_orography = xarray_utils.xarray_nondimensionalize(
        nodal_orography, sim_units
    ).values
    data_coords = xarray_utils.coordinates_from_dataset(dataset)
    assert isinstance(data_coords, coordinates.DinosaurCoordinates)
    nodal_orography = spatial_filter(nodal_orography)
    data_grid = data_coords.horizontal
    if data_grid != self.grid:
      raise ValueError(f'{data_grid=} does not match {self.grid=}.')
    self.orography.value = nodal_orography
