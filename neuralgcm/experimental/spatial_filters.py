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

"""Modules that implement spatial filters."""

import abc
import dataclasses
from typing import Sequence

from dinosaur import filtering
from neuralgcm.experimental import coordinates
from neuralgcm.experimental import typing
from neuralgcm.experimental import units
import numpy as np


class SpatialFilter(abc.ABC):
  """Base class for spatial filters."""

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    """Returns filtered ``inputs``."""


class ModalSpatialFilter(SpatialFilter):
  """Base class for filters."""

  @abc.abstractmethod
  def filter_modal(self, inputs: typing.Pytree) -> typing.Pytree:
    """Returns filtered modal ``inputs``."""


class ExponentialModalFilter(ModalSpatialFilter):
  """Modal filter that removes high frequency components."""

  def __init__(
      self,
      grid: coordinates.SphericalHarmonicGrid | coordinates.LonLatGrid,
      attenuation: float = 16,
      order: int = 18,
      cutoff: float = 0,
  ):
    """See ``dinosaur.filtering.exponential_filter`` for details."""
    self.grid = grid
    self.filter = filtering.exponential_filter(
        grid.ylm_grid, attenuation, order, cutoff)

  def filter_modal(self, inputs: typing.Pytree) -> typing.Pytree:
    return self.filter(inputs)

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    dinosaur_grid = self.grid.ylm_grid
    return dinosaur_grid.to_nodal(self.filter(dinosaur_grid.to_modal(inputs)))

  @classmethod
  def from_timescale(
      cls,
      grid: coordinates.SphericalHarmonicGrid | coordinates.LonLatGrid,
      dt: float | typing.Quantity | typing.Numeric,
      timescale: float | typing.Quantity | typing.Numeric,
      order: int = 18,
      cutoff: float = 0,
      *,
      sim_units: units.SimUnits,
  ):
    """Returns a filter with the given timescale."""
    if isinstance(dt, np.timedelta64):
      dt = units.nondimensionalize_timedelta64(dt, sim_units)
    else:
      dt = units.maybe_nondimensionalize(dt, sim_units)
    if isinstance(timescale, np.timedelta64):
      timescale = units.nondimensionalize_timedelta64(timescale, sim_units)
    else:
      timescale = units.maybe_nondimensionalize(timescale, sim_units)
    return cls(grid, attenuation=(dt / timescale), order=order, cutoff=cutoff)


@dataclasses.dataclass
class SequentialModalFilter(ModalSpatialFilter):
  """Modal filter that applies multiple filters sequentially."""

  filters: Sequence[ModalSpatialFilter]
  grid: coordinates.SphericalHarmonicGrid | coordinates.LonLatGrid

  def filter_modal(self, inputs: typing.Pytree) -> typing.Pytree:
    for modal_filter in self.filters:
      inputs = modal_filter.filter_modal(inputs)
    return inputs

  def __call__(self, inputs: typing.Pytree) -> typing.Pytree:
    dinosaur_grid = self.grid.ylm_grid
    modal_inputs = dinosaur_grid.to_modal(inputs)
    modal_outputs = self.filter_modal(modal_inputs)
    return dinosaur_grid.to_nodal(modal_outputs)
