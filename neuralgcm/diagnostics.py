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
"""Defines `diagnostic` modules that compute diagnostic predictions."""

from collections import abc
from typing import Any, Callable, Protocol

from dinosaur import coordinate_systems
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import typing

import gin
import jax
import jax.numpy as jnp


class DiagnosticFn(Protocol):
  """Implements initialization and computation of model diagnostic fields."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: dict[str, Any],
  ):
    del coords, dt, physics_specs, aux_features

  def __call__(
      self,
      model_state: typing.ModelState,
      physics_tendencies: typing.Pytree,
      forcing: typing.Forcing | None = None,
  ) -> dict[str, jax.Array]:
    """Computes diagnostic field from `model_state` and `physics_tendencies`."""
    ...


DiagnosticModule = Callable[..., DiagnosticFn]


@gin.register
class NoDiagnostics:
  """Diagnostic module that computes no diagnostics."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: dict[str, Any],
  ):
    del coords, dt, physics_specs, aux_features

  def __call__(
      self,
      model_state: typing.ModelState,
      physics_tendencies: typing.Pytree,
      forcing: typing.Forcing | None = None,
  ) -> dict[str, jax.Array]:
    return {}


@gin.register
class CombinedDiagnostics:
  """Computes a combination of multiple diagnostics."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: dict[str, Any],
      diagnostic_modules: abc.Sequence[DiagnosticModule] = gin.REQUIRED,
  ):
    self.diagnostic_fns = [
        module(coords, dt, physics_specs, aux_features)
        for module in diagnostic_modules
    ]

  def __call__(
      self,
      model_state: typing.ModelState,
      physics_tendencies: typing.Pytree,
      forcing: typing.Forcing | None = None,
  ) -> dict[str, jax.Array]:
    diagnostics = {}
    for fn in self.diagnostic_fns:
      new_diagnostics = fn(model_state, physics_tendencies, forcing)
      if any(k in diagnostics for k in new_diagnostics):
        raise ValueError(
            f'{new_diagnostics.keys()} overlaps with {diagnostics.keys()}'
        )
      diagnostics.update(new_diagnostics)
    return diagnostics


@gin.register
class PrecipitationMinusEvaporationDiagnostics:
  """Computes `P-E` by integrating physics_tendencies.

  Depending on the `method` computes either precipitation minus evaporation
  rate, which in ERA5 has units `kg m**-2 s**-1` or time-accumulated value
  in `kg m**-2` if `method == cumulative`.
  """

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: dict[str, Any],
      moisture_species: tuple[str, ...] = ('specific_humidity',),
      method: str = 'rate',
  ):
    del aux_features
    self.coords = coords
    self.dt = dt
    self.physics_specs = physics_specs
    self.moisture_species = moisture_species
    self.method = method
    self.to_nodal_fn = coords.horizontal.to_nodal

  def __call__(
      self,
      model_state: typing.ModelState,
      physics_tendencies: typing.Pytree,
      forcing: typing.Forcing | None = None,
  ) -> typing.Pytree:
    """Computes precipitation minus evaporation."""
    del forcing  # unused
    lsp = model_state.state.log_surface_pressure
    p_surface = jnp.squeeze(jnp.exp(self.to_nodal_fn(lsp)), axis=0)
    moisture_tendencies = [
        v
        for tracer, v in physics_tendencies.tracers.items()
        if tracer in self.moisture_species
    ]
    moisture_tendencies = sum(self.to_nodal_fn(moisture_tendencies))
    scale = p_surface / self.physics_specs.g
    e_minus_p = scale * sigma_coordinates.sigma_integral(
        moisture_tendencies, self.coords.vertical, keepdims=False
    )
    if self.method == 'rate':
      return {'P_minus_E_rate': -e_minus_p}
    elif self.method == 'cumulative':
      # TODO(dkochkov) Address possible precision loss due to small deltas.
      surface_nodal_shape = self.coords.horizontal.nodal_shape
      previous = model_state.diagnostics.get(
          'P_minus_E_cumulative',
          jnp.zeros(surface_nodal_shape))
      return {'P_minus_E_cumulative': previous - (e_minus_p * self.dt)}
    else:
      raise ValueError(f'Unknown {self.method=}, must be `rate`/`cumulative`')


@gin.register
class PrecipitableWaterDiagnostics:
  """Computes cumulative preciptable water in the state."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: dict[str, Any],
      moisture_species: tuple[str, ...] = ('specific_humidity',),
  ):
    del dt, aux_features
    self.coords = coords
    self.physics_specs = physics_specs
    self.moisture_species = moisture_species
    self.to_nodal_fn = coords.horizontal.to_nodal

  def __call__(
      self,
      model_state: typing.ModelState,
      physics_tendencies: typing.Pytree,
      forcing: typing.Forcing | None = None,
  ) -> typing.Pytree:
    """Computes preciptable water."""
    del physics_tendencies, forcing  # unused
    lsp = model_state.state.log_surface_pressure
    p_surface = jnp.squeeze(jnp.exp(self.to_nodal_fn(lsp)), axis=0)
    moisture_tracers = [
        v
        for tracer, v in model_state.tracers.items()
        if tracer in self.moisture_species
    ]
    moisture = sum(self.to_nodal_fn(moisture_tracers))
    water_density = self.physics_specs.nondimensionalize(scales.WATER_DENSITY)
    scale = p_surface / (self.physics_specs.g * water_density)
    water = scale * sigma_coordinates.sigma_integral(
        moisture, self.coords.vertical, keepdims=False
    )
    return {'precipitable_water': water}


@gin.register
class NodalModelDiagnosticsDecoder:
  """Diagnostics decoder that returns elements from model_state.diagnostics."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: dict[str, Any],
  ):
    del dt, aux_features
    self.coords = coords
    self.physics_specs = physics_specs

  def __call__(
      self,
      model_state: typing.ModelState,
      physics_tendencies: typing.Pytree,
      forcing: typing.Forcing | None = None,
  ) -> typing.Pytree:
    """Computes precipitation minus evaporation."""
    del physics_tendencies, forcing  # unused.
    nodal_diagnostics = coordinate_systems.maybe_to_nodal(
        model_state.diagnostics, self.coords)
    return nodal_diagnostics
