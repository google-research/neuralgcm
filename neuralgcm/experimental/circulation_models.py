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

"""Modules that implement time evolution of a model state."""

from typing import Callable

from dinosaur import coordinate_systems
from flax import nnx

from neuralgcm.experimental import coordinates
from neuralgcm.experimental import equations
from neuralgcm.experimental import spatial_filters
from neuralgcm.experimental import time_integrators
from neuralgcm.experimental import typing


# TODO(dkochkov): add typing for subcomponents.
# TODO(dkochkov): consider renaming this to advance_modules/advance_models.
# TODO(dkochkov): use methods from interpolators instead of one in dinosaur.


class DycoreWithParameterization(nnx.Module):
  """Simulation model that combines dynamical core with parameterizations."""

  def __init__(
      self,
      internal_coords: coordinates.DinosaurCoordinates,
      dt: float,
      dycore_equation: nnx.Module,
      parameterization: nnx.Module,
      time_integrator_cls: Callable[..., time_integrators.DinosaurIntegrator],
      modal_filter: spatial_filters.ModalSpatialFilter,
      num_substeps: int = 1,
      dycore_substeps: int = 1,
      io_coords: coordinates.DinosaurCoordinates | None = None,
      remat_substep: bool = False,
  ):
    self.internal_coords = internal_coords
    self.dycore_equation = dycore_equation
    self.modal_filter = modal_filter
    self.parameterization = parameterization
    self.time_integrator_cls = time_integrator_cls
    self.num_substeps = num_substeps
    self.inner_dt = dt / num_substeps
    self.dycore_substeps = dycore_substeps
    self.dycore_dt = self.inner_dt / dycore_substeps
    self.remat_substep = remat_substep
    if io_coords is None:
      self.io_coords = internal_coords
      self.interpolate_fn = None
      self.reverse_interpolate_fn = None
    else:
      self.io_coords = io_coords
      self.interpolate_fn = coordinate_systems.get_spectral_interpolate_fn(
          io_coords.dinosaur_coords, self.internal_coords.dinosaur_coords
      )
      self.reverse_interpolate_fn = (
          coordinate_systems.get_spectral_interpolate_fn(
              self.internal_coords.dinosaur_coords, io_coords.dinosaur_coords
          )
      )

  def step_with_fixed_parameterization_tendencies(
      self,
      prognostics: typing.Pytree,
      tendencies: typing.Pytree,
  ) -> typing.Pytree:
    """Computes `prognostics` evolved in time by `self.inner_dt`."""
    # prognostics, tendencies = self.internal_coords.with_dycore_sharding(
    #     (prognostics, tendencies)
    # )
    parametrization_eq = time_integrators.ExplicitODE.from_functions(
        lambda prognostics: tendencies
    )
    all_equations = (self.dycore_equation, parametrization_eq)
    equation = equations.ComposedODE(all_equations)
    integrator = self.time_integrator_cls(equation, self.dycore_dt)

    def _single_step(x_and_integrator):
      x, integrator = x_and_integrator
      x_next = integrator(x)
      return (self.modal_filter.filter_modal(x_next), integrator)

    step = time_integrators.repeated(_single_step, steps=self.dycore_substeps)
    prognostics, _ = step((prognostics, integrator))
    prognostics = time_integrators.maybe_fix_sim_time_roundoff(
        prognostics, self.inner_dt
    )
    # prognostics = self.internal_coords.with_dycore_sharding(prognostics)
    return prognostics

  def substep(self, prognostics: typing.Pytree):
    # prognostics = self.internal_coords.with_dycore_sharding(prognostics)
    pp_tendency = self.parameterization(prognostics)
    if self.interpolate_fn is not None:
      pp_tendency = self.interpolate_fn(pp_tendency)
      prognostics = self.interpolate_fn(prognostics)
    next_prognostics = self.step_with_fixed_parameterization_tendencies(
        prognostics, pp_tendency
    )
    if self.reverse_interpolate_fn is not None:
      next_prognostics = self.reverse_interpolate_fn(next_prognostics)
    # next_prognostics = self.internal_coords.with_dycore_sharding(
    #     next_prognostics)
    return next_prognostics

  def __call__(self, prognostics: typing.Pytree) -> typing.Pytree:
    """Computes `prognostics` evolved in time by `dt`."""

    def substep_fn(prognostics_and_self):
      prognostics, model = prognostics_and_self
      next_prognostics = model.substep(prognostics)
      return next_prognostics, model

    if self.remat_substep:
      substep_fn = nnx.remat(substep_fn)
    step_fn = time_integrators.repeated(substep_fn, self.num_substeps)
    next_prognostics, _ = step_fn((prognostics, self))
    return next_prognostics


class EquationStep(nnx.Module):
  """Class to wrap an equation and an integrator to perform a single step."""

  def __init__(
      self,
      *,
      dt: float,
      equation: equations.ExplicitODE,
      integrator_cls: Callable[..., time_integrators.DinosaurIntegrator],
  ):
    self.dt = dt
    self.step = integrator_cls(equation, dt)

  def __call__(self, prognostics: typing.Pytree) -> typing.Pytree:
    next_prognostics = self.step(prognostics)
    return time_integrators.maybe_fix_sim_time_roundoff(
        next_prognostics, self.dt
    )
