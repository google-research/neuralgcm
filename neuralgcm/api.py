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
"""Public API for NeuralGCM models."""
from __future__ import annotations
import datetime
import functools
from typing import Any

from dinosaur import coordinate_systems
from dinosaur import typing
from dinosaur import xarray_utils
import haiku as hk
import jax
from jax import tree_util
import jax.numpy as jnp
from neuralgcm import gin_utils
from neuralgcm import model_builder
from neuralgcm import model_utils
from neuralgcm import physics_specifications
import numpy as np
import pandas as pd
import xarray


Params = dict[str, dict[str, jnp.ndarray]]
# TODO(shoyer): make these types more precise
Inputs = Any
Forcings = Any
Outputs = Any
BatchedOutputs = Any
State = Any

TimedeltaLike = str | np.timedelta64 | pd.Timestamp | datetime.timedelta


def _sim_time_from_state(state: State) -> jax.Array:
  """Extract sim_time from model state."""
  # TODO(shoyer): eliminate whichever of these two cases is no longer needed!
  # TODO(shoyer): consider renaming `sim_time` to `time`?
  if isinstance(state, typing.ModelState):
    sim_time = getattr(state.state, 'sim_time', None)
  else:
    sim_time = getattr(state, 'sim_time', None)
  return sim_time


def _calculate_sub_steps(
    timestep: np.timedelta64, duration: TimedeltaLike
) -> int:
  """Calculate the number of time-steps required to simulate a time interval."""
  duration = pd.Timedelta(duration)
  time_step_ratio = duration / timestep
  if abs(time_step_ratio - round(time_step_ratio)) > 1e-6:
    raise ValueError(
        f'non-integral time-step ratio: {duration=} is not a multiple of '
        f'the internal model timestep {timestep}'
    )
  return round(time_step_ratio)


def _prepend_dummy_time_axis(state: typing.Pytree) -> typing.Pytree:
  return tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)


def _static_gin_config(method):
  """Decorator to add static gin config to a method."""
  @functools.wraps(method)
  def _method(self, *args, **kwargs):
    with gin_utils.specific_config(self.gin_config):
      return method(self, *args, **kwargs)
  return _method


@tree_util.register_pytree_node_class
class PressureLevelModel:
  """Inference-only API for models that predict dense data on pressure levels.

  These models are trained on ECMWF ERA5 data on pressure-levels as stored in
  the Copernicus Data Store.

  This class encapsulates the details of defining models (e.g., with Haiku) and
  hence should remain stable even for future NeuralGCM models.
  """

  def __init__(
      self,
      structure: model_builder.WhirlModel,
      params: Params,
      gin_config: str,
  ):
    self._structure = structure
    self._params = params
    self.gin_config = gin_config

  def __repr__(self):
    return (
        f'{self.__class__.__name__}(structure={self._structure},'
        f' params={self._params})'
    )

  @property
  def params(self) -> Params:
    return self._params

  def tree_flatten(self):
    leaves, params_def = tree_util.tree_flatten(self.params)
    return (leaves, (params_def, self._structure, self.gin_config))

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    params_def, structure, gin_config = aux_data
    params = tree_util.tree_unflatten(params_def, leaves)
    return cls(structure, params, gin_config)

  @property
  def timestep(self) -> np.timedelta64:
    """Spacing between internal model timesteps."""
    to_timedelta = (
        self._structure.specs.physics_specs.dimensionalize_timedelta64
    )
    return to_timedelta(self._structure.specs.dt)

  @property
  def data_coords(self) -> coordinate_systems.CoordinateSystem:
    """Coordinate system for input and output data."""
    return self._structure.data_coords

  @property
  def model_coords(self) -> coordinate_systems.CoordinateSystem:
    """Coordinate system for internal model state."""
    return self._structure.coords

  @_static_gin_config
  def data_from_xarray(
      self, dataset: xarray.Dataset
  ) -> tuple[Inputs, Forcings]:
    """Extracts data and forcings from xarray.Dataset."""
    ref_datetime = self._structure.specs.aux_features['reference_datetime']
    dataset = xarray_utils.ds_with_sim_time(
        dataset,
        self._structure.specs.physics_specs,
        reference_datetime=ref_datetime,
    )
    dataset_coords = model_builder.coordinate_system_from_dataset(dataset)
    if not np.allclose(
        dataset_coords.horizontal.longitudes,
        self.data_coords.horizontal.longitudes,
        atol=1e-3,
    ):
      raise ValueError('longitude coordinate mismatch')
    if not np.allclose(
        dataset_coords.horizontal.latitudes,
        self.data_coords.horizontal.latitudes,
        atol=1e-3,
    ):
      raise ValueError('latitude coordinate mismatch')
    if not np.allclose(
        dataset_coords.vertical.centers,
        self.data_coords.vertical.centers,
        atol=1e-3,
    ):
      raise ValueError('pressure level coordinate mismatch')
    return self._structure.from_xarray_fn(dataset)

  def data_to_xarray(
      self,
      data: Inputs | Outputs,
      times: np.ndarray | None,
  ) -> xarray.Dataset:
    """Converts decoded model predictions to xarray.Dataset format."""
    default_to_internal_names_dict = {
        'u_component_of_wind': 'u',
        'v_component_of_wind': 'v',
        'geopotential': 'z',
        'temperature': 't',
        'longitude': 'lon',
        'latitude': 'lat'
    }
    return xarray_utils.data_to_xarray_with_renaming(
        data,
        to_xarray_fn=xarray_utils.data_to_xarray,
        renaming_dict=default_to_internal_names_dict,
        coords=self.data_coords,
        times=times,
    )

  @jax.jit
  @_static_gin_config
  def encode(
      self, inputs: Inputs, forcings: Forcings, rng_key: typing.PRNGKeyArray
  ) -> State:
    """Encode from pressure-level inputs & forcings to model state.

    Args:
      inputs: input data on pressure-levels, as a dict where each entry is an
        array with shape `[level, longitude, latitude]` matching `data_coords`.
      forcings: forcing data on pressure-levels, as a dict where each entry is
        an array with shape `[level, longitude, latitude]` matching
        `data_coords`. Single level data (e.g., sea surface temperature) should
        have a `level` dimension of size 1.
      rng_key: JAX RNG key to use for encoding the state.

    Returns:
      Dynamical core state on sigma levels, where all arrays have dimensions
      `[level, zonal_wavenumber, total_wavenumber]` matching `model_coords`.
    """
    # TODO(langmore): refactor into an API that explicitly takes input random
    # noise rather than an RNG key.
    sim_time = inputs['sim_time']
    inputs, forcings = _prepend_dummy_time_axis((inputs, forcings))
    f = self._structure.forcing_fn(self.params, None, forcings, sim_time)
    return self._structure.encode_fn(self.params, rng_key, inputs, f)

  @jax.jit
  @_static_gin_config
  def advance(
      self, state: State, forcings: Forcings, rng_key: typing.PRNGKeyArray
  ) -> State:
    """Advance model state one timestep forward.

    Args:
      state: dynamical core state on sigma levels, where all arrays have
        dimensions `[level, zonal_wavenumber, total_wavenumber]` matching
        `model_coords`
      forcings: forcing data on pressure-levels, as a dict where each entry is
        an array with shape `[level, longitude, latitude]` matching
        `data_coords`. Single level data (e.g., sea surface temperature) should
        have a `level` dimension of size 1.
      rng_key: JAX RNG key to use for advancing the state.

    Returns:
      State advanced one time-step forward.
    """
    # TODO(shoyer): refactor rng_key into RandomnessState.
    sim_time = _sim_time_from_state(state)
    forcings = _prepend_dummy_time_axis(forcings)
    f = self._structure.forcing_fn(self.params, None, forcings, sim_time)
    state = self._structure.advance_fn(self.params, rng_key, state, f)
    return state

  @jax.jit
  @_static_gin_config
  def decode(self, state: State, forcings: Forcings) -> Outputs:
    """Decode from model state to pressure-level outputs.

    Args:
      state: dynamical core state on sigma levels, where all arrays have
        dimensions `[level, zonal_wavenumber, total_wavenumber]` matching
        `model_coords`.
      forcings: forcing data on pressure-levels, as a dict where each entry is
        an array with shape `[level, longitude, latitude]` matching
        `data_coords`. Single level data (e.g., sea surface temperature) should
        have a `level` dimension of size 1.

    Returns:
      inputs: outputs on pressure-levels, as a dict where each entry is an
        array with shape `[level, longitude, latitude]` matching `data_coords`.
    """
    # TODO(shoyer): consider adding an RNG key?
    sim_time = _sim_time_from_state(state)
    forcings = _prepend_dummy_time_axis(forcings)
    f = self._structure.forcing_fn(self.params, None, forcings, sim_time)
    return self._structure.decode_fn(self.params, None, state, f)

  @functools.partial(
      jax.jit,
      static_argnames=['steps', 'timedelta', 'start_with_input'],
  )
  @_static_gin_config
  def unroll(
      self,
      state: State,
      forcings: Forcings,
      rng_key: typing.PRNGKeyArray,
      *,
      steps: int,
      timedelta: TimedeltaLike | None = None,
      start_with_input: bool = False,
  ) -> tuple[State, BatchedOutputs]:
    """Unroll predictions over many time-steps.

    Usage:

      advanced_state, outputs = model.unroll(
          state, forcings, rng_key, steps=N, post_process_fn=model.decode
      )

    where `advanced_state` is the advanced model state after `N` steps and
    `outputs` is a trajectory of decoded states on pressure-levels with a
    leading dimension of size `N`.

    Args:
      state: initial model state.
      forcings: forcing data over the time-period spanned by the desired output
        trajectory. Should include a leading time-axis, but times can be at any
        desired granularity compatible with the model (e.g., it should be fine
        to supply daily forcing data, even if producing hourly outputs).
      rng_key: random key to use for advancing state.
      steps: number of time-steps to take.
      timedelta: size of each time-step to take, which must be a multiple of the
        internal model timestep. By default uses the internal model timestep.
      start_with_input: if `True`, outputs are at times `[0, ..., (steps
        - 1) * timestep]` relative to the initial time; if `False`, outputs
        are at times `[timestep, ..., steps * timestep]`.

    Returns:
      A tuple of the advanced state at time `steps * timestamp`, and outputs
      with a leading `time` axis at the time-steps specified by
      `start_with_input`.
    """
    if timedelta is None:
      timedelta = self.timestep

    inner_steps = _calculate_sub_steps(self.timestep, timedelta)

    def compute_slice_fwd(state, forcings):
      model = self._structure.model_cls()
      # TODO(shoyer): reimplement via encode/advance/decode, in order to
      # guarantee consistency and allow for more flexible decoding. This would
      # be easiest after moving rng_key into state.
      trajectory_fn = model_utils.decoded_trajectory_with_forcing(
          model, start_with_input=start_with_input
      )
      return trajectory_fn(
          state,
          forcing_data=forcings,
          outer_steps=steps,
          inner_steps=inner_steps,
      )

    compute_slice = hk.transform(compute_slice_fwd)
    return compute_slice.apply(self.params, rng_key, state, forcings)

  @classmethod
  def from_checkpoint(cls, checkpoint: Any) -> PressureLevelModel:
    """Creates a PressureLevelModel from a checkpoint.

    Args:
      checkpoint: dictionary with keys "model_config_str", "aux_ds_dict"
        and "params" that specifies model gin configuration, supplemental
        xarray dataset with model-specific static features, and model
        parameters.

    Returns:
      Instance of a `PressureLevelModel` with weights and configuration
      specified by the checkpoint.
    """
    with gin_utils.specific_config(checkpoint['model_config_str']):
      physics_specs = physics_specifications.get_physics_specs()
      aux_ds = xarray.Dataset.from_dict(checkpoint['aux_ds_dict'])
      data_coords = model_builder.coordinate_system_from_dataset(aux_ds)
      model_specs = model_builder.get_model_specs(
          data_coords, physics_specs, {xarray_utils.XARRAY_DS_KEY: aux_ds})
      whirl_model = model_builder.WhirlModel(
          coords=model_specs.coords,
          dt=model_specs.dt,
          physics_specs=model_specs.physics_specs,
          aux_features=model_specs.aux_features,
          input_coords=data_coords,
          output_coords=data_coords,
      )
      return cls(
          whirl_model, checkpoint['params'], checkpoint['model_config_str']
      )


# TODO(shoyer): consider adding separate subclasses for deterministic and
# stochastic models?
#
