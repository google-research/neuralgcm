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

from collections import abc
import datetime
import functools
from typing import Any, Callable

from dinosaur import coordinate_systems
from dinosaur import scales
from dinosaur import time_integration
from dinosaur import typing
from dinosaur import xarray_utils
import jax
from jax import tree_util
import jax.numpy as jnp
from neuralgcm import gin_utils
from neuralgcm import model_builder
from neuralgcm import physics_specifications
import numpy as np
import pandas as pd
import xarray


ArrayLike = float | np.ndarray | jax.Array
Params = dict[str, dict[str, ArrayLike]]
TimedeltaLike = str | np.timedelta64 | pd.Timestamp | datetime.timedelta
Numeric = float | np.ndarray | jax.Array | xarray.DataArray

# TODO(shoyer): make these types more precise
Inputs = dict[str, ArrayLike]
Forcings = dict[str, ArrayLike]
TemporalForcings = dict[str, ArrayLike]
Outputs = dict[str, jax.Array]
BatchedOutputs = dict[str, jax.Array]
State = Any


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


def _check_variables(
    dataset: xarray.Dataset,
    desired_level_variables: abc.Sequence[str] = (),
    desired_surface_variables: abc.Sequence[str] = (),
):
  """Checks that a dataset has the desired variables."""
  T, Z, X, Y = ('time', 'level', 'longitude', 'latitude')  # pylint: disable=invalid-name

  for k in desired_level_variables:
    if k not in dataset.data_vars:
      raise ValueError(f'expected variable {k} not found')
    dims = dataset[k].dims
    if not (set(dims) == {Z, X, Y} or set(dims) == {T, Z, X, Y}):
      raise ValueError(
          f'expected variable {k} to have dims {(Z, X, Y)} or {(T, Z, X, Y)},'
          f' but got {dims}'
      )

  for k in desired_surface_variables:
    if k not in dataset.data_vars:
      raise ValueError(f'expected variable {k} not found')
    dims = dataset[k].dims
    if not (set(dims) == {X, Y} or set(dims) == {T, X, Y}):
      raise ValueError(
          f'expected variable {k} to have dims {(X, Y)} or {(T, X, Y)},'
          f' but got {dims}'
      )


def _check_coords(
    actual_coords: coordinate_systems.CoordinateSystem,
    desired_coords: coordinate_systems.CoordinateSystem,
) -> None:
  """Checks that a dataset has the desired coordinates."""
  if not np.allclose(
      actual := actual_coords.horizontal.longitudes,
      desired := desired_coords.horizontal.longitudes,
      atol=1e-3,
  ):
    raise ValueError(f'longitude coordinate mismatch: {actual=}, {desired=}')

  if not np.allclose(
      actual := actual_coords.horizontal.latitudes,
      desired := desired_coords.horizontal.latitudes,
      atol=1e-3,
  ):
    raise ValueError(f'latitude coordinate mismatch: {actual=}, {desired=}')

  if actual_coords.vertical is not None and not np.allclose(
      actual := actual_coords.vertical.centers,
      desired := desired_coords.vertical.centers,
      atol=1e-3,
  ):
    raise ValueError(
        f'pressure level coordinate mismatch: {actual=}, {desired=}'
    )


def _rename_if_found(
    dataset: xarray.Dataset, names: dict[str, str]
) -> xarray.Dataset:
  return dataset.rename({k: v for k, v in names.items() if k in dataset})


_ABBREVIATED_NAMES = {
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'geopotential': 'z',
    'temperature': 't',
    'longitude': 'lon',
    'latitude': 'lat',
}
_FULL_NAMES = {v: k for k, v in _ABBREVIATED_NAMES.items()}


def _expand_tracers(inputs: dict) -> dict:
  inputs = inputs.copy()
  inputs.update(inputs.pop('tracers'))
  assert not inputs['diagnostics']
  del inputs['diagnostics']
  return inputs


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

    self._tracer_variables = [
        'specific_humidity',
    ]
    self._input_variables = [
        'geopotential',
        'specific_humidity',
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
    ]
    # Some old model versions do not use cloud variables.
    # TODO(shoyer): remove this once all integration tests are updated.
    cloud_variables = [
        'specific_cloud_ice_water_content',
        'specific_cloud_liquid_water_content',
    ]
    for variable in cloud_variables:
      if variable in self.gin_config:
        self._tracer_variables.append(variable)
        self._input_variables.append(variable)

    self._forcing_variables = [
        'sea_ice_cover',
        'sea_surface_temperature',
    ]

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
  def input_variables(self) -> list[str]:
    """List of variable names required in `inputs` by this model."""
    return list(self._input_variables)

  @property
  def forcing_variables(self) -> list[str]:
    """List of variable names required in `forcings` by this model."""
    return list(self._forcing_variables)

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

  def _check_coords(self, dataset: xarray.Dataset):
    dataset_coords = model_builder.coordinate_system_from_dataset(dataset)
    _check_coords(dataset_coords, self.data_coords)

  def _dataset_with_sim_time(self, dataset: xarray.Dataset) -> xarray.Dataset:
    ref_datetime = self._structure.specs.aux_features['reference_datetime']
    return xarray_utils.ds_with_sim_time(
        dataset,
        self._structure.specs.physics_specs,
        reference_datetime=ref_datetime,
    )

  def _to_abbreviated_names_and_tracers(self, inputs: dict) -> dict:
    inputs = {_ABBREVIATED_NAMES.get(k, k): v for k, v in inputs.items()}
    inputs['tracers'] = {
        k: inputs.pop(k) for k in self._tracer_variables if k in inputs
    }
    inputs['diagnostics'] = {}
    return inputs

  def _from_abbreviated_names_and_tracers(self, outputs: dict) -> dict:
    outputs = {_FULL_NAMES.get(k, k): v for k, v in outputs.items()}
    outputs |= outputs.pop('tracers')
    outputs |= outputs.pop('diagnostics')
    return outputs

  def to_nondim_units(self, value: Numeric, units: str) -> Numeric:
    """Scale a value to the model's internal non-dimensional units."""
    scale_ = self._structure.specs.physics_specs.scale
    units_ = scales.parse_units(units)
    return scale_.nondimensionalize(value * units_)

  def from_nondim_units(self, value: Numeric, units: str) -> Numeric:
    """Scale a value from the model's internal non-dimensional units."""
    scale_ = self._structure.specs.physics_specs.scale
    units_ = scales.parse_units(units)
    return scale_.dimensionalize(value, units_).magnitude

  def datetime64_to_sim_time(self, datetime64: np.ndarray) -> np.ndarray:
    """Converts a datetime64 array to sim_time."""
    ref_datetime = self._structure.specs.aux_features['reference_datetime']
    return xarray_utils.datetime64_to_nondim_time(
        datetime64,
        self._structure.specs.physics_specs,
        reference_datetime=ref_datetime,
    )

  def sim_time_to_datetime64(self, sim_time: np.ndarray) -> np.ndarray:
    """Converts a sim_time array to datetime64."""
    ref_datetime = self._structure.specs.aux_features['reference_datetime']
    return xarray_utils.nondim_time_to_datetime64(
        sim_time,
        self._structure.specs.physics_specs,
        reference_datetime=ref_datetime,
    )

  def _data_from_xarray(
      self, dataset: xarray.Dataset, variables: list[str]
  ) -> dict[str, np.ndarray]:
    self._check_coords(dataset)
    dataset = dataset[variables]
    dataset = self._dataset_with_sim_time(dataset)
    dataset = _rename_if_found(dataset, {'longitude': 'lon', 'latitude': 'lat'})
    return xarray_utils.xarray_to_data_dict(dataset)

  def inputs_from_xarray(
      self, dataset: xarray.Dataset
  ) -> dict[str, np.ndarray]:
    """Extract inputs from an xarray.Dataset."""
    _check_variables(dataset, desired_level_variables=self._input_variables)
    return self._data_from_xarray(dataset, self._input_variables)

  def forcings_from_xarray(
      self, dataset: xarray.Dataset
  ) -> dict[str, np.ndarray]:
    """Extract forcings from an xarray.Dataset."""
    _check_variables(dataset, desired_surface_variables=self._forcing_variables)
    return self._data_from_xarray(dataset, self._forcing_variables)

  def data_from_xarray(
      self, dataset: xarray.Dataset
  ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extracts data and forcings from an xarray.Dataset."""
    inputs = self.inputs_from_xarray(dataset)
    forcings = self.forcings_from_xarray(dataset)
    return (inputs, forcings)

  def data_to_xarray(
      self,
      data: dict[str, ArrayLike],
      times: np.ndarray | None,
      decoded: bool = True,
  ) -> xarray.Dataset:
    """Converts decoded model predictions to xarray.Dataset format.

    Args:
      data: dict of arrays with shapes matching input/outputs or encoded model
        state for this model, i.e., with shape
        `([time,] level, longitude, latitude)`,
        where `[time,]` indicates an optional leading time dimension.
      times: either `None` indicating no leading time dimension on any
        variables, or a coordinate array of times with shape `(time,)`.
      decoded: if `True`, use `self.data_coords` to determine the output
        coordinates; otherwise use `self.model_coords`.

    Returns:
      An xarray.Dataset with appropriate coordinates and dimensions.
    """
    coords = self.data_coords if decoded else self.model_coords
    dataset = xarray_utils.data_to_xarray(data, coords=coords, times=times)
    dataset = _rename_if_found(dataset, {'lon': 'longitude', 'lat': 'latitude'})
    return dataset

  def _squeeze_level_from_forcings(self, forcings: Forcings) -> Forcings:
    # Due to a bug in xarray_to_dynamic_covariate_data, we were accidentally
    # not inserting a level dimension in forcings.
    forcings = dict(forcings)
    for k in self._forcing_variables:
      if k in forcings:
        assert isinstance(forcings[k], (np.ndarray, jax.Array))
        forcings[k] = forcings[k].squeeze(axis=-3)
    return forcings

  @jax.jit
  @_static_gin_config
  def encode(
      self,
      inputs: Inputs,
      forcings: Forcings,
      rng_key: typing.PRNGKeyArray | None = None,
  ) -> State:
    """Encode from pressure-level inputs & forcings to model state.

    Args:
      inputs: input data on pressure-levels, as a dict where each entry is an
        array with shape `[level, longitude, latitude]` matching `data_coords`.
      forcings: forcing data on pressure-levels, as a dict where each entry is
        an array with shape `[level, longitude, latitude]` matching
        `data_coords`. Single level data (e.g., sea surface temperature) should
        have a `level` dimension of size 1.
      rng_key: optional JAX RNG key to use for encoding the state. Required if
        using stochastic models, otherwise ignored.

    Returns:
      Dynamical core state on sigma levels, where all arrays have dimensions
      `[level, zonal_wavenumber, total_wavenumber]` matching `model_coords`.
    """
    sim_time = inputs['sim_time']
    inputs = self._to_abbreviated_names_and_tracers(inputs)
    inputs = _prepend_dummy_time_axis(inputs)
    forcings = self._squeeze_level_from_forcings(forcings)
    forcings = _prepend_dummy_time_axis(forcings)
    f = self._structure.forcing_fn(self.params, None, forcings, sim_time)
    return self._structure.encode_fn(self.params, rng_key, inputs, f)

  @jax.jit
  @_static_gin_config
  def advance(self, state: State, forcings: Forcings) -> State:
    """Advance model state one timestep forward.

    Args:
      state: dynamical core state on sigma levels, where all arrays have
        dimensions `[level, zonal_wavenumber, total_wavenumber]` matching
        `model_coords`
      forcings: forcing data on pressure-levels, as a dict where each entry is
        an array with shape `[level, longitude, latitude]` matching
        `data_coords`. Single level data (e.g., sea surface temperature) should
        have a `level` dimension of size 1.

    Returns:
      State advanced one time-step forward.
    """
    sim_time = _sim_time_from_state(state)
    forcings = self._squeeze_level_from_forcings(forcings)
    forcings = _prepend_dummy_time_axis(forcings)
    f = self._structure.forcing_fn(self.params, None, forcings, sim_time)
    state = self._structure.advance_fn(self.params, None, state, f)
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
      Outputs on pressure-levels, as a dict where each entry is an array with
      shape `[level, longitude, latitude]` matching `data_coords`.
    """
    sim_time = _sim_time_from_state(state)
    forcings = self._squeeze_level_from_forcings(forcings)
    forcings = _prepend_dummy_time_axis(forcings)
    f = self._structure.forcing_fn(self.params, None, forcings, sim_time)
    outputs = self._structure.decode_fn(self.params, None, state, f)
    outputs = self._from_abbreviated_names_and_tracers(outputs)
    return outputs

  @functools.partial(
      jax.jit,
      static_argnames=[
          'steps',
          'timedelta',
          'start_with_input',
          'post_process_fn',
      ],
  )
  @_static_gin_config
  def unroll(
      self,
      state: State,
      forcings: TemporalForcings,
      *,
      steps: int,
      timedelta: TimedeltaLike | None = None,
      start_with_input: bool = False,
      post_process_fn: Callable[[State], Any] | None = None,
  ) -> tuple[State, BatchedOutputs]:
    """Unroll predictions over many time-steps.

    Usage:

      advanced_state, outputs = model.unroll(state, forcings, steps=N)

    where ``advanced_state`` is the advanced model state after ``N`` steps and
    ``outputs`` is a trajectory of decoded states on pressure-levels with a
    leading dimension of size ``N``.

    Args:
      state: initial model state.
      forcings: forcing data over the time-period spanned by the desired output
        trajectory. Should include a leading time-axis, but times can be at any
        desired granularity (e.g., it should be fine to supply daily forcing
        data, even if producing hourly outputs). The nearest forcing in time
        will be used for each internal ``advance()`` and ``decode()`` call.
      steps: number of time-steps to take.
      timedelta: size of each time-step to take, which must be a multiple of the
        internal model timestep. By default uses the internal model timestep.
      start_with_input: if ``True``, outputs are at times ``[0, ...,
        (steps - 1) * timestep]`` relative to the initial time; if ``False``,
        outputs are at times ``[timestep, ..., steps * timestep]``.
      post_process_fn: optional function to apply to each advanced state and
        current forcings to create outputs like
        ``post_process_fn(state, forcings)``, where ``forcings`` does not
        include a time axis. By default, uses ``model.decode``.

    Returns:
      A tuple of the advanced state at time ``steps * timestamp``, and outputs
      with a leading ``time`` axis at the time-steps specified by ``steps``,
      ``timedelta`` and ``start_with_input``.
    """
    if timedelta is None:
      timedelta = self.timestep

    def get_nearest_forcings(sim_time):
      times = forcings['sim_time']
      assert isinstance(times, jax.Array)
      approx_index = jnp.interp(sim_time, times, jnp.arange(times.size))
      index = jnp.round(approx_index).astype(jnp.int32)
      return jax.tree.map(lambda x: x[index, ...], forcings)

    def with_nearest_forcings(func):
      def wrapped(state):
        sim_time = _sim_time_from_state(state)
        forcings = get_nearest_forcings(sim_time)
        return func(state, forcings)
      return wrapped

    if post_process_fn is None:
      post_process_fn = self.decode

    inner_steps = _calculate_sub_steps(self.timestep, timedelta)
    trajectory_func = time_integration.trajectory_from_step(
        with_nearest_forcings(self.advance),
        outer_steps=steps,
        inner_steps=inner_steps,
        start_with_input=start_with_input,
        post_process_fn=with_nearest_forcings(post_process_fn),
    )
    state, outputs = trajectory_func(state)
    return state, outputs

  @classmethod
  def from_checkpoint(cls, checkpoint: Any) -> PressureLevelModel:
    """Creates a PressureLevelModel from a checkpoint.

    Args:
      checkpoint: dictionary with keys "model_config_str", "aux_ds_dict" and
        "params" that specifies model gin configuration, supplemental xarray
        dataset with model-specific static features, and model parameters.

    Returns:
      Instance of a `PressureLevelModel` with weights and configuration
      specified by the checkpoint.
    """
    with gin_utils.specific_config(checkpoint['model_config_str']):
      physics_specs = physics_specifications.get_physics_specs()
      aux_ds = xarray.Dataset.from_dict(checkpoint['aux_ds_dict'])
      data_coords = model_builder.coordinate_system_from_dataset(aux_ds)
      model_specs = model_builder.get_model_specs(
          data_coords, physics_specs, {xarray_utils.XARRAY_DS_KEY: aux_ds}
      )
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
