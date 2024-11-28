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

"""Types used by neuralgcm.experimental API."""
from __future__ import annotations

import abc
import dataclasses
import datetime
import functools
from typing import Any, Callable, Generic, TypeVar

import jax
import jax.numpy as jnp
from neuralgcm.experimental import scales
import numpy as np
import pandas as pd
import tree_math


units = scales.units
#
# Generic types.
#
Array = np.ndarray | jax.Array
Dtype = jax.typing.DTypeLike | Any
Numeric = float | int | Array
PRNGKeyArray = jax.Array
ShapeDtypeStruct = jax.ShapeDtypeStruct
ShapeFloatStruct = functools.partial(ShapeDtypeStruct, dtype=jnp.float32)
Timestep = np.timedelta64 | float
TimedeltaLike = str | np.timedelta64 | pd.Timestamp | datetime.timedelta
Quantity = scales.Quantity

#
# Generic API input/output types.
#
Pytree = Any
PyTreeState = TypeVar('PyTreeState')


#
# Simulation function signatures.
#
StepFn = Callable[[PyTreeState], PyTreeState]
PostProcessFn = Callable[..., Pytree]


class ObservationSpecs(abc.ABC):
  """Base class for observation data specifications.

  Observation specification objects are used to express queries to the model's
  `observe` method. All ObservationSpecs objects should have a corresponding
  ObservationData class that can be used to store the data, both of which must
  be a valid jax pytree.

  In many cases the difference between ObservationSpecs and ObservationData is
  that ObservationSpecs contain supporting values and a set of
  coordax.Coordinate objects while ObservationData is structured similarly but
  uses coordax.Field to store data. This can be reflected by using the same
  object that inherits from both ObservationSpecs and ObservationData and
  supports coordax.Field and coordax.Coordinate for relevant attributes.
  """


class ObservationData(abc.ABC):
  """Base class for observation data."""

  @property
  @abc.abstractmethod
  def get_specs(self) -> ObservationSpecs:
    """Returns ObservationSpecs that describe data specifications."""
    ...


Query = dict[str, ObservationSpecs]
Observation = dict[str, ObservationData]


@tree_math.struct
class ModelState(Generic[PyTreeState]):
  """Simulation state decomposed into prognostic, diagnostic and randomness.

  Attributes:
    prognostics: Prognostic variables describing the simulation state.
    diagnostics: Optional diagnostic values holding diagnostic information.
    randomness: Optional randomness state describing stochasticity of the model.
  """

  prognostics: PyTreeState
  diagnostics: Pytree = dataclasses.field(default_factory=dict)
  randomness: Pytree = dataclasses.field(default_factory=dict)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Randomness:
  """State describing the random process."""

  prng_key: jax.Array
  prng_step: int = 0
  core: Pytree = None

  def tree_flatten(self):
    """Flattens Randomness JAX pytree."""
    leaves = (self.prng_key, self.prng_step, self.core)
    aux_data = ()
    return leaves, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    """Unflattens Randomness from aux_data and leaves."""
    return cls(*leaves, *aux_data)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Timedelta:
  """JAX compatible time duration, stored in days and seconds.

  Like datetime.timedelta, the Timedelta constructor and arithmetic operations
  normalize seconds to fall in the range [0, 24 * 60 * 60). Timedelta objects
  are pytrees, but normalization is skipped inside jax.tree operations because
  JAX uses pytrees with non-numeric types to implement JAX transformations.

  Using integer days and seconds is recommended to avoid loss of precision. With
  int32 days, Timedelta can exactly represent durations over 5 million years.

  The easiest way to create a Timedelta is to use `from_timedelta64`, which
  supports `np.timedelta64` objects and NumPy arrays with a timedelta64 dtype:

    >>> Timedelta.from_timedelta64(np.timedelta64(1, 's'))
    Timedelta(days=0, seconds=1)
  """

  days: Numeric = 0
  seconds: Numeric = 0

  # TODO(shoyer): can we rewrite this a custom JAX dtype, like jax.random.key?

  def __post_init__(self):
    days_delta, seconds = divmod(self.seconds, 24 * 60 * 60)
    self.days = self.days + days_delta
    self.seconds = seconds

  @classmethod
  def from_timedelta64(cls, values: np.timedelta64 | np.ndarray) -> Timedelta:
    seconds = values // np.timedelta64(1, 's')
    # no need to worry about overflow, because timedelta64 represents values
    # internally with int64 and normalization uses native array operations
    return Timedelta(0, seconds)

  def to_timedelta64(self) -> np.timedelta64 | np.ndarray:
    seconds = np.int64(self.days) * 24 * 60 * 60 + np.int64(self.seconds)
    return seconds * np.timedelta64(1, 's')

  def __add__(self, other):
    if not isinstance(other, Timedelta):
      return NotImplemented
    days = self.days + other.days
    seconds = self.seconds + other.seconds
    return Timedelta(days, seconds)

  def __neg__(self):
    return Timedelta(-self.days, -self.seconds)

  def __sub__(self, other):
    if not isinstance(other, Timedelta):
      return NotImplemented
    return self + (-other)

  def __mul__(self, other):
    if not isinstance(other, Numeric):
      return NotImplemented
    return Timedelta(self.days * other, self.seconds * other)

  __rmul__ = __mul__

  # TODO(shoyer): consider adding other methods supported by datetime.timedelta.

  def tree_flatten(self):
    leaves = (self.days, self.seconds)
    aux_data = None
    return leaves, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    assert aux_data is None
    # JAX uses non-numeric values for pytree leaves inside transformations, so
    # we skip __post_init__ by constructing the object directly:
    # https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization
    result = object.__new__(cls)
    result.days, result.seconds = leaves
    return result


_UNIX_EPOCH = np.datetime64('1970-01-01T00:00:00', 's')


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Timestamp:
  """JAX compatible timestamp, stored as a delta from the Unix epoch.

  The easiest way to create a Timestamp is to use `from_datetime64`, which
  supports `np.datetime64` objects and NumPy arrays with a datetime64 dtype:

    >>> Timestamp.from_datetime64(np.datetime64('1970-01-02'))
    Timestamp(delta=Timedelta(days=1, seconds=0))
  """

  delta: Timedelta

  @classmethod
  def from_datetime64(cls, values: np.datetime64 | np.ndarray) -> Timestamp:
    return cls(Timedelta.from_timedelta64(values - _UNIX_EPOCH))

  def to_datetime64(self) -> np.timedelta64 | np.ndarray:
    return self.delta.to_timedelta64() + _UNIX_EPOCH

  def __add__(self, other):
    if not isinstance(other, Timedelta):
      return NotImplemented
    return Timestamp(self.delta + other)

  __radd__ = __add__

  def __sub__(self, other):
    if isinstance(other, Timestamp):
      return self.delta - other.delta
    elif isinstance(other, Timedelta):
      return Timestamp(self.delta - other)
    else:
      return NotImplemented

  def tree_flatten(self):
    leaves = (self.delta,)
    aux_data = None
    return leaves, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    assert aux_data is None
    return cls(*leaves)


#
# API function signatures.
#
PostProcessFn = Callable[..., Any]


#
# Auxiliary types for intermediate computations.
#
@dataclasses.dataclass(eq=True, order=True, frozen=True)
class KeyWithCosLatFactor:
  """Class describing a key by `name` and an integer `factor_order`."""

  name: str
  factor_order: int
