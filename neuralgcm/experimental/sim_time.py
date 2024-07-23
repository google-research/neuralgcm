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
"""Utilities for keeping track of simulation time."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from neuralgcm.experimental import typing


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class SimTime:
  """Representation of elapsed simulated time in days.

  SimTime stores elapsed time in two parts, in order to avoid precision loss
  associated with storing time in a single float32 value:
  - days: the whole number of days that have elapsed
  - fraction: the fraction of a day that has elapsed

  Total time can be obtained by adding days and fraction.

  SimTime can be incremented by a floating point number of days using the
  `increment` method. Compounding errors are avoided by rounding to the nearest
  day when the fraction is close to 1 or 0 (within 1 second).

  This implies two requirements for time-step size:
  1. Increments should always be at least one second.
  2. The fractional part of time increments should evenly divide a small integer
     number of days.

  When converting back and forth to datetime/timedelta types we recommend
  rounding to the nearest second.
  """

  days: typing.Numeric
  fraction: typing.Numeric

  @jax.jit
  def increment(self, delta: typing.Numeric, /) -> SimTime:
    days_delta, frac = divmod(self.fraction + delta, 1)
    days = self.days + days_delta
    epsilon = 1 / (60 * 60 * 24)  # one second, ~1.2e-05
    round_up = frac > 1 - epsilon
    round_down = frac < epsilon
    days = jnp.where(round_up, days + 1, days)
    frac = jnp.where(round_up | round_down, 0.0, frac)
    return SimTime(days=days, fraction=frac)

  def tree_flatten(self):
    leaves = (self.days, self.fraction)
    aux_data = None
    return leaves, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    assert aux_data is None
    return cls(*leaves)
