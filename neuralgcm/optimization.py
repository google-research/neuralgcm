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
"""Configurable optimizers from JAX."""
import collections
import re
from typing import Sequence

import gin
import optax


gin.external_configurable(optax.adabelief, module='optax')
gin.external_configurable(optax.adam, module='optax')
gin.external_configurable(optax.adamw, module='optax')

gin.external_configurable(optax.constant_schedule, module='optax')
gin.external_configurable(optax.join_schedules, module='optax')
gin.external_configurable(optax.piecewise_constant_schedule, module='optax')
gin.external_configurable(optax.exponential_decay, module='optax')
gin.external_configurable(
    optax.warmup_exponential_decay_schedule, module='optax'
)


class OptimizerError(Exception):
  """Raised if a custom Whirl optimizer encounters an error."""


@gin.configurable
def optimizer(value):
  return value


OptState = collections.namedtuple('OptState', ['state', 'params'])


@gin.register
def piecewise_constant_schedule_specified_by_rates(
    rates: Sequence[float],
    boundaries: Sequence[int],
) -> optax.Schedule:
  """Schedule that is piecewise constant and specified by rates (not scales).

  This is similar to optax.piecewise_constant_schedule, which requires users
  to specify "scales" (ratio of old LR to new LR).

  Args:
    rates: Length K sequence of learning rates. `rates[i]` is used for steps
      `0 <= step < boundaries[1]`, for i=0, and
      `boundaries[i-1] <= step < boundaries[i]`, for 0 < i < len(boundaries)
      `boundaries[i-1] <= step < ∞`, for i = len(boundaries)
    boundaries: Length K-1 sequence of boundaries.

  Returns:
    Schedule to pass to optax optimizers.
  """
  return optax.join_schedules(
      schedules=[optax.constant_schedule(r) for r in rates],
      boundaries=boundaries,
  )


@gin.register
def delayed_constant_schedule(
    turn_on_step: int,
    rate: float,
) -> optax.Schedule:
  """Schedule that is zero until `turn_on_step` then `rate` thereafter."""
  return piecewise_constant_schedule_specified_by_rates(
      rates=[0., rate],
      boundaries=[turn_on_step],
  )


@gin.register
def top_level_multi_adam(
    top_level_keys: Sequence[str] = (),
    learning_rates: Sequence[optax.ScalarOrSchedule] = (),
    default_learning_rate: optax.ScalarOrSchedule = 1e-4,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-6,
    raise_if_keys_not_found: bool = True,
) -> optax.GradientTransformation:
  """Uses an Adam optimizer with different learning rates for different params.

  Args:
    top_level_keys: Keys to use non-default learning rates for. A key starting
      with 'REGEX_', such as 'REGEX_cats' will use re.search to find keys, e.g.
      re.search('cats', key).
    learning_rates: Learning rates to use leafs under the `top_level_keys`.
    default_learning_rate: Learning rate to use for keys not in `learning_rates`
    b1: Exponential decay to track the first moment of past gradients.
    b2: Exponential decay to track the second moment of past gradients.
    eps:  A small constant applied to denominator outside of the square root to
      avoid dividing by zero when rescaling.
    raise_if_keys_not_found: Whether to raise if some `top_level_keys` are not
      found in params.

  Returns:
    optax optimizer with learning rate based on top level key in params dict.
  """
  if len(top_level_keys) != len(learning_rates):
    raise ValueError(
        f'{top_level_keys=} had different length than {learning_rates=}'
    )
  if '' in top_level_keys:
    raise ValueError('An empty string "" was found in `top_level_keys`.')

  default_label = 'DEFAULT_LABEL'
  if default_label in top_level_keys:
    raise ValueError(f'{default_label=} should not be in `top_level_keys`')

  def find_matching_top_level_key(param_name: str) -> str:
    """Searches for param_name in top_level_keys, returns the matching key."""
    prefix = 'REGEX_'
    matches = []
    for k in top_level_keys:
      if k.startswith(prefix) and re.search(k.lstrip(prefix), param_name):
        matches.append(k)
      elif k == param_name:
        matches.append(k)
    if not matches:
      return default_label
    elif len(matches) == 1:
      return matches[0]
    else:
      raise ValueError(
          f'{param_name=} had more than 1 ({len(matches)}) match '
          f'({matches}). Only one `top_level_keys` should match, or else we '
          'cannot choose a unique learning rate for these parameters.'
      )

  def get_prefix_labels(params):
    """Makes prefix labels to help optax match params with learning rates."""
    # E.g. if top_level_keys = ['module_A', 'REGEX_special'],
    # and params.keys() = ['module_A', 'special_A', 'special_B', 'module_C'],
    # labels = {
    #   'module_A': 'module_A',
    #   'special_A': 'REGEX_special', 'special_B': 'REGEX_special',
    #   'module_C': 'DEFAULT_LABEL', 'module_D': 'DEFAULT_LABEL',...
    # }
    # E.g. labels tells optax to use the learning rate 'REGEX_special' for
    # parameters under the prefix 'module_C'.
    labels = {
        param_name: find_matching_top_level_key(param_name)
        for param_name in params
    }
    top_level_keys_that_matched = [
        k for k in labels.values() if k != default_label
    ]
    missing_keys = set(top_level_keys).difference(top_level_keys_that_matched)
    if raise_if_keys_not_found and missing_keys:
      raise OptimizerError(
          f'{missing_keys=} not found in params: {sorted(params)}'
      )
    return labels

  def make_adam(lr):
    return optax.adam(lr, b1=b1, b2=b2, eps=eps)

  return optax.multi_transform(
      transforms={
          k: make_adam(lr) for k, lr in zip(top_level_keys, learning_rates)
      }
      | {default_label: make_adam(default_learning_rate)},
      param_labels=get_prefix_labels,
  )
