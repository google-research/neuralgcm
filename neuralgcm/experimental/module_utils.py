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

"""Utilities for manipulating and transforming modules."""

from __future__ import annotations

import dataclasses
import functools
from typing import Iterable, NamedTuple

from flax import nnx


class ModuleAndMethod(NamedTuple):
  module: nnx.Module
  method_name: str


def format_callbacks(callback_specs):
  """Formats callback_specs to standardized format."""
  if isinstance(callback_specs, ModuleAndMethod):
    return callback_specs
  if isinstance(callback_specs, nnx.Module):  # single callback.
    return ModuleAndMethod(callback_specs, '__call__')  # call default method.
  if isinstance(callback_specs, Iterable) and (len(callback_specs) == 2):
    return ModuleAndMethod(*callback_specs)
  raise TypeError(f'Unexpected {type(callback_specs)=}')


def with_callback(
    module,
    callback_specs: ModuleAndMethod,
    method_name: str = '__call__',
):
  """Returns module with `callback_specs.module` attached to `method_name`."""
  base_class = type(module)

  def __init__(self, wrapped_instance, callback_specs):  # pylint: disable=invalid-name
    self.wrapped_instance = wrapped_instance
    self.callback_specs = format_callbacks(callback_specs)

  def __getattr__(self, attr_name):  # pylint: disable=invalid-name
    """Delegate attribute access to the wrapped instance."""
    return getattr(self.wrapped_instance, attr_name)

  @functools.wraps(getattr(base_class, method_name))
  def wrapped_fn(self, *args, **kwargs):
    result = getattr(self.wrapped_instance, method_name)(*args, **kwargs)
    # The reason we use getattr here is because we need to access of method of
    # the callback module that is an attribute of this module. Otherwise nnx
    # would raise an error informing that we are trying to mutate an object that
    # is out of current scope. (This is exactly what would happen if we added
    # a reference to a callback_module.method as attribute of this class.)
    callback_fn = getattr(
        self.callback_specs.module, self.callback_specs.method_name
    )
    callback_fn(result)
    return result

  attrs = {
      '__init__': __init__,
      '__getattr__': __getattr__,
      method_name: wrapped_fn,
  }
  if dataclasses.is_dataclass(base_class):
    for field in dataclasses.fields(base_class):
      attrs[field.name] = property(
          lambda self, field=field: getattr(self.wrapped_instance, field.name)
      )
  cls = type(base_class.__name__ + 'WithCallbacks', (base_class,), attrs)
  return cls(module, callback_specs)


def retrieve_subclass_modules(module, subclass):
  """Returns list of all unique `subclass` instances on `module`."""
  subclass_modules = []
  for _, x in module.iter_modules():
    if isinstance(x, subclass):
      subclass_modules.append(x)
  return subclass_modules
