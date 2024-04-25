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
"""Helper functions for processing and parsing gin configurations."""

import contextlib
import logging
import threading
import gin


_GIN_LOCK = threading.RLock()


def _remove_unknown_reference(gin_config_str: str) -> str:
  """Removes unknown references form `gin_config_str`."""
  # this happens when we have gin MACROS reference not imported objects.
  return '\n'.join([
      line for line in gin_config_str.splitlines()
      if 'gin.config._UnknownConfigurable' not in line
  ])


def parse_gin_config(
    physics_config_str: str,
    model_config_str: str,
    override_physics_configs_from_data: bool,
    gin_bindings: list[str],
):
  """Parses physics_config_str, model_config_str and gin_bindings in order.

  We use skip unknown parameters in model_config_str to avoid errors associated
  with irrelevant training parameters that refer to configurables only imported
  for training.

  Args:
    physics_config_str: gin configuration string for physics_specifications
      object that stores relevant physics constants.
    model_config_str: gin configuration string of the model.
    override_physics_configs_from_data: whether to reparse `physics_config_str`
      after processing `model_config_str`.
    gin_bindings: additional gin configuration strings that will be parsed last.
  """
  gin.parse_config(physics_config_str)
  gin.parse_config(model_config_str, skip_unknown=True)
  if override_physics_configs_from_data:
    gin.parse_config(physics_config_str)
  gin.parse_config(gin_bindings)
  logging.info('Evaluating model with the following config:\n %s',
               gin.config_str())


@contextlib.contextmanager
def specific_config(
    gin_config: str,
    clear_current: bool = True,
    skip_unknown: bool = True,
):
  """Context manager for evaluation of functions with `gin_config`."""
  with _GIN_LOCK:
    # avoid splitting long lines into multiples that may contain unknown refs.
    current_config = gin.config_str(max_line_length=len(gin.config_str()))
    current_config = _remove_unknown_reference(current_config)
    if clear_current:
      gin.clear_config()
    try:
      gin.parse_config(gin_config, skip_unknown=skip_unknown)
      yield
    finally:
      gin.clear_config()
      gin.parse_config(current_config)
