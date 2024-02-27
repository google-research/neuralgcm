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
"""Defines configurable time integrators to be used by models."""

from typing import Callable
from dinosaur import time_integration
from dinosaur import typing
import gin


TimeIntegrator = Callable[
    [time_integration.ImplicitExplicitODE, typing.Numeric], typing.TimeStepFn]


backward_forward_euler = gin.external_configurable(
    time_integration.backward_forward_euler)
crank_nicolson_rk2 = gin.external_configurable(
    time_integration.crank_nicolson_rk2)
crank_nicolson_rk3 = gin.external_configurable(
    time_integration.crank_nicolson_rk3)
crank_nicolson_rk4 = gin.external_configurable(
    time_integration.crank_nicolson_rk4)
imex_rk_sil3 = gin.external_configurable(time_integration.imex_rk_sil3)
semi_implicit_leapfrog = gin.external_configurable(
    time_integration.semi_implicit_leapfrog)
