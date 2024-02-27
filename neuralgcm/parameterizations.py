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
"""Physics parameterization modules that compute non-dynamical tendencies."""

from typing import Any, Callable, Optional
from dinosaur import coordinate_systems
from dinosaur import pytree_utils
from dinosaur import typing
import gin
import haiku as hk
import jax
from neuralgcm import features
from neuralgcm import mappings
from neuralgcm import transforms

FeaturesModule = features.FeaturesModule
Forcing = typing.Forcing
MappingModule = mappings.MappingModule
StepFilterModule = Callable[..., typing.PyTreeStepFilterFn]
TransformModule = typing.TransformModule


@gin.register
class DirectNeuralParameterization(hk.Module):
  """Computes modal physics tendencies from the input state and forcing."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      modal_to_nodal_features_module: FeaturesModule,
      nodal_mapping_module: mappings.MappingModule,
      tendency_transform_module: TransformModule,
      prediction_mask: Optional[typing.Pytree] = None,
      filter_module: Optional[StepFilterModule] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.prediction_mask = prediction_mask
    self.modal_to_nodal_features_fn = modal_to_nodal_features_module(
        coords, dt, physics_specs, aux_features)
    self.nodal_mapping_module = nodal_mapping_module
    self.tendency_transform_fn = tendency_transform_module(
        coords, dt, physics_specs, aux_features)
    if filter_module is not None:
      self.filter_fn = filter_module(
          coords, dt, physics_specs, aux_features)
    else:
      self.filter_fn = lambda _, y: y  # no filtering.

  def __call__(
      self,
      inputs: typing.PyTreeState,
      memory: Optional[typing.Pytree] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.Pytree] = None,
      forcing: Optional[Forcing] = None,
  ) -> typing.PyTreeState:
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    if memory is not None:
      memory, _ = pytree_utils.as_dict(memory)
    prediction_mask = self.prediction_mask
    if prediction_mask is None:
      prediction_mask = pytree_utils.tree_map_over_nonscalars(
          lambda _: True, inputs, scalar_fn=lambda _: False
      )
    prediction_shapes = jax.tree_util.tree_map(
        lambda x, y: x if y else None,
        coordinate_systems.get_nodal_shapes(inputs, self.coords),
        prediction_mask,
    )
    net = self.nodal_mapping_module(prediction_shapes)
    nodal_inputs = self.modal_to_nodal_features_fn(
        inputs, memory=memory, diagnostics=diagnostics, randomness=randomness,
        forcing=forcing,
    )
    nodal_tendencies = net(nodal_inputs)
    nodal_tendencies = self.tendency_transform_fn(nodal_tendencies)
    modal_tendencies = self.coords.horizontal.to_modal(nodal_tendencies)
    modal_tendencies = self.filter_fn(inputs, modal_tendencies)
    return from_dict_fn(modal_tendencies)


@gin.register
class DivCurlNeuralParameterization(hk.Module):
  """Computes modal physics tendencies via `u, v` → `δ, ζ`."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: typing.AuxFeatures,
      modal_to_nodal_features_module: FeaturesModule,
      nodal_mapping_module: mappings.MappingModule,
      tendency_transform_module: TransformModule,
      prediction_mask: Optional[typing.Pytree] = None,
      filter_module: Optional[StepFilterModule] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.coords = coords
    self.prediction_mask = prediction_mask
    self.modal_to_nodal_features_fn = modal_to_nodal_features_module(
        coords, dt, physics_specs, aux_features)
    self.nodal_mapping_module = nodal_mapping_module
    self.tendency_transform_fn = tendency_transform_module(
        coords, dt, physics_specs, aux_features)
    self.get_nodal_shape_fn = (
        lambda x: coordinate_systems.get_nodal_shapes(x, coords))
    self.to_div_curl_fn = transforms.ToModalWithDivCurlTransform(
        coords, dt, physics_specs, aux_features)
    if filter_module is not None:
      self.filter_fn = filter_module(
          coords, dt, physics_specs, aux_features)
    else:
      self.filter_fn = lambda _, y: y  # no filtering.

  def __call__(
      self,
      inputs: typing.PyTreeState,
      memory: Optional[typing.Pytree] = None,
      diagnostics: Optional[typing.Pytree] = None,
      randomness: Optional[typing.Pytree] = None,
      forcing: Optional[Forcing] = None,
  ) -> typing.PyTreeState:
    inputs = self.coords.with_dycore_sharding(inputs)
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)
    if memory is not None:
      memory = self.coords.with_dycore_sharding(memory)
      memory, _ = pytree_utils.as_dict(memory)
    prediction_mask = self.prediction_mask
    if prediction_mask is None:
      prediction_mask = pytree_utils.tree_map_over_nonscalars(
          lambda _: True, inputs, scalar_fn=lambda _: False
      )
    prediction_shapes = jax.tree_util.tree_map(
        lambda x, y: self.get_nodal_shape_fn(x) if y else None,
        inputs,
        prediction_mask,
    )
    prediction_shapes['u'] = prediction_shapes.pop('divergence')
    prediction_shapes['v'] = prediction_shapes.pop('vorticity')
    net = self.nodal_mapping_module(prediction_shapes)
    nodal_inputs = self.modal_to_nodal_features_fn(
        inputs, memory=memory, diagnostics=diagnostics, randomness=randomness,
        forcing=forcing,
    )
    nodal_inputs = self.coords.dycore_to_physics_sharding(nodal_inputs)
    nodal_tendencies = net(nodal_inputs)
    nodal_tendencies = self.coords.physics_to_dycore_sharding(nodal_tendencies)
    nodal_tendencies = self.tendency_transform_fn(nodal_tendencies)
    modal_tendencies = self.to_div_curl_fn(nodal_tendencies)
    modal_tendencies = self.filter_fn(inputs, modal_tendencies)
    outputs = from_dict_fn(modal_tendencies)
    outputs = self.coords.with_dycore_sharding(outputs)
    return outputs
