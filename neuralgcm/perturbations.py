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
"""Implementation of perturbation modules."""
import abc
from typing import Any, Callable
from dinosaur import coordinate_systems
from dinosaur import pytree_utils
from dinosaur import spherical_harmonic
from dinosaur import typing
import gin
import jax
import jax.numpy as jnp

from neuralgcm import transforms

Pytree = typing.Pytree
PerturbationFn = Callable[..., Pytree]
PerturbationModule = Callable[..., PerturbationFn]


_ALLOWED_PERTURBATION_BASIS = (
    # Converts vorticity/divergence to u/v then perturbs.
    'uv',

    # Perturbs in whatever the state is in (typically vorticity/divergence).
    'generic',
)

# We ♥ λ's
# pylint: disable=g-long-lambda


@gin.register
class NoPerturbation:
  """No-op perturbation that introduces no perturbation to `inputs`."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Any,
  ):
    """Initializes a random field."""
    del coords, dt, physics_specs, aux_features  # unused.

  def __call__(
      self,
      inputs: typing.Pytree,
      state: typing.Pytree,
      randomness: typing.Pytree,
  ) -> typing.Pytree:
    """Updates the state of a random field."""
    del state, randomness  # unused.
    return inputs


class BasePerturbation(abc.ABC):
  """Base class for perturbations."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Any,
      randomness_transform_module: transforms.TransformModule = (
          transforms.IdentityTransform
      ),
      return_modal: bool = True,
      perturbation_basis: str = 'generic',
  ):
    """Initializes module to perturb random fields.

    Args:
      coords: Model coordinate system.
      dt: Time step.
      physics_specs:
      aux_features:
      randomness_transform_module: Module that transforms jax.Array of random
        variables before converting to nodal.
      return_modal: Whether results should be returned in modal space.
      perturbation_basis: Whether to perturb wind in "uv" or "generic" basis.
    """
    self.coords = coords
    self.randomness_transform_fn = randomness_transform_module(
        coords, dt, physics_specs, aux_features
    )
    self.return_modal = return_modal
    self.to_modal = coords.horizontal.to_modal
    self.to_nodal = coords.horizontal.to_nodal
    self.maybe_to_modal = lambda tr: coordinate_systems.maybe_to_modal(
        tr, coords
    )
    self.maybe_to_nodal = lambda tr: coordinate_systems.maybe_to_nodal(
        tr, coords
    )
    if perturbation_basis not in _ALLOWED_PERTURBATION_BASIS:
      raise ValueError(
          f'{perturbation_basis=} which was not in '
          f'{_ALLOWED_PERTURBATION_BASIS=}'
      )
    self.perturbation_basis = perturbation_basis

  def __call__(
      self,
      inputs: typing.Pytree,
      state: typing.Pytree,
      randomness: typing.Pytree,
  ) -> typing.Pytree:
    """Updates the state of a random field."""
    del state  # unused.
    # TODO(dkochkov) allow pytree randomness in addition to broadcasting option.

    if self.perturbation_basis == 'generic':
      return self._perturb_in_generic_coordinates(inputs, randomness)
    elif self.perturbation_basis == 'uv':
      return self._perturb_in_uv_coordinates(inputs, randomness)

  @abc.abstractmethod
  def _perturb_core(
      self,
      inputs: typing.Pytree,
      randomness: typing.Pytree,
  ) -> typing.Pytree:
    """Perturbs inputs using randomness."""

  def _perturb_in_generic_coordinates(
      self,
      inputs: typing.Pytree,
      randomness: typing.Pytree,
  ) -> typing.Pytree:
    """Perturb `inputs` in (vorticity, divergence) coordinate system."""
    nodal_inputs = self.maybe_to_nodal(inputs)
    nodal_randomness = self.maybe_to_nodal(randomness)

    nodal_randomness = self.randomness_transform_fn(
        pytree_utils.tree_map_over_nonscalars(
            # Broadcast randomness so that self.randomness_transform_fn can use
            # the shape of x to determine what to do.
            lambda x: jnp.broadcast_to(nodal_randomness, x.shape),
            nodal_inputs,
            scalar_fn=jnp.zeros_like,
        )
    )

    perturbed_nodal_inputs = self._perturb_core(nodal_inputs, nodal_randomness)
    if self.return_modal:
      return self.to_modal(perturbed_nodal_inputs)
    else:
      return perturbed_nodal_inputs

  def _perturb_in_uv_coordinates(
      self,
      inputs: typing.Pytree,
      randomness: typing.Pytree,
  ) -> typing.Pytree:
    """Perturb `inputs` in (u, v) coordinate system."""
    inputs, from_dict_fn = pytree_utils.as_dict(inputs)

    # Remove vorticity/divergence from inputs and replace with u/v.
    vordiv = self.maybe_to_modal({
        'vorticity': inputs.pop('vorticity'),
        'divergence': inputs.pop('divergence'),
    })
    u_nodal, v_nodal = spherical_harmonic.vor_div_to_uv_nodal(
        grid=self.coords.horizontal,
        vorticity=vordiv['vorticity'],
        divergence=vordiv['divergence'],
        clip=True,
    )
    nodal_inputs = self.maybe_to_nodal(inputs)  # Recall we popped vor/div.
    nodal_inputs['u'] = u_nodal
    nodal_inputs['v'] = v_nodal

    # Perturb in u/v space
    nodal_randomness = self.maybe_to_nodal(randomness)
    nodal_randomness = self.randomness_transform_fn(
        pytree_utils.tree_map_over_nonscalars(
            # Broadcast randomness so that self.randomness_transform_fn can use
            # the shape of x to determine what to do.
            lambda x: jnp.broadcast_to(nodal_randomness, x.shape),
            nodal_inputs,
            scalar_fn=jnp.zeros_like,
        )
    )
    perturbed_nodal_inputs = self._perturb_core(nodal_inputs, nodal_randomness)

    # Transform perturbed u/v to vor/div (modal).
    vorticity, divergence = spherical_harmonic.uv_nodal_to_vor_div_modal(
        grid=self.coords.horizontal,
        u_nodal=perturbed_nodal_inputs.pop('u'),
        v_nodal=perturbed_nodal_inputs.pop('v'),
        clip=True,
    )

    # Insert vorticity/divergence into perturbed_inputs in the right space.
    if self.return_modal:
      perturbed_inputs = self.to_modal(perturbed_nodal_inputs)
      perturbed_inputs['vorticity'] = vorticity
      perturbed_inputs['divergence'] = divergence
    else:
      perturbed_inputs = perturbed_nodal_inputs.copy()
      perturbed_inputs['vorticity'] = self.to_nodal(vorticity)
      perturbed_inputs['divergence'] = self.to_nodal(divergence)

    return from_dict_fn(perturbed_inputs)


@gin.register
class ProportionalPerturbation(BasePerturbation):
  """Perturbation that scales inputs by 1 + randomness."""

  def _perturb_core(
      self,
      inputs: typing.Pytree,
      randomness: typing.Pytree,
  ) -> typing.Pytree:
    """Multiplies inputs by (1 + randomness)."""
    return jax.tree_util.tree_map(lambda x, y: x * (1 + y), inputs, randomness)
