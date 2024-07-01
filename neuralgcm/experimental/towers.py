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
"""Modules that define stacks of neural-net layers acting on spatial arrays."""

import functools
from typing import Callable, Protocol, Sequence

from flax import nnx
import jax
from neuralgcm.experimental import typing


Array = typing.Array


class UnaryTower(Protocol):
  """Protocol for towers that transform array -> array preserving spatial shape.

  This is a protocol, so we can use any module that implements the same
  signature without complicating the inheretance hierarchy.
  """

  def __init__(self, input_size: int, output_size: int, **kwargs):
    ...

  def __call__(self, inputs: jax.Array) -> jax.Array:
    ...


UnaryTowerFactory = Callable[..., UnaryTower]


# TODO(dkochkov) consider passing spatial axes to parameterize # of vmap needed.


class ColumnTower(nnx.Module):
  """Tower that operates on latent/level axis with shared NN across all lat/lon.

  input shape: [nn_input_shape, lon, lat],
  output shape: [nn_output_shape, lon, lat].
  """

  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      column_net_factory: nnx.Module,
      apply_remat: bool = False,
      rngs: nnx.Rngs,
  ):
    self.column_network = column_net_factory(input_size, output_size, rngs=rngs)
    self.apply_remat = apply_remat

  def __call__(self, inputs: Array) -> Array:
    """Applies Column tower to inputs."""
    vmap = functools.partial(
        nnx.vmap, in_axes=-1, out_axes=-1, state_axes={}, split_rngs=False)
    mapped_column_net = vmap(vmap(self.column_network))
    if self.apply_remat:
      mapped_column_net = nnx.remat(mapped_column_net)
    return mapped_column_net(inputs)


class EpdTower(nnx.Module):
  """EPD tower module parameterized by encode/process/decode towers."""

  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      latent_size: int,
      num_process_blocks: int,
      encode_tower_factory: UnaryTowerFactory,
      process_tower_factory: UnaryTowerFactory,
      decode_tower_factory: UnaryTowerFactory,
      post_encode_activation: Callable[[Array], Array] | None = None,
      pre_decode_activation: Callable[[Array], Array] | None = None,
      final_activation: Callable[[Array], Array] | None = None,
      rngs: nnx.Rngs,
  ):
    self.input_size = input_size
    self.output_size = output_size
    self.latent_size = latent_size
    self.post_encode_activation = post_encode_activation
    self.pre_decode_activation = pre_decode_activation
    self.final_activation = final_activation
    self.encoder = encode_tower_factory(input_size, latent_size, rngs=rngs)
    self.decoder = decode_tower_factory(latent_size, output_size, rngs=rngs)
    self.process_towers = tuple([
        process_tower_factory(latent_size, latent_size, rngs=rngs)
        for _ in range(num_process_blocks)
    ])

  def __call__(self, inputs: Array) -> Array:
    """Applies EPD tower to inputs."""
    encoded = self.encoder(inputs)
    if self.post_encode_activation is not None:
      encoded = self.post_encode_activation(encoded)
    current = encoded
    for process_block in self.process_towers:
      current = current + process_block(current)
    if self.pre_decode_activation is not None:
      current = self.pre_decode_activation(current)
    out = self.decoder(current)
    if self.final_activation is not None:
      return self.final_activation(out)
    return out


class ChainedTower(nnx.Module):
  """Chains multiple towers together."""

  def __init__(
      self,
      input_size: int,
      output_size: int,
      *,
      intermediate_sizes: tuple[int | str, ...],
      tower_modules: Sequence[UnaryTowerFactory],
      rngs: nnx.Rngs,
  ):
    self.input_size = input_size
    self.output_size = output_size
    replace_values = {'input_size': input_size, 'output_size': output_size}
    replace_str = lambda s: replace_values[s]
    intermediate_sizes = tuple(
        replace_str(s) if isinstance(s, str) else s for s in intermediate_sizes
    )
    output_sizes = (*intermediate_sizes, output_size)
    input_sizes = (input_size, *intermediate_sizes)
    zip_vals = zip(input_sizes, output_sizes, tower_modules, strict=True)
    self.towers = tuple(
        tower_factory(d_in, d_out, rngs=rngs)
        for d_in, d_out, tower_factory in zip_vals
    )

  def __call__(self, inputs: Array) -> Array:
    current = inputs
    for tower in self.towers:
      current = tower(current)
    return current
