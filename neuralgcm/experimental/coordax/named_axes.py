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
"""Array with optional named axes, inspired by Penzai's NamedArray.

This module is intended to be nearly a drop-in replacement for Penzai's
NamedArray, but with an alternative, simpler implementation. Dimensions are
specified by a tuple, where each element is either a string or `None`. `None` is
used to indicate strictly positional dimensions.

Some code and documentation is adapted from penzai.core.named_axes.
"""
import textwrap
from typing import Self

import jax
import jax.numpy as jnp


_VALID_PYTREE_OPS = (
    'JAX pytree operations on NamedArray objects are only valid when they'
    ' insert new leading dimensions, or trim unnamed leading dimensions. The'
    ' sizes and positions (from the end) of all named dimensions must be'
    ' preserved.'
)


@jax.tree_util.register_pytree_node_class
class NamedArray:
  """Array with optionally named axes.

  Axis names are either a string or None, indicating an unnamed axis.

  Attributes:
    data: the underlying data array.
    dims: tuple of dimension names or None, with the same length as data.ndim.
      All dimension names must be unique.
  """

  _data: jnp.ndarray
  _dims: tuple[str | None, ...]

  def __init__(self, data: jax.typing.ArrayLike, dims: tuple[str | None, ...]):
    data = jnp.asarray(data)
    if data.ndim != len(dims):
      raise ValueError(f'{data.ndim=} != {len(dims)=}')
    named_dims = [dim for dim in dims if dim is not None]
    if len(set(named_dims)) < len(named_dims):
      raise ValueError('dimension names may not be repeated: {dims}')
    self._data = data
    self._dims = dims

  @property
  def data(self) -> jnp.ndarray:
    """Data associated with this array."""
    return self._data

  @property
  def dims(self) -> tuple[str | None, ...]:
    """Dimension names of this array."""
    return self._dims

  @property
  def ndim(self) -> int:
    """Number of dimensions in the array, including postional and named axes."""
    return self.data.ndim

  @property
  def shape(self) -> tuple[int, ...]:
    """Shape of the array, including positional and named axes."""
    return self.data.shape

  @property
  def positional_shape(self) -> tuple[int, ...]:
    """Shape of the array with all named axes removed."""
    return tuple(
        size for dim, size in zip(self.dims, self.data.shape) if dim is None
    )

  @property
  def named_shape(self) -> dict[str, int]:
    """Mapping from dimension names to sizes."""
    return {
        dim: size
        for dim, size in zip(self.dims, self.data.shape)
        if dim is not None
    }

  def __repr__(self) -> str:
    indent = lambda x: textwrap.indent(x, prefix=' ' * 13)[13:]
    return textwrap.dedent(f"""\
    {type(self).__name__}(
        data={indent(repr(self.data))},
        dims={self.dims},
    )""")

  def tree_flatten(self):
    """Flatten this object for JAX pytree operations."""
    if isinstance(self.data, jnp.ndarray):
      size_info = tuple(self.named_shape.items())
    else:
      # Arrays unflattened from non-ndarray leaves are flattened inside vmap.
      # The resulting treedef is ignored.
      size_info = None
    return [self.data], (self.dims, size_info)

  @classmethod
  def _new_unvalidated(
      cls, data: jnp.ndarray, dims: tuple[str | None, ...]
  ) -> Self:
    """Create a new NamedArray, without validating data and dims."""
    # pylint: disable=protected-access
    obj = super().__new__(cls)
    obj._data = data
    obj._dims = dims
    return obj

  @classmethod
  def _new_with_padded_or_trimmed_dims(
      cls, data: jnp.ndarray, dims: tuple[str | None, ...]
  ) -> Self:
    """Create a new NamedArray, padding or trimming dims to match data.ndim."""
    assert isinstance(data, jnp.ndarray)
    if len(dims) <= data.ndim:
      dims = (None,) * (data.ndim - len(dims)) + dims
    else:
      trimmed_dims = dims[: -data.ndim]
      if any(dim is not None for dim in trimmed_dims):
        raise ValueError(
            'cannot trim named dimensions when unflattening to a NamedArray:'
            f' {trimmed_dims}. {_VALID_PYTREE_OPS} If you are using vmap or'
            ' scan, the first dimension must be unnamed.'
        )
      dims = dims[-data.ndim :]
    return cls(data, dims)

  @classmethod
  def tree_unflatten(cls, treedef, leaves: list[jax.Array | object]) -> Self:
    """Unflatten this object for JAX pytree operations."""
    dims, size_info = treedef
    [data] = leaves

    if not isinstance(data, jnp.ndarray):
      # JAX builds pytrees with non-ndarray leaves inside some transformations,
      # such as vmap, for handling the in_axes argument.
      return cls._new_unvalidated(data, dims)

    # Restored NamedArray objects may have additional or removed leading
    # dimensions, if produced with scan or vmap.
    result = cls._new_with_padded_or_trimmed_dims(data, dims)
    assert size_info is not None
    named_shape = dict(size_info)
    if result.named_shape != named_shape:
      raise ValueError(
          'named shape mismatch when unflattening to a NamedArray: '
          f'{result.named_shape} != {named_shape}. {_VALID_PYTREE_OPS}'
      )
    return result

  def tag(self, *dims: str) -> Self:
    """Attaches dimension names to the positional axes of an array.

    Args:
      *dims: axis names to assign to each positional axis in the array. Must
        have exactly the same length as the number of unnamed axes in the array.

    Raises:
      ValueError: If the wrong number of dimensions are provided.

    Returns:
      A NamedArray with the given names assigned to the positional axes, and no
      remaining positional axes.
    """
    if len(dims) != len(self.positional_shape):
      pos_ndim = len(self.positional_shape)
      raise ValueError(
          'there must be exactly as many dimensions given to `tag` as there'
          f' are positional axes in the array, but got {dims} for '
          f'{pos_ndim} positional {"axis" if pos_ndim == 1 else "axes"}.'
      )

    if any(not isinstance(name, str) for name in dims):
      raise TypeError(f'dimension names must be strings: {dims}')

    dim_queue = list(reversed(dims))
    new_dims = tuple(
        dim_queue.pop() if dim is None else dim for dim in self.dims
    )
    assert not dim_queue
    return type(self)(self.data, new_dims)

  def untag(self, *dims: str) -> Self:
    """Removes the requested dimension names.

    `untag` can only be called on a `NamedArray` that does not have any
    positional axes. It produces a new `NamedArray` where the axes with the
    requested dimension names are now treated as positional instead.

    Args:
      *dims: axis names to make positional, in the order they should appear in
        the positional array.

    Raises:
      ValueError: if the provided axis ordering is not valid.

    Returns:
      A named array with the given dimensions converted to positional axes.
    """
    if self.positional_shape:
      raise ValueError(
          '`untag` cannot be used to introduce positional axes for a NamedArray'
          ' that already has positional axes. Please assign names to the'
          ' existing positional axes first using `tag`.'
      )

    named_shape = self.named_shape
    if any(dim not in named_shape for dim in dims):
      raise ValueError(
          f'cannot untag {dims} because they are not a subset of the current '
          f'named dimensions {tuple(self.dims)}'
      )

    ordered = tuple(sorted(dims, key=self.dims.index))
    if ordered != dims:
      raise ValueError(
          f'cannot untag {dims} because they do not appear in the order of '
          f'the current named dimensions {ordered}'
      )

    untagged = set(dims)
    new_dims = tuple(None if dim in untagged else dim for dim in self.dims)
    return type(self)(self.data, new_dims)
