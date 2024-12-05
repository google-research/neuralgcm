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
import functools
import textwrap
import types
from typing import Any, Self

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

  def order_as(self, *dims: str | types.EllipsisType) -> Self:
    """Reorder the dimensions of an array.

    All dimensions must be named. Use `tag` first to name any positional axes.

    Args:
      *dims: dimension names that appear on this array, in the desired order on
        the result. `...` may be used once, to indicate all other dimensions in
        order of appearance on this array.

    Returns:
      Array with transposed data and reordered dimensions, as indicated.
    """
    if any(dim is None for dim in self.dims):
      raise ValueError(
          'cannot reorder the dimensions of an array with unnamed '
          f'dimensions: {self.dims}'
      )

    ellipsis_count = sum(dim is ... for dim in dims)
    if ellipsis_count > 1:
      raise ValueError(
          f'dimension names contain multiple ellipses (...): {dims}'
      )
    elif ellipsis_count == 1:
      explicit_dims = {dim for dim in dims if dim is not ...}
      implicit_dims = tuple(
          dim for dim in self.dims if dim not in explicit_dims
      )
      i = dims.index(...)
      dims = dims[:i] + implicit_dims + dims[i + 1 :]

    order = tuple(self.dims.index(dim) for dim in dims)
    return type(self)(self.data.transpose(order), dims)


def _collect_named_shape(
    leaves_and_paths: list[tuple[jax.tree_util.KeyPath, Any]],
    source_description: str,
) -> dict[str, int]:
  """Collect shared named_shape, or raise an informative error."""
  known_sizes = {}
  bad_dims = []
  for _, leaf in leaves_and_paths:
    if isinstance(leaf, NamedArray):
      for name, size in leaf.named_shape.items():
        if name in known_sizes:
          if known_sizes[name] != size and name not in bad_dims:
            bad_dims.append(name)
        else:
          known_sizes[name] = size

  if bad_dims:
    shapes_str = []
    for keypath, leaf in leaves_and_paths:
      if isinstance(leaf, NamedArray):
        if keypath[0] == jax.tree_util.SequenceKey(0):
          prefix = 'args'
        else:
          assert keypath[0] == jax.tree_util.SequenceKey(1)
          prefix = 'kwargs'
        path = jax.tree_util.keystr(keypath[1:])
        shapes_str.append(f'  {prefix}{path}.named_shape == {leaf.named_shape}')
    shapes_message = '\n'.join(shapes_str)

    raise ValueError(
        f'Inconsistent sizes in a call to {source_description} for dimensions '
        f'{bad_dims}:\n{shapes_message}'
    )

  return known_sizes


def nmap(fun):
  """Automatically vectorizes ``fun`` over named dimensions.

  ``nmap`` is a "named dimension vectorizing map". It wraps an ordinary
  positional-axis-based function so that it accepts NamedArrays as input and
  produces NamedArrays as output, and vectorizes over all of named dimensions,
  calling the original function with positionally-indexed slices corresponding
  to each argument's `positional_shape`.

  Unlike `jax.vmap`, the axes to vectorize over are inferred
  automatically from the named dimensions in the NamedArray inputs, rather
  than being specified as part of the mapping transformation. Specifically, each
  dimension name that appears in any of the arguments is vectorized over jointly
  across all arguments that include that dimension, and is then included as a
  named dimension in the output. To make an axis visible to ``fun``, you can
  call
  `untag` on the argument and pass the axis name(s) of interest; ``fun`` will
  then see those axes as positional axes instead of mapping over them.

  `untag` and ``nmap`` are together the primary ways to apply individual
  operations to axes of a NamedArray. `tag` can then be used on the result to
  re-bind names to positional axes.

  Within ``fun``, any mapped-over axes will be accessible using standard JAX
  collective operations like ``psum``, although doing this is usually
  unnecessary.

  Args:
    fun: Function to vectorize by name. This can take arbitrary arguments (even
      non-JAX-arraylike arguments or "static" axis sizes), but must produce a
      PyTree of JAX ArrayLike outputs.

  Returns:
    An automatically-vectorized version of ``fun``, which can optionally be
    called with NamedArrays instead of ordinary arrays, and which will always
    return NamedArrays for each of its output leaves. Any argument (or PyTree
    leaf of an argument) that is a NamedArray will have its named dimensions
    vectorized over; ``fun`` will then be called with batch tracers
    corresponding to slices of the input array that are shaped like
    ``named_array_arg.positional_shape``. Every named dimension that
    appeared in any input will also appear as the trailing dimensions of every
    output.
  """

  @functools.wraps(fun)
  def wrapped(*args, **kwargs):

    leaves_and_paths, treedef = jax.tree_util.tree_flatten_with_path(
        (args, kwargs),
        is_leaf=lambda node: isinstance(node, NamedArray),
    )
    leaves = [leaf for _, leaf in leaves_and_paths]

    named_shape = _collect_named_shape(
        leaves_and_paths, source_description=f'nmap({fun})'
    )
    all_dims = tuple(named_shape.keys())

    leaf_dims: list[list[str | None]] = [
        list(leaf.dims) if isinstance(leaf, NamedArray) else []
        for leaf in leaves
    ]
    all_in_axes = []
    for vmap_dim in all_dims:
      in_axes = []
      for dims in leaf_dims:
        if vmap_dim in dims:
          axis = dims.index(vmap_dim)
          del dims[axis]
        else:
          axis = None
        in_axes.append(axis)
      all_in_axes.append(in_axes)

    def vectorized_fun(leaf_data):
      args, kwargs = jax.tree.unflatten(treedef, leaf_data)
      return fun(*args, **kwargs)

    for out_axis in reversed(range(-len(all_dims), 0)):
      vectorized_fun = jax.vmap(
          vectorized_fun,
          in_axes=(all_in_axes[out_axis],),
          out_axes=out_axis,
          axis_name=all_dims[out_axis],
      )

    leaf_data = [
        leaf.data if isinstance(leaf, NamedArray) else leaf for leaf in leaves
    ]
    result = vectorized_fun(leaf_data)

    def wrap_output(data: jnp.ndarray) -> NamedArray:
      dims = (None,) * (data.ndim - len(all_dims)) + all_dims
      return NamedArray(data, dims)

    return jax.tree.map(wrap_output, result)

  return wrapped
