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
"""Defines ``Field`` and base ``Coordinate`` classes that define coordax API.

``Coordinate`` objects define a discretization schema, dimension names and
provide methods & coordinate field values to facilitate computations.
``Field`` objects keep track of positional and named dimensions of an array.
Named dimensions of a ``Field`` are associated with coordinates that describe
their discretization.

Current implementation of ``Field`` and associated helper methods like `cmap`
(coordinate-map) are based on `penzai.core.named_axes` library.
"""

from __future__ import annotations

import abc
import collections
import dataclasses
import functools
import operator
from typing import Any, Callable, Hashable, Mapping, Sequence, TypeAlias, TypeGuard

import jax
import jax.numpy as jnp
import numpy as np
from penzai.core import named_axes
from penzai.core import struct


AxisName: TypeAlias = Hashable
Pytree: TypeAlias = Any


class Coordinate(abc.ABC):
  """Abstract class for coordinate objects.

  Coordinate subclasses are expected to obey several invariants:
  1. Dimension names may not be repeated: `len(set(dims)) == len(dims)`
  2. All dimensions must be named: `len(shape) == len(dims)`
  """

  @property
  @abc.abstractmethod
  def dims(self) -> tuple[AxisName, ...]:
    """Dimension names of the coordinate."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def shape(self) -> tuple[int, ...]:
    """Shape of the coordinate."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def fields(self) -> dict[AxisName, Field]:
    """A maps from field names to their values."""

  @property
  def sizes(self) -> dict[AxisName, int]:
    """Sizes of all dimensions on this coordinate."""
    return dict(zip(self.dims, self.shape))

  @property
  def ndim(self) -> int:
    """Dimensionality of the coordinate."""
    return len(self.dims)


@dataclasses.dataclass
class ArrayKey:
  """Wrapper for a numpy array to make it hashable."""

  value: np.ndarray

  def __eq__(self, other):
    return (
        isinstance(self, ArrayKey)
        and self.value.dtype == other.value.dtype
        and self.value.shape == other.value.shape
        and (self.value == other.value).all()
    )

  def __ne__(self, other):
    return not self == other

  def __hash__(self) -> int:
    return hash((self.value.shape, self.value.tobytes()))


@struct.pytree_dataclass
class SelectedAxis(Coordinate, struct.Struct):
  """Coordinate that exposes one dimension of a multidimensional coordinate."""

  coordinate: Coordinate
  axis: int

  def __post_init__(self):
    if self.axis >= self.coordinate.ndim:
      raise ValueError(
          f'Dimension {self.axis=} of {self.coordinate=} is out of bounds'
      )

  @property
  def dims(self) -> tuple[AxisName, ...]:
    """Dimension names of the coordinate."""
    return (self.coordinate.dims[self.axis],)

  @property
  def shape(self) -> tuple[int, ...]:
    """Shape of the coordinate."""
    return (self.coordinate.shape[self.axis],)

  @property
  def fields(self) -> dict[AxisName, Field]:
    """A maps from field names to their values."""
    return self.coordinate.fields

  @property
  def ndim(self) -> int:
    """Dimensionality of the coordinate."""
    return 1

  def __repr__(self):
    return f'coordax.SelectedAxis({self.coordinate!r}, axis={self.axis})'


def consolidate_coordinates(*coordinates: Coordinate) -> tuple[Coordinate, ...]:
  """Consolidates coordinates without SelectedAxis objects, if possible."""
  axes = []
  result = []

  def reset_axes():
    result.extend(axes)
    axes[:] = []

  def append_axis(c):
    axes.append(c)
    if len(axes) == c.coordinate.ndim:
      # sucessful consolidation
      result.append(c.coordinate)
      axes[:] = []

  for c in coordinates:
    if isinstance(c, SelectedAxis) and c.axis == 0:
      # new SelectedAxis to consider consolidating
      reset_axes()
      append_axis(c)
    elif (
        isinstance(c, SelectedAxis)
        and axes
        and c.axis == len(axes)
        and c.coordinate == axes[-1].coordinate
    ):
      # continued SelectedAxis to consolidate
      append_axis(c)
    else:
      # coordinate cannot be consolidated
      reset_axes()
      result.append(c)

  reset_axes()

  return tuple(result)


@struct.pytree_dataclass
class CartesianProduct(Coordinate, struct.Struct):
  """Coordinate defined as the outer product of independent coordinates."""

  coordinates: tuple[Coordinate, ...] = dataclasses.field(
      metadata={'pytree_node': False}
  )

  def __post_init__(self):
    new_coordinates = []
    for c in self.coordinates:
      if isinstance(c, CartesianProduct):
        # c.coordinates are already Axiss where needed.
        new_coordinates.extend(c.coordinates)
      elif c.ndim > 1:
        for i in range(c.ndim):
          new_coordinates.append(SelectedAxis(c, i))
      else:
        new_coordinates.append(c)
    combined_coordinates = consolidate_coordinates(*new_coordinates)
    if len(combined_coordinates) <= 1:
      raise ValueError('CartesianProduct must contain more than 1 component')
    existing_dims = collections.Counter()
    for c in new_coordinates:
      existing_dims.update(c.dims)
    repeated_dims = [dim for dim, count in existing_dims.items() if count > 1]
    if repeated_dims:
      raise ValueError(f'CartesianProduct components contain {repeated_dims=}')
    object.__setattr__(self, 'coordinates', tuple(new_coordinates))

  def __eq__(self, other):
    if not isinstance(other, CartesianProduct):
      return len(self.coordinates) == 1 and self.coordinates[0] == other
    return isinstance(other, CartesianProduct) and all(
        self.coordinates[i] == other.coordinates[i]
        for i in range(len(self.coordinates))
    )

  @property
  def dims(self):
    return sum([c.dims for c in self.coordinates], start=tuple())

  @property
  def shape(self) -> tuple[int, ...]:
    """Returns the shape of the coordinate axes."""
    return sum([c.shape for c in self.coordinates], start=tuple())

  @property
  def fields(self) -> dict[AxisName, Field]:
    """Returns a mapping from field names to their values."""
    return functools.reduce(
        operator.or_, [c.fields for c in self.coordinates], {}
    )


@struct.pytree_dataclass
class NamedAxis(Coordinate, struct.Struct):
  """One dimensional coordinate that only dimension size."""

  name: AxisName = dataclasses.field(metadata={'pytree_node': False})
  size: int = dataclasses.field(metadata={'pytree_node': False})

  @property
  def dims(self) -> tuple[AxisName, ...]:
    return (self.name,)

  @property
  def shape(self) -> tuple[int, ...]:
    return (self.size,)

  @property
  def fields(self) -> dict[AxisName, Field]:
    return {}

  def __repr__(self):
    return f'coordax.NamedAxis({self.name!r}, size={self.size})'


# TODO(dkochkov): consider using @struct.pytree_dataclass here and storing
# tuple values instead of np.ndarray (which could be exposed as a property).
@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class LabeledAxis(Coordinate):  # pytype: disable=final-error
  """One dimensional coordinate with custom coordinate values."""

  name: AxisName = dataclasses.field(metadata={'pytree_node': False})
  ticks: np.ndarray = dataclasses.field(metadata={'pytree_node': False})

  @property
  def dims(self) -> tuple[AxisName, ...]:
    return (self.name,)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.ticks.shape

  @property
  def fields(self) -> dict[AxisName, Field]:
    return {self.name: wrap(self.ticks, self)}

  def _components(self):
    return (self.name, ArrayKey(self.ticks))

  def tree_flatten(self):
    """Flattens LabeledAxis."""
    aux_data = self._components()
    return (), aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, leaves):
    """Unflattens LabeledAxis."""
    del leaves  # unused
    name, array_key = aux_data
    return cls(name=name, ticks=array_key.value)

  def __eq__(self, other):
    return (
        isinstance(other, LabeledAxis)
        and self._components() == other._components()
    )

  def __ne__(self, other):
    return not self == other

  def __hash__(self) -> int:
    return hash(self._components())

  def __repr__(self):
    return f'coordax.LabeledAxis({self.name!r}, ticks={self.ticks!r})'


def compose_coordinates(*coordinates: Coordinate) -> Coordinate:
  """Composes `coords` into a single coordinate system by cartesian product."""
  if not coordinates:
    raise ValueError('No coordinates provided.')
  coordinate_axes = []
  for c in coordinates:
    if isinstance(c, CartesianProduct):
      coordinate_axes.extend(c.coordinates)
    else:
      coordinate_axes.append(c)
  coordinates = consolidate_coordinates(*coordinate_axes)
  if len(coordinates) == 1:
    return coordinates[0]
  return CartesianProduct(coordinates)


def _dimension_names(*names: AxisName | Coordinate) -> tuple[AxisName, ...]:
  """Returns a tuple of dimension names from a list of names or coordinates."""
  dims_or_name_tuple = lambda x: x.dims if isinstance(x, Coordinate) else (x,)
  return sum([dims_or_name_tuple(c) for c in names], start=tuple())


def _validate_matching_coords(
    *axis_order: AxisName | Coordinate,
    coords: dict[AxisName, Coordinate],
):
  """Validates `axis_order` entries match with coordinates in `coords`."""
  coordinates = []
  for axis in axis_order:
    if isinstance(axis, Coordinate):
      if isinstance(axis, CartesianProduct):
        for c in axis.coordinates:
          coordinates.append(c)
      elif axis.ndim > 1:
        for i in range(axis.ndim):
          coordinates.append(SelectedAxis(axis, i))
      else:
        coordinates.append(axis)
  for c in coordinates:
    for dim in c.dims:
      if dim not in coords:
        raise ValueError(f'{dim=} of {c} not found in {coords.keys()=}')
      if coords[dim] != c:
        raise ValueError(f'Coordinate {c} does not match {coords[dim]=}')


def is_field(value) -> TypeGuard[Field]:
  """Returns True if `value` is of type `Field`."""
  return isinstance(value, Field)


def is_positional_prefix_field(f: Field) -> bool:
  """Returns True if positional axes of `f` are in prefix order."""
  return isinstance(f.named_array, named_axes.NamedArray)


def cmap(fun: Callable[..., Any]) -> Callable[..., Any]:
  """Vectorizes `fun` over coordinate dimensions of ``Field`` inputs.

  Args:
    fun: Function to vectorize over coordinate dimensions.

  Returns:
    A vectorized version of `fun` that applies original `fun` to locally
    positional dimensions in inputs, while vectorizing over all coordinate
    dimensions. All dimensions over which `fun` is vectorized will be present in
    every output.
  """
  if hasattr(fun, '__name__'):
    fun_name = fun.__name__
  else:
    fun_name = repr(fun)
  if hasattr(fun, '__doc__'):
    fun_doc = fun.__doc__
  else:
    fun_doc = None
  return _cmap_with_doc(fun, fun_name, fun_doc)


def _cmap_with_doc(
    fun: Callable[..., Any], fun_name: str, fun_doc: str | None = None
) -> Callable[..., Any]:
  """Builds a coordinate-vectorized wrapped function with a docstring."""

  @functools.wraps(fun)
  def wrapped_fun(*args, **kwargs):
    leaves, treedef = jax.tree.flatten((args, kwargs), is_leaf=is_field)
    field_leaves = [leaf for leaf in leaves if is_field(leaf)]
    all_coords = {}
    for field in field_leaves:
      for dim_name, c in field.coords.items():
        if dim_name in all_coords and all_coords[dim_name] != c:
          other = all_coords[dim_name]
          raise ValueError(f'Coordinates {c=} != {other=} use same {dim_name=}')
        else:
          all_coords[dim_name] = c
    named_array_leaves = [x.named_array if is_field(x) else x for x in leaves]
    fun_on_named_arrays = named_axes.nmap(fun)
    na_args, na_kwargs = jax.tree.unflatten(treedef, named_array_leaves)
    result = fun_on_named_arrays(*na_args, **na_kwargs)

    def _wrap_field(leaf):
      if isinstance(leaf, named_axes.NamedArray):
        return Field(
            named_array=leaf,
            coords={k: all_coords[k] for k in leaf.named_axes.keys()},
        )
      else:
        assert isinstance(leaf, named_axes.NamedArrayView)
        return Field(
            named_array=leaf,
            coords={k: all_coords[k] for k in leaf.data_axis_for_name.keys()},
        )

    return jax.tree.map(_wrap_field, result, is_leaf=named_axes.is_namedarray)

  docstr = (
      f'Dimension-vectorized version of `{fun_name}`. Takes similar arguments'
      f' as `{fun_name}` but accepts and returns Fields in place of arrays.'
  )
  if fun_doc:
    docstr += f'\n\nOriginal documentation:\n\n{fun_doc}'
  wrapped_fun.__doc__ = docstr
  return wrapped_fun


def _swapped_binop(binop):
  """Swaps the order of operations for a binary operation."""

  def swapped(x, y):
    return binop(y, x)

  return swapped


def _wrap_scalar_conversion(scalar_conversion):
  """Wraps a scalar conversion operator on a Field."""

  def wrapped_scalar_conversion(self: Field):
    if self.named_shape or self.positional_shape:
      raise ValueError(
          f'Cannot convert a non-scalar Field with {scalar_conversion}'
      )
    return scalar_conversion(self.unwrap())

  return wrapped_scalar_conversion


def _wrap_array_method(name):
  """Wraps an array method on a Field."""

  def func(array, *args, **kwargs):
    return getattr(array, name)(*args, **kwargs)

  array_method = getattr(jax.Array, name)
  wrapped_func = cmap(func)
  functools.update_wrapper(
      wrapped_func,
      array_method,
      assigned=('__name__', '__qualname__', '__annotations__'),
      updated=(),
  )
  wrapped_func.__module__ = __name__
  wrapped_func.__doc__ = (
      'Name-vectorized version of array method'
      f' `{name} <numpy.ndarray.{name}>`. Takes similar arguments as'
      f' `{name} <numpy.ndarray.{name}>` but accepts and returns Fields'
      ' (or FieldViews) in place of regular arrays.'
  )
  return wrapped_func


def _wrap_array_or_scalar(inputs: float | np.ndarray | jax.Array) -> Field:
  """Helper function that wraps inputs in a fully positional Field."""
  if isinstance(inputs, float):
    data_array = jnp.array(inputs)
  elif isinstance(inputs, np.ndarray):
    data_array = inputs
  else:
    data_array = jnp.asarray(inputs)
  named_array = named_axes.NamedArray(
      named_axes=collections.OrderedDict(), data_array=data_array
  )
  wrapped = Field(
      named_array=named_array,
      coords={},
  )
  return wrapped


@struct.pytree_dataclass
class Field(struct.Struct):
  """An array with a combination of positional and named dimensions.

  Attributes:
    named_array: A named array that tracks positional and named dimensions.
    coords: A mapping from dimension names to their coordinate objects.
  """

  # TODO(shoyer): Consider storing an ndarray and dims separately, and creating
  # the Penzai NamedArray on the fly. This would be make Penzai more of an
  # implementation detail instead of part of the public API.
  # Potential downside: would need a separate wrapper for NamedArrayView.
  # e.g.,
  #   data: jax.Array
  #   dims: tuple[AxisName, ...]
  #   coord: dict[AxisName, Coordinate]
  # or
  #   data: jax.Array
  #   coords: tuple[Coordinate, ...]

  named_array: named_axes.NamedArray | named_axes.NamedArrayView
  coords: dict[AxisName, Coordinate] = dataclasses.field(
      metadata={'pytree_node': False}
  )

  @property
  def data(self) -> np.ndarray | jax.Array:
    """The value of the underlying data array."""
    return self.named_array.data_array

  @property
  def dtype(self) -> np.dtype:
    """The dtype of the field."""
    return self.named_array.dtype

  def check_valid(self) -> None:
    """Checks that the field coordinates and dimension names are consistent."""
    data_dims = set(self.named_array.named_shape.keys())
    keys_dims = set(self.coords.keys())
    coord_dims = set(sum([c.dims for c in self.coords.values()], start=()))
    if (data_dims != keys_dims) or (data_dims != coord_dims):
      raise ValueError(
          'Field dimension names must be the same across keys and coordinates'
          f' , got {data_dims=}, {keys_dims=} and {keys_dims=}.'
      )
    self.named_array.check_valid()

  @property
  def named_shape(self) -> Mapping[AxisName, int]:
    """A mapping of axis names to their sizes."""
    return self.named_array.named_shape

  @property
  def positional_shape(self) -> tuple[int, ...]:
    """A tuple of axis sizes for any anonymous axes."""
    return self.named_array.positional_shape

  @property
  def shape(self) -> tuple[int, ...]:
    """A tuple of axis sizes of the underlying data array."""
    return self.named_array.data_array.shape

  @property
  def dims(self) -> tuple[AxisName | int, ...]:
    """A tuple of indices and dimension names in the data layout order."""
    if isinstance(self.named_array, named_axes.NamedArray):
      postional_dims = tuple(range(len(self.positional_shape)))
      return postional_dims + tuple(self.named_array.named_axes.keys())
    elif isinstance(self.named_array, named_axes.NamedArrayView):
      data_axis_for_logical_axis = self.named_array.data_axis_for_logical_axis
      to_name = {v: k for k, v in self.named_array.data_axis_for_name.items()}
      to_logical = {v: k for k, v in enumerate(data_axis_for_logical_axis)}
      axis_to_dim = to_logical | to_name
      return tuple(
          axis_to_dim[i] for i in range(len(self.named_array.data_shape))
      )
    else:
      raise TypeError(
          f'{type(self.named_array)=} is not of expected NamedArray type.'
      )

  @property
  def coord_fields(self) -> dict[AxisName, Field]:
    """A mapping from coordinate field names to their values."""
    return functools.reduce(
        operator.or_, [c.fields for c in self.coords.values()], {}
    )

  def unwrap(self, *names: AxisName | Coordinate) -> jax.Array:
    """Extracts the underlying data from fully positional or named field."""
    return self.named_array.unwrap(*_dimension_names(*names))

  def with_positional_prefix(self) -> Field:
    """Returns a `Field` with positional axes moved to the front."""
    return Field(
        named_array=self.named_array.with_positional_prefix(),
        coords=self.coords,
    )

  def as_view(self) -> Field:
    """Returns a `Field` with a view representation of the underlying data."""
    return Field(
        named_array=self.named_array.as_namedarrayview(),
        coords=self.coords,
    )

  def untag(self, *axis_order: AxisName | Coordinate) -> Field:
    """Returns a view of the field with the requested axes made positional."""
    _validate_matching_coords(*axis_order, coords=self.coords)
    untag_dims = _dimension_names(*axis_order)
    result = Field(
        named_array=self.named_array.untag(*untag_dims),
        coords={k: v for k, v in self.coords.items() if k not in untag_dims},
    )
    result.check_valid()
    return result

  def tag(self, *names: AxisName | Coordinate) -> Field:
    """Returns a Field with attached coordinates to the positional axes."""
    tag_dims = _dimension_names(*names)
    tagged_array = self.named_array.tag(*tag_dims)
    coords = {}
    coords.update(self.coords)
    for c in names:
      if isinstance(c, Coordinate):
        if isinstance(c, CartesianProduct):
          for sub_c in c.coordinates:
            coords[sub_c.dims[0]] = sub_c
        else:
          for i, dim in enumerate(c.dims):
            coords[dim] = c if c.ndim == 1 else SelectedAxis(c, i)
      else:
        coords[c] = NamedAxis(c, size=tagged_array.named_shape[c])
    result = Field(named_array=tagged_array, coords=coords)
    result.check_valid()
    return result

  @property
  def ndim(self) -> int:
    return len(self.dims)

  def untag_prefix(self, *axis_order: AxisName | Coordinate) -> Field:
    """Returns a field with requested axes made front positional axes."""
    _validate_matching_coords(*axis_order, coords=self.coords)
    untag_dims = _dimension_names(*axis_order)
    result = Field(
        named_array=self.named_array.untag_prefix(*untag_dims),
        coords={k: v for k, v in self.coords.items() if k not in untag_dims},
    )
    result.check_valid()
    return result

  def tag_prefix(self, *axis_order: AxisName | Coordinate) -> Field:
    """Returns a field with coords attached to the first positional axes."""
    tag_dims = _dimension_names(*axis_order)
    tagged_array = self.named_array.tag_prefix(*tag_dims)
    coords = {}
    coords.update(self.coords)
    for c in axis_order:
      if isinstance(c, Coordinate):
        for dim in c.dims:
          coords[dim] = c
      else:
        coords[c] = NamedAxis(c, size=tagged_array.named_shape[c])
    result = Field(named_array=tagged_array, coords=coords)
    result.check_valid()
    return result

  def tag_suffix(self, *axis_order: AxisName | Coordinate) -> Field:
    """Returns a field with coords attached to the last positional axes."""
    n_tmp = len(self.positional_shape) - len(_dimension_names(*axis_order))
    tmp_dims = [named_axes.TmpPosAxisMarker() for _ in range(n_tmp)]
    return self.tag(*tmp_dims, *axis_order).untag_prefix(*tmp_dims)

  def untag_suffix(self, *axis_order: AxisName | Coordinate) -> Field:
    """Returns a field with requested axes made last positional axes."""
    n_tmp = len(self.positional_shape)
    tmp_dims = [named_axes.TmpPosAxisMarker() for _ in range(n_tmp)]
    return self.tag(*tmp_dims).untag(*tmp_dims, *axis_order)

  # Note: Can't call this "transpose" like Xarray, to avoid conflicting with the
  # positional only ndarray method.
  def order_as(self, *axis_order: AxisName | Coordinate) -> Field:
    """Returns a field with the axes in the given order."""
    _validate_matching_coords(*axis_order, coords=self.coords)
    ordered_dims = _dimension_names(*axis_order)
    ordered_array = self.named_array.order_as(*ordered_dims)
    result = Field(named_array=ordered_array, coords=self.coords)
    result.check_valid()
    return result

  def order_like(self, other: Field) -> Field:
    """Returns a field with the axes in the same order as `other`."""
    self.check_valid()
    other.check_valid()
    # To be able to order-like coordinates of two fields must match.
    _validate_matching_coords(*other.coords.values(), coords=self.coords)
    return Field(
        named_array=self.named_array.order_like(other.named_array),
        coords=self.coords,
    )

  def broadcast_to(
      self,
      positional_shape: Sequence[int] = (),
      named_shape: Mapping[AxisName, int] | tuple[Coordinate, ...] = (),
  ) -> Field:
    """Returns a field with positional and named shapes broadcasted."""
    if isinstance(named_shape, tuple):
      # TODO(dkochkov): Do we need to support broadcast_to with new coordinates?
      _validate_matching_coords(*named_shape, coords=self.coords)
      coordinates = [c for c in named_shape if isinstance(c, Coordinate)]
      dims = _dimension_names(*coordinates)
      shapes = sum([c.shape for c in coordinates], start=tuple())
      named_shape = {dim: size for dim, size in zip(dims, shapes)}
    return Field(
        named_array=self.named_array.broadcast_to(
            positional_shape, named_shape
        ),
        coords=self.coords,
    )

  def broadcast_like(self, other: Field | jax.typing.ArrayLike) -> Field:
    """Returns a field broadcasted to the shape of `other`."""
    if isinstance(other, Field):
      return self.broadcast_to(other.positional_shape, other.named_shape)
    else:
      shape = jnp.shape(other)
      return self.broadcast_to(shape, {})

  def __getitem__(self, indexer) -> Field:
    """Retrieves slices from an indexer, as in pz.core.named_axes indexing."""
    self.check_valid()
    # TODO(shoyer): consider disabling positional indexing, because it's
    # ambiguous whether it refers to positional-only or all axes.
    return Field(
        named_array=self.named_array.at[indexer].get(), coords=self.coords
    )

  # Iteration. Note that we *must* implement this to avoid Python simply trying
  # to run __getitem__ until it raises IndexError, because we won't raise
  # IndexError (since JAX clips array indices).
  def __iter__(self):
    if not self.positional_shape:
      raise ValueError('Cannot iterate over an array with no positional axes.')
    for i in range(self.positional_shape[0]):
      yield self[i]

  @classmethod
  def wrap(
      cls, array: jax.typing.ArrayLike | float, *names: AxisName | Coordinate
  ) -> Field:
    """Wraps a positional array as a ``Field``."""
    wrapped = _wrap_array_or_scalar(array)
    if names:
      return wrapped.tag(*names)
    else:
      return wrapped

  # TODO(shoyer): after updating the data model, the __repr__ should look like
  # either:
  #
  # Field(
  #     data=array(...),
  #     coords=(
  #         NamedAxis("time"),
  #         LatLonGrid(...),
  #     ),
  # )
  #
  # or:
  #
  # Field(
  #     data=array(...),
  #     dims=('time', 'lon', 'lat'),
  #     coords={
  #         'time': NamedAxis("time"),
  #         'lon': SelectedAxis(LonLatGrid(...), axis=0),
  #         'lat': SelectedAxis(LonLatGrid(...), axis=1),
  #     }
  # )

  # TODO(shoyer): restore a custom repr, once it's clear that we handle axis
  # order consistently with penzai.
  # def __repr__(self):
  #   indent = '    '
  #   data_repr = textwrap.indent(repr(self.data), prefix=indent)
  #   coordinates = consolidate_coordinates(*self.coords.values())
  #   coords_repr = indent + f',\n{indent}'.join(
  #       repr(c).removeprefix('coordax.') for c in coordinates
  #   )
  #   return f'coordax.Field.wrap(\n{data_repr},\n{coords_repr},\n)'

  # Convenience wrappers: Elementwise infix operators.
  __lt__ = _cmap_with_doc(operator.lt, 'jax.Array.__lt__')
  __le__ = _cmap_with_doc(operator.le, 'jax.Array.__le__')
  __eq__ = _cmap_with_doc(operator.eq, 'jax.Array.__eq__')
  __ne__ = _cmap_with_doc(operator.ne, 'jax.Array.__ne__')
  __ge__ = _cmap_with_doc(operator.ge, 'jax.Array.__ge__')
  __gt__ = _cmap_with_doc(operator.gt, 'jax.Array.__gt__')

  __add__ = _cmap_with_doc(operator.add, 'jax.Array.__add__')
  __sub__ = _cmap_with_doc(operator.sub, 'jax.Array.__sub__')
  __mul__ = _cmap_with_doc(operator.mul, 'jax.Array.__mul__')
  __truediv__ = _cmap_with_doc(operator.truediv, 'jax.Array.__truediv__')
  __floordiv__ = _cmap_with_doc(operator.floordiv, 'jax.Array.__floordiv__')
  __mod__ = _cmap_with_doc(operator.mod, 'jax.Array.__mod__')
  __divmod__ = _cmap_with_doc(divmod, 'jax.Array.__divmod__')
  __pow__ = _cmap_with_doc(operator.pow, 'jax.Array.__pow__')
  __lshift__ = _cmap_with_doc(operator.lshift, 'jax.Array.__lshift__')
  __rshift__ = _cmap_with_doc(operator.rshift, 'jax.Array.__rshift__')
  __and__ = _cmap_with_doc(operator.and_, 'jax.Array.__and__')
  __or__ = _cmap_with_doc(operator.or_, 'jax.Array.__or__')
  __xor__ = _cmap_with_doc(operator.xor, 'jax.Array.__xor__')

  __radd__ = _cmap_with_doc(_swapped_binop(operator.add), 'jax.Array.__radd__')
  __rsub__ = _cmap_with_doc(_swapped_binop(operator.sub), 'jax.Array.__rsub__')
  __rmul__ = _cmap_with_doc(_swapped_binop(operator.mul), 'jax.Array.__rmul__')
  __rtruediv__ = _cmap_with_doc(
      _swapped_binop(operator.truediv), 'jax.Array.__rtruediv__'
  )
  __rfloordiv__ = _cmap_with_doc(
      _swapped_binop(operator.floordiv), 'jax.Array.__rfloordiv__'
  )
  __rmod__ = _cmap_with_doc(_swapped_binop(operator.mod), 'jax.Array.__rmod__')
  __rdivmod__ = _cmap_with_doc(_swapped_binop(divmod), 'jax.Array.__rdivmod__')
  __rpow__ = _cmap_with_doc(_swapped_binop(operator.pow), 'jax.Array.__rpow__')
  __rlshift__ = _cmap_with_doc(
      _swapped_binop(operator.lshift), 'jax.Array.__rlshift__'
  )
  __rrshift__ = _cmap_with_doc(
      _swapped_binop(operator.rshift), 'jax.Array.__rrshift__'
  )
  __rand__ = _cmap_with_doc(_swapped_binop(operator.and_), 'jax.Array.__rand__')
  __ror__ = _cmap_with_doc(_swapped_binop(operator.or_), 'jax.Array.__ror__')
  __rxor__ = _cmap_with_doc(_swapped_binop(operator.xor), 'jax.Array.__rxor__')

  __abs__ = _cmap_with_doc(operator.abs, 'jax.Array.__abs__')
  __neg__ = _cmap_with_doc(operator.neg, 'jax.Array.__neg__')
  __pos__ = _cmap_with_doc(operator.pos, 'jax.Array.__pos__')
  __invert__ = _cmap_with_doc(operator.inv, 'jax.Array.__invert__')

  # Convenience wrappers: Scalar conversions.
  __bool__ = _wrap_scalar_conversion(bool)
  __complex__ = _wrap_scalar_conversion(complex)
  __int__ = _wrap_scalar_conversion(int)
  __float__ = _wrap_scalar_conversion(float)
  __index__ = _wrap_scalar_conversion(operator.index)

  # elementwise operations
  astype = _wrap_array_method('astype')
  clip = _wrap_array_method('clip')
  conj = _wrap_array_method('conj')
  conjugate = _wrap_array_method('conjugate')
  imag = _wrap_array_method('imag')
  real = _wrap_array_method('real')
  round = _wrap_array_method('round')
  view = _wrap_array_method('view')

  # Intentionally not included: anything that acts on a subset of axes or takes
  # an axis as an argument (e.g., mean). It is ambiguous whether these should
  # act over positional or named axes.

  # maybe include some of below with names that signify positional nature?
  # reshape = _wrap_array_method('reshape')
  # squeeze = _wrap_array_method('squeeze')
  # transpose = _wrap_array_method('transpose')
  # T = _wrap_array_method('T')
  # mT = _wrap_array_method('mT')  # pylint: disable=invalid-name


wrap = Field.wrap


def wrap_like(array: jax.Array, other: Field) -> Field:
  """Wraps `array` with the same coordinates as `other`."""
  other.check_valid()
  if array.shape != other.shape:
    raise ValueError(f'{array.shape=} and {other.shape=} must be equal')
  return Field(
      named_array=dataclasses.replace(other.named_array, data_array=array),
      coords=other.coords,
  )
