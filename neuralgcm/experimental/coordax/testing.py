"""Testing utilities for coordax."""

from typing import Mapping

import chex
import jax
from neuralgcm.experimental.coordax import core
import numpy as np


AxisName = core.AxisName


def assert_field_properties(
    actual: core.Field,
    data: np.ndarray | jax.Array | None = None,
    dims: tuple[AxisName, ...] | None = None,
    shape: tuple[int, ...] | None = None,
    coord_field_keys: set[AxisName] | None = None,
    named_shape: Mapping[AxisName, int] | None = None,
    positional_shape: tuple[int, ...] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
  """Asserts that a Field has expected properties."""
  if data is not None:
    np.testing.assert_allclose(actual.data, data, rtol=rtol, atol=atol)
  if dims is not None:
    chex.assert_equal(actual.dims, dims)
  if shape is not None:
    chex.assert_shape(actual, shape)
  if coord_field_keys is not None:
    chex.assert_trees_all_equal(
        set(actual.coord_fields.keys()), coord_field_keys
    )
  if named_shape is not None:
    chex.assert_equal(actual.named_shape, named_shape)
  if positional_shape is not None:
    chex.assert_equal(actual.positional_shape, positional_shape)


def assert_fields_allclose(
    actual: core.Field,
    desired: core.Field,
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
  """Asserts that two Fields are close and have matching coordinates."""
  assert_field_properties(
      actual=actual,
      data=desired.data,
      dims=desired.dims,
      shape=desired.shape,
      coord_field_keys=set(desired.coord_fields.keys()),
      named_shape=desired.named_shape,
      positional_shape=desired.positional_shape,
      rtol=rtol,
      atol=atol,
  )
