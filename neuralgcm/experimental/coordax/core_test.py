"""Tests core methods in the coordax API."""

import functools
import operator
from typing import Any, Callable

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from neuralgcm.experimental.coordax import core
from neuralgcm.experimental.coordax import testing
import numpy as np


class CoreTest(parameterized.TestCase):
  """Tests core methods in the coordax API."""

  @parameterized.named_parameters(
      dict(
          testcase_name='view_with_name',
          array=np.arange(5 * 3).reshape((5, 3)),
          tags=('i', 'j'),
          untags=('i',),
          expected_dims=(0, 'j'),
          expected_named_shape={'j': 3},
          expected_positional_shape=(5,),
          expected_coord_field_keys=set(),
      ),
      dict(
          testcase_name='view_with_name_and_coord',
          array=np.arange(5 * 3).reshape((5, 3, 1)),
          tags=('i', 'j', core.NamedCoordinate('k', np.arange(1))),
          untags=('j',),
          expected_dims=('i', 0, 'k'),
          expected_named_shape={'i': 5, 'k': 1},
          expected_positional_shape=(3,),
          expected_coord_field_keys=set(['k']),
      ),
      dict(
          testcase_name='tag_prefix',
          array=np.arange(5 * 3).reshape((5, 3, 1)),
          tags=(core.NamedCoordinate('k', np.arange(5)), 'j'),
          untags=(),  # untag can only be used on fully labeled fields.
          expected_dims=('k', 'j', 0),
          expected_named_shape={'k': 5, 'j': 3},
          expected_positional_shape=(1,),
          expected_coord_field_keys=set(['k']),
      ),
  )
  def test_field_properties(
      self,
      array: np.ndarray,
      tags: tuple[str | core.Coordinate, ...],
      untags: tuple[str | core.Coordinate, ...],
      expected_dims: tuple[str | int, ...],
      expected_named_shape: dict[str, int],
      expected_positional_shape: tuple[int, ...],
      expected_coord_field_keys: set[str],
  ):
    """Tests that field properties are correctly set."""
    field = core.wrap(array)
    if len(tags) == array.ndim:
      field = field.tag(*tags)
    else:
      field = field.tag_prefix(*tags)
    if untags:
      field = field.untag(*untags)
    testing.assert_field_properties(
        actual=field,
        data=array,
        dims=expected_dims,
        named_shape=expected_named_shape,
        positional_shape=expected_positional_shape,
        coord_field_keys=expected_coord_field_keys,
    )
    with self.subTest('broadcast_to_positional'):
      new_positional_shape = (7,) + field.positional_shape
      broadcasted = field.broadcast_to(positional_shape=new_positional_shape)
      testing.assert_field_properties(
          actual=broadcasted,
          named_shape=expected_named_shape,
          positional_shape=new_positional_shape,
          coord_field_keys=expected_coord_field_keys,
      )
      field.check_valid()

  @parameterized.named_parameters(
      dict(
          testcase_name='sum_simple',
          field_a=core.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3))),
          field_b=core.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3))),
          op=operator.add,
          expected_result=core.wrap(
              np.arange(2 * 3 * 4).reshape((2, 4, 3)) * 2
          ),
      ),
      dict(
          testcase_name='sum_aligned',
          field_a=core.wrap(np.arange(2 * 3).reshape((2, 3)), 'x', 'y'),
          field_b=core.wrap(np.arange(2 * 3)[::-1].reshape((3, 2)), 'y', 'x'),
          op=operator.add,
          expected_result=core.wrap(np.array([[5, 4, 3], [7, 6, 5]]), 'x', 'y'),
      ),
      dict(
          testcase_name='product_aligned',
          field_a=core.wrap(np.arange(2 * 3).reshape((2, 3))).tag_prefix('x'),
          field_b=core.wrap(np.arange(2), 'x'),
          op=operator.mul,
          expected_result=core.wrap(
              np.arange(2 * 3).reshape((2, 3)) * np.array([[0], [1]])
          ).tag_prefix('x'),
      ),
  )
  def test_field_binary_ops(
      self,
      field_a: core.Field,
      field_b: core.Field,
      op: Callable[[core.Field, core.Field], core.Field],
      expected_result: core.Field,
  ):
    """Tests that field binary ops work as expected."""
    actual = op(field_a, field_b)
    testing.assert_fields_allclose(actual=actual, desired=expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name='_name_&_name',
          array=np.arange(4),
          tags=('idx',),
          untags=('idx',),
      ),
      dict(
          testcase_name='coord_&_name',
          array=np.arange(4),
          tags=(core.NameOnlyCoordinate('idx', 4),),
          untags=('idx',),
      ),
      dict(
          testcase_name='coord_&_coord',
          array=np.arange(4),
          tags=(core.NameOnlyCoordinate('idx', 4),),
          untags=(core.NameOnlyCoordinate('idx', 4),),
      ),
      dict(
          testcase_name='names_&_partial_name',
          array=np.arange(2 * 3).reshape((2, 3)),
          tags=('x', 'y'),
          untags=('x',),
          full_unwrap=False,
      ),
      dict(
          testcase_name='names_&_partial_coord',
          array=np.arange(2 * 3).reshape((2, 3)),
          tags=('x', 'y'),
          untags=(core.NameOnlyCoordinate('y', 3),),
          full_unwrap=False,
      ),
      dict(
          testcase_name='names_&_coords',
          array=np.arange(2 * 3).reshape((2, 3)),
          tags=('x', 'y'),
          untags=(
              core.NameOnlyCoordinate('x', 2),
              core.NameOnlyCoordinate('y', 3),
          ),
          full_unwrap=True,
      ),
      dict(
          testcase_name='names_&_product_coord',
          array=np.arange(2 * 3).reshape((2, 3)),
          tags=('x', 'y'),
          untags=(
              core.compose_coordinates(
                  core.NameOnlyCoordinate('x', 2),
                  core.NameOnlyCoordinate('y', 3),
              ),
          ),
          full_unwrap=True,
      ),
      dict(
          testcase_name='product_coord_&_product_coord',
          array=np.arange(2 * 3).reshape((2, 3)),
          tags=(
              core.compose_coordinates(
                  core.NameOnlyCoordinate('x', 2),
                  core.NameOnlyCoordinate('y', 3),
              ),
          ),
          untags=(
              core.compose_coordinates(
                  core.NameOnlyCoordinate('x', 2),
                  core.NameOnlyCoordinate('y', 3),
              ),
          ),
          full_unwrap=True,
      ),
      dict(
          testcase_name='mixed_&_names',
          array=np.arange(2 * 3 * 4).reshape((2, 4, 3)),
          tags=('x', core.NameOnlyCoordinate('y', 4), 'z'),
          untags=('y', 'z'),
          full_unwrap=False,
      ),
      dict(
          testcase_name='mixed_&_wrong_names',
          array=np.arange(2 * 3 * 4).reshape((2, 4, 3)),
          tags=('x', core.NameOnlyCoordinate('y_prime', 4), 'z'),
          untags=('y', 'z'),
          full_unwrap=False,
          should_raise_on_untag=True,
      ),
      dict(
          testcase_name='coord_&_wrong_coord_value',
          array=np.arange(9),
          tags=(
              core.NamedCoordinate(
                  'z',
                  np.arange(9),
              ),
          ),
          untags=(core.NamedCoordinate('z', np.arange(9) + 1),),
          full_unwrap=False,
          should_raise_on_untag=True,
      ),
  )
  def test_tag_then_untag_by(
      self,
      array: np.ndarray,
      tags: tuple[str | core.Coordinate, ...],
      untags: tuple[str | core.Coordinate, ...],
      should_raise_on_untag: bool = False,
      full_unwrap: bool = True,
  ):
    """Tests that tag and untag on Field work as expected."""
    with self.subTest('tag'):
      field = core.wrap(array, *tags)
      expected_dims = sum(
          [
              tag.dims if isinstance(tag, core.Coordinate) else (tag,)
              for tag in tags
          ],
          start=tuple(),
      )
      chex.assert_trees_all_equal(field.dims, expected_dims)

    with self.subTest('untag'):
      if should_raise_on_untag:
        with self.assertRaises(ValueError):
          field.untag(*untags)
      else:
        untagged = field.untag(*untags)
        if full_unwrap:
          unwrapped = untagged.unwrap()
          np.testing.assert_array_equal(unwrapped, array)

  @parameterized.named_parameters(
      dict(
          testcase_name='full_by_names',
          f=core.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3))),
          tags=('x', 'y', 'z'),
          expected_dims=('x', 'y', 'z'),
          expected_coord_field_keys=set(),
      ),
      dict(
          testcase_name='partial_by_names',
          f=core.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3))),
          tags=('x', 'y'),
          expected_dims=('x', 'y', 0),
          expected_coord_field_keys=set(),
      ),
      dict(
          testcase_name='partial_by_name&coord',
          f=core.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3))),
          tags=('x', core.NamedCoordinate('y', np.arange(4))),
          expected_dims=('x', 'y', 0),
          expected_coord_field_keys=set(['y']),
      ),
      dict(
          testcase_name='partial_by_product_coord',
          f=core.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3))),
          tags=(
              core.compose_coordinates(
                  core.NamedCoordinate('x', np.arange(2)),
                  core.NamedCoordinate('z', np.linspace(0, 1, 4)),
              ),
          ),
          expected_dims=('x', 'z', 0),
          expected_coord_field_keys=set(['x', 'z']),
      ),
  )
  def test_tag_prefix(
      self,
      f: core.Field,
      tags: tuple[str | tuple[str, ...] | core.Coordinate, ...],
      expected_dims: tuple[str | int, ...],
      expected_coord_field_keys: set[str],
  ):
    """Tests that tag_prefix works as expected."""
    tagged_f = f.tag_prefix(*tags)
    testing.assert_field_properties(
        actual=tagged_f,
        dims=expected_dims,
        coord_field_keys=expected_coord_field_keys,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='cos(x)',
          inputs=core.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3))),
          tags=('x', 'y', 'z'),
          untags=('x',),
          fn=jnp.cos,
          expected_dims=('y', 'z', 0),
          expected_coord_field_keys=set(),
          result_fn=lambda x: jnp.cos(x).transpose((1, 2, 0)),
      ),
      dict(
          testcase_name='norm(x, z)',
          inputs=core.wrap(np.arange(2 * 3 * 5).reshape((2, 3, 5))),
          tags=('x', core.NamedCoordinate('y', np.arange(3)), 'z'),
          fn=jnp.linalg.norm,
          untags=('x', 'z'),
          expected_dims=('y',),
          expected_coord_field_keys=set(['y']),
          result_fn=lambda x: jnp.linalg.norm(x, axis=(0, 2)),
      ),
  )
  def test_cmap(
      self,
      inputs: core.Field,
      tags: tuple[str | core.Coordinate, ...],
      untags: tuple[str | core.Coordinate, ...],
      fn: Callable[..., Any],
      expected_dims: tuple[str | int, ...],
      expected_coord_field_keys: set[str],
      result_fn: Callable[..., Any],
  ):
    """Tests that cmap works as expected."""
    input_f = inputs.tag_prefix(*tags).untag(*untags)
    actual = core.cmap(fn)(input_f)
    expected_values = result_fn(inputs.data)
    testing.assert_field_properties(
        actual=actual,
        data=expected_values,
        dims=expected_dims,
        shape=expected_values.shape,
        coord_field_keys=expected_coord_field_keys,
    )

  def test_jax_transforms(self):
    """Tests that vmap/scan work with Field with leading positional axes."""
    coords = core.NamedCoordinate('x', np.array([2, 3, 7]))
    batch, length = 4, 10
    vmap_axis = core.NameOnlyCoordinate('i', batch)
    scan_axis = core.NamedCoordinate('timedelta', np.arange(length))

    def initialize(data):
      return core.wrap(data, coords)

    def body_fn(c, _):
      return (c + 1, c)

    with self.subTest('scan'):
      data = np.zeros(coords.shape)
      init = initialize(data)
      _, scanned = jax.lax.scan(body_fn, init, length=length)
      scanned = scanned.tag_prefix(scan_axis)
      testing.assert_field_properties(
          actual=scanned,
          dims=('timedelta', 'x'),
          shape=(length,) + coords.shape,
      )

    with self.subTest('vmap'):
      batch_data = np.zeros(vmap_axis.shape + coords.shape)
      batch_init = jax.vmap(initialize)(batch_data).tag(vmap_axis)
      testing.assert_field_properties(
          batch_init, dims=('i', 'x'), shape=batch_data.shape
      )

    with self.subTest('vmap_of_scan'):
      batch_data = np.zeros(vmap_axis.shape + coords.shape)
      batch_init = jax.vmap(initialize)(batch_data).tag(vmap_axis)
      # we ensure that inputs to scan/vmap have positional prefix, since
      # otherwise shape changes are not allowed.
      batch_init = batch_init.untag(vmap_axis).with_positional_prefix()
      scan_fn = functools.partial(jax.lax.scan, body_fn, length=length)
      _, scanned = jax.vmap(scan_fn, in_axes=0)(batch_init)
      scanned = scanned.tag_prefix(vmap_axis, scan_axis)
      testing.assert_field_properties(
          actual=scanned,
          dims=('i', 'timedelta', 'x'),
          shape=(batch, length) + coords.shape,
      )

    with self.subTest('scan_of_vmap'):
      batch_data = np.zeros(vmap_axis.shape + coords.shape)
      batch_init = jax.vmap(initialize)(batch_data).tag(vmap_axis)
      # we ensure that inputs to scan/vmap have positional prefix, since
      # otherwise shape changes are not allowed.
      batch_init = batch_init.untag(vmap_axis).with_positional_prefix()
      vmaped_body_fn = jax.vmap(body_fn)
      scan_fn = functools.partial(jax.lax.scan, vmaped_body_fn, length=length)
      _, scanned = scan_fn(batch_init)
      scanned = scanned.tag_prefix(scan_axis, vmap_axis)
      testing.assert_field_properties(
          actual=scanned,
          dims=('timedelta', 'i', 'x'),
          shape=(length, batch) + coords.shape,
      )


if __name__ == '__main__':
  absltest.main()
