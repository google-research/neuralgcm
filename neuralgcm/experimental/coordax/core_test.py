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
import functools
import operator
from typing import Any, Callable

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from neuralgcm.experimental import coordax
from neuralgcm.experimental.coordax import core
from neuralgcm.experimental.coordax import testing
import numpy as np


class CoreTest(parameterized.TestCase):

  PRODUCT_XY = coordax.CartesianProduct(
      (coordax.NamedAxis('x', 2), coordax.NamedAxis('y', 3))
  )

  @parameterized.named_parameters(
      dict(
          testcase_name='empty',
          coordinates=(),
          expected=(),
      ),
      dict(
          testcase_name='single_other_axis',
          coordinates=(coordax.NamedAxis('x', 2),),
          expected=(coordax.NamedAxis('x', 2),),
      ),
      dict(
          testcase_name='single_selected_axis',
          coordinates=(
              coordax.SelectedAxis(coordax.NamedAxis('x', 2), axis=0),
          ),
          expected=(coordax.NamedAxis('x', 2),),
      ),
      dict(
          testcase_name='pair_of_other_axes',
          coordinates=(
              coordax.NamedAxis('x', 2),
              coordax.LabeledAxis('y', np.arange(3)),
          ),
          expected=(
              coordax.NamedAxis('x', 2),
              coordax.LabeledAxis('y', np.arange(3)),
          ),
      ),
      dict(
          testcase_name='pair_of_selections_correct',
          coordinates=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
          ),
          expected=(PRODUCT_XY,),
      ),
      dict(
          testcase_name='pair_of_selections_wrong_order',
          coordinates=(
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
          ),
          expected=(
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
          ),
      ),
      dict(
          testcase_name='selection_incomplete',
          coordinates=(coordax.SelectedAxis(PRODUCT_XY, axis=0),),
          expected=(coordax.SelectedAxis(PRODUCT_XY, axis=0),),
      ),
      dict(
          testcase_name='selections_with_following',
          coordinates=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
              coordax.NamedAxis('z', 4),
          ),
          expected=(
              PRODUCT_XY,
              coordax.NamedAxis('z', 4),
          ),
      ),
      dict(
          testcase_name='selections_with_preceeding',
          coordinates=(
              coordax.NamedAxis('z', 4),
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
          ),
          expected=(
              coordax.NamedAxis('z', 4),
              PRODUCT_XY,
          ),
      ),
      dict(
          testcase_name='selections_split',
          coordinates=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.NamedAxis('z', 4),
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
          ),
          expected=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.NamedAxis('z', 4),
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
          ),
      ),
      dict(
          testcase_name='two_selected_axes_consolidate_after',
          coordinates=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.SelectedAxis(coordax.NamedAxis('x', 4), axis=0),
          ),
          expected=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.NamedAxis('x', 4),
          ),
      ),
      dict(
          testcase_name='two_selected_axes_consolidate_before',
          coordinates=(
              coordax.SelectedAxis(coordax.NamedAxis('x', 4), axis=0),
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
          ),
          expected=(
              coordax.NamedAxis('x', 4),
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
          ),
      ),
  )
  def test_consolidate_coordinates(self, coordinates, expected):
    actual = core.consolidate_coordinates(*coordinates)
    self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='selected_axes_compoents_merge',
          coordinates=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
          ),
          expected=PRODUCT_XY,
      ),
      dict(
          testcase_name='selected_axis_simplified',
          coordinates=(
              coordax.SelectedAxis(coordax.NamedAxis('x', 4), axis=0),
              coordax.NamedAxis('z', 7),
          ),
          expected=coordax.CartesianProduct(
              (coordax.NamedAxis('x', 4), coordax.NamedAxis('z', 7))
          ),
      ),
      dict(
          testcase_name='cartesian_product_unraveled',
          coordinates=(
              coordax.NamedAxis('x', 7),
              coordax.CartesianProduct(
                  (coordax.NamedAxis('y', 7), coordax.NamedAxis('z', 4))
              ),
          ),
          expected=coordax.CartesianProduct((
              coordax.NamedAxis('x', 7),
              coordax.NamedAxis('y', 7),
              coordax.NamedAxis('z', 4),
          )),
      ),
      dict(
          testcase_name='consolidate_over_parts',
          coordinates=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.CartesianProduct((
                  coordax.SelectedAxis(PRODUCT_XY, axis=1),
                  coordax.NamedAxis('z', 4)
              )),
          ),
          expected=coordax.CartesianProduct((
              coordax.NamedAxis('x', 2),
              coordax.NamedAxis('y', 3),
              coordax.NamedAxis('z', 4),
          )),
      ),
  )
  def test_compose_coordinates(self, coordinates, expected):
    actual = core.compose_coordinates(*coordinates)
    self.assertEqual(actual, expected)

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
          tags=('i', 'j', coordax.LabeledAxis('k', np.arange(1))),
          untags=('j',),
          expected_dims=('i', 0, 'k'),
          expected_named_shape={'i': 5, 'k': 1},
          expected_positional_shape=(3,),
          expected_coord_field_keys=set(['k']),
      ),
      dict(
          testcase_name='tag_prefix',
          array=np.arange(5 * 3).reshape((5, 3, 1)),
          tags=(coordax.LabeledAxis('k', np.arange(5)), 'j'),
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
      tags: tuple[str | coordax.Coordinate, ...],
      untags: tuple[str | coordax.Coordinate, ...],
      expected_dims: tuple[str | int, ...],
      expected_named_shape: dict[str, int],
      expected_positional_shape: tuple[int, ...],
      expected_coord_field_keys: set[str],
  ):
    """Tests that field properties are correctly set."""
    field = coordax.wrap(array)
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
          field_a=coordax.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3))),
          field_b=coordax.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3))),
          op=operator.add,
          expected_result=coordax.wrap(
              np.arange(2 * 3 * 4).reshape((2, 4, 3)) * 2
          ),
      ),
      dict(
          testcase_name='sum_aligned',
          field_a=coordax.wrap(np.arange(2 * 3).reshape((2, 3)), 'x', 'y'),
          field_b=coordax.wrap(
              np.arange(2 * 3)[::-1].reshape((3, 2)), 'y', 'x'
          ),
          op=operator.add,
          expected_result=coordax.wrap(
              np.array([[5, 4, 3], [7, 6, 5]]), 'x', 'y'
          ),
      ),
      dict(
          testcase_name='product_aligned',
          field_a=coordax.wrap(np.arange(2 * 3).reshape((2, 3))).tag_prefix(
              'x'
          ),
          field_b=coordax.wrap(np.arange(2), 'x'),
          op=operator.mul,
          expected_result=coordax.wrap(
              np.arange(2 * 3).reshape((2, 3)) * np.array([[0], [1]])
          ).tag_prefix('x'),
      ),
  )
  def test_field_binary_ops(
      self,
      field_a: coordax.Field,
      field_b: coordax.Field,
      op: Callable[[coordax.Field, coordax.Field], coordax.Field],
      expected_result: coordax.Field,
  ):
    """Tests that field binary ops work as expected."""
    actual = op(field_a, field_b)
    actual.check_valid()
    testing.assert_fields_allclose(actual=actual, desired=expected_result)

  def test_field_repr(self):
    expected = (
        "Field(named_array=<NamedArray int64(| x:2, y:3) "
        "(wrapping numpy.ndarray)>, coords={'x': NamedAxis(name='x', size=2), "
        "'y': LabeledAxis(name='y', ticks=np.array([7, 8, 9]))})"
    )
    actual = coordax.wrap(
        np.array([[1, 2, 3], [4, 5, 6]]),
        'x',
        coordax.LabeledAxis('y', np.array([7, 8, 9])),
    )
    self.assertEqual(repr(actual), expected)

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
          tags=(coordax.NamedAxis('idx', 4),),
          untags=('idx',),
      ),
      dict(
          testcase_name='coord_&_coord',
          array=np.arange(4),
          tags=(coordax.NamedAxis('idx', 4),),
          untags=(coordax.NamedAxis('idx', 4),),
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
          untags=(coordax.NamedAxis('y', 3),),
          full_unwrap=False,
      ),
      dict(
          testcase_name='names_&_coords',
          array=np.arange(2 * 3).reshape((2, 3)),
          tags=('x', 'y'),
          untags=(
              coordax.NamedAxis('x', 2),
              coordax.NamedAxis('y', 3),
          ),
          full_unwrap=True,
      ),
      dict(
          testcase_name='names_&_product_coord',
          array=np.arange(2 * 3).reshape((2, 3)),
          tags=('x', 'y'),
          untags=(
              coordax.compose_coordinates(
                  coordax.NamedAxis('x', 2),
                  coordax.NamedAxis('y', 3),
              ),
          ),
          full_unwrap=True,
      ),
      dict(
          testcase_name='product_coord_&_product_coord',
          array=np.arange(2 * 3).reshape((2, 3)),
          tags=(
              coordax.compose_coordinates(
                  coordax.NamedAxis('x', 2),
                  coordax.NamedAxis('y', 3),
              ),
          ),
          untags=(
              coordax.compose_coordinates(
                  coordax.NamedAxis('x', 2),
                  coordax.NamedAxis('y', 3),
              ),
          ),
          full_unwrap=True,
      ),
      dict(
          testcase_name='mixed_&_names',
          array=np.arange(2 * 3 * 4).reshape((2, 4, 3)),
          tags=('x', coordax.NamedAxis('y', 4), 'z'),
          untags=('y', 'z'),
          full_unwrap=False,
      ),
      dict(
          testcase_name='mixed_&_wrong_names',
          array=np.arange(2 * 3 * 4).reshape((2, 4, 3)),
          tags=('x', coordax.NamedAxis('y_prime', 4), 'z'),
          untags=('y', 'z'),
          full_unwrap=False,
          should_raise_on_untag=True,
      ),
      dict(
          testcase_name='coord_&_wrong_coord_value',
          array=np.arange(9),
          tags=(
              coordax.LabeledAxis(
                  'z',
                  np.arange(9),
              ),
          ),
          untags=(coordax.LabeledAxis('z', np.arange(9) + 1),),
          full_unwrap=False,
          should_raise_on_untag=True,
      ),
  )
  def test_tag_then_untag_by(
      self,
      array: np.ndarray,
      tags: tuple[str | coordax.Coordinate, ...],
      untags: tuple[str | coordax.Coordinate, ...],
      should_raise_on_untag: bool = False,
      full_unwrap: bool = True,
  ):
    """Tests that tag and untag on Field work as expected."""
    with self.subTest('tag'):
      field = coordax.wrap(array, *tags)
      expected_dims = sum(
          [
              tag.dims if isinstance(tag, coordax.Coordinate) else (tag,)
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
          f=coordax.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3))),
          tags=('x', 'y', 'z'),
          expected_dims=('x', 'y', 'z'),
          expected_coord_field_keys=set(),
      ),
      dict(
          testcase_name='partial_by_names',
          f=coordax.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3))),
          tags=('x', 'y'),
          expected_dims=('x', 'y', 0),
          expected_coord_field_keys=set(),
      ),
      dict(
          testcase_name='partial_by_name&coord',
          f=coordax.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3))),
          tags=('x', coordax.LabeledAxis('y', np.arange(4))),
          expected_dims=('x', 'y', 0),
          expected_coord_field_keys=set(['y']),
      ),
      dict(
          testcase_name='partial_by_product_coord',
          f=coordax.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3))),
          tags=(
              coordax.compose_coordinates(
                  coordax.LabeledAxis('x', np.arange(2)),
                  coordax.LabeledAxis('z', np.linspace(0, 1, 4)),
              ),
          ),
          expected_dims=('x', 'z', 0),
          expected_coord_field_keys=set(['x', 'z']),
      ),
  )
  def test_tag_prefix(
      self,
      f: coordax.Field,
      tags: tuple[str | tuple[str, ...] | coordax.Coordinate, ...],
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
          testcase_name='partial_by_name_tail',
          f=coordax.wrap(np.arange(2 * 3).reshape((2, 3)), 'x', 'y'),
          untags=('y',),
          prefix_expected_dims=('x', 0),
          suffix_expected_dims=('x', 0),
          expected_coord_field_keys=set(),
      ),
      dict(
          testcase_name='partial_by_name_head',
          f=coordax.wrap(np.arange(2 * 3).reshape((2, 3)), 'x', 'y'),
          untags=('x',),
          prefix_expected_dims=(0, 'y'),
          suffix_expected_dims=(0, 'y'),
          expected_coord_field_keys=set(),
      ),
      dict(
          testcase_name='partial_by_coord',
          f=coordax.wrap(
              np.arange(2 * 3 * 5).reshape((2, 3, 5)),
              'x',
              coordax.LabeledAxis('y', np.arange(3)),
              'z',
          ),
          untags=('x', 'z'),
          prefix_expected_dims=(0, 'y', 1),
          suffix_expected_dims=(0, 'y', 1),
          expected_coord_field_keys=set(['y']),
      ),
      dict(
          testcase_name='with_positional_prefix_partial_by_name',
          f=coordax.wrap(
              np.arange(2 * 3 * 5).reshape((2, 3, 5)),
              'x',
              coordax.LabeledAxis('y', np.arange(3)),
              'z',
          ).untag('x'),
          untags=('z',),
          prefix_expected_dims=(1, 'y', 0),
          suffix_expected_dims=(0, 'y', 1),
          expected_coord_field_keys=set(['y']),
      ),
      dict(
          testcase_name='mixed_partial_by_name',
          f=coordax.wrap(np.ones((1, 1, 1, 1)), 'x', 'y', 'z', 'w').untag(
              'y', 'w'
          ),
          untags=('z',),
          prefix_expected_dims=('x', 1, 0, 2),
          suffix_expected_dims=('x', 0, 2, 1),
          expected_coord_field_keys=set(),
      ),
  )
  def test_untag(
      self,
      f: coordax.Field,
      untags: tuple[str | tuple[str, ...] | coordax.Coordinate, ...],
      prefix_expected_dims: tuple[str | int, ...],
      suffix_expected_dims: tuple[str | int, ...],
      expected_coord_field_keys: set[str],
  ):
    """Tests that untag_prefix works as expected."""
    with self.subTest('prefix'):
      untagged_f = f.untag_prefix(*untags)
      testing.assert_field_properties(
          actual=untagged_f,
          dims=prefix_expected_dims,
          coord_field_keys=expected_coord_field_keys,
      )
    with self.subTest('suffix'):
      untagged_f = f.untag_suffix(*untags)
      testing.assert_field_properties(
          actual=untagged_f,
          dims=suffix_expected_dims,
          coord_field_keys=expected_coord_field_keys,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='cos(x)',
          inputs=coordax.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3))),
          tags=('x', 'y', 'z'),
          untags=('x',),
          fn=jnp.cos,
          expected_dims=('y', 'z', 0),
          expected_coord_field_keys=set(),
          result_fn=lambda x: jnp.cos(x).transpose((1, 2, 0)),
      ),
      dict(
          testcase_name='norm(x, z)',
          inputs=coordax.wrap(np.arange(2 * 3 * 5).reshape((2, 3, 5))),
          tags=('x', coordax.LabeledAxis('y', np.arange(3)), 'z'),
          fn=jnp.linalg.norm,
          untags=('x', 'z'),
          expected_dims=('y',),
          expected_coord_field_keys=set(['y']),
          result_fn=lambda x: jnp.linalg.norm(x, axis=(0, 2)),
      ),
  )
  def test_cmap(
      self,
      inputs: coordax.Field,
      tags: tuple[str | coordax.Coordinate, ...],
      untags: tuple[str | coordax.Coordinate, ...],
      fn: Callable[..., Any],
      expected_dims: tuple[str | int, ...],
      expected_coord_field_keys: set[str],
      result_fn: Callable[..., Any],
  ):
    """Tests that cmap works as expected."""
    input_f = inputs.tag_prefix(*tags).untag(*untags)
    actual = coordax.cmap(fn)(input_f)
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
    coords = coordax.LabeledAxis('x', np.array([2, 3, 7]))
    batch, length = 4, 10
    vmap_axis = coordax.NamedAxis('i', batch)
    scan_axis = coordax.LabeledAxis('timedelta', np.arange(length))

    def initialize(data):
      return coordax.wrap(data, coords)

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
