# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests neuralgcm coordinate classes."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import coordinates
import numpy as np


class CoordinatesTest(parameterized.TestCase):
  """Tests that coordinate have expected shapes and dims."""

  @parameterized.named_parameters(
      dict(
          testcase_name='spherical_harmonic',
          coords=coordinates.SphericalHarmonicGrid.TL31(),
          expected_dims=('longitude_wavenumber', 'total_wavenumber'),
          expected_shape=(64, 33),
      ),
      dict(
          testcase_name='lon_lat',
          coords=coordinates.LonLatGrid.T21(),
          expected_dims=('longitude', 'latitude'),
          expected_shape=(64, 32),
      ),
      dict(
          testcase_name='product_of_levels',
          coords=cx.compose_coordinates(
              coordinates.SigmaLevels.equidistant(4),
              coordinates.PressureLevels([50, 100, 200, 800, 1000]),
              coordinates.LayerLevels(3),
          ),
          expected_dims=('sigma', 'pressure', 'layer_index'),
          expected_shape=(4, 5, 3),
      ),
      dict(
          testcase_name='sigma_spherical_harmonic_product',
          coords=cx.compose_coordinates(
              coordinates.SigmaLevels.equidistant(4),
              coordinates.SphericalHarmonicGrid.T21(),
          ),
          expected_dims=('sigma', 'longitude_wavenumber', 'total_wavenumber'),
          expected_shape=(4, 44, 23),
      ),
      dict(
          testcase_name='dinosaur_primitive_equation_coords',
          coords=coordinates.DinosaurCoordinates(
              horizontal=coordinates.SphericalHarmonicGrid.T21(),
              vertical=coordinates.SigmaLevels.equidistant(4),
          ),
          expected_dims=('sigma', 'longitude_wavenumber', 'total_wavenumber'),
          expected_shape=(4, 44, 23),
      ),
      dict(
          testcase_name='batched_trajectory',
          coords=cx.compose_coordinates(
              cx.NamedAxis('batch', 7),
              coordinates.TimeDelta(np.arange(5) / 2),
              coordinates.PressureLevels([100, 200, 800, 1000]),
              coordinates.LonLatGrid.T21(),
          ),
          expected_dims=(
              'batch',
              'timedelta',
              'pressure',
              'longitude',
              'latitude',
          ),
          expected_shape=(7, 5, 4, 64, 32),
      ),
  )
  def test_coordinates(
      self,
      coords: cx.Coordinate,
      expected_dims: tuple[str, ...],
      expected_shape: tuple[int, ...],
  ):
    """Tests that coordinates are pytrees and have expected shape and dims."""
    with self.subTest('pytree_roundtrip'):
      leaves, tree_def = jax.tree.flatten(coords)
      reconstructed = jax.tree.unflatten(tree_def, leaves)
      self.assertEqual(reconstructed, coords)

    with self.subTest('dims'):
      self.assertEqual(coords.dims, expected_dims)

    with self.subTest('shape'):
      self.assertEqual(coords.shape, expected_shape)


if __name__ == '__main__':
  absltest.main()
