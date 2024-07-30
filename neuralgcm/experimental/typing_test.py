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
from absl.testing import absltest
import jax
import jax.numpy as jnp
from neuralgcm.experimental import typing
import numpy as np


class TimedeltaTest(absltest.TestCase):

  def test_defaults(self):
    actual = typing.Timedelta()
    expected = typing.Timedelta(0, 0)
    self.assertEqual(actual, expected)

  def test_normalization(self):
    actual = typing.Timedelta(0, 24 * 60 * 60)
    expected = typing.Timedelta(1, 0)
    self.assertEqual(actual, expected)

    actual = typing.Timedelta(0, -1)
    expected = typing.Timedelta(-1, 86399)
    self.assertEqual(actual, expected)

  def test_from_timedelta64(self):
    expected = typing.Timedelta(365, 0)
    actual = typing.Timedelta.from_timedelta64(np.timedelta64(365, 'D'))
    self.assertEqual(actual, expected)

  def test_to_timedelta64(self):
    delta = typing.Timedelta(365, 0)
    expected = np.timedelta64(365, 'D')
    actual = delta.to_timedelta64()
    self.assertEqual(actual, expected)

  def test_addition(self):
    delta = typing.Timedelta(1, 12 * 60 * 60)
    actual = delta + delta
    expected = typing.Timedelta(3, 0)
    self.assertEqual(actual, expected)

    with self.assertRaises(TypeError):
      expected + 60

  def test_negation(self):
    delta = typing.Timedelta(1, 6 * 60 * 60)
    actual = -delta
    expected = typing.Timedelta(-2, 18 * 60 * 60)
    self.assertEqual(actual, expected)

  def test_subtraction(self):
    delta = typing.Timedelta(1, 12 * 60 * 60)
    actual = delta - delta
    expected = typing.Timedelta(0, 0)
    self.assertEqual(actual, expected)

  def test_multiplication(self):
    delta = typing.Timedelta(1, 12 * 60 * 60)
    expected = typing.Timedelta(3, 0)

    actual = delta * 2
    self.assertEqual(actual, expected)

    actual = 2 * delta
    self.assertEqual(actual, expected)

    with self.assertRaises(TypeError):
      delta * delta

  def test_pytree(self):
    # pytree transformations do not normalize seconds
    delta = typing.Timedelta(1, 12 * 60 * 60)
    expected = typing.Timedelta(2)
    expected.seconds = 24 * 60 * 60  # cannot construct directly
    actual = jax.tree.map(lambda x: 2 * x, delta)
    self.assertEqual(actual, expected)

  def test_vmap(self):
    delta = typing.Timedelta(days=jnp.arange(2), seconds=jnp.arange(2))
    result = jax.vmap(lambda x: x)(delta)
    self.assertIsInstance(result, typing.Timedelta)
    np.testing.assert_array_equal(result.days, delta.days)
    np.testing.assert_array_equal(result.seconds, delta.seconds)


class TimestampTest(absltest.TestCase):

  def test_from_datetime64(self):
    expected = typing.Timestamp(typing.Timedelta(365, 0))
    actual = typing.Timestamp.from_datetime64(np.datetime64('1971-01-01'))
    self.assertEqual(expected, actual)

  def test_to_datetime64(self):
    timestamp = typing.Timestamp(typing.Timedelta(365, 0))
    expected = np.datetime64('1971-01-01')
    actual = timestamp.to_datetime64()
    self.assertEqual(expected, actual)

  def test_datetime64_through_int32_roundtrip(self):
    original = np.datetime64('2024-01-01T00:00:00', 'ns')
    timestamp = typing.Timestamp.from_datetime64(original)
    timestamp_int32 = jax.tree.map(np.int32, timestamp)
    restored = timestamp_int32.to_datetime64()
    self.assertEqual(original, restored)

  def test_addition(self):
    delta = typing.Timedelta(1, 0)
    timestamp = typing.Timestamp(delta)
    expected = typing.Timestamp(delta * 2)

    actual = timestamp + delta
    self.assertEqual(actual, expected)

    actual = delta + timestamp
    self.assertEqual(actual, expected)

    with self.assertRaises(TypeError):
      timestamp + 1

    with self.assertRaises(TypeError):
      timestamp + timestamp

  def test_subtraction(self):
    first = typing.Timestamp.from_datetime64(
        np.datetime64('2024-01-01T00:00:00')
    )
    second = typing.Timestamp.from_datetime64(
        np.datetime64('2023-01-01T00:00:00')
    )
    delta = typing.Timedelta(365, 0)
    actual = first - delta
    self.assertEqual(actual, second)
    actual = -delta + first
    self.assertEqual(actual, second)
    actual = first - second
    self.assertEqual(actual, delta)
    actual = second - first
    self.assertEqual(actual, -delta)

  def test_vmap(self):
    stamp = typing.Timestamp(
        typing.Timedelta(days=jnp.arange(2), seconds=jnp.arange(2))
    )
    result = jax.vmap(lambda x: x)(stamp)
    self.assertIsInstance(result, typing.Timestamp)
    np.testing.assert_array_equal(result.delta.days, stamp.delta.days)
    np.testing.assert_array_equal(result.delta.seconds, stamp.delta.seconds)


if __name__ == '__main__':
  absltest.main()
