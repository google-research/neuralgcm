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

"""Tests time integrator modules and supporting utilities."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from neuralgcm.experimental import time_integrators


@dataclasses.dataclass
class MockStep(nnx.Module):
  """Mock step module for testing."""

  delta: nnx.Param = dataclasses.field(default_factory=lambda: nnx.Param(0.5))
  cumulative: nnx.Intermediate = dataclasses.field(
      default_factory=lambda: nnx.Intermediate(0.0)
  )
  last_output: nnx.Intermediate = dataclasses.field(
      default_factory=lambda: nnx.Intermediate(0.0)
  )

  def half_increment(self, x):
    output = x + self.delta.value
    self.last_output = nnx.Intermediate(output)
    return output

  def __call__(self, x):
    result = self.half_increment(self.half_increment(x))
    self.cumulative = nnx.Intermediate(self.cumulative.value + result)
    return result


# TODO(dkochkov) Consider using namedtuple for state to improve readability.
class TimeIntegrationUtilsTest(parameterized.TestCase):
  """Tests for time integration utility function."""

  @parameterized.named_parameters(
      dict(
          testcase_name='nnx_module',
          step_fn=lambda mc: (mc[0], mc[0](mc[1])),
          get_input=lambda model, c: (model, c),
          get_expected_output=lambda mc: mc[1],
          inner_steps=3,
          expected_result=3.0,
          expected_state_vars={'cumulative': 6.0, 'last_output': 3.0},
      ),
      dict(
          testcase_name='nnx_module_method',
          step_fn=lambda mc: (mc[0], mc[0].half_increment(mc[1])),
          get_input=lambda model, c: (model, c),
          get_expected_output=lambda mc: mc[1],
          inner_steps=3,
          expected_result=1.5,
          expected_state_vars={'cumulative': 0.0, 'last_output': 1.5},
      ),
      dict(
          testcase_name='pure_function',
          step_fn=lambda c: c + 1,
          get_input=lambda model, c: c,
          get_expected_output=lambda c: c,
          inner_steps=3,
          expected_result=3.0,
          expected_state_vars=None,
      ),
  )
  def test_repeated(
      self,
      step_fn,
      get_input,
      get_expected_output,
      inner_steps: int,
      expected_result: float,
      expected_state_vars: dict[str, float] | None,
  ):
    """Tests output_shape, number and initialization of params in MlpUniform."""
    model = MockStep()
    repeated_step_fn = time_integrators.repeated(step_fn, inner_steps)
    inputs = get_input(model, 0.0)
    outputs = repeated_step_fn(inputs)

    with self.subTest('expected_result'):
      actual = get_expected_output(outputs)
      self.assertAlmostEqual(actual, expected_result)

    if expected_state_vars is not None:
      for k, expected in expected_state_vars.items():
        with self.subTest(f'expected_state_var_{k}'):
          actual = nnx.state(model, nnx.Intermediate)[k].value
          self.assertAlmostEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
