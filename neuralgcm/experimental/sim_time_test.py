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
from absl.testing import parameterized
from jax import lax
from neuralgcm.experimental import sim_time


def _error_in_days(years, time_step_in_minutes):
  days = 365.24219 * years
  steps_per_day = 24 * 60 / time_step_in_minutes
  steps = round(steps_per_day * days)
  dt = 1 / steps_per_day
  result, _ = lax.scan(
      lambda t, _: (t.increment(dt), None),
      init=sim_time.SimTime(0, 0),
      length=steps,
  )
  expected_days = steps / steps_per_day
  return float(result.days) + float(result.fraction) - expected_days


class SimTimeTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(time_step_in_minutes=1),
      dict(time_step_in_minutes=2.5),
      dict(time_step_in_minutes=10),
      dict(time_step_in_minutes=60),
      dict(time_step_in_minutes=6 * 60),
      dict(time_step_in_minutes=24 * 60),
  )
  def test_error(self, time_step_in_minutes):
    years = 200
    error = _error_in_days(years, time_step_in_minutes)
    tolerance = 0.5 / (60 * 60 * 24)  # 0.5 seconds
    self.assertLess(abs(error), tolerance)


if __name__ == "__main__":
  absltest.main()
