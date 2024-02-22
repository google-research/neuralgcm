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
from absl.testing import absltest
from absl.testing import parameterized
import jax
from neuralgcm import optimization
import numpy as np
import optax


class PiecewiseConstantScheduleTest(parameterized.TestCase):

  def test_3_leg_schedule(self):
    params = {'theta': 0.0}
    gradients = jax.tree_util.tree_map(np.ones_like, params)

    schedule = optimization.piecewise_constant_schedule_specified_by_rates(
        rates=[0.0, 1.0, 10.0],
        boundaries=[2, 4],
    )
    optimizer = optax.sgd(schedule)
    opt_state = optimizer.init(params)

    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    self.assertEqual({'theta': 0.0}, params)  # 0 - 0*1 = 0

    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    self.assertEqual({'theta': 0.0}, params)  # 0 - 0*1 = 0

    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    self.assertEqual({'theta': -1.}, params)  # 0 - 1*1 = -1

    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    self.assertEqual({'theta': -2.}, params)  # -1 - 1*1 = -2

    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    self.assertEqual({'theta': -12.}, params)  # -2 - 10*1 = -12


class TopLevelMultiRateTest(parameterized.TestCase):

  def _make_exponential_decay(self, lr: float):
    return optax.warmup_exponential_decay_schedule(
        init_value=lr,
        peak_value=2 * lr,
        warmup_steps=2,
        transition_steps=2,
        decay_rate=0.1,
    )

  @parameterized.named_parameters(
      dict(testcase_name='Default'),
      dict(testcase_name='ExponentialDecay', use_exponential_decay=True),
  )
  def test_rates_adjusted_well(self, use_exponential_decay=False):
    if use_exponential_decay:
      make_lr = self._make_exponential_decay
    else:
      make_lr = lambda lr: lr

    params = {
        'Top_A': {'w': 0.0, 'b': 0.0},
        'Top_B': {'w': 0.0, 'b': 0.0},
        'my_cats_breath': {'w': 0.0, 'b': 0.0},
        'cats_kill_birds': {'w': 5.0, 'b': 8.0},
    }
    gradients = jax.tree_util.tree_map(np.ones_like, params)

    # Give Top_A lr=100, cats lr=1, and all others lr=1e-3
    optimizer = optimization.top_level_multi_adam(
        top_level_keys=['Top_A', 'REGEX_cats'],
        learning_rates=[make_lr(100), make_lr(1)],
        default_learning_rate=make_lr(1e-3),
    )
    state = optimizer.init(params)
    updates, unused_new_state = optimizer.update(gradients, state, params)
    new_params = optax.apply_updates(params, updates)

    # We took one step, so the final parameter values should be the initial
    # minus 1x learning rate (up to some Adam initialization noise).
    np.testing.assert_allclose(0 - 100, new_params['Top_A']['w'], rtol=1e-5)
    np.testing.assert_allclose(0 - 100, new_params['Top_A']['b'], rtol=1e-5)
    np.testing.assert_allclose(
        0 - 1, new_params['my_cats_breath']['w'], rtol=1e-5
    )
    np.testing.assert_allclose(
        0 - 1, new_params['my_cats_breath']['b'], rtol=1e-5
    )
    np.testing.assert_allclose(
        5 - 1, new_params['cats_kill_birds']['w'], rtol=1e-5
    )
    np.testing.assert_allclose(
        8 - 1, new_params['cats_kill_birds']['b'], rtol=1e-5
    )
    np.testing.assert_allclose(0 - 1e-3, new_params['Top_B']['w'], rtol=1e-5)
    np.testing.assert_allclose(0 - 1e-3, new_params['Top_B']['b'], rtol=1e-5)

  def test_missing_keys_handling(self):
    params = {'Top_A': {'w': 0.0, 'b': 0.0}, 'Top_B': {'w': 0.0, 'b': 0.0}}

    with self.subTest('Raises if raise_if_keys_not_found=True'):
      optimizer = optimization.top_level_multi_adam(
          top_level_keys=['Top_A', 'NOT_IN_PARAMS'],
          learning_rates=[1, 2],
          raise_if_keys_not_found=True,
      )
      with self.assertRaisesRegex(optimization.OptimizerError, 'NOT_IN_PARAMS'):
        optimizer.init(params)

    with self.subTest('Does not raise if raise_if_keys_not_found=False'):
      optimizer = optimization.top_level_multi_adam(
          top_level_keys=['Top_A', 'NOT_IN_PARAMS'],
          learning_rates=[1, 2],
          raise_if_keys_not_found=False,
      )
      optimizer.init(params)  # Should not raise


if __name__ == '__main__':
  absltest.main()
