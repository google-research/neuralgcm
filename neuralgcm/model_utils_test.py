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
"""Tests for model_utils.py."""

from absl.testing import absltest
from absl.testing import parameterized
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import pytree_utils
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import typing
from dinosaur import xarray_utils
import haiku as hk
import jax
from jax import tree_util
import jax.numpy as jnp
from neuralgcm import forcings
from neuralgcm import model_builder
from neuralgcm import model_utils
import numpy as np


tree_map = jax.tree_util.tree_map
tree_leaves = jax.tree_util.tree_leaves
units = scales.units
Pytree = typing.Pytree

EPS_f32 = jnp.finfo(jnp.float32).eps


class SimpleModelFromTrajectory:
  def __init__(self, trajectory_fn):
    self.trajectory = trajectory_fn
    self.encode = lambda x, forcing: x
    self.forcing_fn = lambda x, t: {}


class WithForcingTest(parameterized.TestCase):
  """Tests with_forcing method."""

  # TODO(pnorgaard): consider moving to a gcm test_util file
  def _assert_tree(
      self,
      method: typing.Callable[[typing.Any, typing.Any], None],
      expected: Pytree,
      actual: Pytree) -> None:
    expected_leaves, expected_treedef = tree_util.tree_flatten(expected)
    actual_leaves, actual_treedef = tree_util.tree_flatten(actual)
    self.assertEqual(actual_treedef, expected_treedef)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
      method(actual_leaf, expected_leaf)

  def assertTreeAllClose(self, expected: Pytree, actual: Pytree) -> None:
    self._assert_tree(np.testing.assert_allclose, expected, actual)

  def setUp(self):
    super().setUp()
    # make model specifications
    n_sigma_layers = 3
    self.coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(n_sigma_layers)
    )
    self.physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    self.one_hour = self.physics_specs.nondimensionalize(units.hour)
    self.dt = self.one_hour
    ref_datetime = np.datetime64('1970-01-01T00:00:00')
    self.aux_features = {xarray_utils.REFERENCE_DATETIME_KEY: ref_datetime}

    # make forcing data
    n_times = 120
    self.sim_time_data = self.one_hour * np.arange(n_times)
    surface_shape = (n_times,) + self.coords.surface_nodal_shape
    surface_scalar = np.arange(n_times).reshape((n_times, 1, 1, 1)) * np.ones(
        surface_shape)
    self.forcing_data = {'surface_scalar': surface_scalar,
                         'sim_time': self.sim_time_data}
    self.inputs_to_units_mapping = {'surface_scalar': 'dimensionless',
                                    'sim_time': 'dimensionless'}

  def test_with_forcing_state_without_time(self):
    # specify input, a PE state that does not have 'sim_time'
    # forcing_fn from NoForcing module accepts any input for "sim_time", and
    # returns an empty dictionary.
    ones = np.ones(self.coords.nodal_shape)
    surface_ones = np.ones(self.coords.surface_nodal_shape)
    input_state = primitive_equations.State(
        vorticity=ones,
        divergence=ones,
        temperature_variation=ones,
        log_surface_pressure=surface_ones,
        tracers={'specific_humidity': ones})

    def model_fwd(x, forcing_data):

      # function to be wrapped
      def advance(x, forcing):
        del forcing
        return x

      advance_wrapped = model_utils.with_forcing(
          advance,
          forcing_fn=forcings.NoForcing(
              self.coords, self.dt, self.physics_specs, self.aux_features),
          forcing_data=forcing_data,
      )
      return advance_wrapped(x)

    model = hk.without_apply_rng(hk.transform(model_fwd))
    params = model.init(jax.random.PRNGKey(42), input_state, self.forcing_data)
    output_state = jax.jit(model.apply)(params, input_state, self.forcing_data)
    # advance(x, forcing) just returns x, so expect input/output to be equal
    self.assertTreeAllClose(input_state, output_state)

  def test_with_forcing_single_time(self):
    # specify input, a PE state with a single time value
    ones = np.ones(self.coords.nodal_shape)
    surface_ones = np.ones(self.coords.surface_nodal_shape)
    input_state = primitive_equations.StateWithTime(
        vorticity=ones,
        divergence=ones,
        temperature_variation=ones,
        log_surface_pressure=surface_ones,
        sim_time=self.sim_time_data[-1],
        tracers={'specific_humidity': ones})

    def model_fwd(x, forcing_data):

      # function to be wrapped
      def advance(x, forcing):
        # multiplying together ndim=3 arrays (skip sim_time scalar)
        # surface_scalar broadcasts with vars defined on levels
        return pytree_utils.tree_map_over_nonscalars(
            lambda var: var * forcing['surface_scalar'], x)

      advance_wrapped = model_utils.with_forcing(
          advance,
          forcing_fn=forcings.DynamicDataForcing(
              self.coords, self.dt, self.physics_specs, self.aux_features,
              inputs_to_units_mapping=self.inputs_to_units_mapping),
          forcing_data=forcing_data,
      )
      return advance_wrapped(x)

    model = hk.without_apply_rng(hk.transform(model_fwd))
    params = model.init(jax.random.PRNGKey(42), input_state, self.forcing_data)
    output_state = jax.jit(model.apply)(params, input_state, self.forcing_data)
    # at sime_time=2, forcing values are 2 everywhere
    expected = pytree_utils.tree_map_over_nonscalars(
        lambda x: self.forcing_data['surface_scalar'][-1] * x, input_state)
    self.assertTreeAllClose(output_state, expected)

  def test_with_forcing_multiple_times(self):
    # specify input, a PE state with a multiple time values
    n_times = 4
    time_slice = slice(-n_times, None)
    sim_time = self.sim_time_data[time_slice]
    ones = np.ones((n_times,) + self.coords.nodal_shape)
    surface_ones = np.ones((n_times,) + self.coords.surface_nodal_shape)
    input_state = primitive_equations.StateWithTime(
        vorticity=ones,
        divergence=ones,
        temperature_variation=ones,
        log_surface_pressure=surface_ones,
        sim_time=sim_time,
        tracers={'specific_humidity': ones})

    def model_fwd(x, forcing_data):

      # function to be wrapped
      def encode(x, forcing):
        # multiplying together ndim=4 arrays (skip sim_time 1d array)
        # surface_scalar broadcasts with vars defined on levels
        return pytree_utils.tree_map_where(
            condition_fn=lambda x: x.ndim >= 2,
            f=lambda var: var * forcing['surface_scalar'],
            g=lambda x: x,
            x=x,
        )

      encode_wrapped = model_utils.with_forcing(
          encode,
          forcing_fn=forcings.DynamicDataForcing(
              self.coords, self.dt, self.physics_specs, self.aux_features,
              inputs_to_units_mapping=self.inputs_to_units_mapping),
          forcing_data=forcing_data,
      )
      return encode_wrapped(x)

    model = hk.without_apply_rng(hk.transform(model_fwd))
    params = model.init(jax.random.PRNGKey(42), input_state, self.forcing_data)
    output_state = jax.jit(model.apply)(params, input_state, self.forcing_data)
    multiplier = self.forcing_data['surface_scalar'][time_slice]
    expected = pytree_utils.tree_map_where(
        condition_fn=lambda x: x.ndim >= 2,
        f=lambda x: multiplier * x,
        g=lambda x: x,
        x=input_state,
        )
    self.assertTreeAllClose(output_state, expected)


class TrajectoryWithStopGradientsTest(parameterized.TestCase):

  def _linear_growth_trajectory_fn(self, theta):
    """X[t] = X[t-1] + theta."""

    def trajectory_fn(
        x, forcing_data, outer_steps, inner_steps, start_with_input=False
    ):
      legs = [x]
      for _ in range(outer_steps):
        legs.append(
            tree_map(
                lambda x_i: x_i
                + theta**inner_steps
                + forcing_data.get('force', 0),
                legs[-1],
            )
        )
      final_state = legs[-1]
      legs = legs[:-1] if start_with_input else legs[1:]

      trajectory = tree_map(
          lambda *items: jnp.stack(items),
          legs[0],
          *legs[1:],
      )
      return final_state, trajectory

    return trajectory_fn

  def test_final_state_and_slice_correct_simple(self):
    theta = 3
    outer_steps = 2

    model = SimpleModelFromTrajectory(
        # X[t] = X[0] + t * theta
        self._linear_growth_trajectory_fn(theta)
    )
    def call_trajectory_fn(start_with_input):
      trajectory_fn = (
          model_utils.trajectory_with_inputs_and_forcing_and_stop_gradients(
              model,
              num_init_frames=1,
              start_with_input=start_with_input,
              stop_gradient_outer_steps=(),
          )
      )
      x = {'X': jnp.array(0.0)}  # Start at 0 ==> X[t] = t * theta
      return trajectory_fn(
          x, forcing_data={}, outer_steps=outer_steps, inner_steps=1
      )

    full_trajectory = 0 + np.arange(outer_steps + 1) * theta

    with self.subTest('start_with_input=True'):
      final_state, trajectory = call_trajectory_fn(start_with_input=True)
      self.assertEqual(final_state, {'X': full_trajectory[-1]})
      self.assertCountEqual(trajectory, ['X'])
      np.testing.assert_array_equal(trajectory['X'], full_trajectory[:-1])

    with self.subTest('start_with_input=False'):
      final_state, trajectory = call_trajectory_fn(start_with_input=False)
      self.assertEqual(final_state, {'X': full_trajectory[-1]})
      self.assertCountEqual(trajectory, ['X'])
      np.testing.assert_array_equal(trajectory['X'], full_trajectory[1:])

  @parameterized.named_parameters(
      dict(
          testcase_name='NoStopGrad_YesStartWithInput',
          outer_steps=3,
          stop_gradient_outer_steps=(),
          start_with_input=True,
      ),
      dict(
          testcase_name='StopGrad13_YesStartWithInput',
          outer_steps=5,
          stop_gradient_outer_steps=(1, 3),
          start_with_input=True,
      ),
      dict(
          testcase_name='NoStopGrad_NoStartWithInput',
          outer_steps=3,
          stop_gradient_outer_steps=(),
          start_with_input=False,
      ),
      dict(
          testcase_name='StopGrad13_NoStartWithInput',
          outer_steps=5,
          stop_gradient_outer_steps=(1, 3),
          start_with_input=False,
      ),
      dict(
          testcase_name='StopGrad5_NoStartWithInput',
          outer_steps=5,
          stop_gradient_outer_steps=(5,),
          start_with_input=False,
      ),
  )
  def test_trajectory_matches_non_stop_grad_function(
      self,
      outer_steps,
      stop_gradient_outer_steps,
      start_with_input,
      inner_steps=2,
      force=3,
  ):
    # Test against a function where the gradients can be computed by hand.
    x = {'X': jnp.array(2.0)}

    model = SimpleModelFromTrajectory(
        self._linear_growth_trajectory_fn(theta=1.3)
    )

    final_state, y = (
        model_utils.trajectory_with_inputs_and_forcing_and_stop_gradients(
            model,
            num_init_frames=1,
            start_with_input=start_with_input,
            stop_gradient_outer_steps=stop_gradient_outer_steps,
        )
    )(
        x,
        forcing_data={'force': force},
        outer_steps=outer_steps,
        inner_steps=inner_steps,
    )

    final_state_nostopgrad, y_nostopgrad = (
        model_utils.trajectory_with_inputs_and_forcing(
            model,
            num_init_frames=1,
            start_with_input=start_with_input,
        )
    )(
        x,
        forcing_data={'force': force},
        outer_steps=outer_steps,
        inner_steps=inner_steps,
    )

    self.assertCountEqual(['X'], final_state)
    np.testing.assert_array_equal(final_state['X'], final_state_nostopgrad['X'])
    np.testing.assert_array_equal(y['X'], y_nostopgrad['X'])

  @parameterized.named_parameters(
      dict(
          testcase_name='NoStopGrad_YesStartWithInput',
          outer_steps=3,
          stop_gradient_outer_steps=(),
          start_with_input=True,
      ),
      dict(
          testcase_name='StopGrad13_YesStartWithInput',
          outer_steps=5,
          stop_gradient_outer_steps=(1, 3),
          start_with_input=True,
      ),
      dict(
          testcase_name='StopGrad1_YesStartWithInput',
          outer_steps=5,
          stop_gradient_outer_steps=(1,),
          start_with_input=True,
      ),
      dict(
          testcase_name='StopGrad2_YesStartWithInput',
          outer_steps=5,
          stop_gradient_outer_steps=(2,),
          start_with_input=True,
      ),
      dict(
          testcase_name='StopGrad5_YesStartWithInput',
          outer_steps=5,
          stop_gradient_outer_steps=(5,),
          start_with_input=True,
      ),
      dict(
          testcase_name='NoStopGrad_NoStartWithInput',
          outer_steps=3,
          stop_gradient_outer_steps=(),
          start_with_input=False,
      ),
      dict(
          testcase_name='StopGrad13_NoStartWithInput',
          outer_steps=5,
          stop_gradient_outer_steps=(1, 3),
          start_with_input=False,
      ),
      dict(
          testcase_name='StopGrad1_NoStartWithInput',
          outer_steps=5,
          stop_gradient_outer_steps=(1,),
          start_with_input=False,
      ),
      dict(
          testcase_name='StopGrad2_NoStartWithInput',
          outer_steps=5,
          stop_gradient_outer_steps=(2,),
          start_with_input=False,
      ),
      dict(
          testcase_name='StopGrad5_NoStartWithInput',
          outer_steps=5,
          stop_gradient_outer_steps=(5,),
          start_with_input=False,
      ),
  )
  def test_stop_gradient_gives_expected_results(
      self, outer_steps, stop_gradient_outer_steps, start_with_input
  ):
    # Test against a function where the gradients can be computed by hand.

    def start_with_input_slicer(iterable):
      """Slice iterable to either `start_with_input` or not."""
      iterable = np.array(list(iterable))
      return iterable[:-1] if start_with_input else iterable[1:]

    # Weight the loss by time. This ensures the test pinpoints errors that occur
    # at different times.
    time_weights = start_with_input_slicer(10 ** np.arange(1, outer_steps + 2))

    def loss_fn(theta):
      """Loss = 0.5 * Sum(weight[t] * X[t]²), with stop grads."""
      x = {'X': jnp.array(theta)}
      model = SimpleModelFromTrajectory(
          self._linear_growth_trajectory_fn(theta)
      )
      trajectory_fn = (
          model_utils.trajectory_with_inputs_and_forcing_and_stop_gradients(
              model,
              num_init_frames=1,
              start_with_input=start_with_input,
              stop_gradient_outer_steps=stop_gradient_outer_steps,
          )
      )
      unused_final_state, y = trajectory_fn(
          x, forcing_data={}, outer_steps=outer_steps, inner_steps=1
      )
      loss_parts = tree_map(lambda xi: 0.5 * jnp.sum(time_weights * xi**2), y)
      return sum(tree_leaves(loss_parts))

    theta = 2.0
    loss = loss_fn(theta)
    grad_loss = jax.grad(loss_fn)(theta)

    # X[t] = theta + t * theta.
    expected_trajectory = (
        theta + start_with_input_slicer(range(outer_steps + 1)) * theta
    )
    expected_loss = 0.5 * np.sum(time_weights * expected_trajectory**2)
    self.assertEqual(loss, expected_loss)

    # The gradient can be computed explicitly:
    #  Loss = Sum(Lossₜ),
    #  with t in {0, ..., outer_steps - 1} if start_with_input,
    #  and t in {1, ..., outer_steps} otherwise.
    # With Wₜ the time weights, we have
    #  Lossₜ = Wₜ Xₜ² / 2
    # and of course Xₜ is a function of (Xₜ₋₁, θ). That is,
    #   Xₜ = Xₜ(Xₜ₋₁, θ) = Xₜ₋₁ + θ, with X₀ = θ.
    # therefore,
    #  d Lossₜ / dθ = (∂Lossₜ/∂Xₜ) (dXₜ/dθ)
    #               = (Wₜ Xₜ) (∂Xₜ/∂θ + ∂Xₜ/∂Xₜ₋₁ dXₜ₋₁/dθ)
    # This recurses until you get to t=0, where dX₀/dθ = 1.
    # For t = 0 dXₜ/dθ = 1.
    # For t > 0 ∂Xₜ/∂θ ≡ 1, and when there is NO stop gradient on Xₜ₋₁
    # ∂Xₜ/∂Xₜ₋₁ ≡ 1 as well. When there is a stop gradient on Xₜ₋₁, then
    # although ∂Xₜ/∂θ = 1, the gradient stops so ∂Xₜ/∂Xₜ₋₁ = 0.  This means
    #  d Lossₜ / dθ = (Wₜ Xₜ) * Kₜ
    # where Kₜ is the number of steps (starting from t and moving backwards)
    # until you reach a stop gradient. E.g. if Kₜ = 2, then it means there was
    # no stop gradient at t-1, but there was one at t-2. Notice whether or not
    # there is a stop gradient at t does not affect Kₜ.
    expected_grad = 0
    for t_i in range(outer_steps):
      # t_i indexes into trajectory.  t is the actual time at index t_i.
      t = t_i if start_with_input else t_i + 1
      n_non_stop_grad_contributing_terms = 1  # account for ∂Xₜ/∂θ
      for s in reversed(range(t)):  # [t-1, t-2, ..., 0]
        if s in stop_gradient_outer_steps:
          break
        # if s not in stop_gradient_outer_steps, ∂Xₛ₊₁/∂Xₛ = 1,
        # and of course ∂Xₛ/∂θ = 1 as well, so add a term.
        n_non_stop_grad_contributing_terms += 1
      expected_grad += (
          time_weights[t_i]
          * expected_trajectory[t_i]
          * n_non_stop_grad_contributing_terms
      )
    self.assertEqual(expected_grad, grad_loss)


class ComputeRepresentationsTest(parameterized.TestCase):
  """Tests functions that compute representations."""

  def setUp(self):
    super().setUp()
    self.data_sigma_layers = 4
    self.model_sigma_layers = self.data_sigma_layers - 1
    self.model_coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(self.model_sigma_layers)
    )
    self.data_coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.T21(),
        sigma_coordinates.SigmaCoordinates.equidistant(self.data_sigma_layers)
    )
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    dt = physics_specs.nondimensionalize(units.hour)
    ref_datetime = np.datetime64('1970-01-01T00:00:00')
    aux_features = {xarray_utils.REFERENCE_DATETIME_KEY: ref_datetime}
    self.forcing_data = {}

    class MockDynamicalSystem(model_builder.DynamicalSystem):
      def encode(self, x, forcing):
        del forcing
        # mock encoder that slices away all but last layers.
        return tree_util.tree_map(lambda y: y[-1, slice(0, -1), ...], x)

      def decode(self, x, forcing):
        del forcing
        # mock decoding by concatenating top layer.
        return tree_util.tree_map(lambda y: jnp.concatenate([y, y[-1:]]), x)

      def advance(self, x, forcing):
        return tree_util.tree_map(lambda y: y + 1.0, x)

      def forcing_fn(self, forcing_data, sim_time):
        del forcing_data, sim_time
        return {}

    self.whirl_model = model_builder.WhirlModel(
        self.model_coords, dt, physics_specs, aux_features, self.data_coords,
        self.data_coords, MockDynamicalSystem
    )

  def test_representation_shapes(self):
    n_steps = 6

    def compute_representations_fwd(initial_state, targets):
      model = self.whirl_model.model_cls()
      _, predicted_trajectory = model.trajectory(
          initial_state, n_steps, forcing_data={}, start_with_input=True)
      predictions, targets = (
          model_utils.compute_prediction_and_target_representations(
              predicted_trajectory, targets, {}, model))
      return predictions, targets

    model_nodal_shape = self.model_coords.nodal_shape
    data_nodal_shape = self.data_coords.nodal_shape
    state = {'a': np.ones(model_nodal_shape)}
    targets_state = {'a': np.ones((1,) + data_nodal_shape) * np.expand_dims(
        np.arange(n_steps), axis=(1, 2, 3))}
    model = hk.without_apply_rng(hk.transform(compute_representations_fwd))
    actual_predictions, actual_targets = model.apply(None, state, targets_state)

    model_modal_shape = self.model_coords.modal_shape
    data_modal_shape = self.data_coords.modal_shape
    expected_shapes = typing.TrajectoryRepresentations(
        data_nodal_trajectory=np.array((n_steps,) + data_nodal_shape),
        data_modal_trajectory=np.array((n_steps,) + data_modal_shape),
        model_nodal_trajectory=np.array((n_steps,) + model_nodal_shape),
        model_modal_trajectory=np.array((n_steps,) + model_modal_shape))
    actual_predictions_shapes = pytree_utils.shape_structure(actual_predictions)
    actual_targets_shapes = pytree_utils.shape_structure(actual_targets)
    predictions_msg = f'{actual_predictions_shapes=} vs {expected_shapes=}'
    targets_msg = f'{actual_targets_shapes=} vs {expected_shapes=}'
    for x, y, expected in zip(
        jax.tree_util.tree_leaves(actual_predictions_shapes),
        jax.tree_util.tree_leaves(actual_targets_shapes),
        jax.tree_util.tree_leaves(expected_shapes)):
      np.testing.assert_array_equal(x, expected, err_msg=predictions_msg)
      np.testing.assert_array_equal(y, expected, err_msg=targets_msg)


class SafeSqrtTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='0', x=0, expected_grad=0),
      dict(testcase_name='eps_/_2', x=EPS_f32 / 2, expected_grad=0),
      dict(
          testcase_name='eps_x_2',
          x=EPS_f32 * 2,
          expected_grad=0.5 / np.sqrt(2 * EPS_f32),
      ),
      dict(testcase_name='1', x=1.0, expected_grad=0.5),
  )
  def test_on_1d_vector(self, x, expected_grad):
    x = jnp.array(x, dtype=jnp.float32)
    y = model_utils.safe_sqrt(x)
    np.testing.assert_array_equal(y, jnp.sqrt(x))
    dy_dx = jax.grad(model_utils.safe_sqrt)(x)
    np.testing.assert_array_equal(dy_dx, expected_grad)

  def test_grad_x_squared_is_sign_x_unless_x_tiny(self):
    x = jnp.array([-2., -0.5, -EPS_f32 / 2, 0., EPS_f32 / 2, 0.5, 2.])
    norm = lambda x: model_utils.safe_sqrt(x**2)
    dnorm_dx_fn = jax.grad(norm)
    dnorm_dx_vals = jax.vmap(dnorm_dx_fn)(x)
    expected_vals = np.sign(x)
    expected_vals[2:5] = 0
    np.testing.assert_array_equal(dnorm_dx_vals, expected_vals)


if __name__ == '__main__':
  absltest.main()
