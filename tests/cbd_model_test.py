from typing import Any, Callable, Tuple
import pytest
import jax.numpy as jnp
from jax import random, jit
import cbclib_v2 as cbc
from cbclib_v2.annotations import IntArray, KeyArray
from cbclib_v2.test_util import check_gradient, Criterion, TestSetup, TestState, TestModel

class TestCBDModel():
    EPS: float = 5e-7

    @pytest.fixture
    def patterns(self, key: KeyArray, model: TestModel, int_state: cbc.jax.InternalState,
                 num_lines: int) -> cbc.jax.Patterns:
        keys = random.split(key, 4)
        center = model.lens.zero_order(int_state.lens)

        length = random.uniform(keys[0], (num_lines,), jnp.float32, 1.5e-3, 1.5e-2)
        x = random.uniform(keys[2], (num_lines,), jnp.float32,
                           TestSetup.roi[2] * TestSetup.x_pixel_size,
                           TestSetup.roi[3] * TestSetup.x_pixel_size)
        y = random.uniform(keys[3], (num_lines,), jnp.float32,
                           TestSetup.roi[0] * TestSetup.y_pixel_size,
                           TestSetup.roi[1] * TestSetup.y_pixel_size)
        phi = jnp.arctan2(y - center[1], x - center[0])
        angles = phi + jnp.pi / 2 + random.uniform(keys[1], (num_lines,), jnp.float32,
                                                   -jnp.pi / 50, jnp.pi / 50)

        lines = jnp.stack((x - 0.5 * length * jnp.cos(angles),
                           y - 0.5 * length * jnp.sin(angles),
                           x + 0.5 * length * jnp.cos(angles),
                           y + 0.5 * length * jnp.sin(angles)), axis=-1)
        index = jnp.concatenate((jnp.full((num_lines // 2), 0),
                                 jnp.full((num_lines - num_lines // 2), 1)))
        return cbc.jax.Patterns(lines=lines, index=index)

    @pytest.fixture
    def data(self, key: KeyArray, patterns: cbc.jax.Patterns, model: TestModel,
             int_state: cbc.jax.InternalState, num_points: int
             ) -> cbc.jax.CBData:
        return model.init_data(key, patterns, num_points, int_state)

    @pytest.fixture
    def pupil_loss(self, model: TestModel, num_lines: int) -> Criterion:
        return jit(model.pupil_loss(num_lines))

    @pytest.fixture
    def line_loss(self, model: TestModel, num_lines: int) -> Criterion:
        return jit(model.line_loss(num_lines))

    def check_loss(self, f: Criterion, data: cbc.jax.CBData, state: TestState):
        def loss(state):
            return f(data, state)

        check_gradient(loss, (state,), eps=self.EPS)

    @pytest.mark.slow
    @pytest.mark.parametrize('num_lines,num_points', [(200, 4)])
    def test_model_gradients(self, key: KeyArray, data: cbc.jax.CBData,
                             pupil_loss: Criterion, line_loss: Criterion):
        self.check_loss(line_loss, data, TestState.random(key))
        self.check_loss(pupil_loss, data, TestState.random(key))
