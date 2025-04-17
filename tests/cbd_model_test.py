import pytest
import jax.numpy as jnp
from jax import random, jit
import cbclib_v2 as cbc
from cbclib_v2.annotations import ArrayNamespace, JaxNumPy, KeyArray
from cbclib_v2.test_util import check_gradient, Criterion, TestSetup, TestState, TestModel

class TestCBDModel():
    EPS: float = 5e-7

    @pytest.fixture
    def xp(self) -> ArrayNamespace:
        return JaxNumPy

    @pytest.fixture
    def state(self, xp: ArrayNamespace) -> TestState:
        return TestState(TestSetup.lens(xp), TestSetup.xtal(xp), TestSetup.z(xp))

    @pytest.fixture
    def int_state(self, model: TestModel, state: TestState) -> cbc.jax.InternalState:
        return model.to_internal(state)

    @pytest.fixture
    def patterns(self, key: KeyArray, model: TestModel, int_state: cbc.jax.InternalState,
                 num_lines: int, xp: ArrayNamespace) -> cbc.jax.Patterns:
        keys = xp.asarray(random.split(key, 4))
        center = model.lens.zero_order(int_state.lens, xp)

        length = xp.asarray(random.uniform(keys[0], (num_lines,), xp.float32, 1.5e-3, 1.5e-2))
        x = xp.asarray(random.uniform(keys[2], (num_lines,), jnp.float32,
                                      TestSetup.roi[2] * TestSetup.x_pixel_size,
                                      TestSetup.roi[3] * TestSetup.x_pixel_size))
        y = xp.asarray(random.uniform(keys[3], (num_lines,), jnp.float32,
                                      TestSetup.roi[0] * TestSetup.y_pixel_size,
                                      TestSetup.roi[1] * TestSetup.y_pixel_size))
        phi = xp.arctan2(y - center[1], x - center[0])
        angles = phi + xp.pi / 2 + random.uniform(keys[1], (num_lines,), xp.float32,
                                                  -xp.pi / 50, xp.pi / 50)

        lines = xp.stack((x - 0.5 * length * xp.cos(angles), y - 0.5 * length * xp.sin(angles),
                          x + 0.5 * length * xp.cos(angles), y + 0.5 * length * xp.sin(angles)),
                         axis=-1)
        index = xp.concatenate((xp.full((num_lines // 2), 0),
                                xp.full((num_lines - num_lines // 2), 1)))
        return cbc.jax.Patterns(lines=lines, index=index)

    @pytest.fixture
    def data(self, key: KeyArray, patterns: cbc.jax.Patterns, model: TestModel,
             int_state: cbc.jax.InternalState, num_points: int
             ) -> cbc.jax.CBData:
        return model.init_data(key, patterns, num_points, int_state)

    @pytest.fixture
    def pupil_loss(self, model: TestModel, num_lines: int, xp: ArrayNamespace) -> Criterion:
        return jit(model.pupil_loss(num_lines, xp=xp))

    @pytest.fixture
    def line_loss(self, model: TestModel, num_lines: int, xp: ArrayNamespace) -> Criterion:
        return jit(model.line_loss(num_lines, xp=xp))

    def check_loss(self, f: Criterion, data: cbc.jax.CBData, state: TestState):
        def loss(state):
            return f(data, state)

        check_gradient(loss, (state,), eps=self.EPS, rtol=1e-1, atol=1e-2)

    @pytest.mark.slow
    @pytest.mark.parametrize('num_lines,num_points', [(10, 4)])
    def test_model_gradients(self, key: KeyArray, data: cbc.jax.CBData,
                             pupil_loss: Criterion, line_loss: Criterion):
        self.check_loss(line_loss, data, TestState.random(key))
        self.check_loss(pupil_loss, data, TestState.random(key))
