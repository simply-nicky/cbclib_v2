import pytest
import jax.numpy as jnp
from jax import random, jit
from cbclib_v2.annotations import ArrayNamespace, JaxNumPy, KeyArray
from cbclib_v2.indexer import CBData, CBDModel, Patterns
from cbclib_v2.test_util import check_gradient, Criterion, FullState, TestSetup

class TestCBDModel():
    EPS: float = 5e-7

    @pytest.fixture
    def xp(self) -> ArrayNamespace:
        return JaxNumPy

    @pytest.fixture
    def state(self, xp: ArrayNamespace) -> FullState:
        return FullState(TestSetup.xtal(xp), TestSetup.fixed_pupil_setup(xp))

    @pytest.fixture
    def patterns(self, key: KeyArray, model: CBDModel, state: FullState, num_lines: int,
                 xp: ArrayNamespace) -> Patterns:
        keys = xp.asarray(random.split(key, 4))
        center = model.lens.zero_order(state.lens, xp)

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
        return Patterns(lines=lines, index=index)

    @pytest.fixture
    def data(self, key: KeyArray, patterns: Patterns, model: CBDModel, state: FullState,
             num_points: int) -> CBData:
        return model.init_data_random(key, patterns, num_points, state)

    @pytest.fixture
    def pupil_loss(self, model: CBDModel, xp: ArrayNamespace) -> Criterion:
        return jit(model.pupil_loss(xp=xp))

    @pytest.fixture
    def line_loss(self, model: CBDModel, xp: ArrayNamespace) -> Criterion:
        return jit(model.line_loss(xp=xp))

    def check_loss(self, f: Criterion, data: CBData, state: FullState):
        def loss(state):
            return f(data, state)

        check_gradient(loss, (state,), eps=self.EPS, rtol=1e-1, atol=1e-2)

    @pytest.mark.slow
    @pytest.mark.parametrize('num_lines,num_points', [(10, 4)])
    def test_model_gradients(self, key: KeyArray, data: CBData, pupil_loss: Criterion,
                             line_loss: Criterion):
        self.check_loss(line_loss, data, FullState.random(key))
        self.check_loss(pupil_loss, data, FullState.random(key))
