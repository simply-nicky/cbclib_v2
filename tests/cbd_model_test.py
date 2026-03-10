from typing import Callable
import pytest
from jax import jit, tree
from cbclib_v2 import default_rng, field, State
from cbclib_v2.annotations import AnyGenerator, AnyNamespace, Generator, JaxArray, JaxNamespace, JaxNumPy, RealArray
from cbclib_v2.indexer import BaseState, CBData, CBDModel, FixedPupilSetup, Patterns, XtalState
from cbclib_v2.indexer import random_state
from cbclib_v2.test_util import check_gradient, TestSetup

Criterion = Callable[[CBData, BaseState,], RealArray]

REL_TOL = 0.025

def random_xtal(xp: AnyNamespace) -> Callable[[AnyGenerator], XtalState]:
    return random_state(TestSetup.xtal(xp),
                        tree.map(lambda val: REL_TOL * val, TestSetup.xtal(xp)))

def random_setup(xp: AnyNamespace) -> Callable[[AnyGenerator], FixedPupilSetup]:
    return random_state(TestSetup.fixed_pupil_setup(xp),
                        tree.map(lambda val: REL_TOL * val, TestSetup.fixed_pupil_setup(xp)))

class FullState(BaseState, State, random=True):
    xtal    : XtalState = field(random=random_xtal(JaxNumPy))
    setup   : FixedPupilSetup = field(random=random_setup(JaxNumPy))

class TestCBDModel():
    EPS: float = 5e-7

    @pytest.fixture
    def xp(self) -> JaxNamespace:
        return JaxNumPy

    @pytest.fixture
    def rng(self, xp: JaxNamespace) -> Generator[JaxArray]:
        return default_rng(42, xp)

    @pytest.fixture
    def state(self, xp: JaxNamespace) -> FullState:
        return FullState(TestSetup.xtal(xp), TestSetup.fixed_pupil_setup(xp))

    @pytest.fixture
    def patterns(self, rng: Generator[JaxArray], model: CBDModel, state: FullState,
                 num_lines: int, xp: JaxNamespace) -> Patterns:
        center = model.lens.zero_order(state.lens, xp)

        length = rng.uniform(1.5e-3, 1.5e-2, (num_lines,))
        x = rng.uniform(TestSetup.roi[2] * TestSetup.x_pixel_size,
                        TestSetup.roi[3] * TestSetup.x_pixel_size, (num_lines,))
        y = rng.uniform(TestSetup.roi[0] * TestSetup.y_pixel_size,
                        TestSetup.roi[1] * TestSetup.y_pixel_size, (num_lines,))
        phi = xp.atan2(y - center[1], x - center[0])
        angles = phi + xp.pi / 2 + rng.uniform(-xp.pi / 50, xp.pi / 50, (num_lines,))

        lines = xp.stack((x - 0.5 * length * xp.cos(angles), y - 0.5 * length * xp.sin(angles),
                          x + 0.5 * length * xp.cos(angles), y + 0.5 * length * xp.sin(angles)),
                         axis=-1)
        index = xp.concat((xp.full((num_lines // 2), 0), xp.full((num_lines - num_lines // 2), 1)))
        return Patterns(lines=lines, index=index)

    @pytest.fixture
    def data(self, rng: Generator[JaxArray], patterns: Patterns, model: CBDModel, state: FullState,
             num_points: int) -> CBData:
        return model.init_data_random(rng, patterns, num_points, state)

    @pytest.fixture
    def pupil_loss(self, model: CBDModel, xp: JaxNamespace) -> Criterion:
        return jit(model.pupil_loss(xp=xp))

    @pytest.fixture
    def line_loss(self, model: CBDModel, xp: JaxNamespace) -> Criterion:
        return jit(model.line_loss(xp=xp))

    def check_loss(self, f: Criterion, data: CBData, state: FullState):
        def loss(state):
            return f(data, state)

        check_gradient(loss, (state,), eps=self.EPS, rtol=1e-1, atol=1e-2)

    @pytest.mark.slow
    @pytest.mark.parametrize('num_lines,num_points', [(10, 4)])
    def test_model_gradients(self, rng: Generator[JaxArray], data: CBData, pupil_loss: Criterion,
                             line_loss: Criterion):
        self.check_loss(line_loss, data, FullState.random(rng))
        self.check_loss(pupil_loss, data, FullState.random(rng))
