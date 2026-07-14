from typing import Callable
import pytest
from jax import jit, tree
from cbclib_v2 import default_rng, field, State
from cbclib_v2.annotations import (AnyGenerator, AnyNamespace, Generator, JaxArray, JaxNamespace,
                                   JaxNumPy, NDArray, NumPy, NumPyNamespace, RealArray)
from cbclib_v2.indexer import (BaseState, CBDLoss, CBDPoints, CBData, CBDModel, CBDataBest,
                               FixedPupilSetup, FixedState, MaskedLaueVectors, MillerWithRLP,
                               Patterns, ResolvedState, XtalState, random_state)
from cbclib_v2.test_util import check_close, check_gradient, TestSetup

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
    def resolved(self, state: FullState, xp: JaxNamespace) -> ResolvedState:
        return state.resolve(xp)

    @pytest.fixture
    def patterns(self, rng: Generator[JaxArray], model: CBDModel, resolved: ResolvedState,
                 num_lines: int, xp: JaxNamespace) -> Patterns:
        center = model.lens.zero_order(resolved.setup.lens, xp)

        length = rng.uniform(1.5e-3, 1.5e-2, (num_lines,))
        x = rng.uniform(TestSetup.roi[2] * TestSetup.x_pixel_size,
                        TestSetup.roi[3] * TestSetup.x_pixel_size, (num_lines,))
        y = rng.uniform(TestSetup.roi[0] * TestSetup.y_pixel_size,
                        TestSetup.roi[1] * TestSetup.y_pixel_size, (num_lines,))
        phi = xp.atan2(y - center[..., 1], x - center[..., 0])
        angles = phi + xp.pi / 2 + rng.uniform(-xp.pi / 50, xp.pi / 50, (num_lines,))

        lines = xp.stack((x - 0.5 * length * xp.cos(angles), y - 0.5 * length * xp.sin(angles),
                          x + 0.5 * length * xp.cos(angles), y + 0.5 * length * xp.sin(angles)),
                         axis=-1)
        index = xp.concat((xp.full((num_lines // 2), 0), xp.full((num_lines - num_lines // 2), 1)))
        return Patterns(lines=lines, index=index)

    @pytest.fixture
    def data(self, rng: Generator[JaxArray], patterns: Patterns, model: CBDModel,
             resolved: ResolvedState, num_points: int) -> CBData:
        return model.init_data_random(rng, patterns, num_points, resolved)

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

class TestCBDModelSimulationWorkflow:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def state(self, xp: NumPyNamespace) -> FixedState:
        return FixedState(TestSetup.xtal(xp), TestSetup.fixed_setup())

    @pytest.fixture
    def resolved(self, state: FixedState, xp: NumPyNamespace) -> ResolvedState:
        return state.resolve(xp)

    @pytest.fixture
    def q_abs(self) -> float:
        return 0.08

    @pytest.fixture
    def aperture_rlp(self, model: CBDModel, q_abs: float, resolved: ResolvedState,
                        xp: NumPyNamespace) -> MillerWithRLP:
        return model.hkl_in_aperture(q_abs, resolved, xp)

    @pytest.fixture
    def final_state(self, state: FixedState) -> FixedState:
        return state

    @pytest.fixture
    def laue(self, model: CBDModel, aperture_rlp: MillerWithRLP, resolved: ResolvedState,
             xp: NumPyNamespace) -> MaskedLaueVectors:
        return model.lens.source_lines(aperture_rlp, resolved.setup.lens, xp)

    def test_resolved(self, resolved: ResolvedState) -> None:
        assert resolved.xtal.basis.shape[-2:] == (3, 3)
        assert resolved.setup.lens.foc_pos.shape[-1:] == (3,)
        assert resolved.setup.lens.pupil_roi.shape[-1:] == (4,)
        assert resolved.setup.z.shape == (1,)

    def test_aperture_rlp(self, aperture_rlp: MillerWithRLP, state: FixedState,
                          q_abs: float, xp: NumPyNamespace) -> None:
        assert aperture_rlp.hkl.shape[0] > 0
        assert aperture_rlp.hkl.shape[-1] == 3
        assert xp.all(aperture_rlp.index >= 0)
        assert xp.all(aperture_rlp.index < state.xtal.basis.shape[0])

        assert xp.all(xp.abs(xp.acos(aperture_rlp.source_points[..., 2])) < q_abs)

    def test_source_lines(self, model: CBDModel, laue: MaskedLaueVectors, resolved: ResolvedState,
                          xp: NumPyNamespace) -> None:
        valid = xp.broadcast_to(laue.mask, laue.kin.shape[:-1])
        q = xp.broadcast_to(laue.q, laue.kout.shape)
        q_abs = xp.sum(q[valid]**2, axis=-1)
        kdotq = xp.sum(laue.kin[valid] * q[valid], axis=-1)
        odotq = xp.sum(laue.source_line[valid] * q[valid], axis=-1)
        kmin = model.lens.kin_min(resolved.setup.lens, xp)[..., :2]
        kmax = model.lens.kin_max(resolved.setup.lens, xp)[..., :2]

        assert xp.any(valid)
        check_close(xp.sum(laue.kin[valid]**2, axis=-1), xp.ones(valid.sum()))
        check_close(q[valid], laue.kout[valid] - laue.kin[valid])
        check_close(kdotq, -0.5 * q_abs)
        check_close(odotq, -0.5 * q_abs)
        assert xp.all(laue.kin[..., :2][valid] >= kmin)
        assert xp.all(laue.kin[..., :2][valid] <= kmax)

class TestCBDModelLossWorkflow:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def rng(self, xp: NumPyNamespace) -> Generator[NDArray]:
        return default_rng(42, xp)

    @pytest.fixture
    def state(self, xp: NumPyNamespace) -> FixedState:
        return FixedState(TestSetup.xtal(xp), TestSetup.fixed_setup())

    @pytest.fixture
    def resolved(self, state: FixedState, xp: NumPyNamespace) -> ResolvedState:
        return state.resolve(xp)

    def make_patterns(self, rng: Generator[NDArray], model: CBDModel, resolved: ResolvedState,
                      xp: NumPyNamespace, num_lines: int) -> Patterns:
        center = model.lens.zero_order(resolved.setup.lens, xp)
        length = rng.uniform(1.5e-3, 1.5e-2, (num_lines,))
        x = rng.uniform(TestSetup.roi[2] * TestSetup.x_pixel_size,
                        TestSetup.roi[3] * TestSetup.x_pixel_size, (num_lines,))
        y = rng.uniform(TestSetup.roi[0] * TestSetup.y_pixel_size,
                        TestSetup.roi[1] * TestSetup.y_pixel_size, (num_lines,))
        phi = xp.atan2(y - center[..., 1], x - center[..., 0])
        angles = phi + xp.pi / 2 + rng.uniform(-xp.pi / 50, xp.pi / 50, (num_lines,))
        lines = xp.stack((x - 0.5 * length * xp.cos(angles),
                          y - 0.5 * length * xp.sin(angles),
                          x + 0.5 * length * xp.cos(angles),
                          y + 0.5 * length * xp.sin(angles)), axis=-1)
        index = xp.concat((xp.full((num_lines // 2), 0),
                           xp.full((num_lines - num_lines // 2), 1)))
        return Patterns(lines=lines, index=index)

    @pytest.fixture
    def patterns(self, rng: Generator[NDArray], model: CBDModel, resolved: ResolvedState,
                 xp: NumPyNamespace) -> Patterns:
        return self.make_patterns(rng, model, resolved, xp, num_lines=8)

    @pytest.fixture
    def initialized_data(self, model: CBDModel, patterns: Patterns,
                         resolved: ResolvedState) -> CBData:
        return model.init_data(patterns, resolved)

    @pytest.fixture
    def best_data(self, model: CBDModel, initialized_data: CBData) -> CBDataBest:
        return model.keep_best(initialized_data, 0.5)

    @pytest.fixture
    def line_loss(self, model: CBDModel, xp: NumPyNamespace) -> CBDLoss:
        return model.line_loss(loss='l1', xp=xp)

    @pytest.fixture
    def projected_points(self, line_loss: CBDLoss, best_data: CBDataBest,
                         resolved: ResolvedState, xp: NumPyNamespace) -> CBDPoints:
        return line_loss.project_data(best_data, resolved, xp)

    def test_init_data(self, initialized_data: CBData, patterns: Patterns,
                       model: CBDModel, resolved: ResolvedState, xp: NumPyNamespace) -> None:
        q1, q2 = model.patterns_to_q(patterns, resolved.setup, xp)
        hkl1 = model.xtal.q_to_hkl(q1, resolved.xtal, xp)
        hkl2 = model.xtal.q_to_hkl(q2, resolved.xtal, xp)
        closest = xp.asarray(xp.round(xp.stack((hkl1.hkl, hkl2.hkl), axis=1)), dtype=int)
        matches = xp.all(initialized_data.miller.hkl[:, None] == closest[:, :, None], axis=-1)

        assert initialized_data.miller.hkl.shape[:-1] == initialized_data.points.points.shape[:-2]
        assert initialized_data.points.points.shape[-2:] == (2, 2)
        assert initialized_data.points.index.shape == patterns.index.shape
        assert xp.all(xp.any(matches, axis=-1))

    def test_keep_best(self, best_data: CBDataBest, initialized_data: CBData,
                       xp: NumPyNamespace) -> None:
        assert best_data.mask.shape == initialized_data.points.index.shape
        assert best_data.mask.dtype == xp.dtype(bool)
        assert xp.any(best_data.mask)
        assert xp.all(best_data.miller.hkl == initialized_data.miller.hkl)
        assert xp.all(best_data.points.points == initialized_data.points.points)

    def test_project_data(self, projected_points: CBDPoints, initialized_data: CBData,
                          xp: NumPyNamespace) -> None:
        assert projected_points.kin.shape == initialized_data.miller.hkl.shape[:-1] + (2, 3)
        assert projected_points.kout.shape == initialized_data.points.points.shape[:-1] + (3,)
        assert xp.all(xp.isfinite(projected_points.kin))
        assert xp.all(xp.isfinite(projected_points.kout))
        assert xp.all(xp.isfinite(projected_points.q))
        check_close(xp.sum(projected_points.kout**2, axis=-1),
                    xp.ones(projected_points.kout.shape[:-1]))
        check_close(projected_points.q, projected_points.kout - projected_points.kin)

    def test_loss_value(self, line_loss: CBDLoss, initialized_data: CBData,
                        best_data: CBDataBest, state: FixedState,
                        resolved: ResolvedState, xp: NumPyNamespace) -> None:
        points = line_loss.project_data(initialized_data, resolved, xp)
        distances = xp.min(line_loss.distance_matrix(points, resolved, xp), axis=-1)
        indices = xp.lexsort((distances, initialized_data.points.index), axis=0)
        sorted_distances = distances[indices]
        actual = line_loss.distances(best_data, points, resolved, xp)
        values, counts = xp.unique(initialized_data.points.index, return_counts=True)

        expected_mask = xp.concat([xp.arange(size) < 0.5 * size for size in counts])
        expected = sorted_distances * expected_mask

        assert xp.all(values == xp.arange(state.xtal.basis.shape[0]))
        assert xp.all(best_data.mask == expected_mask)
        assert xp.all(actual[~best_data.mask] == 0.0)
        check_close(actual, expected)
        start = 0
        for size in counts:
            stop = start + int(size)
            kept = sorted_distances[start:start + int(size) // 2]
            discarded = sorted_distances[start + int(size) // 2:stop]
            assert xp.all(kept <= discarded[0])
            start = stop

        value = line_loss(best_data, state)
        assert value.shape == ()
        assert xp.isfinite(value)
        check_close(value, expected.mean())

    def test_per_pattern(self, line_loss: CBDLoss, best_data: CBDataBest,
                         state: FixedState, xp: NumPyNamespace) -> None:
        values = line_loss.per_pattern(best_data, state)
        counts = xp.asarray([xp.sum(best_data.points.index == index)
                             for index in range(state.xtal.basis.shape[0])])
        value = line_loss(best_data, state)

        assert values.shape == (state.xtal.basis.shape[0],)
        assert xp.all(xp.isfinite(values))
        check_close(value, xp.sum(values * counts) / xp.sum(counts))
