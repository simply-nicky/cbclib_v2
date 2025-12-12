import pytest
import jax.numpy as jnp
from jax import random
import cbclib_v2 as cbc
from cbclib_v2.annotations import ArrayNamespace, KeyArray, RealArray, NumPy
from cbclib_v2.indexer import (CBData, CBDIndexer, CBDLoss, CBDModel, CircleState, MillerWithRLP,
                               Patterns, PointsWithK, TiltOverAxisState, UCA, project_to_rect)
from cbclib_v2.test_util import FixedState, TestSetup, check_close

class TestCBDIndexer():
    EPS : float = 1e-7

    @pytest.fixture
    def xp(self) -> ArrayNamespace:
        return NumPy

    @pytest.fixture
    def state(self, xp: ArrayNamespace) -> FixedState:
        return FixedState(TestSetup.xtal(xp))

    @pytest.fixture(params=[10,])
    def num_lines(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[4,])
    def num_points(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[0.3])
    def q_abs(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def patterns(self, key: KeyArray, indexer: CBDIndexer, state: FixedState, num_lines: int,
                 xp: ArrayNamespace) -> Patterns:
        keys = xp.asarray(random.key_data(random.split(key, 4)))
        center = indexer.lens.zero_order(state.lens, xp)

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
    def points(self, indexer: CBDIndexer, patterns: Patterns, state: FixedState,
               xp: ArrayNamespace) -> PointsWithK:
        return indexer.points_to_kout(patterns.sample(xp.full(patterns.shape[0], 0.5)), state, xp)

    @pytest.fixture
    def all_rlp(self, indexer: CBDIndexer, patterns: Patterns, q_abs: float, state: FixedState,
                xp: ArrayNamespace) -> MillerWithRLP:
        hkl = indexer.xtal.hkl_in_ball(q_abs, state.xtal, xp)
        iterator = indexer.xtal.hkl_range(patterns.index_array.unique(), hkl, state.xtal, xp)
        return MillerWithRLP.concatenate(list(iterator))

    @pytest.fixture
    def patterns_uca(self, indexer: CBDIndexer, patterns: Patterns, points: PointsWithK,
                     state: FixedState, xp: ArrayNamespace) -> UCA:
        return indexer.patterns_to_uca(patterns, points, state, xp)

    @pytest.fixture
    def candidates(self, indexer: CBDIndexer, all_rlp: MillerWithRLP, patterns_uca: UCA,
                   xp: ArrayNamespace) -> MillerWithRLP:
        return indexer.candidates(all_rlp, patterns_uca, xp)[0]

    @pytest.fixture
    def uca(self, indexer: CBDIndexer, all_rlp: MillerWithRLP, patterns_uca: UCA,
            xp: ArrayNamespace) -> UCA:
        return indexer.candidates(all_rlp, patterns_uca, xp)[1]

    @pytest.fixture
    def data(self, key: KeyArray, patterns: Patterns, model: CBDModel, state: FixedState,
             num_points: int) -> CBData:
        return model.init_data_random(key, patterns, num_points, state)

    @pytest.fixture
    def pupil_loss(self, model: CBDModel, xp: ArrayNamespace) -> CBDLoss:
        return model.pupil_loss(xp=xp)

    @pytest.fixture
    def solution(self, model: CBDModel, pupil_loss: CBDLoss, data: CBData, state: FixedState,
                 xp: ArrayNamespace) -> MillerWithRLP:
        miller = pupil_loss.index(data, state)
        return model.xtal.hkl_to_q(miller, state.xtal, xp)

    def test_candidates(self, patterns_uca: UCA, solution: MillerWithRLP, points: PointsWithK,
                        candidates: MillerWithRLP, uca: UCA, xp: ArrayNamespace):
        xp = cbc.array_namespace(patterns_uca, solution, points, candidates, uca)
        resolution = xp.sum(solution.q**2, axis=-1)
        idxs = xp.where((patterns_uca.min_resolution < resolution) &
                        (resolution < patterns_uca.max_resolution))[0]
        idxs2, uca_idxs = xp.where(xp.all((solution.hkl[..., None, :] == candidates.hkl) &
                                          (points.kout[..., None, :] == uca.kout), axis=-1))
        assert xp.all(idxs == idxs2)
        assert all(xp.all(lhs == rhs) for lhs, rhs in zip(patterns_uca[idxs].to_dict().values(),
                                                          uca[uca_idxs].to_dict().values()))

    @pytest.fixture
    def circles(self, indexer: CBDIndexer, candidates: MillerWithRLP, uca: UCA,
                xp: ArrayNamespace) -> CircleState:
        return indexer.intersection(candidates, uca, xp)

    def test_intersection(self, circles: CircleState, candidates: MillerWithRLP, uca: UCA,
                          xp: ArrayNamespace):
        theta = xp.linspace(0, 2 * xp.pi)[:, None]
        pts = circles.points(theta)
        resolution = xp.broadcast_to(xp.sum(candidates.q**2, axis=-1), pts.shape[:-1])
        check_close(xp.sum(pts**2, axis=-1), resolution)
        check_close(xp.sum((pts - uca.kout)**2, axis=-1), xp.ones(pts.shape[:-1]))

    @pytest.fixture
    def endpoints(self, indexer: CBDIndexer, circles: CircleState, uca: UCA,
                  xp: ArrayNamespace) -> RealArray:
        return indexer.uca_endpoints(circles, uca, xp)

    def test_endpoints(self, circles: CircleState, endpoints: RealArray, candidates: MillerWithRLP,
                       uca: UCA, xp: ArrayNamespace):
        pts = circles.points(endpoints)
        check_close(xp.sum(pts**2, axis=-1),
                    xp.broadcast_to(xp.sum(candidates.q**2, axis=-1), pts.shape[:-1]))
        check_close(xp.sum((pts - uca.kout)**2, axis=-1), xp.ones(pts.shape[:-1]))
        proj = project_to_rect(pts[..., :2], uca.q_min[..., :2], uca.q_max[..., :2])
        check_close(pts[..., :2], proj)

    @pytest.fixture
    def midpoints(self, circles: CircleState, endpoints: RealArray, xp: ArrayNamespace
                  ) -> RealArray:
        return circles.points(xp.mean(endpoints, axis=0))

    @pytest.fixture
    def tilts(self, indexer: CBDIndexer, candidates: MillerWithRLP, midpoints: RealArray,
              xp: ArrayNamespace):
        return indexer.rotations(candidates, midpoints, xp)

    def test_rotations(self, tilts: TiltOverAxisState, candidates: MillerWithRLP,
                       midpoints: RealArray, xp: ArrayNamespace):
        pts = (tilts.to_tilt().to_rotation() @ candidates.q[..., None, None, :])[..., 0, :]
        check_close(xp.broadcast_to(midpoints[..., None, :], pts.shape), pts)
