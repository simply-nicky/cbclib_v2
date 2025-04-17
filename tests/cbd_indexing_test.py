import pytest
import jax.numpy as jnp
from jax import random
import cbclib_v2 as cbc
from cbclib_v2.annotations import ArrayNamespace, KeyArray, RealArray, NumPy
from cbclib_v2.test_util import check_close, TestSetup, TestModel, TestState

class TestCBDIndexer():
    EPS : float = 1e-7

    @pytest.fixture
    def xp(self) -> ArrayNamespace:
        return NumPy

    @pytest.fixture
    def state(self, xp: ArrayNamespace) -> TestState:
        return TestState(TestSetup.lens(xp), TestSetup.xtal(xp), TestSetup.z(xp))

    @pytest.fixture
    def int_state(self, model: TestModel, state: TestState) -> cbc.jax.InternalState:
        return model.to_internal(state)

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
    def indexer(self) -> cbc.jax.CBDIndexer:
        return cbc.jax.CBDIndexer()

    @pytest.fixture
    def points(self, indexer: cbc.jax.CBDIndexer, patterns: cbc.jax.Patterns,
               int_state: cbc.jax.InternalState, xp: ArrayNamespace) -> cbc.jax.PointsWithK:
        return indexer.points_to_kout(patterns.sample(xp.full(patterns.shape[0], 0.5)), int_state)

    @pytest.fixture
    def all_rlp(self, indexer: cbc.jax.CBDIndexer, q_abs: float, int_state: cbc.jax.InternalState,
                xp: ArrayNamespace) -> cbc.jax.MillerWithRLP:
        rlp = indexer.xtal.miller_in_ball(q_abs, int_state.xtal, xp)
        return indexer.xtal.hkl_to_q(rlp, int_state.xtal)

    @pytest.fixture
    def patterns_uca(self, indexer: cbc.jax.CBDIndexer, patterns: cbc.jax.Patterns,
                     points: cbc.jax.PointsWithK, int_state: cbc.jax.InternalState) -> cbc.jax.UCA:
        return indexer.patterns_to_uca(patterns, points, int_state)

    @pytest.fixture
    def candidates(self, indexer: cbc.jax.CBDIndexer, all_rlp: cbc.jax.MillerWithRLP,
                   patterns_uca: cbc.jax.UCA) -> cbc.jax.MillerWithRLP:
        return indexer.candidates(all_rlp, patterns_uca)[0]

    @pytest.fixture
    def uca(self, indexer: cbc.jax.CBDIndexer, all_rlp: cbc.jax.MillerWithRLP,
            patterns_uca: cbc.jax.UCA) -> cbc.jax.UCA:
        return indexer.candidates(all_rlp, patterns_uca)[1]

    @pytest.fixture
    def data(self, key: KeyArray, patterns: cbc.jax.Patterns, model: TestModel,
             int_state: cbc.jax.InternalState, num_points: int) -> cbc.jax.CBData:
        return model.init_data(key, patterns, num_points, int_state)

    @pytest.fixture
    def pupil_loss(self, model: TestModel, num_lines: int) -> cbc.jax.CBDLoss:
        return model.pupil_loss(num_lines)

    @pytest.fixture
    def solution(self, model: TestModel, pupil_loss: cbc.jax.CBDLoss, data: cbc.jax.CBData,
                 int_state: cbc.jax.InternalState) -> cbc.jax.MillerWithRLP:
        miller = pupil_loss.index(data, int_state)
        return model.xtal.hkl_to_q(miller, int_state.xtal)

    def test_indexing_candidates(self, patterns_uca: cbc.jax.UCA, solution: cbc.jax.MillerWithRLP,
                                 points: cbc.jax.PointsWithK, candidates: cbc.jax.MillerWithRLP,
                                 uca: cbc.jax.UCA, xp: ArrayNamespace):
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
    def circles(self, indexer: cbc.jax.CBDIndexer, candidates: cbc.jax.MillerWithRLP,
                uca: cbc.jax.UCA) -> cbc.jax.CircleState:
        return indexer.intersection(candidates, uca)

    def test_intersection(self, indexer: cbc.jax.CBDIndexer, circles: cbc.jax.CircleState,
                          candidates: cbc.jax.MillerWithRLP, uca: cbc.jax.UCA, xp: ArrayNamespace):
        theta = xp.linspace(0, 2 * xp.pi)[:, None]
        pts = indexer.circle(theta, circles)
        resolution = xp.broadcast_to(xp.sum(candidates.q**2, axis=-1), pts.shape[:-1])
        check_close(xp.sum(pts**2, axis=-1), resolution)
        check_close(xp.sum((pts - uca.kout)**2, axis=-1), xp.ones(pts.shape[:-1]))

    @pytest.fixture
    def endpoints(self, indexer: cbc.jax.CBDIndexer, circles: cbc.jax.CircleState, uca: cbc.jax.UCA
                  ) -> RealArray:
        return indexer.uca_endpoints(circles, uca)

    def test_endpoints(self, indexer: cbc.jax.CBDIndexer, circles: cbc.jax.CircleState,
                       endpoints: RealArray, candidates: cbc.jax.MillerWithRLP, uca: cbc.jax.UCA,
                       xp: ArrayNamespace):
        pts = indexer.circle(endpoints, circles)
        check_close(xp.sum(pts**2, axis=-1),
                    xp.broadcast_to(xp.sum(candidates.q**2, axis=-1), pts.shape[:-1]))
        check_close(xp.sum((pts - uca.kout)**2, axis=-1), xp.ones(pts.shape[:-1]))
        proj = cbc.jax.project_to_rect(pts[..., :2], uca.q_min[..., :2], uca.q_max[..., :2])
        check_close(pts[..., :2], proj)

    @pytest.fixture
    def phi(self, xp: ArrayNamespace) -> RealArray:
        return xp.linspace(-2 * xp.pi, 2 * xp.pi, 10)

    @pytest.fixture
    def midpoints(self, indexer: cbc.jax.CBDIndexer, circles: cbc.jax.CircleState,
                  endpoints: RealArray, xp: ArrayNamespace) -> RealArray:
        return indexer.circle(xp.mean(endpoints, axis=0), circles)

    @pytest.fixture
    def tilts(self, indexer: cbc.jax.CBDIndexer, phi: RealArray, candidates: cbc.jax.MillerWithRLP,
              midpoints: RealArray):
        return indexer.rotations(phi, candidates, midpoints)

    def test_rotations(self, tilts: cbc.jax.TiltOverAxisState, candidates: cbc.jax.MillerWithRLP,
                       midpoints: RealArray, xp: ArrayNamespace):
        pts = cbc.jax.TiltOverAxis()(candidates.q[..., None, :], tilts)[..., 0, :]
        check_close(xp.broadcast_to(midpoints, pts.shape), pts)
