import pytest
from jax import random
from cbclib_v2.annotations import ArrayNamespace, KeyArray, NumPy, RealArray
from cbclib_v2.indexer import (CBDModel, CBDPoints, LaueVectors, Miller, MillerWithRLP, Rotation,
                               RotationState, XtalCell, XtalState)
from cbclib_v2.test_util import FixedState, TestSetup, check_close

class TestCBDSetup():
    @pytest.fixture
    def xp(self) -> ArrayNamespace:
        return NumPy

    @pytest.fixture
    def state(self, xp: ArrayNamespace) -> FixedState:
        return FixedState(TestSetup.xtal(xp))

    def skew_symmetric(self, vec: RealArray, xp: ArrayNamespace) -> RealArray:
        return xp.cross(xp.identity(vec.shape[-1]), vec[..., None, :])

    def rodriguez_formula(self, angles: RealArray, xp: ArrayNamespace) -> RealArray:
        axis = xp.stack([xp.sin(angles[..., 1]) * xp.cos(angles[..., 2]),
                         xp.sin(angles[..., 1]) * xp.sin(angles[..., 2]),
                         xp.cos(angles[..., 1])], axis=-1)
        skew = self.skew_symmetric(axis, xp)
        I = 1.0 * xp.broadcast_to(xp.identity(skew.shape[-1]), skew.shape)
        S = xp.sin(angles[..., 0])[..., None, None]
        C = xp.cos(angles[..., 0])[..., None, None]
        return I + S * skew + (1.0 - C) * (skew @ skew)

    @pytest.fixture
    def xtal(self, state: FixedState) -> XtalState:
        return state.xtal

    @pytest.fixture
    def ormatrix(self, xtal: XtalState) -> RotationState:
        return xtal.orientation_matrix

    @pytest.fixture
    def cell(self, xtal: XtalState) -> XtalCell:
        return xtal.unit_cell

    @pytest.fixture(params=[0.3,])
    def q_abs(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[200,])
    def num_points(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def miller(self, key: KeyArray, q_abs: float, num_points: int, model: CBDModel,
               state: FixedState) -> Miller:
        miller = model.hkl_in_aperture(q_abs, state)
        return miller[random.choice(key, miller.hkl.shape[0], (num_points,))]

    @pytest.fixture
    def rlp(self, miller: Miller, model: CBDModel,
            state: FixedState, xp: ArrayNamespace) -> MillerWithRLP:
        return model.xtal.hkl_to_q(miller, state.xtal, xp)

    @pytest.fixture
    def laue(self, rlp: MillerWithRLP, model: CBDModel, state: FixedState,
             xp: ArrayNamespace) -> LaueVectors:
        return model.lens.source_lines(rlp, state.lens, xp)

    @pytest.fixture
    def points(self, laue: LaueVectors, model: CBDModel, state: FixedState,
               xp: ArrayNamespace) -> CBDPoints:
        return model.kout_to_points(laue, state, xp)

    def text_xtal_to_cell(self, xtal: XtalState, ormatrix: RotationState,
                          cell: XtalCell, xp: ArrayNamespace):
        basis = cell.to_basis()
        check_close(xp.linalg.det(ormatrix.matrix), xp.array(1.0))
        check_close(Rotation()(basis.basis, ormatrix), xtal.basis)

    def test_xtal_to_spherical(self, xtal: XtalState):
        r, theta, phi = xtal.to_spherical()
        basis = XtalState.import_spherical(r, theta, phi).basis
        check_close(xtal.basis, basis)

    def text_reciprocate_xtal(self, xtal: XtalState):
        check_close(xtal.basis, xtal.reciprocate().reciprocate().basis)

    def test_cell_to_xtal(self, cell: XtalCell):
        new_cell = cell.to_basis().unit_cell
        check_close(cell.angles, new_cell.angles)
        check_close(cell.lengths, new_cell.lengths)

    def test_hkl_and_q(self, miller: Miller, rlp: MillerWithRLP,
                       model: CBDModel, state: FixedState, xp: ArrayNamespace):
        rlp = model.xtal.q_to_hkl(rlp, state.xtal, xp)
        assert xp.all(rlp.hkl_indices == miller.hkl_indices)

    def test_laue(self, laue: LaueVectors, model: CBDModel,
                  state: FixedState, xp: ArrayNamespace):
        check_close(xp.broadcast_to(laue.q, laue.kout.shape), laue.kout - laue.kin)
        check_close(laue.kin, model.lens.project_to_pupil(laue.kin, state.lens, xp))

    def test_points_and_kout(self, laue: LaueVectors, points: CBDPoints, model: CBDModel,
                             state: FixedState, xp: ArrayNamespace):
        check_close(model.points_to_kout(points, state, xp).kout, laue.kout)

    def test_rotation_to_tilt(self, ormatrix: RotationState, xp: ArrayNamespace):
        tilt = ormatrix.to_tilt()
        check_close(tilt.to_rotation().matrix, self.rodriguez_formula(tilt.angles, xp))
        check_close(ormatrix.matrix, tilt.to_rotation().matrix)

    def test_rotation_to_tilt_over_axis(self, ormatrix: RotationState):
        tilt_over_axis = ormatrix.to_tilt().to_tilt_over_axis()
        check_close(ormatrix.matrix, tilt_over_axis.to_tilt().to_rotation().matrix)
