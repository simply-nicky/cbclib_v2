import pytest
from cbclib_v2 import default_rng
from cbclib_v2.annotations import Generator, NDArray, NumPyNamespace, NumPy, RealArray
from cbclib_v2.indexer import (CBDModel, CBDPoints, MaskedLaueVectors, Miller, MillerWithRLP,
                               FixedApertureState, FixedPupilState, FixedState, RotationState,
                               ResolvedState, XtalCell, XtalState)
from cbclib_v2.test_util import TestSetup, check_close

class TestCBDSetup():
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

    def skew_symmetric(self, vec: RealArray, xp: NumPyNamespace) -> RealArray:
        return xp.linalg.cross(xp.eye(vec.shape[-1]), vec[..., None, :])

    def rodriguez_formula(self, angles: RealArray, xp: NumPyNamespace) -> RealArray:
        axis = xp.stack([xp.sin(angles[..., 1]) * xp.cos(angles[..., 2]),
                         xp.sin(angles[..., 1]) * xp.sin(angles[..., 2]),
                         xp.cos(angles[..., 1])], axis=-1)
        skew = self.skew_symmetric(axis, xp)
        I = 1.0 * xp.broadcast_to(xp.eye(skew.shape[-1]), skew.shape)
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
    def miller(self, rng: Generator[NDArray], q_abs: float, num_points: int, model: CBDModel,
               resolved: ResolvedState, xp: NumPyNamespace) -> Miller:
        miller = model.hkl_in_aperture(q_abs, resolved, xp)
        idxs = rng.choice(miller.hkl.shape[0], size=(num_points,))
        return miller[idxs]

    @pytest.fixture
    def rlp(self, miller: Miller, model: CBDModel,
            state: FixedState, xp: NumPyNamespace) -> MillerWithRLP:
        return model.xtal.hkl_to_q(miller, state.xtal, xp)

    @pytest.fixture
    def laue(self, rlp: MillerWithRLP, model: CBDModel, resolved: ResolvedState,
             xp: NumPyNamespace) -> MaskedLaueVectors:
        return model.lens.source_lines(rlp, resolved.setup.lens, xp)

    @pytest.fixture
    def points(self, laue: MaskedLaueVectors, model: CBDModel, resolved: ResolvedState,
               xp: NumPyNamespace) -> CBDPoints:
        return model.kout_to_points(laue, resolved.setup, xp)

    def text_xtal_to_cell(self, xtal: XtalState, ormatrix: RotationState,
                          cell: XtalCell, xp: NumPyNamespace):
        basis = cell.to_basis()
        check_close(xp.linalg.det(ormatrix.matrix), xp.array(1.0))
        check_close(ormatrix @ basis.basis, xtal.basis)

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
                       model: CBDModel, state: FixedState, xp: NumPyNamespace):
        rlp = model.xtal.q_to_hkl(rlp, state.xtal, xp)
        assert xp.all(rlp.hkl_indices == miller.hkl_indices)

    def test_laue(self, laue: MaskedLaueVectors, model: CBDModel,
                  resolved: ResolvedState, xp: NumPyNamespace):
        check_close(xp.broadcast_to(laue.q, laue.kout.shape), laue.kout - laue.kin)
        valid = xp.broadcast_to(laue.mask, laue.kin.shape[:-1])
        kin = model.lens.project_to_pupil(laue.kin, laue.index, resolved.setup.lens, xp)
        check_close(laue.kin[valid], kin[valid])

    def test_points_and_kout(self, laue: MaskedLaueVectors, points: CBDPoints,
                             model: CBDModel, resolved: ResolvedState, xp: NumPyNamespace):
        valid = xp.broadcast_to(laue.mask, laue.kout.shape[:-1])
        kout = model.points_to_kout(points, resolved.setup, xp).kout
        check_close(kout[valid], laue.kout[valid])

    def test_rotation_to_tilt(self, ormatrix: RotationState, xp: NumPyNamespace):
        tilt = ormatrix.to_tilt()
        check_close(tilt.to_rotation().matrix, self.rodriguez_formula(tilt.angles, xp))
        check_close(ormatrix.matrix, tilt.to_rotation().matrix)

    def test_rotation_to_tilt_over_axis(self, ormatrix: RotationState):
        tilt_over_axis = ormatrix.to_tilt().to_tilt_over_axis()
        check_close(ormatrix.matrix, tilt_over_axis.to_tilt().to_rotation().matrix)

    def test_fixed_pupil_state_from_resolved(self, resolved: ResolvedState, xp: NumPyNamespace):
        restored = FixedPupilState.from_resolved(resolved)
        converted = restored.resolve(xp)

        check_close(converted.xtal.basis, resolved.xtal.basis)
        check_close(converted.setup.lens.foc_pos, resolved.setup.lens.foc_pos)
        check_close(converted.setup.lens.pupil_roi, resolved.setup.lens.pupil_roi)
        check_close(converted.setup.z, resolved.setup.z)

    def test_fixed_aperture_state_from_resolved(self, resolved: ResolvedState, xp: NumPyNamespace):
        restored = FixedApertureState.from_resolved(resolved)
        converted = restored.resolve(xp)

        check_close(converted.xtal.basis, resolved.xtal.basis)
        check_close(converted.setup.lens.foc_pos, resolved.setup.lens.foc_pos)
        check_close(converted.setup.lens.pupil_roi, resolved.setup.lens.pupil_roi)
        check_close(converted.setup.z, resolved.setup.z)
