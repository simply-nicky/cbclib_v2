import numpy as np
import jax.numpy as jnp
import pytest
from jax import random
import cbclib_v2 as cbc
from cbclib_v2.annotations import KeyArray, RealArray
from cbclib_v2.test_util import TestModel, check_close

class TestCBDSetup():
    def skew_symmetric(self, vec: RealArray) -> RealArray:
        return jnp.cross(jnp.identity(vec.shape[-1]), vec[..., None, :])

    def rodriguez_formula(self, angles: RealArray) -> RealArray:
        axis = jnp.stack([jnp.sin(angles[..., 1]) * jnp.cos(angles[..., 2]),
                          jnp.sin(angles[..., 1]) * jnp.sin(angles[..., 2]),
                          jnp.cos(angles[..., 1])], axis=-1)
        skew = self.skew_symmetric(axis)
        I = jnp.broadcast_to(jnp.identity(skew.shape[-1]), skew.shape)
        S = jnp.sin(angles[..., 0])[..., None, None]
        C = jnp.cos(angles[..., 0])[..., None, None]
        return I + S * skew + (1 - C) * (skew @ skew)

    @pytest.fixture
    def xtal(self, int_state: cbc.jax.InternalState) -> cbc.jax.XtalState:
        return int_state.xtal

    @pytest.fixture
    def ormatrix(self, xtal: cbc.jax.XtalState) -> cbc.jax.RotationState:
        return xtal.orientation_matrix()

    @pytest.fixture
    def cell(self, xtal: cbc.jax.XtalState) -> cbc.jax.XtalCell:
        return xtal.lattice_constants()

    @pytest.fixture(params=[0.3,])
    def q_abs(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[200,])
    def num_points(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def miller(self, key: KeyArray, q_abs: float, num_points: int, model: TestModel,
               int_state: cbc.jax.InternalState) -> cbc.jax.Miller:
        miller = model.hkl_in_aperture(q_abs, int_state)
        return miller.filter(random.choice(key, miller.hkl.shape[0], (num_points,)))

    @pytest.fixture
    def rlp(self, miller: cbc.jax.Miller, model: TestModel,
            int_state: cbc.jax.InternalState) -> cbc.jax.MillerWithRLP:
        return model.xtal.hkl_to_q(miller, int_state.xtal)

    @pytest.fixture
    def laue(self, rlp: cbc.jax.MillerWithRLP, model: TestModel,
             int_state: cbc.jax.InternalState) -> cbc.jax.LaueVectors:
        return model.lens.source_lines(rlp, int_state.lens)

    @pytest.fixture
    def points(self, laue: cbc.jax.LaueVectors, model: TestModel,
               int_state: cbc.jax.InternalState) -> cbc.jax.CBDPoints:
        return model.kout_to_points(laue, int_state)

    def text_xtal_to_cell(self, xtal: cbc.jax.XtalState, ormatrix: cbc.jax.RotationState,
                          cell: cbc.jax.XtalCell):
        basis = cell.to_basis()
        check_close(jnp.linalg.det(ormatrix.matrix), jnp.array(1.0))
        check_close(cbc.jax.Rotation().apply(basis, ormatrix).basis, xtal.basis)

    def test_xtal_to_spherical(self, xtal: cbc.jax.XtalState):
        r, theta, phi = xtal.to_spherical()
        basis = cbc.jax.XtalState.import_spherical(r, theta, phi).basis
        check_close(xtal.basis, basis)

    def text_reciprocate_xtal(self, xtal: cbc.jax.XtalState):
        check_close(xtal.basis, xtal.reciprocate().reciprocate().basis)

    def test_cell_to_xtal(self, cell: cbc.jax.XtalCell):
        new_cell = cell.to_basis().lattice_constants()
        check_close(cell.angles, new_cell.angles)
        check_close(cell.lengths, new_cell.lengths)

    def test_hkl_and_q(self, miller: cbc.jax.Miller, rlp: cbc.jax.MillerWithRLP,
                       model: TestModel, int_state: cbc.jax.InternalState):
        rlp = model.xtal.q_to_hkl(rlp, int_state.xtal)
        assert np.all(rlp.hkl_indices == miller.hkl_indices)

    def test_laue(self, laue: cbc.jax.LaueVectors, model: TestModel,
                  int_state: cbc.jax.InternalState):
        check_close(jnp.broadcast_to(laue.q, laue.kout.shape), laue.kout - laue.kin)
        check_close(laue.kin, model.lens.project_to_pupil(laue.kin, int_state.lens))

    def test_points_and_kout(self, laue: cbc.jax.LaueVectors, points: cbc.jax.CBDPoints,
                             model: TestModel, int_state: cbc.jax.InternalState):
        check_close(model.points_to_kout(points, int_state).kout, laue.kout)

    def test_rotation_to_tilt(self, ormatrix: cbc.jax.RotationState):
        tilt = ormatrix.to_tilt()
        check_close(tilt.to_rotation().matrix, self.rodriguez_formula(tilt.angles))
        check_close(ormatrix.matrix, tilt.to_rotation().matrix)

    def test_rotation_to_tilt_over_axis(self, ormatrix: cbc.jax.RotationState):
        tilt_over_axis = ormatrix.to_tilt().to_tilt_over_axis()
        check_close(ormatrix.matrix, tilt_over_axis.to_tilt().to_rotation().matrix)
