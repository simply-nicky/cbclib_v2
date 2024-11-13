import numpy as np
import jax.numpy as jnp
import pytest
from jax import random
import cbclib_v2 as cbc
from cbclib_v2.annotations import KeyArray
from cbclib_v2.test_util import TestModel, check_close

class TestCBDSetup():
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
