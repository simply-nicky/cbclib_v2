from typing import Callable
import numpy as np
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads
from cbclib_v2.jax import (euler_angles, euler_matrix, tilt_angles, tilt_matrix, det_to_k,
                           k_to_det, k_to_smp)
from cbclib_v2.annotations import JaxIntArray, JaxRealArray, Shape

class TestJaxPrimitives():
    Generator = Callable[[], JaxRealArray]

    @pytest.fixture(params=[(5,), (5, 5)])
    def n_samples(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture
    def src(self, rng: np.random.Generator) -> JaxRealArray:
        return jnp.asarray(rng.random(3))

    @pytest.fixture
    def generate_mats(self, rng: np.random.Generator, n_samples: Shape) -> Generator:
        return lambda : jnp.asarray(rng.random(n_samples + (3, 3)))

    @pytest.fixture
    def generate_vecs(self, rng: np.random.Generator, n_samples: Shape) -> Generator:
        return lambda : jnp.asarray(rng.random(n_samples + (3,)))

    @pytest.fixture
    def generate_coords(self, rng: np.random.Generator, n_samples: Shape) -> Generator:
        return lambda : jnp.asarray(rng.random(n_samples))

    @pytest.fixture
    def idxs(self, n_samples: Shape) -> JaxIntArray:
        return jnp.reshape(jnp.arange(np.prod(n_samples)), n_samples)

    def check_gradient(self, f, args, **static_args):
        def wrapper(*args):
            return f(*args, **static_args)
        check_grads(wrapper, args, order=1, modes='rev', atol=1e-1, rtol=1e-3)

    def test_euler_angles(self, generate_mats: Generator):
        self.check_gradient(euler_angles, (generate_mats(),))

    def test_euler_matrix(self, generate_vecs: Generator):
        self.check_gradient(euler_matrix, (generate_vecs(),))

    def test_tilt_angles(self, generate_mats: Generator):
        self.check_gradient(tilt_angles, (generate_mats(),))

    def test_tilt_matrix(self, generate_vecs: Generator):
        self.check_gradient(tilt_matrix, (generate_vecs(),))

    def test_det_to_k(self, generate_coords: Generator, generate_vecs: Generator,
                      idxs: JaxIntArray):
        self.check_gradient(det_to_k, (generate_coords(), generate_coords(),
                                       generate_vecs()), idxs=idxs)

    def test_k_to_det(self, generate_vecs: Generator, idxs: JaxIntArray):
        self.check_gradient(k_to_det, (generate_vecs(), generate_vecs()), idxs=idxs)

    def test_k_to_smp(self, generate_coords: Generator, generate_vecs: Generator,
                      src: JaxRealArray, idxs: JaxIntArray):
        self.check_gradient(k_to_smp, (generate_vecs(), generate_coords(), src),
                            idxs=idxs)
