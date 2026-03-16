from math import prod
from typing import Callable
import pytest
from cbclib_v2 import default_rng
from cbclib_v2.indexer import (euler_angles, euler_matrix, tilt_angles, tilt_matrix, det_to_k,
                               k_to_det, k_to_smp)
from cbclib_v2.annotations import (Generator, JaxNamespace, IntArray, JaxArray, JaxNumPy, RealArray,
                                   Shape)
from cbclib_v2.test_util import check_gradient

class TestJaxPrimitives():
    GenFunc = Callable[[], RealArray]

    @pytest.fixture
    def xp(self) -> JaxNamespace:
        return JaxNumPy

    @pytest.fixture
    def rng(self, xp: JaxNamespace) -> Generator[JaxArray]:
        return default_rng(42, xp)

    @pytest.fixture(params=[(5,), (5, 5)])
    def n_samples(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture
    def src(self, rng: Generator, xp: JaxNamespace) -> RealArray:
        return xp.asarray(rng.random(3))

    @pytest.fixture
    def generate_mats(self, rng: Generator, n_samples: Shape, xp: JaxNamespace) -> GenFunc:
        return lambda : xp.asarray(rng.random(n_samples + (3, 3)))

    @pytest.fixture
    def generate_vecs(self, rng: Generator, n_samples: Shape, xp: JaxNamespace) -> GenFunc:
        return lambda : xp.asarray(rng.random(n_samples + (3,)))

    @pytest.fixture
    def generate_coords(self, rng: Generator, n_samples: Shape, xp: JaxNamespace) -> GenFunc:
        return lambda : xp.asarray(rng.random(n_samples))

    @pytest.fixture
    def idxs(self, n_samples: Shape, xp: JaxNamespace) -> IntArray:
        return xp.reshape(xp.arange(prod(n_samples)), n_samples)

    def check_gradient(self, f, args, **static_args):
        check_gradient(f, args, atol=1e-1, rtol=1e-3, **static_args)

    def test_euler_angles(self, generate_mats: GenFunc, xp: JaxNamespace):
        self.check_gradient(euler_angles, (generate_mats(),), xp=xp)

    def test_euler_matrix(self, generate_vecs: GenFunc, xp: JaxNamespace):
        self.check_gradient(euler_matrix, (generate_vecs(),), xp=xp)

    def test_tilt_angles(self, generate_mats: GenFunc, xp: JaxNamespace):
        self.check_gradient(tilt_angles, (generate_mats(),), xp=xp)

    def test_tilt_matrix(self, generate_vecs: GenFunc, xp: JaxNamespace):
        self.check_gradient(tilt_matrix, (generate_vecs(),), xp=xp)

    def test_det_to_k(self, generate_coords: GenFunc, generate_vecs: GenFunc, idxs: IntArray,
                      xp: JaxNamespace):
        xy = xp.stack((generate_coords(), generate_coords()), axis=-1)
        self.check_gradient(det_to_k, (xy, generate_vecs()), idxs=idxs, xp=xp)

    def test_k_to_det(self, generate_vecs: GenFunc, idxs: IntArray, xp: JaxNamespace):
        self.check_gradient(k_to_det, (generate_vecs(), generate_vecs()), idxs=idxs, xp=xp)

    def test_k_to_smp(self, generate_coords: GenFunc, generate_vecs: GenFunc, src: RealArray,
                      idxs: IntArray, xp: JaxNamespace):
        z = generate_coords()
        z = xp.reshape(xp.reshape(z, -1)[xp.reshape(idxs, -1)], idxs.shape)
        self.check_gradient(k_to_smp, (generate_vecs(), z, src), xp=xp)
