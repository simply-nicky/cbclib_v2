from math import comb
from typing import Tuple
import pytest
from jax import jacfwd, jit
from numpy.polynomial.legendre import leggauss
from cbclib_v2.annotations import AnyNamespace, JaxNumPy, NumPy, RealArray
from cbclib_v2.scaler import LegendreBasis, LegendreFactory
from cbclib_v2.test_util import check_close

class TestLegendreBasis:
    @pytest.fixture(params=[NumPy, JaxNumPy], ids=['numpy', 'jax'])
    def xp(self, request: pytest.FixtureRequest) -> AnyNamespace:
        return request.param

    @pytest.fixture(params=[0, 2, 4])
    def basis(self, request: pytest.FixtureRequest,
              xp: AnyNamespace) -> LegendreBasis:
        return LegendreFactory.create(request.param, xp)

    @pytest.fixture
    def quadrature(self, basis: LegendreBasis, xp: AnyNamespace
                   ) -> Tuple[RealArray, RealArray]:
        nodes, weights = leggauss(basis.degree + 1)
        nodes = xp.asarray((nodes + 1.0) / 2.0)
        weights = xp.asarray(weights / 2.0)
        u, v = xp.meshgrid(nodes, nodes, indexing='ij')
        points = xp.stack((u, v), axis=-1).reshape((-1, 2))
        area_weights = xp.asarray(weights[:, None] * weights[None, :]).reshape(-1)
        return points, area_weights

    def test_orthonormality(self, basis: LegendreBasis,
                           quadrature: Tuple[RealArray, RealArray], xp: AnyNamespace) -> None:
        points, weights = quadrature
        values = basis.evaluate(points)
        # Gauss quadrature integrates every product of these modes exactly.
        gram = values.T @ (weights[:, None] * values)
        check_close(gram, xp.eye(len(basis.modes)))

    def test_low_degree_modes(self, xp: AnyNamespace) -> None:
        points = xp.asarray([[0.0, 0.5], [0.25, 0.75], [1.0, 1.0]])
        u, v = 2.0 * points[:, 0] - 1.0, 2.0 * points[:, 1] - 1.0
        expected = xp.stack((xp.ones_like(u), 3.0**0.5 * u, 3.0**0.5 * v,
                             5.0**0.5 * (3.0 * u**2 - 1.0) / 2.0, 3.0 * u * v,
                             5.0**0.5 * (3.0 * v**2 - 1.0) / 2.0), axis=-1)
        check_close(LegendreFactory.create(2, xp).evaluate(points), expected)

    def test_reuse(self, basis: LegendreBasis,
                   quadrature: Tuple[RealArray, RealArray], xp: AnyNamespace) -> None:
        points, _ = quadrature
        # Reusing one basis for individual points preserves the batched evaluation.
        values = xp.stack(tuple(basis.evaluate(point) for point in points), axis=0)
        check_close(values, basis.evaluate(points))

    @pytest.mark.parametrize('coordinates', [(0.25, 0.75), (0.0, 1.0)])
    def test_coordinate_gradient(self, coordinates: Tuple[float, float]) -> None:
        xp = JaxNumPy
        point = xp.asarray(coordinates)
        basis = LegendreFactory.create(1, xp)
        gradient = jit(jacfwd(lambda points, basis: basis.evaluate(points)))(point, basis)
        # Shifted linear modes have constant derivatives of twice their normalisation.
        expected = xp.asarray([[0.0, 0.0], [2.0 * 3.0**0.5, 0.0],
                               [0.0, 2.0 * 3.0**0.5]])
        check_close(gradient, expected)

class TestLegendreFactory:
    @pytest.mark.parametrize('degree', [0, 1, 2, 3, 4, 5, 12, 25])
    def test_exact_coefficients(self, degree: int) -> None:
        # Shifted Legendre coefficients obey the exact binomial formula.
        expected = tuple((-1)**(degree + k) * comb(degree, k) * comb(degree + k, k)
                         for k in range(degree + 1))
        assert LegendreFactory.coefficients(degree) == expected

    def test_shared_cache(self) -> None:
        first = LegendreFactory()
        second = LegendreFactory()
        row = first.coefficients(7)
        second.coefficients(9)
        # Extending the global cache retains the existing exact coefficient rows.
        assert second.coefficients(7) is row
