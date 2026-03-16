from math import prod
from typing import Tuple
import pytest
from cbclib_v2 import default_rng
from cbclib_v2.annotations import (Generator, NDArray, NDIntArray, NDRealArray, NumPy,
                                   NumPyNamespace, Shape)
from cbclib_v2.label import Structure
from cbclib_v2.ndimage import draw_lines
from cbclib_v2.streak_finder import PatternStreakFinder, PeaksList, Pattern
from cbclib_v2.test_util import check_close, p_value, p0_values, Streak

class TestStreakFinder():
    ATOL: float = 1e-8
    RTOL: float = 1e-5

    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def rng(self, xp: NumPyNamespace) -> Generator[NDArray]:
        return default_rng(42, xp)

    def center_of_mass(self, x: NDIntArray, y: NDIntArray, val: NDRealArray,
                       xp: NumPyNamespace) -> NDRealArray:
        return xp.sum(xp.stack((x, y), axis=-1) * val[..., None], axis=0) / xp.sum(val)

    def covariance_matrix(self, x: NDIntArray, y: NDIntArray, val: NDRealArray, xp: NumPyNamespace
                          ) -> NDRealArray:
        pts = xp.stack((x, y), axis=-1)
        ctr = self.center_of_mass(x, y, val, xp)
        return xp.sum((pts[..., None, :] * pts[..., None] - ctr[None, :] * ctr[:, None]) * \
                      val[..., None, None], axis=0) / xp.sum(val)

    def line(self, x: NDIntArray, y: NDIntArray, val: NDRealArray, xp: NumPyNamespace
             ) -> NDRealArray:
        ctr = self.center_of_mass(x, y, val, xp)
        mat = self.covariance_matrix(x, y, val, xp)
        eigval, eigvec = xp.linalg.eigh(mat)
        return xp.stack((ctr + 2 * xp.sqrt(xp.log(2) * eigval[-1]) * eigvec[-1],
                         ctr - 2 * xp.sqrt(xp.log(2) * eigval[-1]) * eigvec[-1]))

    @pytest.fixture(params=[40])
    def n_lines(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[(80, 100)])
    def shape(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture(params=[15.0])
    def length(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[1.0])
    def width(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def centers(self, rng: Generator[NDArray], n_lines: int, shape: Shape, xp: NumPyNamespace
                ) -> NDRealArray:
        array : NDRealArray = rng.random((2, n_lines))
        return xp.array([[shape[-1]], [shape[-2]]]) * array

    @pytest.fixture
    def lines(self, rng: Generator[NDArray], n_lines: int, centers: NDRealArray,
              length: float, width: float, xp: NumPyNamespace) -> NDRealArray:
        lengths = length * rng.random((n_lines,))
        thetas = 2 * xp.pi * rng.random((n_lines,))
        x0, y0 = centers
        return xp.stack((x0 - 0.5 * lengths * xp.cos(thetas),
                         y0 - 0.5 * lengths * xp.sin(thetas),
                         x0 + 0.5 * lengths * xp.cos(thetas),
                         y0 + 0.5 * lengths * xp.sin(thetas),
                         width * xp.ones(n_lines)), axis=1)

    @pytest.fixture(params=[0.25])
    def noise(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[0.15])
    def vmin(self, request: pytest.FixtureRequest, noise: float) -> float:
        return request.param + noise

    @pytest.fixture(params=[0.8])
    def xtol(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def image(self, rng: Generator[NDArray], lines: NDRealArray, shape: Shape, vmin: float,
              xp: NumPyNamespace) -> NDRealArray:
        noise = 0.25 * vmin * rng.random(shape)
        return draw_lines(xp.zeros(shape), lines, kernel='biweight') + noise

    @pytest.fixture(params=[(1, 2)])
    def structure(self, request: pytest.FixtureRequest) -> Structure:
        radius, connectivity = request.param
        return Structure([radius,] * 2, connectivity)

    @pytest.fixture(params=[3])
    def min_size(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def finder(self, image: NDRealArray, structure: Structure, min_size: int
               ) -> PatternStreakFinder:
        return PatternStreakFinder(image, structure, min_size)

    @pytest.fixture(params=[5])
    def npts(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def peaks(self, finder: PatternStreakFinder, vmin: float, npts: int) -> PeaksList:
        return finder.detect_peaks(vmin, npts)

    @pytest.mark.xfail(raises=IndexError)
    def test_peaks_list(self, peaks: PeaksList):
        return peaks[len(peaks)]

    @pytest.fixture
    def result(self, finder: PatternStreakFinder, peaks: PeaksList, vmin: float, xtol: float
               ) -> Pattern:
        return finder.detect_streaks(peaks, xtol, vmin)[0]

    @pytest.mark.xfail(raises=IndexError)
    def test_streak_list(self, result: Pattern):
        return result[len(result)]

    @pytest.fixture
    def streak(self, rng: Generator[NDArray], result: Pattern) -> Streak:
        index = int(rng.integers(0, len(result)))
        return result[index]

    def get_pixels(self, x: int, y: int, finder: PatternStreakFinder, xp: NumPyNamespace
                   ) -> Tuple[NDIntArray, NDIntArray]:
        shifts : NDIntArray = xp.asarray(list(finder.structure), dtype=int)
        coords = xp.asarray([y, x]) + shifts
        inbound = xp.all(coords >= 0 & (coords < xp.array(finder.data.shape[-2:])), axis=-1)
        coords = coords[inbound]
        return coords[:, 1], coords[:, 0]

    def get_line(self, x: int, y: int, image: NDRealArray, finder: PatternStreakFinder,
                 xp: NumPyNamespace) -> NDRealArray:
        xs, ys = self.get_pixels(x, y, finder, xp)
        return self.line(xs, ys, image[ys, xs], xp)

    def test_streak_points(self, streak: Streak, image: NDRealArray, finder: PatternStreakFinder,
                           xp: NumPyNamespace):
        ends = xp.stack([self.get_line(ctr[0], ctr[1], image, finder, xp)
                         for ctr in streak.centers])
        streak_ends = xp.array(streak.ends).reshape((-1, 2, 2))
        check_close(xp.sort(ends, axis=-2), xp.sort(streak_ends, axis=-2))

        indices = []
        for center in streak.centers:
            x, y = self.get_pixels(center[0], center[1], finder, xp)
            indices.append(xp.ravel_multi_index((y, x), finder.data.shape[-2:]))

        indices = xp.unique(xp.concat(indices), axis=0)
        assert xp.all(xp.array(list(streak.region)) == indices)

    @pytest.fixture
    def p0(self, image: NDRealArray, vmin: float) -> NDRealArray:
        return p0_values(image, vmin)

    def test_p0_values(self, image: NDRealArray, vmin: float, p0: NDRealArray, xp: NumPyNamespace):
        n_signal = xp.sum(image >= vmin, axis=(-2, -1))
        check_close(p0, n_signal / prod(image.shape[-2:]))

    def test_p_values(self, result: Pattern, image: NDRealArray, p0: NDRealArray, xtol: float,
                      vmin: float, min_size: int, xp: NumPyNamespace):
        p_values = p_value(result, image, float(p0.item()), xtol, vmin)
        assert xp.all(p_values < xp.log(p0) * min_size)

    def test_central_line(self, streak: Streak, image: NDRealArray, xp: NumPyNamespace):
        line = streak.line()
        tau = xp.array(line[2:]) - xp.array(line[:2])
        centers = xp.array(streak.centers)
        indices = xp.array(list(streak.region))
        y, x = xp.unravel_index(indices, image.shape[-2:])
        center = self.center_of_mass(x, y, image[y, x], xp)
        prods = xp.sum((centers - center) * tau, axis=-1)
        central_line = xp.concat((centers[xp.argmin(prods)], centers[xp.argmax(prods)]))
        assert xp.all(central_line == xp.asarray(streak.central_line()))

    def test_negative_image(self, image: NDRealArray, structure: Structure,
                            min_size: int, peaks: PeaksList, xtol: float):
        finder = PatternStreakFinder(-image, structure, min_size)
        patterns = list(finder.detect_streaks(peaks, xtol, 0.0))
        assert len(patterns[0]) == 0
