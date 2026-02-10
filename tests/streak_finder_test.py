from math import prod
from typing import Tuple
import pytest
from cbclib_v2.annotations import (Generator, NDArray, NDBoolArray, NDIntArray, NDRealArray,
                                   NumPy, NumPyNamespace, Shape)
from cbclib_v2.label import Structure
from cbclib_v2.ndimage import draw_lines
from cbclib_v2.streak_finder import PatternStreakFinder, PeaksList, Streak, Pattern, p_value
from cbclib_v2.test_util import check_close

class TestStreakFinder():
    ATOL: float = 1e-8
    RTOL: float = 1e-5

    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def rng(self, cpu_rng: Generator[NDArray]) -> Generator[NDArray]:
        return cpu_rng

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
        return xp.array([[shape[-1]], [shape[-2]]]) * rng.random((2, n_lines))

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

    @pytest.fixture(params=[0.05])
    def num_bad(self, request: pytest.FixtureRequest, shape: Shape) -> int:
        return int(request.param * prod(shape))

    @pytest.fixture
    def mask(self, rng: Generator[NDArray], shape: Shape, num_bad: int, xp: NumPyNamespace
             ) -> NDBoolArray:
        mask = xp.ones(shape, dtype=bool)
        indices = xp.unravel_index(rng.choice(mask.size, num_bad, replace=False), mask.shape)
        mask[indices] = False
        return mask

    @pytest.fixture(params=[(1, 2)])
    def structure(self, request: pytest.FixtureRequest) -> Structure:
        radius, connectivity = request.param
        return Structure([radius,] * 2, connectivity)

    @pytest.fixture(params=[3])
    def min_size(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def finder(self, image: NDRealArray, mask: NDBoolArray, structure: Structure,
               min_size: int) -> PatternStreakFinder:
        return PatternStreakFinder(image, mask, structure, min_size)

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
        coords = xp.stack((y, x), axis=-1) + xp.array(list(finder.structure))
        inbound = xp.all(coords >= 0 & (coords < xp.array(finder.mask.shape)), axis=-1)
        coords = coords[inbound]
        mask = finder.mask[coords[:, 0], coords[:, 1]]
        return coords[mask, 1], coords[mask, 0]

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

        pts = xp.concat([xp.stack(self.get_pixels(ctr[0], ctr[1], finder, xp), axis=-1)
                         for ctr in streak.centers])
        pts = xp.unique(pts, axis=0)
        pts = pts[xp.lexsort((pts[:, 1], pts[:, 0]))]
        assert xp.all(xp.stack([streak.x, streak.y], axis=-1) == pts)

    def test_mask(self, result: Pattern, finder: PatternStreakFinder, xp: NumPyNamespace):
        for streak in result:
            assert xp.all(finder.mask[streak.y, streak.x])

    def test_p_values(self, result: Pattern, image: NDRealArray, mask: NDBoolArray, xtol: float,
                      vmin: float, min_size: int, xp: NumPyNamespace):
        p_values, prob = p_value(result, image, mask, xtol, vmin)
        assert xp.all(p_values < xp.log(prob) * min_size)

    def test_result_probability(self, result: Pattern, image: NDRealArray, mask: NDBoolArray,
                                xtol: float, vmin: float, xp: NumPyNamespace):
        _, prob = p_value(result, image, mask, xtol, vmin)
        index = xp.searchsorted(xp.sort(image[mask]), vmin)
        check_close(1 - index / mask.sum(), xp.asarray(prob))

    def test_central_line(self, streak: Streak, image: NDRealArray, xp: NumPyNamespace):
        line = streak.line()
        tau = xp.array(line[2:]) - xp.array(line[:2])
        centers = xp.array(streak.centers)
        center = self.center_of_mass(xp.asarray(streak.x), xp.asarray(streak.y),
                                     image[streak.y, streak.x], xp)
        prods = xp.sum((centers - center) * tau, axis=-1)
        central_line = xp.concat((centers[xp.argmin(prods)], centers[xp.argmax(prods)]))
        assert xp.all(central_line == xp.asarray(streak.central_line()))

    def test_negative_image(self, image: NDRealArray, mask: NDBoolArray, structure: Structure,
                            min_size: int, peaks: PeaksList, xtol: float):
        finder = PatternStreakFinder(-image, mask, structure, min_size)
        streaks = finder.detect_streaks(peaks, xtol, 0.0)[0]
        assert len(streaks) == 0
