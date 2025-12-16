from math import prod
from typing import Tuple
import numpy as np
import pytest
from cbclib_v2.annotations import NDBoolArray, NDIntArray, NDRealArray, Shape
from cbclib_v2.label import Structure2D
from cbclib_v2.ndimage import draw_lines
from cbclib_v2.streak_finder import PatternStreakFinder, PeaksList, Streak, Pattern, p_value
from cbclib_v2.test_util import check_close

class TestStreakFinder():
    ATOL: float = 1e-8
    RTOL: float = 1e-5

    def center_of_mass(self, x: NDIntArray, y: NDIntArray, val: NDRealArray) -> NDRealArray:
        return np.sum(np.stack((x, y), axis=-1) * val[..., None], axis=0) / np.sum(val)

    def covariance_matrix(self, x: NDIntArray, y: NDIntArray, val: NDRealArray) -> NDRealArray:
        pts = np.stack((x, y), axis=-1)
        ctr = self.center_of_mass(x, y, val)
        return np.sum((pts[..., None, :] * pts[..., None] - ctr[None, :] * ctr[:, None]) * \
                      val[..., None, None], axis=0) / np.sum(val)

    def line(self, x: NDIntArray, y: NDIntArray, val: NDRealArray) -> NDRealArray:
        ctr = self.center_of_mass(x, y, val)
        mat = self.covariance_matrix(x, y, val)
        eigval, eigvec = np.linalg.eigh(mat)
        return np.stack((ctr + 2 * np.sqrt(np.log(2) * eigval[-1]) * eigvec[-1],
                         ctr - 2 * np.sqrt(np.log(2) * eigval[-1]) * eigvec[-1]))

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
    def centers(self, rng: np.random.Generator, n_lines: int, shape: Shape) -> NDRealArray:
        return np.array([[shape[-1]], [shape[-2]]]) * rng.random((2, n_lines))

    @pytest.fixture
    def lines(self, rng: np.random.Generator, n_lines: int, centers: NDRealArray,
              length: float, width: float) -> NDRealArray:
        lengths = length * rng.random((n_lines,))
        thetas = 2 * np.pi * rng.random((n_lines,))
        x0, y0 = centers
        return np.stack((x0 - 0.5 * lengths * np.cos(thetas),
                         y0 - 0.5 * lengths * np.sin(thetas),
                         x0 + 0.5 * lengths * np.cos(thetas),
                         y0 + 0.5 * lengths * np.sin(thetas),
                         width * np.ones(n_lines)), axis=1)

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
    def image(self, lines: NDRealArray, shape: Shape, vmin: float,
              rng: np.random.Generator) -> NDRealArray:
        noise = 0.25 * vmin * rng.random(shape)
        return draw_lines(lines, shape, kernel='biweight') + noise

    @pytest.fixture(params=[0.05])
    def num_bad(self, request: pytest.FixtureRequest, shape: Shape) -> int:
        return int(request.param * prod(shape))

    @pytest.fixture
    def mask(self, shape: Shape, num_bad: int, rng: np.random.Generator) -> NDBoolArray:
        mask = np.ones(shape, dtype=bool)
        indices = np.unravel_index(rng.choice(mask.size, num_bad, replace=False), mask.shape)
        mask[indices] = False
        return mask

    @pytest.fixture(params=[(1, 2)])
    def structure(self, request: pytest.FixtureRequest) -> Structure2D:
        radius, rank = request.param
        return Structure2D(radius, rank)

    @pytest.fixture(params=[3])
    def min_size(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def finder(self, image: NDRealArray, mask: NDBoolArray, structure: Structure2D,
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
    def streak(self, rng: np.random.Generator, result: Pattern) -> Streak:
        index = int(rng.integers(0, len(result)))
        return result[index]

    def get_pixels(self, x: int, y: int, finder: PatternStreakFinder
                   ) -> Tuple[NDIntArray, NDIntArray]:
        xs, ys = np.array(finder.structure.x) + x, np.array(finder.structure.y) + y
        is_inbound = (xs >= 0) & (xs < finder.mask.shape[1]) & (ys >= 0) & (ys < finder.mask.shape[0])
        xs, ys = xs[is_inbound], ys[is_inbound]
        mask = finder.mask[ys, xs]
        return xs[mask], ys[mask]

    def get_line(self, x: int, y: int, image: NDRealArray, finder: PatternStreakFinder
                 ) -> NDRealArray:
        xs, ys = self.get_pixels(x, y, finder)
        return self.line(xs, ys, image[ys, xs])

    def test_streak_points(self, streak: Streak, image: NDRealArray, finder: PatternStreakFinder):
        ends = np.stack([self.get_line(ctr[0], ctr[1], image, finder) for ctr in streak.centers])
        streak_ends = np.array(streak.ends).reshape((-1, 2, 2))
        check_close(np.sort(ends, axis=-2), np.sort(streak_ends, axis=-2))

        pts = np.concatenate([np.stack(self.get_pixels(ctr[0], ctr[1], finder), axis=-1)
                              for ctr in streak.centers])
        pts = np.unique(pts, axis=0)
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        assert np.all(np.stack([streak.x, streak.y], axis=-1) == pts)

    def test_mask(self, result: Pattern, finder: PatternStreakFinder):
        for streak in result:
            assert np.all(finder.mask[streak.y, streak.x])

    def test_p_values(self, result: Pattern, image: NDRealArray,
                     mask: NDBoolArray, xtol: float, vmin: float, min_size: int):
        p_values, prob = p_value(result, image, mask, xtol, vmin)
        assert np.all(p_values < np.log(prob) * min_size)

    def test_result_probability(self, result: Pattern, image: NDRealArray,
                                mask: NDBoolArray, xtol: float, vmin: float):
        _, prob = p_value(result, image, mask, xtol, vmin)
        index = np.searchsorted(np.sort(image[mask]), vmin)
        check_close(1 - index / mask.sum(), np.asarray(prob))

    def test_central_line(self, streak: Streak, image: NDRealArray):
        line = streak.line()
        tau = np.array(line[2:]) - np.array(line[:2])
        centers = np.array(streak.centers)
        center = self.center_of_mass(np.asarray(streak.x), np.asarray(streak.y),
                                     image[streak.y, streak.x])
        prods = np.sum((centers - center) * tau, axis=-1)
        central_line = np.concatenate((centers[np.argmin(prods)], centers[np.argmax(prods)]))
        assert np.all(central_line == np.asarray(streak.central_line()))

    def test_negative_image(self, image: NDRealArray, mask: NDBoolArray, structure: Structure2D,
                            min_size: int, peaks: PeaksList, xtol: float):
        finder = PatternStreakFinder(-image, mask, structure, min_size)
        streaks = finder.detect_streaks(peaks, xtol, 0.0)[0]
        assert len(streaks) == 0
