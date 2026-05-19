from math import prod
from typing import Tuple
import pytest
from cbclib_v2 import Lines, default_rng
from cbclib_v2.annotations import (CPArray, CuPy, CuPyNamespace, Generator, IntArray, NDArray,
                                   NumPy, NumPyNamespace, RealArray, Shape)
from cbclib_v2.label import CPLabelResult, LabelResult, NPLabelResult, Structure, label, p_values
from cbclib_v2.ndimage import draw_lines
from cbclib_v2.streak_finder import PeakLabels, PatternStreakFinder, Streaks, streak_labels
from cbclib_v2.streak_finder import n_signal, detect_peaks, peak_labels
from cbclib_v2.test_util import check_close

TestGenerator = Generator[NDArray] | Generator[CPArray]
TestNamespace = NumPyNamespace | CuPyNamespace

class TestNewStreakFinder:
    def binned_shape(self, shape: Tuple[int, ...], radius: int) -> Tuple[int, ...]:
        return shape[:-2] + ((shape[-2] + radius - 1) // radius, (shape[-1] + radius - 1) // radius)

    def center_of_mass(self, x: IntArray, y: IntArray, val: RealArray,
                       xp: TestNamespace) -> RealArray:
        points = xp.sum(xp.stack((x, y), axis=-1) * val[..., None], axis=-2)
        return points / xp.sum(val, axis=-1)[..., None]

    def covariance_matrix(self, x: IntArray, y: IntArray, val: RealArray, xp: TestNamespace
                          ) -> RealArray:
        vec = xp.stack((x, y), axis=-1) - self.center_of_mass(x, y, val, xp)[..., None, :]
        matrix = xp.sum(vec[..., None, :] * vec[..., None] * val[..., None, None], axis=-3)
        return matrix / xp.sum(val, axis=-1)[..., None, None]

    def line(self, x: IntArray, y: IntArray, val: RealArray, xp: TestNamespace) -> RealArray:
        ctr = self.center_of_mass(x, y, val, xp)
        mat = self.covariance_matrix(x, y, val, xp)
        mu_xx, mu_xy, mu_yy = mat[..., 0, 0], mat[..., 0, 1], mat[..., 1, 1]
        theta = 0.5 * xp.atan2(2 * mu_xy, (mu_xx - mu_yy))

        tau = xp.zeros(ctr.shape, dtype=ctr.dtype)
        tau[..., 0] = xp.cos(theta)
        tau[..., 1] = xp.sin(theta)

        delta = xp.sqrt(4 * mu_xy**2 + (mu_xx - mu_yy)**2)
        hw = xp.sqrt(2 * xp.log(2) * (mu_xx + mu_yy + delta))
        return xp.concat((ctr + hw[..., None] * tau, ctr - hw[..., None] * tau), axis=-1)

    def lgamma(self, x: int | IntArray, xp: TestNamespace) -> RealArray:
        if xp is NumPy:
            from scipy.special import gammaln
            return gammaln(x)
        if xp is CuPy:
            import cupyx.scipy.special as cpx
            return cpx.gammaln(x)
        raise ValueError(f"Unknown Array API: {xp}")

    def logbinom(self, n: int | IntArray, k: int | IntArray, p: float | RealArray,
                 xp: TestNamespace) -> RealArray:
        log_p = xp.log(p)
        log_q = xp.log1p(-p)
        log_term = self.lgamma(n + 1, xp) - self.lgamma(k + 1, xp)
        log_term -= self.lgamma(n - k + 1, xp)
        log_term += k * log_p + (n - k) * log_q

        i = xp.arange(int(k), int(n)) + 1
        if i.size == 0:
            return log_term

        log_ratio = xp.log(n - i + 1) - xp.log(i) + log_p - log_q
        log_offsets = xp.concat((xp.zeros(1), xp.cumsum(log_ratio)))
        log_terms = log_term + log_offsets
        max_term = xp.max(log_terms)
        return max_term + xp.log(xp.sum(xp.exp(log_terms - max_term)))

    def labels_and_index(self, labeled: LabelResult, xp: TestNamespace
                         ) -> Tuple[IntArray, IntArray]:
        if isinstance(labeled, CPLabelResult):
            return labeled.labels, labeled.index
        if isinstance(labeled, NPLabelResult):
            index = xp.arange(1, len(labeled.regions) + 1)
            return labeled.to_array(index), index
        raise TypeError("Unknown LabelResult type")

    @pytest.fixture(params=['cpu', 'gpu'])
    def platform(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture
    def xp(self, platform: str) -> TestNamespace:
        if platform == 'cpu':
            return NumPy
        if platform == 'gpu':
            if CuPy is None:
                pytest.skip("CuPy is not available")
            return CuPy
        raise ValueError(f"Unknown platform: {platform}")

    @pytest.fixture
    def rng(self, xp: TestNamespace) -> TestGenerator:
        return default_rng(42, xp)

    @pytest.fixture(params=[40, 0])
    def n_lines(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def ndim(self) -> int:
        return 2

    @pytest.fixture(params=[(3, 80, 100)])
    def shape(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture(params=[15.0])
    def length(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[1.0])
    def width(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def centers(self, rng: TestGenerator, n_lines: int, shape: Shape, xp: TestNamespace
                ) -> RealArray:
        array = rng.random((2, n_lines))
        return xp.array([[shape[-1]], [shape[-2]]]) * array

    @pytest.fixture
    def frames(self, rng: TestGenerator, shape: Shape, ndim: int, n_lines: int) -> IntArray:
        return rng.integers(0, prod(shape[:-ndim]) - 1, size=n_lines)

    @pytest.fixture
    def lines(self, rng: TestGenerator, shape: Shape, ndim: int, n_lines: int,
              length: float, xp: TestNamespace) -> Lines:
        lengths = length * rng.random((n_lines,))
        pt0 = xp.array(shape[:-ndim - 1:-1]) * rng.random((n_lines, ndim))
        vec = rng.normal(xp.zeros(ndim), size=(n_lines, ndim))
        pt1 = pt0 + vec * (lengths / xp.sqrt(xp.sum(vec**2, axis=-1)))[:, None]
        return Lines(xp.concat((pt0, pt1), axis=-1))

    @pytest.fixture(params=[0.4])
    def vmin(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def noise(self, rng: TestGenerator, vmin: float, shape: Shape) -> RealArray:
        return 0.25 * vmin * rng.random(shape)

    @pytest.fixture(params=[0.8])
    def xtol(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def image(self, lines: Lines, width: float, frames: IntArray, shape: Shape, noise: RealArray,
              xp: TestNamespace) -> RealArray:
        return draw_lines(xp.zeros(shape), lines.to_lines(width), frames, kernel='biweight') + noise

    @pytest.fixture(params=[(1, 2)])
    def structure(self, request: pytest.FixtureRequest, shape: Shape) -> Structure:
        radius, connectivity = request.param
        return Structure([0,] * (len(shape) - 2) + [radius,] * 2, connectivity)

    @pytest.fixture
    def finder(self, image: RealArray, structure: Structure, vmin: float) -> PatternStreakFinder:
        return PatternStreakFinder(image, structure=structure, vmin=vmin)

    @pytest.fixture(params=[5])
    def npts(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def regions(self, finder: PatternStreakFinder, npts: int, structure: Structure) -> LabelResult:
        return finder.detect_regions(npts, structure)

    @pytest.fixture
    def peak_indices(self, finder: PatternStreakFinder, regions: LabelResult) -> IntArray:
        return detect_peaks(finder.data, regions, finder.structure.connectivity, finder.vmin)

    def test_peak_indices(self, peak_indices: IntArray, finder: PatternStreakFinder,
                          xp: TestNamespace):
        assert peak_indices.shape == self.binned_shape(finder.data.shape,
                                                       finder.structure.connectivity)

        mask = (peak_indices >= 0) & (peak_indices < finder.data.size)
        peaks = peak_indices[mask]
        peak_pts = xp.stack(xp.unravel_index(peaks, finder.data.shape), axis=-1)
        bin_pts = xp.copy(peak_pts)
        bin_pts[..., -2:] //= finder.structure.connectivity
        assert xp.all(bin_pts == xp.stack(xp.where(mask), axis=-1))

        assert xp.all(finder.data.ravel()[peaks] > finder.vmin)

        y_shifts = xp.zeros((3, peak_pts.shape[-1]), dtype=peak_pts.dtype)
        y_shifts[:, -2] = xp.array([-1, 0, 1], dtype=peak_pts.dtype)
        x_shifts = xp.zeros((3, peak_pts.shape[-1]), dtype=peak_pts.dtype)
        x_shifts[:, -1] = xp.array([-1, 0, 1], dtype=peak_pts.dtype)

        y_nbr_pts = peak_pts[:, None, :] + y_shifts[None, :, :]
        x_nbr_pts = peak_pts[:, None, :] + x_shifts[None, :, :]
        y_indices = xp.argmax(finder.data[tuple(y_nbr_pts.T)], axis=0)
        x_indices = xp.argmax(finder.data[tuple(x_nbr_pts.T)], axis=0)

        assert xp.all(y_indices == 1)
        assert xp.all(x_indices == 1)

    @pytest.fixture
    def peak_labels(self, peak_indices: IntArray, finder: PatternStreakFinder
                    ) -> Tuple[PeakLabels, IntArray]:
        return peak_labels(peak_indices, finder.data, finder.structure.connectivity)

    @pytest.fixture
    def labels(self, peak_labels: Tuple[PeakLabels, IntArray]) -> PeakLabels:
        return peak_labels[0]

    @pytest.fixture
    def peaks(self, peak_labels: Tuple[PeakLabels, IntArray]) -> IntArray:
        return peak_labels[1]

    def test_peak_labels(self, labels: PeakLabels, peaks: IntArray, peak_indices: IntArray,
                         finder: PatternStreakFinder, xp: TestNamespace):
        assert labels.labels.shape == self.binned_shape(finder.data.shape,
                                                        finder.structure.connectivity)
        assert peaks.size == labels.n_good
        assert xp.all(peaks[labels.n_labels:] == -1)

        indices = peak_indices[(peak_indices >= 0) & (peak_indices < finder.data.size)]
        assert xp.all(indices == peaks[labels.labels[labels.labels > 0] - 1])

        values = finder.data.ravel()[peaks[xp.arange(labels.n_seeds)]]
        assert xp.all(values >= finder.vmin)
        assert xp.all(values[1:] <= values[:-1])

    @pytest.fixture
    def fit_linelets_result(self, finder: PatternStreakFinder, labels: PeakLabels, peaks: IntArray
                        ) -> Tuple[RealArray, PeakLabels]:
        return finder.fit_linelets(labels, peaks)

    @pytest.fixture
    def linelets(self, fit_linelets_result: Tuple[RealArray, PeakLabels]) -> RealArray:
        return fit_linelets_result[0]

    @pytest.fixture
    def new_labels(self, fit_linelets_result: Tuple[RealArray, PeakLabels]) -> PeakLabels:
        return fit_linelets_result[1]

    def get_pixels(self, coord: IntArray, finder: PatternStreakFinder, xp: TestNamespace
                   ) -> Tuple[IntArray, RealArray]:
        coords = xp.asarray(list(finder.structure), dtype=int) + coord[..., None, :]
        inbound = xp.all((coords >= 0) & (coords < xp.array(finder.data.shape)), axis=-1)
        values = xp.zeros(coords.shape[:-1], dtype=finder.data.dtype)
        values[inbound] = finder.data[tuple(coords[inbound].T)]
        return coords, values

    def test_linelets(self, linelets: RealArray, new_labels: PeakLabels, peaks: IntArray,
                      finder: PatternStreakFinder, xp: TestNamespace):
        assert linelets.shape == (new_labels.n_good, 4)
        assert xp.all(linelets[new_labels.n_labels:] == 0)

        coords = xp.stack(xp.unravel_index(peaks[:new_labels.n_labels], finder.data.shape), axis=-1)
        pixels, values = self.get_pixels(coords, finder, xp)
        lines = self.line(pixels[..., -1], pixels[..., -2], values, xp)
        assert xp.all(xp.isclose(linelets[:new_labels.n_labels], lines))

    @pytest.fixture
    def streaks(self, finder: PatternStreakFinder, new_labels: PeakLabels, peaks: IntArray,
                linelets: RealArray, xtol: float) -> Streaks:
        return finder.detect_streaks(new_labels, peaks, linelets, xtol)

    def test_streaks(self, streaks: Streaks, new_labels: PeakLabels, linelets: RealArray, xtol: float,
                     xp: TestNamespace):
        assert len(streaks) == new_labels.n_seeds

        for label, streak in zip(range(1, new_labels.n_seeds + 1), streaks):
            seed = xp.where(new_labels.labels.ravel() == label)[0][0]
            bins = xp.asarray(streak.indices)
            assert bins[0] == seed

            # Check that each index has it's own adjacent child
            coords = xp.stack(xp.unravel_index(bins, new_labels.labels.shape), axis=-1)
            dists = xp.abs(coords[..., None, :] - coords)
            assert xp.all(xp.any(xp.all(dists <= 1, axis=-1), axis=-1))

            # Check lines self-consistency
            lines = Lines(linelets[new_labels.labels.ravel()[bins] - 1])
            line = Lines(xp.asarray(streak.line(new_labels.labels, linelets)))
            assert xp.all(line.distance(lines.points) < xtol)

    @pytest.fixture
    def n_signal(self, finder: PatternStreakFinder, streaks: Streaks, new_labels: PeakLabels,
                 peaks: IntArray) -> IntArray:
        return n_signal(streaks, new_labels, peaks, finder.data, finder.structure, finder.vmin)

    @pytest.fixture
    def ranks(self, n_signal: IntArray, xp: TestNamespace) -> IntArray:
        indices = xp.arange(n_signal.size)
        order = xp.lexsort(xp.stack((indices, -n_signal)))
        ranks = xp.empty(order.shape, dtype=indices.dtype)
        ranks[order] = indices
        return ranks

    def test_n_signal(self, n_signal: IntArray, streaks: Streaks, new_labels: PeakLabels,
                      peaks: IntArray, finder: PatternStreakFinder, xp: TestNamespace):
        assert n_signal.shape == (new_labels.n_seeds,)

        shifts = xp.array(list(finder.structure), dtype=int)
        for streak, num_points in zip(streaks, n_signal):
            bins = xp.asarray(streak.indices)
            peak_indices = peaks[new_labels.labels.ravel()[bins] - 1]
            peak_crds = xp.stack(xp.unravel_index(peak_indices, finder.data.shape), axis=-1)
            coords = peak_crds[..., None, :] + shifts
            inbound = xp.all((coords >= 0) & (coords < xp.array(finder.data.shape)), axis=-1)
            footprint = xp.unique(xp.ravel_multi_index(coords[inbound].T, finder.data.shape))

            assert xp.sum(finder.data.ravel()[footprint] >= finder.vmin) == num_points

    @pytest.fixture
    def streak_labels(self, finder: PatternStreakFinder, streaks: Streaks, ranks: IntArray,
                      new_labels: PeakLabels, peaks: IntArray, xp: TestNamespace) -> IntArray:
        out = xp.zeros(finder.data.shape, dtype=peaks.dtype)
        return streak_labels(out, streaks, ranks, new_labels, peaks, finder.structure)

    def test_streak_labels(self, streak_labels: IntArray, streaks: Streaks, new_labels: PeakLabels,
                           peaks: IntArray, ranks: IntArray, finder: PatternStreakFinder,
                           xp: TestNamespace):
        assert streak_labels.shape == finder.data.shape

        shifts = xp.array(list(finder.structure), dtype=int)
        for rank, streak in zip(ranks, streaks):
            bins = xp.asarray(streak.indices)
            peak_indices = peaks[new_labels.labels.ravel()[bins] - 1]
            peak_crds = xp.stack(xp.unravel_index(peak_indices, finder.data.shape), axis=-1)
            coords = peak_crds[..., None, :] + shifts
            inbound = xp.all((coords >= 0) & (coords < xp.array(finder.data.shape)), axis=-1)
            footprint = xp.unique(xp.ravel_multi_index(coords[inbound].T, finder.data.shape))

            bin_ranks = streak_labels.ravel()[footprint]
            assert xp.all(bin_ranks <= rank + 1)

    @pytest.fixture
    def labeled(self, streak_labels: IntArray) -> LabelResult:
        return label(streak_labels, Structure([0,] * (streak_labels.ndim - 2) + [1, 1], 1))

    @pytest.fixture
    def detected(self, labeled: LabelResult, finder: PatternStreakFinder) -> RealArray:
        return finder.line_fit(labeled)

    @pytest.fixture
    def p_values(self, labeled: LabelResult, detected: RealArray, finder: PatternStreakFinder,
                 xtol: float) -> RealArray:
        return p_values(labeled, detected, finder.data, finder.p0, finder.vmin, xtol)

    def test_p_values(self, p_values: RealArray, labeled: LabelResult, detected: RealArray,
                      finder: PatternStreakFinder, xtol: float, xp: TestNamespace):
        assert p_values.size == detected.shape[0]

        labels, index = self.labels_and_index(labeled, xp)
        for label, line, p_val in zip(index, detected, p_values):
            indices = xp.stack(xp.where(labels == label), axis=-1)
            dists = Lines(line).distance(indices[..., ::-1])
            indices = indices[dists < xtol]
            values = finder.data[tuple(indices.T)]
            expected = self.logbinom(values.size, xp.sum(values >= finder.vmin), finder.p0, xp)
            check_close(expected, p_val)
