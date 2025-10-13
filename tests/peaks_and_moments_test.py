from math import prod
from typing import List, Tuple
import numpy as np
import pytest
from cbclib_v2.annotations import NDBoolArray, NDIntArray, NDRealArray, Shape
from cbclib_v2.label import PointSet2D, Pixels2DDouble, Structure2D, label
from cbclib_v2.streak_finder import Peaks, PeaksList, detect_peaks, filter_peaks
from cbclib_v2.test_util import check_close

class TestPeaksAndMoments():
    def to_tuple(self, pixels: Pixels2DDouble) -> Tuple[List[int], List[int], List[float]]:
        return (pixels.x, pixels.y, pixels.value)

    def moments(self, pixels: Pixels2DDouble) -> NDRealArray:
        return np.concatenate(([pixels.total_mass(),], pixels.mean(), pixels.moment_of_inertia()))

    def central_moments(self, pixels: Pixels2DDouble) -> NDRealArray:
        return np.concatenate((pixels.center_of_mass(), pixels.covariance_matrix()))

    def center_of_mass(self, x: NDIntArray, y: NDIntArray, val: NDRealArray) -> NDRealArray:
        return np.sum(np.stack((x, y), axis=-1) * val[..., None], axis=0) / np.sum(val)

    def covariance_matrix(self, x: NDIntArray, y: NDIntArray, val: NDRealArray) -> NDRealArray:
        pts = np.stack((x, y), axis=-1)
        ctr = self.center_of_mass(x, y, val)
        return np.sum((pts[..., None, :] * pts[..., None] - ctr[None, :] * ctr[:, None]) * \
                      val[..., None, None], axis=0) / np.sum(val)

    @pytest.fixture(params=[(100, 100)])
    def shape(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture(params=[0.0])
    def vmin(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[1.0])
    def vmax(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def threshold(self, vmin: float, vmax: float) -> float:
        return 0.9 * (vmax - vmin) + vmin

    @pytest.fixture
    def image(self, rng: np.random.Generator, shape: Shape, vmin: float, vmax: float
              ) -> NDRealArray:
        return (vmax - vmin) * rng.random(shape) + vmin

    @pytest.fixture(params=[0.05])
    def num_bad(self, request: pytest.FixtureRequest, shape: Shape) -> int:
        return int(prod(shape) * request.param)

    @pytest.fixture
    def mask(self, shape: Shape, num_bad: int, rng: np.random.Generator) -> NDBoolArray:
        mask = np.ones(shape, dtype=bool)
        indices = np.unravel_index(rng.choice(mask.size, num_bad, replace=False), mask.shape)
        mask[indices] = False
        return mask

    @pytest.fixture
    def peaks(self, image: NDRealArray, mask: NDBoolArray, threshold: float) -> Peaks:
        return detect_peaks(image, mask, radius=3, vmin=threshold)[0]

    @pytest.fixture(params=[100,])
    def n_keys(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[5,])
    def vrange(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def keys(self, rng: np.random.Generator, n_keys: int, shape: Shape) -> NDIntArray:
        x = rng.integers(0, shape[-1], size=n_keys)
        y = rng.integers(0, shape[-2], size=n_keys)
        return np.stack((x, y), axis=-1)

    def test_peaks_find_range(self, peaks: Peaks, keys: NDIntArray, vrange: int):
        points = np.array(list(peaks))
        for key in keys:
            nearest = np.array(peaks.find_range(key[0], key[1], vrange))
            if nearest.size:
                dist = np.sum((key - nearest)**2)
                assert np.min(np.sum((points - key)**2, axis=-1)) == dist
                assert dist < vrange * vrange
            else:
                assert np.min(np.sum((points - key)**2, axis=-1)) >= vrange * vrange

    def test_peaks(self, peaks: Peaks, image: NDRealArray, mask: NDBoolArray, threshold: float):
        points = np.stack((peaks.x, peaks.y), axis=-1)
        assert np.all(mask[points[..., 1], points[..., 0]])
        assert np.all(image[points[..., 1], points[..., 0]] > threshold)
        x_neighbours = points[:, None, :] + np.array([[-1, 0], [0, 0], [1, 0]])
        y_neighbours = points[:, None, :] + np.array([[0, -1], [0, 0], [0, 1]])
        x_indices = np.argmax(image[x_neighbours[..., 1], x_neighbours[..., 0]], axis=-1)
        y_indices = np.argmax(image[y_neighbours[..., 1], y_neighbours[..., 0]], axis=-1)
        assert np.all((x_indices == 1) | (y_indices == 1))

    @pytest.fixture(params=[(3, 3), ])
    def connectivity(self, request: pytest.FixtureRequest) -> Structure2D:
        return Structure2D(request.param[0], request.param[1])

    @pytest.fixture(params=[8,])
    def npts(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def filtered(self, peaks: Peaks, image: NDRealArray, mask: NDBoolArray, connectivity: Structure2D,
                 threshold: float, npts: int) -> Peaks:
        filtered = PeaksList()
        filtered.append(peaks)
        filter_peaks(filtered, image, mask, connectivity, threshold, npts)
        return filtered[0]

    def test_filtered(self, peaks: Peaks, filtered: Peaks, image: NDRealArray, mask: NDBoolArray,
                      connectivity: Structure2D, threshold: float, npts: int):
        regions = label((image > threshold) & mask, connectivity, PointSet2D(peaks.x, peaks.y), npts)
        peak_pts = np.stack((peaks.x, peaks.y), axis=-1)
        pts = np.concatenate([np.stack((region.x, region.y), axis=-1) for region in regions])
        peak_pts = peak_pts[np.any(np.all(peak_pts[:, None, :] == pts[None], axis=-1), axis=-1)]
        peak_pts = peak_pts[np.lexsort((peak_pts[:, 1], peak_pts[:, 0]))]
        filtered_pts = np.stack((filtered.x, filtered.y), axis=-1)
        filtered_pts = filtered_pts[np.lexsort((filtered_pts[:, 1], filtered_pts[:, 0]))]
        assert np.all(filtered_pts == peak_pts)

    @pytest.fixture(params=[10,])
    def rank(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[30,])
    def n_pts(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def structure(self, rank: int) -> Structure2D:
        return Structure2D(rank, rank)

    @pytest.fixture
    def seeds(self, rng: np.random.Generator, n_pts: int, rank: int, shape: Shape) -> NDIntArray:
        return rng.integers((rank, rank), (shape[1] - rank, shape[0] - rank), size=(n_pts, 2))

    @pytest.fixture
    def points(self, seeds: NDIntArray, structure: Structure2D) -> NDIntArray:
        return np.stack((structure.x + seeds[:, None, 1], structure.y + seeds[:, None, 0]), axis=-1)

    @pytest.fixture
    def regions(self, image: NDRealArray, points: NDIntArray) -> List[Pixels2DDouble]:
        return [Pixels2DDouble(pts[:, 0], pts[:, 1], image[pts[:, 1], pts[:, 0]]) for pts in points]

    def test_pixels_merge(self, regions: List[Pixels2DDouble]):
        rsum = Pixels2DDouble().merge(regions[0])
        assert self.to_tuple(rsum) == self.to_tuple(regions[0])
        check_close(self.moments(rsum), self.moments(regions[0]))
        check_close(self.central_moments(rsum), self.central_moments(regions[0]))

        rsum = rsum.merge(regions[0])
        assert self.to_tuple(rsum) == self.to_tuple(regions[0])
        check_close(self.moments(rsum), self.moments(regions[0]))
        check_close(self.central_moments(rsum), self.central_moments(regions[0]))

    def test_pixels(self, image: NDRealArray, points: NDIntArray, regions: List[Pixels2DDouble]):
        all_pixels = Pixels2DDouble()
        for region in regions:
            all_pixels.merge(region)
        pts = np.unique(points.reshape((-1, 2)), axis=0)
        assert np.all(all_pixels.x == pts[..., 0])
        assert np.all(all_pixels.y == pts[..., 1])
        assert np.all(all_pixels.value == image[pts[..., 1], pts[..., 0]])

        total_mass = np.sum(image[pts[..., 1], pts[..., 0]])
        mean = np.sum(pts * image[pts[..., 1], pts[..., 0], None], axis=0)
        inertia = np.sum(pts[..., None, :] * pts[..., None] * \
                         image[pts[..., 1], pts[..., 0], None, None], axis=0)
        check_close(self.moments(all_pixels),
                    np.concatenate(([total_mass,], mean, np.ravel(inertia))))

        ctr = self.center_of_mass(pts[..., 0], pts[..., 1], image[pts[..., 1], pts[..., 0]])
        mat = self.covariance_matrix(pts[..., 0], pts[..., 1], image[pts[..., 1], pts[..., 0]])
        check_close(self.central_moments(all_pixels), np.concatenate((ctr, mat.ravel())))
