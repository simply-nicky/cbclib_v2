from math import prod
from typing import List, Tuple
import pytest
from cbclib_v2 import set_at
from cbclib_v2.annotations import (Generator, NDArray, NDBoolArray, NDIntArray, NDRealArray, NumPy,
                                   NumPyNamespace, Shape)
from cbclib_v2.label import CPLabelResult, Pixels2DDouble, Structure, label
from cbclib_v2.streak_finder import Peaks, PeaksList, detect_peaks, filter_peaks
from cbclib_v2.test_util import check_close

class TestPeaksAndMoments():
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def rng(self, cpu_rng: Generator[NDArray]) -> Generator[NDArray]:
        return cpu_rng

    def to_tuple(self, pixels: Pixels2DDouble) -> Tuple[List[int], List[int], List[float]]:
        return (pixels.x, pixels.y, pixels.value)

    def moments(self, pixels: Pixels2DDouble, xp: NumPyNamespace) -> NDRealArray:
        return xp.concat(([pixels.total_mass(),], pixels.mean(), pixels.moment_of_inertia()))

    def central_moments(self, pixels: Pixels2DDouble, xp: NumPyNamespace) -> NDRealArray:
        return xp.concat((pixels.center_of_mass(), pixels.covariance_matrix()))

    def center_of_mass(self, x: NDIntArray, y: NDIntArray, val: NDRealArray, xp: NumPyNamespace
                       ) -> NDRealArray:
        return xp.sum(xp.stack((x, y), axis=-1) * val[..., None], axis=0) / xp.sum(val)

    def covariance_matrix(self, x: NDIntArray, y: NDIntArray, val: NDRealArray, xp: NumPyNamespace
                          ) -> NDRealArray:
        pts = xp.stack((x, y), axis=-1)
        ctr = self.center_of_mass(x, y, val, xp)
        return xp.sum((pts[..., None, :] * pts[..., None] - ctr[None, :] * ctr[:, None]) * \
                      val[..., None, None], axis=0) / xp.sum(val)

    @pytest.fixture(params=[(120, 80), (10, 10, 45, 35)])
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
    def images(self, rng: Generator[NDArray], shape: Shape, vmin: float, vmax: float
               ) -> NDRealArray:
        return (vmax - vmin) * rng.random(shape) + vmin

    @pytest.fixture
    def image(self, images: NDRealArray) -> NDRealArray:
        return images.reshape((-1,) + images.shape[-2:])[0]

    @pytest.fixture(params=[0.05])
    def num_bad(self, request: pytest.FixtureRequest, shape: Shape) -> int:
        return int(prod(shape) * request.param)

    @pytest.fixture
    def masks(self, rng: Generator[NDArray], shape: Shape, num_bad: int, xp: NumPyNamespace
              ) -> NDBoolArray:
        mask = xp.ones(shape, dtype=bool)
        indices = xp.unravel_index(rng.choice(mask.size, num_bad, replace=False), mask.shape)
        return set_at(mask, indices, False)

    @pytest.fixture
    def mask(self, masks: NDBoolArray) -> NDBoolArray:
        return masks.reshape((-1,) + masks.shape[-2:])[0]

    @pytest.fixture
    def peaks(self, images: NDRealArray, masks: NDBoolArray, threshold: float) -> Peaks:
        return detect_peaks(images, masks, radius=3, vmin=threshold)[0]

    @pytest.fixture(params=[100,])
    def n_keys(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[5,])
    def vrange(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def keys(self, rng: Generator[NDArray], n_keys: int, shape: Shape, xp: NumPyNamespace
             ) -> NDIntArray:
        x = rng.integers(0, shape[-1], size=n_keys)
        y = rng.integers(0, shape[-2], size=n_keys)
        return xp.stack((x, y), axis=-1)

    def test_peaks_find_range(self, peaks: Peaks, keys: NDIntArray, vrange: int,
                              xp: NumPyNamespace):
        points = xp.array(list(peaks))
        for key in keys:
            nearest = xp.array(peaks.find_range(key[0], key[1], vrange))
            if nearest.size:
                dist = xp.sum((key - nearest)**2)
                assert xp.min(xp.sum((points - key)**2, axis=-1)) == dist
                assert dist < vrange * vrange
            else:
                assert xp.min(xp.sum((points - key)**2, axis=-1)) >= vrange * vrange

    def test_peaks(self, peaks: Peaks, image: NDRealArray, mask: NDBoolArray, threshold: float,
                   xp: NumPyNamespace):
        points = xp.stack((peaks.x, peaks.y), axis=-1)
        assert xp.all(mask[points[..., 1], points[..., 0]])
        assert xp.all(image[points[..., 1], points[..., 0]] > threshold)
        x_neighbours = points[:, None, :] + xp.array([[-1, 0], [0, 0], [1, 0]])
        y_neighbours = points[:, None, :] + xp.array([[0, -1], [0, 0], [0, 1]])
        x_indices = xp.argmax(image[x_neighbours[..., 1], x_neighbours[..., 0]], axis=-1)
        y_indices = xp.argmax(image[y_neighbours[..., 1], y_neighbours[..., 0]], axis=-1)
        assert xp.all((x_indices == 1) | (y_indices == 1))

    @pytest.fixture(params=[3,])
    def connectivity(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def structure(self, connectivity: int) -> Structure:
        return Structure([connectivity,] * 2, connectivity)

    @pytest.fixture(params=[8,])
    def npts(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def filtered(self, peaks: Peaks, image: NDRealArray, mask: NDBoolArray,
                 structure: Structure, threshold: float, npts: int) -> Peaks:
        filtered = PeaksList()
        filtered.append(peaks)
        filter_peaks(filtered, image, mask, structure, threshold, npts)
        return filtered[0]

    def test_filtered(self, peaks: Peaks, filtered: Peaks, image: NDRealArray, mask: NDBoolArray,
                      structure: Structure, threshold: float, npts: int, xp: NumPyNamespace):
        labeled = label((image > threshold) & mask, structure, npts)
        if isinstance(labeled, CPLabelResult):
            raise TypeError("label result is on GPU, expected CPU")

        peak_pts = xp.stack((peaks.x, peaks.y), axis=-1)
        pts = xp.concat([xp.stack(xp.unravel_index(list(region), image.shape)[::-1], axis=-1)
                         for region in labeled.regions])
        peak_pts = peak_pts[xp.any(xp.all(peak_pts[:, None, :] == pts[None], axis=-1), axis=-1)]
        peak_pts = peak_pts[xp.lexsort((peak_pts[:, 1], peak_pts[:, 0]))]
        filtered_pts = xp.stack((filtered.x, filtered.y), axis=-1)
        filtered_pts = filtered_pts[xp.lexsort((filtered_pts[:, 1], filtered_pts[:, 0]))]
        assert xp.all(filtered_pts == peak_pts)

    @pytest.fixture(params=[30,])
    def n_pts(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def seeds(self, rng: Generator[NDArray], image: NDRealArray, n_pts: int) -> NDIntArray:
        return rng.choice(prod(image.shape), size=n_pts, replace=False)

    @pytest.fixture
    def points(self, seeds: NDIntArray, image: NDRealArray, structure: Structure, xp: NumPyNamespace
               ) -> NDIntArray:
        seed_indices = xp.stack(xp.unravel_index(seeds, image.shape), axis=-1)
        shifts = xp.array(list(structure))
        points = seed_indices[:, None, :] + shifts[None, :, :]
        inbound = xp.all((points >= 0) & (points < xp.asarray(image.shape)[None, None, :]), axis=(-2, -1))
        return points[inbound, :, ::-1]

    @pytest.fixture
    def indices(self, points: NDIntArray, image: NDRealArray, xp: NumPyNamespace) -> NDIntArray:
        indices = xp.arange(prod(image.shape)).reshape(image.shape)
        return indices[tuple(points.reshape(-1, len(image.shape)).T[::-1])]

    @pytest.fixture
    def regions(self, image: NDRealArray, points: NDIntArray) -> List[Pixels2DDouble]:
        return [Pixels2DDouble(pts[:, 0], pts[:, 1], image[pts[:, 1], pts[:, 0]]) for pts in points]

    def test_pixels_merge(self, regions: List[Pixels2DDouble], xp: NumPyNamespace):
        rsum = Pixels2DDouble().merge(regions[0])
        assert self.to_tuple(rsum) == self.to_tuple(regions[0])
        check_close(self.moments(rsum, xp), self.moments(regions[0], xp))
        check_close(self.central_moments(rsum, xp), self.central_moments(regions[0], xp))

        rsum = rsum.merge(regions[0])
        assert self.to_tuple(rsum) == self.to_tuple(regions[0])
        check_close(self.moments(rsum, xp), self.moments(regions[0], xp))
        check_close(self.central_moments(rsum, xp), self.central_moments(regions[0], xp))

    def test_pixels(self, image: NDRealArray, indices: NDIntArray, regions: List[Pixels2DDouble],
                    xp: NumPyNamespace):
        all_pixels = Pixels2DDouble()
        for region in regions:
            all_pixels.merge(region)

        indices = xp.unique_values(indices)
        pts = xp.stack(xp.unravel_index(indices, image.shape)[::-1], axis=-1)
        pts = pts[xp.lexsort((pts[:, 1], pts[:, 0]))]

        assert xp.all(all_pixels.x == pts[..., 0])
        assert xp.all(all_pixels.y == pts[..., 1])
        assert xp.all(all_pixels.value == image[pts[..., 1], pts[..., 0]])

        total_mass = xp.sum(image[pts[..., 1], pts[..., 0]])
        mean = xp.sum(pts * image[pts[..., 1], pts[..., 0], None], axis=0)
        inertia = xp.sum(pts[..., None, :] * pts[..., None] * \
                         image[pts[..., 1], pts[..., 0], None, None], axis=0)
        check_close(self.moments(all_pixels, xp),
                    xp.concat(([total_mass,], mean, xp.reshape(inertia, -1))))

        ctr = self.center_of_mass(pts[..., 0], pts[..., 1], image[pts[..., 1], pts[..., 0]], xp)
        mat = self.covariance_matrix(pts[..., 0], pts[..., 1], image[pts[..., 1], pts[..., 0]], xp)
        check_close(self.central_moments(all_pixels, xp), xp.concat((ctr, mat.reshape(-1))))

    def test_3d_image(self, images: NDRealArray, masks: NDBoolArray, structure: Structure,
                      threshold: float, npts: int, xp: NumPyNamespace):
        if len(images.shape) <= 2:
            pytest.skip(f"Skipping image with a shape {images.shape} because len(shape) <= 2")

        all_peaks = detect_peaks(images, masks, radius=3, vmin=threshold)
        filtered = PeaksList()
        filtered.extend(all_peaks)
        filter_peaks(filtered, images, masks, structure, threshold, npts)

        for index, (image, mask) in enumerate(zip(images, masks)):
            peaks = detect_peaks(image, mask, radius=3, vmin=threshold)
            num_modules = image.size // prod(image.shape[-2:])
            other_peaks = all_peaks[index * num_modules:(index + 1) * num_modules]
            assert xp.all(peaks.index() == other_peaks.index())
            assert xp.all(peaks.x() == other_peaks.x())
            assert xp.all(peaks.y() == other_peaks.y())

            filter_peaks(peaks, image, mask, structure, threshold, npts)
            other_peaks = filtered[index * num_modules:(index + 1) * num_modules]
            assert xp.all(peaks.index() == other_peaks.index())
            assert xp.all(peaks.x() == other_peaks.x())
            assert xp.all(peaks.y() == other_peaks.y())
