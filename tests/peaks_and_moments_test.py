from math import prod
from typing import List
import pytest
from scipy.ndimage import maximum_filter
from cbclib_v2 import default_rng
from cbclib_v2.annotations import (Generator, NDArray, NDIntArray, NDRealArray, NumPy,
                                   NumPyNamespace, Shape)
from cbclib_v2.label import CPLabelResult, Region, Regions, Structure, label
from cbclib_v2.streak_finder import Peaks, PeaksList, detect_peaks, filter_peaks
from cbclib_v2.test_util import check_close, local_maxima, Pixels2D

class TestPeaksAndMoments():
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def rng(self, xp: NumPyNamespace) -> Generator[NDArray]:
        return default_rng(42, xp)

    def center_of_mass(self, x: NDIntArray, y: NDIntArray, val: NDRealArray, xp: NumPyNamespace
                       ) -> NDRealArray:
        return xp.sum(xp.stack((y, x), axis=-1) * val[..., None], axis=0) / xp.sum(val)

    def covariance_matrix(self, x: NDIntArray, y: NDIntArray, val: NDRealArray, xp: NumPyNamespace
                          ) -> NDRealArray:
        pts = xp.stack((y, x), axis=-1)
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
    def peaks(self, images: NDRealArray, threshold: float) -> Peaks:
        return detect_peaks(images, radius=3, vmin=threshold)[0]

    @pytest.fixture(params=[100,])
    def n_keys(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[5,])
    def vrange(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def keys(self, rng: Generator[NDArray], n_keys: int, shape: Shape) -> NDIntArray:
        return rng.integers(0, prod(shape[-2:]), size=n_keys)

    def test_peaks_find_range(self, peaks: Peaks, keys: NDIntArray, vrange: int,
                              xp: NumPyNamespace):
        indices = list(peaks)
        points = xp.stack(xp.unravel_index(indices, peaks.shape), axis=-1)

        for index in keys:
            point = xp.array(xp.unravel_index(index, peaks.shape))
            nearest_idx = peaks.find_range(index, vrange)
            if nearest_idx >= 0:
                nearest = xp.array(xp.unravel_index(nearest_idx, peaks.shape))
                dist = xp.sum((point - nearest)**2)
                assert xp.min(xp.sum((points - point)**2, axis=-1)) == dist
                assert dist < vrange * vrange
            else:
                assert xp.min(xp.sum((points - point)**2, axis=-1)) >= vrange * vrange

    def test_peaks(self, peaks: Peaks, image: NDRealArray, threshold: float,
                   xp: NumPyNamespace):
        indices = list(peaks)
        points = xp.stack(xp.unravel_index(indices, peaks.shape), axis=-1)

        assert xp.all(image[points[..., 0], points[..., 1]] > threshold)
        y_neighbours = points[:, None, :] + xp.array([[-1, 0], [0, 0], [1, 0]])
        x_neighbours = points[:, None, :] + xp.array([[0, -1], [0, 0], [0, 1]])
        x_indices = xp.argmax(image[x_neighbours[..., 0], x_neighbours[..., 1]], axis=-1)
        y_indices = xp.argmax(image[y_neighbours[..., 0], y_neighbours[..., 1]], axis=-1)
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
    def filtered(self, peaks: Peaks, image: NDRealArray,
                 structure: Structure, threshold: float, npts: int) -> Peaks:
        filtered = PeaksList()
        filtered.append(peaks)
        filter_peaks(filtered, image, structure, threshold, npts)
        return filtered[0]

    def test_filtered(self, peaks: Peaks, filtered: Peaks, image: NDRealArray,
                      structure: Structure, threshold: float, npts: int, xp: NumPyNamespace):
        labeled = label(image > threshold, structure, npts)
        if isinstance(labeled, CPLabelResult):
            raise TypeError("label result is on GPU, expected CPU")

        peak_indices = xp.array(list(peaks))
        region_indices = xp.concat([list(region) for region in labeled.regions], axis=0)
        is_labeled = xp.any(peak_indices[:, None] == region_indices[None], axis=-1)
        labeled_peaks = peak_indices[is_labeled]

        assert xp.all(xp.array(list(filtered)) == labeled_peaks)

    @pytest.fixture(params=[30,])
    def n_pts(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def seeds(self, rng: Generator[NDArray], shape: Shape, n_pts: int) -> NDIntArray:
        return rng.choice(prod(shape[-2:]), size=n_pts, replace=False)

    @pytest.fixture
    def indices(self, seeds: NDIntArray, structure: Structure, shape: Shape, xp: NumPyNamespace
               ) -> NDIntArray:
        points = xp.stack(xp.unravel_index(seeds, shape[-2:]), axis=-1)
        shifts = xp.array(list(structure))
        points = points[:, None, :] + shifts[None, :, :]
        inbound = xp.all((points >= 0) & (points < xp.asarray(shape[-2:])[None, None, :]), axis=-1)
        all_points = points[inbound]
        return xp.ravel_multi_index(all_points.T, shape[-2:])

    @pytest.fixture
    def regions(self, seeds: NDIntArray, structure: Structure, shape: Shape) -> Regions:
        return Regions(Region(seed, structure, shape[-2:]) for seed in seeds)

    @pytest.fixture
    def pixels(self, image: NDRealArray, regions: Regions) -> List[Pixels2D]:
        return [Pixels2D(region, image) for region in regions]

    def test_pixels_merge(self, pixels: List[Pixels2D], image: NDRealArray):
        rsum = Pixels2D()
        rsum.merge(pixels[0], image)
        assert list(rsum.region) == list(pixels[0].region)
        check_close(rsum.moment_of_inertia(), pixels[0].moment_of_inertia())
        check_close(rsum.center_of_mass(), pixels[0].center_of_mass())

        rsum.merge(pixels[0], image)
        assert list(rsum.region) == list(pixels[0].region)
        check_close(rsum.moment_of_inertia(), pixels[0].moment_of_inertia())
        check_close(rsum.center_of_mass(), pixels[0].center_of_mass())

    def test_pixels(self, image: NDRealArray, indices: NDIntArray, pixels: List[Pixels2D],
                    xp: NumPyNamespace):
        all_pixels = Pixels2D()
        for region in pixels:
            all_pixels.merge(region, image)

        all_indices = xp.sort(xp.unique_values(indices))
        points = xp.stack(xp.unravel_index(all_indices, image.shape), axis=-1)

        assert xp.all(xp.asarray(list(all_pixels.region)) == all_indices)

        total_mass = xp.sum(image[points[..., 0], points[..., 1]])
        check_close(all_pixels.total_mass(), total_mass)
        mean = xp.sum(points * image[points[..., 0], points[..., 1], None], axis=0)
        check_close(all_pixels.mean(), mean)
        inertia = xp.sum(points[..., None, :] * points[..., None] * \
                         image[points[..., 0], points[..., 1], None, None], axis=0)
        check_close(all_pixels.moment_of_inertia(), inertia.reshape(-1))

        ctr = self.center_of_mass(points[..., 1], points[..., 0],
                                  image[points[..., 0], points[..., 1]], xp)
        check_close(all_pixels.center_of_mass(), ctr)
        mat = self.covariance_matrix(points[..., 1], points[..., 0],
                                     image[points[..., 0], points[..., 1]], xp)
        check_close(all_pixels.covariance_matrix(), mat.reshape(-1))

    def test_3d_image(self, images: NDRealArray, structure: Structure, threshold: float, npts: int,
                      xp: NumPyNamespace):
        if len(images.shape) <= 2:
            pytest.skip(f"Skipping image with a shape {images.shape} because len(shape) <= 2")

        all_peaks = detect_peaks(images, radius=3, vmin=threshold)
        filtered = PeaksList()
        filtered.extend(all_peaks)
        filter_peaks(filtered, images, structure, threshold, npts)

        for index, image in enumerate(images):
            peaks = detect_peaks(image, radius=3, vmin=threshold)
            num_modules = image.size // prod(image.shape[-2:])
            other_peaks = all_peaks[index * num_modules:(index + 1) * num_modules]
            assert xp.all(peaks.index() == other_peaks.index())
            assert xp.all(peaks.to_array() == other_peaks.to_array())

            filter_peaks(peaks, image, structure, threshold, npts)
            other_peaks = filtered[index * num_modules:(index + 1) * num_modules]
            assert xp.all(peaks.index() == other_peaks.index())
            assert xp.all(peaks.to_array() == other_peaks.to_array())

    @pytest.fixture
    def vicinity(self, image: NDRealArray) -> Structure:
        return Structure([0,] * (image.ndim - 2) + [1, 1], 1)

    def test_local_maxima(self, image: NDRealArray, vicinity: Structure, xp: NumPyNamespace):
        out = local_maxima(image, structure=vicinity)
        filtered = maximum_filter(image, footprint=vicinity.to_array(), mode='constant',
                                  cval=xp.inf)

        out2 = xp.where(xp.asarray(image == filtered).reshape(-1))

        assert xp.all(out == out2)

    @pytest.fixture
    def peaks_list(self, image: NDRealArray) -> PeaksList:
        return detect_peaks(image, radius=1, vmin=0.0)

    def test_detect_peaks(self, image: NDRealArray, peaks_list: PeaksList, vicinity: Structure,
                          xp: NumPyNamespace):
        for image, peaks in zip(xp.reshape(image, (-1,) + image.shape[-2:]), peaks_list):
            filtered = maximum_filter(image, footprint=vicinity.to_array(), mode='constant',
                                      cval=xp.inf)
            expected_peaks = xp.where(xp.reshape(image == filtered, -1))
            peaks_array = xp.array(list(peaks))
            assert xp.all(peaks_array == expected_peaks)
