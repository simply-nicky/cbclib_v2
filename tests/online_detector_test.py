from pathlib import Path
from typing import Tuple
import numpy as np
import pytest
from cbclib_v2 import CrystData, read_crystfel
from cbclib_v2._src.crystfel import Detector
from cbclib_v2._src.array_api import asnumpy
from cbclib_v2.annotations import (CuPy, CuPyNamespace, IntArray, NDIntArray, NDRealArray, NumPy,
                                   NumPyNamespace, RealArray)
from cbclib_v2.label import Structure

class TestOnlineDetector:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def center(self) -> Tuple[float, float]:
        return (0.0, 0.0)

    @pytest.fixture
    def n_bins(self) -> int:
        return 4

    @pytest.fixture
    def geometry(self, tmp_path: Path) -> Detector:
        geom_file = tmp_path / "online.geom"
        geom_file.write_text(
            "photon_energy = 12.0keV\n"
            "clen = 0.1m\n"
            "res = 1\n"
            "panel/corner_x = -0.5\n"
            "panel/corner_y = -0.5\n"
            "panel/fs = +1.0x+0.0y\n"
            "panel/ss = +0.0x+1.0y\n"
            "panel/min_fs = 0\n"
            "panel/max_fs = 2\n"
            "panel/min_ss = 0\n"
            "panel/max_ss = 2\n"
        )
        return read_crystfel(str(geom_file))

    @pytest.fixture
    def radial_index(self, geometry: Detector, center: Tuple[float, float],
                     n_bins: int) -> IntArray:
        return geometry.radial_index(center, n_bins)

    @pytest.fixture
    def structure(self) -> Structure:
        return Structure([0, 1, 1], 1)

    @pytest.fixture
    def smooth_data(self, xp: NumPyNamespace) -> RealArray:
        return xp.asarray([[
            [10.0, 20.0, 22.0],
            [12.0, 24.0, 30.0],
            [14.0, 32.0, 34.0],
        ]])

    @pytest.fixture
    def outlier_data(self, xp: NumPyNamespace) -> RealArray:
        return xp.asarray([[
            [10.0, 20.0, 22.0],
            [12.0, 200.0, 30.0],
            [14.0, 32.0, 34.0],
        ]])

    @pytest.fixture
    def interval_data(self, xp: NumPyNamespace) -> RealArray:
        return xp.arange(9, dtype=float).reshape(1, 3, 3)

    @pytest.fixture
    def region_data(self, xp: NumPyNamespace) -> RealArray:
        return xp.asarray([[
            [10.0, 20.0, 22.0],
            [12.0, 100.0, 30.0],
            [14.0, 32.0, 34.0],
        ]])

    @pytest.fixture
    def unequal_geometry(self, tmp_path: Path) -> Detector:
        geom_file = tmp_path / "unequal_panels.geom"
        geom_file.write_text(
            "photon_energy = 12.0keV\n"
            "clen = 0.1m\n"
            "res = 1\n"
            "p0/corner_x = -0.5\n"
            "p0/corner_y = -0.5\n"
            "p0/fs = +1.0x+0.0y\n"
            "p0/ss = +0.0x+1.0y\n"
            "p0/min_fs = 0\n"
            "p0/max_fs = 1\n"
            "p0/min_ss = 0\n"
            "p0/max_ss = 1\n"
            "p1/corner_x = -0.5\n"
            "p1/corner_y = +1.5\n"
            "p1/fs = +1.0x+0.0y\n"
            "p1/ss = +0.0x+1.0y\n"
            "p1/min_fs = 0\n"
            "p1/max_fs = 1\n"
            "p1/min_ss = 2\n"
            "p1/max_ss = 4\n"
        )
        return read_crystfel(str(geom_file))

    @pytest.fixture
    def unequal_n_bins(self) -> int:
        return 6

    @pytest.fixture
    def unequal_radial_index(self, unequal_geometry: Detector, center: Tuple[float, float],
                             unequal_n_bins: int) -> IntArray:
        return unequal_geometry.radial_index(center, unequal_n_bins)

    @pytest.fixture
    def unequal_data(self, xp: NumPyNamespace) -> RealArray:
        return xp.arange(10, dtype=float).reshape(1, 5, 2)

    def expected_profiles(self, data: NDRealArray, radial_index: NDIntArray,
                          n_bins: int) -> Tuple[NDRealArray, NDRealArray, NDIntArray]:
        whitefield, std, counts = [], [], []
        for frame in np.reshape(data, (-1,) + radial_index.shape):
            frame_whitefield, frame_std, frame_counts = [], [], []
            for radius_bin in range(n_bins):
                values = frame[radial_index == radius_bin]
                frame_counts.append(values.size)
                if values.size:
                    frame_whitefield.append(values.mean())
                    frame_std.append(values.std())
                else:
                    frame_whitefield.append(0.0)
                    frame_std.append(0.0)
            whitefield.append(frame_whitefield)
            std.append(frame_std)
            counts.append(frame_counts)
        return np.asarray(whitefield), np.asarray(std), np.asarray(counts)

    def test_profiles_without_clipping(self, smooth_data: RealArray, radial_index: IntArray,
                                       n_bins: int, structure: Structure) -> None:
        detector = CrystData(data=smooth_data).online_detector(structure, radial_index, n_bins)
        profiles = detector.profiles(n_iter=0)
        expected_whitefield, expected_std, expected_counts = self.expected_profiles(
            asnumpy(smooth_data), asnumpy(radial_index), n_bins
        )

        assert profiles.whitefield.shape == (1, n_bins)
        assert profiles.std.shape == (1, n_bins)
        assert profiles.counts.shape == (1, n_bins)
        np.testing.assert_allclose(asnumpy(profiles.whitefield), expected_whitefield)
        np.testing.assert_allclose(asnumpy(profiles.std), expected_std)
        np.testing.assert_array_equal(asnumpy(profiles.counts), expected_counts)

    def test_profiles_clip_bright_outlier(self, outlier_data: RealArray,
                                          radial_index: IntArray, n_bins: int,
                                          structure: Structure) -> None:
        detector = CrystData(data=outlier_data).online_detector(structure, radial_index, n_bins)
        profiles = detector.profiles(clip_snr=0.5, n_iter=1)

        assert asnumpy(profiles.counts)[0, 2] == 4
        assert asnumpy(profiles.whitefield)[0, 2] == pytest.approx(
            (22.0 + 30.0 + 14.0 + 32.0) / 4.0
        )

    def test_interval_matches_subsampled_radial_bins(self, interval_data: RealArray,
                                                     radial_index: IntArray, n_bins: int,
                                                     structure: Structure) -> None:
        detector = CrystData(data=interval_data).online_detector(structure, radial_index, n_bins)
        profiles = detector.profiles(interval=2, n_iter=0)

        assert int(asnumpy(profiles.counts).sum()) == 5

    def test_profiles_support_unequal_panel_sizes(self, unequal_data: RealArray,
                                                  unequal_radial_index: IntArray,
                                                  unequal_n_bins: int,
                                                  structure: Structure) -> None:
        detector = CrystData(data=unequal_data).online_detector(
            structure, unequal_radial_index, unequal_n_bins
        )
        profiles = detector.profiles(n_iter=0)
        expected_whitefield, expected_std, expected_counts = self.expected_profiles(
            asnumpy(unequal_data), asnumpy(unequal_radial_index), unequal_n_bins
        )

        assert int(asnumpy(profiles.counts).sum()) == 10
        np.testing.assert_allclose(asnumpy(profiles.whitefield), expected_whitefield)
        np.testing.assert_allclose(asnumpy(profiles.std), expected_std)
        np.testing.assert_array_equal(asnumpy(profiles.counts), expected_counts)

    @pytest.mark.parametrize("dtype", [np.uint32, np.int32])
    def test_detect_regions_uses_radial_threshold(self, region_data: RealArray,
                                                  radial_index: IntArray, n_bins: int,
                                                  structure: Structure, dtype: type) -> None:
        data = region_data.astype(dtype)
        detector = CrystData(data=data).online_detector(structure, radial_index, n_bins)
        profiles = detector.profiles(clip_snr=0.5, n_iter=1)
        labeled = detector.detect_regions(min_snr=5.0, npts=1, profiles=profiles)

        assert labeled.index.size == 1
        assert asnumpy(labeled.labels)[0, 1, 1] == 1

    def test_npts_filters_small_regions(self, region_data: RealArray,
                                        radial_index: IntArray, n_bins: int,
                                        structure: Structure) -> None:
        detector = CrystData(data=region_data).online_detector(structure, radial_index, n_bins)
        profiles = detector.profiles(clip_snr=0.5, n_iter=1)
        labeled = detector.detect_regions(min_snr=5.0, npts=3, profiles=profiles)

        assert labeled.index.size == 0

class TestCudaOnlineDetector(TestOnlineDetector):
    @pytest.fixture
    def xp(self) -> CuPyNamespace:
        if CuPy is None:
            pytest.skip("CuPy is not available")
        return CuPy

    @pytest.fixture
    def radial_index(self, geometry: Detector, center: Tuple[float, float],
                     n_bins: int) -> IntArray:
        if CuPy is None:
            pytest.skip("CuPy is not available")
        return geometry.radial_index(center, n_bins, xp=CuPy)

    @pytest.fixture
    def unequal_radial_index(self, unequal_geometry: Detector, center: Tuple[float, float],
                             unequal_n_bins: int) -> IntArray:
        if CuPy is None:
            pytest.skip("CuPy is not available")
        return unequal_geometry.radial_index(center, unequal_n_bins, xp=CuPy)

    def expected_profiles(self, data: NDRealArray, radial_index: NDIntArray,
                          n_bins: int) -> Tuple[NDRealArray, NDRealArray, NDIntArray]:
        return super().expected_profiles(asnumpy(data), asnumpy(radial_index), n_bins)

    def test_cuda_geometry_matches_cpu(self, geometry: Detector, center: Tuple[float, float],
                                       n_bins: int) -> None:
        if CuPy is None:
            pytest.skip("CuPy is not available")

        np.testing.assert_allclose(
            asnumpy(geometry.pixel_map(xp=CuPy)),
            geometry.pixel_map(xp=NumPy)
        )
        np.testing.assert_allclose(
            asnumpy(geometry.radii(center, xp=CuPy)),
            geometry.radii(center, xp=NumPy)
        )
        np.testing.assert_array_equal(
            asnumpy(geometry.radial_index(center, n_bins, xp=CuPy)),
            geometry.radial_index(center, n_bins, xp=NumPy)
        )
