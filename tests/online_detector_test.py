from pathlib import Path
from typing import Tuple
import pytest
from cbclib_v2 import CrystData, read_crystfel
from cbclib_v2 import asnumpy, Detector
from cbclib_v2.annotations import (CuPy, CuPyNamespace, IntArray, NumPy, NumPyNamespace,
                                   RealArray)
from cbclib_v2.label import Structure
from cbclib_v2.test_util import check_close

TestNamespace = CuPyNamespace | NumPyNamespace

class TestOnlineDetector:
    @pytest.fixture(params=['cpu', 'gpu'])
    def platform(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture
    def xp(self, platform: str) -> TestNamespace:
        if platform == 'cpu':
            return NumPy
        elif platform == 'gpu':
            return CuPy
        else:
            raise ValueError(f"Unknown platform: {platform}")

    @pytest.fixture
    def center(self) -> Tuple[int, int]:
        return (0, 0)

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
    def radial_index(self, xp: TestNamespace, geometry: Detector, center: Tuple[int, int],
                     n_bins: int) -> IntArray:
        out = xp.empty(geometry.shape[-2:], dtype=xp.int32)
        return geometry.radial_index(out, center, n_bins)

    @pytest.fixture
    def structure(self) -> Structure:
        return Structure([0, 1, 1], 1)

    @pytest.fixture
    def smooth_data(self, xp: TestNamespace) -> RealArray:
        return xp.asarray([[
            [10.0, 20.0, 22.0],
            [12.0, 24.0, 30.0],
            [14.0, 32.0, 34.0],
        ]], dtype=xp.float32)

    @pytest.fixture
    def outlier_data(self, xp: TestNamespace) -> RealArray:
        return xp.asarray([[
            [10.0, 20.0, 22.0],
            [12.0, 200.0, 30.0],
            [14.0, 32.0, 34.0],
        ]], dtype=xp.float32)

    @pytest.fixture
    def interval_data(self, xp: TestNamespace) -> RealArray:
        return xp.arange(9, dtype=xp.float32).reshape(1, 3, 3)

    @pytest.fixture
    def region_data(self, xp: TestNamespace) -> RealArray:
        return xp.asarray([[
            [10.0, 20.0, 22.0],
            [12.0, 100.0, 30.0],
            [14.0, 32.0, 34.0],
        ]], dtype=xp.float32)

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
    def unequal_radial_index(self, xp: TestNamespace, unequal_geometry: Detector,
                             center: Tuple[int, int], unequal_n_bins: int) -> IntArray:
        out = xp.empty(unequal_geometry.shape[-2:], dtype=xp.int64)
        return unequal_geometry.radial_index(out, center, unequal_n_bins)

    @pytest.fixture
    def unequal_data(self, xp: TestNamespace) -> RealArray:
        return xp.arange(10, dtype=float).reshape(1, 5, 2)

    def expected_profiles(self, data: RealArray, radial_index: IntArray, n_bins: int,
                          xp: TestNamespace) -> Tuple[RealArray, RealArray, IntArray]:
        whitefield, std, counts = [], [], []
        for frame in xp.reshape(data, (-1,) + radial_index.shape):
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
        return xp.asarray(whitefield), xp.asarray(std), xp.asarray(counts)

    def test_without_clipping(self, geometry: Detector, center: Tuple[int, int],
                              smooth_data: RealArray, radial_index: IntArray,
                              n_bins: int, structure: Structure, xp: TestNamespace):
        pixel_out = xp.empty((3,) + geometry.shape[-2:], dtype=xp.float32)
        radius_out = xp.empty(geometry.shape[-2:], dtype=xp.float32)

        pixel_result = geometry.pixel_map(pixel_out)
        radius_result = geometry.radii(radius_out, center)

        assert pixel_result.dtype == xp.float32
        assert radius_result.dtype == xp.float32
        assert radial_index.dtype == xp.int32
        default_pixel_out = xp.empty((3,) + geometry.shape[-2:], dtype=xp.float64)
        default_radius_out = xp.empty(geometry.shape[-2:], dtype=xp.float64)
        default_pixel = (
            geometry.pixel_map(default_pixel_out) if xp is xp else geometry.pixel_map(pixel_out)
        )
        default_radius = (
            geometry.radii(default_radius_out, center) if xp is xp
            else geometry.radii(radius_out, center)
        )
        check_close(pixel_result, default_pixel)
        check_close(radius_result, default_radius)

        detector = CrystData(data=smooth_data).online_detector(structure, radial_index, n_bins)
        profiles = detector.profiles(n_iter=0)
        expected_whitefield, expected_std, expected_counts = self.expected_profiles(
            smooth_data, radial_index, n_bins, xp
        )

        assert profiles.whitefield.shape == (1, n_bins)
        assert profiles.std.shape == (1, n_bins)
        assert profiles.counts.shape == (1, n_bins)
        check_close(profiles.whitefield, expected_whitefield)
        check_close(profiles.std, expected_std, rtol=1e-6)
        assert xp.all(profiles.counts == expected_counts)

    def test_clip_bright_outlier(self, outlier_data: RealArray, radial_index: IntArray,
                                 n_bins: int, structure: Structure):
        detector = CrystData(data=outlier_data).online_detector(structure, radial_index, n_bins)
        profiles = detector.profiles(clip_snr=0.5, n_iter=1)

        assert profiles.counts[0, 2] == 4
        assert float(profiles.whitefield[0, 2]) == pytest.approx((22.0 + 30.0 + 14.0 + 32.0) / 4.0)

    def test_interval(self, interval_data: RealArray, radial_index: IntArray,
                      n_bins: int, structure: Structure):
        detector = CrystData(data=interval_data).online_detector(structure, radial_index, n_bins)
        profiles = detector.profiles(interval=2, n_iter=0)

        assert int(profiles.counts.sum()) == 5

    def test_unequal_panel(self, unequal_data: RealArray, unequal_radial_index: IntArray,
                           unequal_n_bins: int, structure: Structure, xp: TestNamespace):
        detector = CrystData(data=unequal_data).online_detector(
            structure, unequal_radial_index, unequal_n_bins
        )
        profiles = detector.profiles(n_iter=0)
        expected_whitefield, expected_std, expected_counts = self.expected_profiles(
            unequal_data, unequal_radial_index, unequal_n_bins, xp)

        assert unequal_radial_index.dtype == xp.int64
        assert xp.sum(profiles.counts) == 10
        check_close(profiles.whitefield, expected_whitefield)
        check_close(profiles.std, expected_std)
        assert xp.all(profiles.counts == expected_counts)

    @pytest.mark.parametrize("dtype", [NumPy.uint32, NumPy.int32])
    def test_radial_threshold(self, region_data: RealArray, radial_index: IntArray,
                              n_bins: int, structure: Structure, dtype: type):
        data = region_data.astype(dtype)
        detector = CrystData(data=data).online_detector(structure, radial_index, n_bins)
        profiles = detector.profiles(clip_snr=0.5, n_iter=1)
        labeled = detector.detect_regions(min_snr=5.0, npts=1, profiles=profiles)

        assert labeled.index.size == 1
        assert asnumpy(labeled.labels)[0, 1, 1] == 1

    def test_filter(self, region_data: RealArray, radial_index: IntArray,
                    n_bins: int, structure: Structure):
        detector = CrystData(data=region_data).online_detector(structure, radial_index, n_bins)
        profiles = detector.profiles(clip_snr=0.5, n_iter=1)
        labeled = detector.detect_regions(min_snr=5.0, npts=3, profiles=profiles)

        assert labeled.index.size == 0
