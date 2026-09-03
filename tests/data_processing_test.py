from dataclasses import dataclass
from typing import Literal, cast
import pytest
from cbclib_v2 import CrystMetadata, LSQData
from cbclib_v2.annotations import (BoolArray, CuPy, CuPyNamespace, IntArray, NumPy,
                                   NumPyNamespace, RealArray)
from cbclib_v2.scripts import ScalingParameters, scale_background
from cbclib_v2.test_util import check_close

TestNamespace = NumPyNamespace | CuPyNamespace

@dataclass(frozen=True)
class ProjectionCase:
    metadata: CrystMetadata
    data: RealArray
    expected: RealArray

@dataclass(frozen=True)
class MaskCase:
    metadata: CrystMetadata
    data: RealArray
    pixels: tuple[IntArray, ...]

class BackendSuite:
    @pytest.fixture(params=['cpu', 'gpu'])
    def platform(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture
    def xp(self, platform: str) -> TestNamespace:
        if platform == 'cpu':
            return NumPy
        if CuPy is None:
            pytest.skip('CuPy is not available')
        return CuPy

class TestCrystMetadataProjection(BackendSuite):
    @pytest.fixture
    def lsq_case(self, xp: TestNamespace) -> ProjectionCase:
        flatfield = xp.ones((2, 3)) * 4.0
        fields = xp.asarray([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
                             [[0.0, 1.0, 1.0], [1.0, 0.0, -1.0]]])
        expected = xp.asarray([[2.0, -1.0], [-0.5, 3.0]])
        data = flatfield + xp.tensordot(expected, fields, axes=((-1,), (0,)))
        data = data.copy()
        data[..., 1, 2] = 1000.0
        mask = xp.asarray([[True, True, True], [True, True, False]])
        metadata = CrystMetadata(flatfield=flatfield, eigen_field=fields, mask=mask)
        return ProjectionCase(metadata, data, expected)

    @pytest.fixture
    def rejection_case(self, xp: TestNamespace) -> ProjectionCase:
        flatfield = xp.ones((2, 5)) * 5.0
        fields = xp.ones((1, 2, 5))
        data = xp.stack((flatfield + 2.0, flatfield - 1.0), axis=0)
        data = data.copy()
        data[0, 0, 0] += 20.0
        data[1, 0, 0] -= 5.0
        metadata = CrystMetadata(flatfield=flatfield, eigen_field=fields,
                                 mask=xp.ones((2, 5), dtype=bool),
                                 std=xp.ones((2, 5)) * 0.5)
        expected = xp.asarray([[2.0], [-1.5]])
        return ProjectionCase(metadata, data, expected)

    @pytest.fixture
    def std_case(self, xp: TestNamespace) -> ProjectionCase:
        flatfield = xp.ones((1, 10)) * 5.0
        data = flatfield[None] + 1.0
        data = data.copy()
        data[0, 0, 0] += 5.0
        metadata = CrystMetadata(flatfield=flatfield,
                                 eigen_field=xp.ones((1, 1, 10)),
                                 std=xp.zeros((1, 10)))
        return ProjectionCase(metadata, data, xp.asarray([[1.5]]))

    @pytest.fixture
    def flatfield_case(self, xp: TestNamespace) -> ProjectionCase:
        flatfield = xp.asarray([[1.0, 2.0], [3.0, 4.0]])
        expected = xp.asarray([[2.0], [3.0]])
        data = expected[:, :, None] * xp.reshape(flatfield, (1, -1))
        data = xp.reshape(data, (2, 2, 2)).copy()
        data[..., 1, 1] = 1000.0
        mask = xp.asarray([[True, True], [True, False]])
        return ProjectionCase(CrystMetadata(flatfield=flatfield, mask=mask), data, expected)

    @pytest.fixture
    def singular_case(self, xp: TestNamespace) -> ProjectionCase:
        metadata = CrystMetadata(flatfield=xp.zeros((2, 2)),
                                 eigen_field=xp.ones((2, 2, 2)))
        data = xp.ones((1, 2, 2)) * 4.0
        return ProjectionCase(metadata, data, xp.asarray([[2.0, 2.0]]))

    @pytest.fixture
    def mask_case(self, xp: TestNamespace) -> MaskCase:
        flatfield = xp.reshape(xp.arange(12), (3, 4)) + 5.0
        fields = xp.stack((xp.ones((3, 4)), flatfield / 10.0), axis=0)
        coefficients = xp.asarray([[2.0, -0.5]])
        data = flatfield + xp.tensordot(coefficients, fields, axes=((-1,), (0,)))
        metadata = CrystMetadata(flatfield=flatfield, eigen_field=fields)
        pixels = xp.unravel_index(xp.asarray([0, 3, 7, 11], dtype=int), flatfield.shape)
        return MaskCase(metadata, data, pixels)

    @pytest.fixture
    def empty_metadata(self, xp: TestNamespace) -> CrystMetadata:
        return CrystMetadata(flatfield=xp.ones((2, 2)))

    @pytest.fixture
    def empty_data(self, xp: TestNamespace) -> RealArray:
        return xp.ones((1, 2, 2))

    def test_lsq(self, lsq_case: ProjectionCase):
        result = lsq_case.metadata.project(lsq_case.data, n_iter=1)
        check_close(result.projection, lsq_case.expected)

    def test_rejection(self, rejection_case: ProjectionCase):
        result = rejection_case.metadata.project(rejection_case.data, clip_snr=3.0,
                                                 n_iter=2)
        check_close(result.projection, rejection_case.expected)

    def test_std_min(self, std_case: ProjectionCase):
        result = std_case.metadata.project(std_case.data, clip_snr=10.0,
                                           n_iter=2, std_min=1.0)
        check_close(result.projection, std_case.expected)

    def test_flatfield(self, flatfield_case: ProjectionCase):
        result = flatfield_case.metadata.project(flatfield_case.data, n_iter=1)
        check_close(result.projection, flatfield_case.expected)
        expected = flatfield_case.expected[..., None] * flatfield_case.metadata.flatfield
        check_close(result.apply(flatfield_case.metadata), expected)

    def test_singular(self, singular_case: ProjectionCase):
        result = singular_case.metadata.project(singular_case.data, n_iter=1)
        check_close(result.projection, singular_case.expected)

    def test_pixel_selection(self, mask_case: MaskCase):
        projection = mask_case.metadata.project(mask_case.data, n_iter=1)
        selected = projection.apply(mask_case.metadata[mask_case.pixels])
        full = projection.apply(mask_case.metadata)
        check_close(selected, full[(...,) + mask_case.pixels])

    def test_iterations(self, empty_metadata: CrystMetadata, empty_data: RealArray):
        with pytest.raises(ValueError, match='n_iter must be at least one'):
            empty_metadata.project(empty_data, n_iter=0)

    def test_missing_std(self, empty_metadata: CrystMetadata, empty_data: RealArray):
        with pytest.raises(ValueError, match='No std'):
            empty_metadata.project(empty_data, n_iter=2)

class TestScaleBackground(BackendSuite):
    @pytest.fixture
    def sampling_case(self, xp: TestNamespace) -> ProjectionCase:
        flatfield = xp.ones((4, 5)) * 3.0
        fields = xp.stack((xp.ones((4, 5)),
                           xp.reshape(xp.arange(20), (4, 5)) / 20.0), axis=0)
        expected = xp.asarray([[2.0, -0.5], [-1.0, 1.5]])
        data = flatfield + xp.tensordot(expected, fields, axes=((-1,), (0,)))
        metadata = CrystMetadata(flatfield=flatfield, eigen_field=fields)
        return ProjectionCase(metadata, data, expected)

    @pytest.fixture
    def empty_metadata(self, xp: TestNamespace) -> CrystMetadata:
        return CrystMetadata(flatfield=xp.ones((2, 2)))

    @pytest.fixture
    def empty_data(self, xp: TestNamespace) -> RealArray:
        return xp.ones((1, 2, 2))

    def test_sampling(self, sampling_case: ProjectionCase, xp: TestNamespace):
        frames = xp.asarray([5, 8])
        params = ScalingParameters(method='robust-lsq', good_fields=(0, 1), n_iter=1,
                                   n_pixels=7)

        first = scale_background(frames, sampling_case.data, sampling_case.metadata, params)
        second = scale_background(frames, sampling_case.data, sampling_case.metadata, params)

        # A sufficient deterministic pixel sample recovers the exact PCA background.
        check_close(first.whitefield, sampling_case.data)
        check_close(second.whitefield, first.whitefield)

    @pytest.mark.parametrize('n_pixels', [0, 5])
    def test_pixels(self, empty_metadata: CrystMetadata, empty_data: RealArray,
                    xp: TestNamespace, n_pixels: int):
        params = ScalingParameters(method='robust-lsq', n_iter=1, n_pixels=n_pixels)
        with pytest.raises(ValueError, match='Invalid n_pixels'):
            scale_background(xp.asarray([0]), empty_data, empty_metadata, params)

class TestLSQData(BackendSuite):
    @pytest.fixture
    def coefficients(self, xp: TestNamespace) -> RealArray:
        return xp.asarray([[2.0, -1.0], [-1.0, 3.0]])

    @pytest.fixture
    def W(self, xp: TestNamespace) -> RealArray:
        return xp.asarray([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]])

    @pytest.fixture
    def mask(self, xp: TestNamespace) -> BoolArray:
        return xp.asarray([[True, True, False], [True, True, False]])

    @pytest.fixture
    def data(self, coefficients: RealArray, W: RealArray, xp: TestNamespace) -> LSQData:
        y = xp.sum(coefficients[..., None] * W, axis=1)
        return LSQData(y=y, W=W)

    @pytest.fixture
    def masked(self, data: LSQData, mask: BoolArray) -> LSQData:
        return data.apply_mask(mask)

    def test_mask(self, masked: LSQData, xp: TestNamespace):
        check_close(masked.y, xp.asarray([[2.0, -1.0, 0.0], [-1.0, 3.0, 0.0]]))

    def test_solve(self, masked: LSQData, coefficients: RealArray):
        check_close(masked.solve(), coefficients)

    def test_frame_design(self, coefficients: RealArray, xp: TestNamespace):
        W = xp.asarray([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
                        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]])
        y = xp.sum(coefficients[..., None] * W, axis=1)
        check_close(LSQData(y=y, W=W).solve(), coefficients)

    def test_frame_shape(self, xp: TestNamespace):
        with pytest.raises(ValueError, match='W must have a shape'):
            LSQData(y=xp.ones((2, 3)), W=xp.ones((3, 2, 3)))

class TestScalingParameters:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def flatfield(self, xp: NumPyNamespace) -> RealArray:
        return xp.ones((2, 2)) * 5.0

    @pytest.fixture
    def metadata(self, flatfield: RealArray, xp: NumPyNamespace) -> CrystMetadata:
        return CrystMetadata(flatfield=flatfield, eigen_field=xp.ones((1, 2, 2)),
                             mask=xp.ones((2, 2), dtype=bool), std=xp.ones((2, 2)))

    @pytest.fixture
    def images(self, flatfield: RealArray) -> RealArray:
        return (flatfield + 2.0)[None]

    def test_projection(self, metadata: CrystMetadata, images: RealArray,
                        xp: NumPyNamespace):
        params = ScalingParameters(method='robust-lsq', clip_snr=4.0,
                                   n_iter=1, std_min=0.5, n_pixels=3)
        data = scale_background(xp.asarray([7]), images, metadata, params)
        check_close(data.whitefield, images)

    def test_no_scale(self, flatfield: RealArray, xp: NumPyNamespace):
        metadata = CrystMetadata(flatfield=flatfield)
        params = ScalingParameters(method='no-scale')
        data = scale_background(0, xp.ones((1, 2, 2)) * 8.0, metadata, params)
        check_close(data.whitefield, flatfield)

    def test_method(self, flatfield: RealArray, xp: NumPyNamespace):
        method = cast(Literal['no-scale', 'robust-lsq'], 'lsq')
        params = ScalingParameters(method=method)
        metadata = CrystMetadata(flatfield=flatfield)
        with pytest.raises(ValueError, match='Invalid method keyword'):
            scale_background(0, xp.ones((1, 2, 2)), metadata, params)
