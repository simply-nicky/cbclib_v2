import sys
from math import prod
from typing import Callable, Dict, Tuple
import pytest
from cbclib_v2 import add_at, default_rng, Lines
from cbclib_v2.annotations import (CPArray, CuPy, CuPyNamespace, Generator, IntArray, NDArray,
                                   NumPy, NumPyNamespace, RealArray, Shape)
from cbclib_v2.ndimage import accumulate_lines, draw_lines, write_lines
from cbclib_v2.test_util import check_close

TestNamespace = NumPyNamespace | CuPyNamespace
TestGenerator = Generator[NDArray] | Generator[CPArray]
Kernel = Callable[[RealArray, RealArray], RealArray]
WriteResult = Tuple[IntArray, IntArray, RealArray]

@pytest.mark.parametrize("ndim,shape", [(2, (4, 3, 16, 22)), (3, (2, 20, 16, 22))])
class TestDrawLine():
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

    def kernel_dict(self, xp: TestNamespace) -> Dict[str, Kernel]:
        def gaussian(x, sigma):
            return xp.where(xp.abs(x) < sigma,
                            xp.exp(-(3 * x / sigma)**2 / 2) / xp.sqrt(2 * xp.pi), 0)
        def rectangular(x, sigma):
            return xp.where(xp.abs(x) < sigma, 1, 0)

        return {'gaussian': gaussian, 'rectangular': rectangular}

    @pytest.fixture(params=[43,])
    def n_lines(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def out(self, shape: Shape, xp: TestNamespace) -> RealArray:
        return xp.zeros(shape)

    @pytest.fixture(params=[10.0])
    def length(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[2.0])
    def width(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=['gaussian', 'rectangular'])
    def kernel(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture(params=[2.0,])
    def max_val(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[10,])
    def n_terms(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def terms(self, rng: TestGenerator, n_terms: int, n_lines: int) -> IntArray:
        return rng.integers(0, n_terms - 1, size=n_lines)

    @pytest.fixture
    def frames(self, rng: TestGenerator, shape: Shape, ndim: int, n_terms: int) -> IntArray:
        return rng.integers(0, prod(shape[:-ndim]) - 1, size=n_terms)

    @pytest.fixture
    def indices(self, terms: IntArray, frames: IntArray) -> IntArray:
        return frames[terms]

    @pytest.fixture
    def lines(self, rng: TestGenerator, shape: Shape, ndim: int, n_lines: int,
              length: float, xp: TestNamespace) -> Lines:
        lengths = length * rng.random((n_lines,))
        pt0 = xp.array(shape[:-ndim - 1:-1]) * rng.random((n_lines, ndim))
        vec = rng.normal(xp.zeros(ndim), size=(n_lines, ndim))
        pt1 = pt0 + vec * (lengths / xp.sqrt(xp.sum(vec**2, axis=-1)))[:, None]
        return Lines(xp.concat((pt0, pt1), axis=-1))

    @pytest.fixture
    def image(self, out: RealArray, lines: Lines, width: float, indices: IntArray, max_val: float,
              kernel: str) -> RealArray:
        return draw_lines(out, lines.lines, indices, width=width, max_val=max_val, kernel=kernel)

    @pytest.fixture
    def accumulated(self, out: RealArray, lines: Lines, width: float, terms: IntArray,
                    frames: IntArray, max_val: float, kernel: str) -> RealArray:
        return accumulate_lines(out, lines.lines, terms, frames, width=width, max_val=max_val,
                                kernel=kernel)

    def test_ref_count(self, lines: Lines, width: float, shape: Shape, ndim: int,
                       xp: TestNamespace):
        out = xp.zeros(shape[-ndim:])
        lines_array = lines.lines
        out_refcount = sys.getrefcount(out)
        lines_refcount = sys.getrefcount(lines_array)

        out = draw_lines(out, lines_array, width=width)

        assert out_refcount == sys.getrefcount(out)
        assert lines_refcount == sys.getrefcount(lines_array)

    def test_empty_lines(self, shape: Shape, ndim: int, xp: TestNamespace):
        image = draw_lines(xp.zeros(shape[-ndim:]), xp.zeros((0, 2 * ndim)), width=1.0)
        assert xp.sum(image) == 0.0

    def test_image_wrong_size_lines(self, out: RealArray, lines: Lines, width: float,
                                    indices: IntArray):
        with pytest.raises(ValueError, match="idxs has an invalid size"):
            _ = draw_lines(out, lines.lines[::2], indices, width=width)

    def test_image_index_range(self, out: RealArray, lines: Lines, width: float,
                               indices: IntArray, ndim: int, xp: TestNamespace):
        n_frames = prod(out.shape[:-ndim])
        for value in (-1, n_frames):
            with pytest.raises(IndexError, match="idxs range"):
                _ = draw_lines(out, lines.lines, xp.full_like(indices, value), width=width)

    def test_image_strided_indices(self, out: RealArray, lines: Lines, width: float,
                                   indices: IntArray, xp: TestNamespace):
        storage = xp.stack((indices, xp.zeros_like(indices)), axis=-1).reshape(-1)

        image = draw_lines(out, lines.lines, storage[::2], width=width)
        expected = draw_lines(xp.zeros_like(out), lines.lines, indices, width=width)

        check_close(image, expected)

    def test_zero_width(self, out: RealArray, lines: Lines, indices: IntArray, kernel: str,
                        xp: TestNamespace):
        image = draw_lines(out, lines.lines, indices, width=0.0, kernel=kernel)

        assert xp.sum(image) == 0

    def test_negative_width(self, out: RealArray, lines: Lines, indices: IntArray, kernel: str,
                            xp: TestNamespace):
        image = draw_lines(out, lines.lines, indices, width=-1.0, kernel=kernel)

        assert xp.sum(image) == 0

    @pytest.mark.slow
    def test_max_val(self, image: RealArray, n_lines: int, max_val: float, xp: TestNamespace):
        assert xp.min(image) == 0
        assert xp.all(xp.max(image, axis=(-2, -1)) <= n_lines * max_val)

    @pytest.fixture
    def image_numpy(self, lines: Lines, width: float, indices: IntArray, shape: Shape, ndim: int,
                    max_val: float, kernel: str, xp: TestNamespace) -> RealArray:
        kernel_func = self.kernel_dict(xp)[kernel]
        pts = xp.meshgrid(*(xp.arange(length) for length in shape[-ndim:]), indexing='ij')
        pts = xp.stack(pts[::-1], axis=-1)[..., None, :]

        frames = []
        for fnum in range(prod(shape[:-ndim])):
            lns = lines[indices == fnum]
            dist = pts - lns.project(pts)
            frame = max_val * kernel_func(xp.sqrt(xp.sum(dist**2, axis=-1)), xp.asarray(width))
            frames.append(xp.sum(frame, axis=-1))
        return xp.stack(frames).reshape(shape)

    def test_draw_line_image(self, image: RealArray, image_numpy: RealArray):
        check_close(image, image_numpy)

    def test_accumulate_lines(self, accumulated, image_numpy: RealArray):
        check_close(accumulated, image_numpy)

@pytest.mark.parametrize("ndim,shape", [(2, (2, 9, 11)), (3, (2, 7, 9, 11))])
class TestWriteLines:
    @pytest.fixture(params=['cpu', 'gpu'])
    def xp(self, request: pytest.FixtureRequest) -> TestNamespace:
        if request.param == 'cpu':
            return NumPy
        if CuPy is None:
            pytest.skip("CuPy is not available")
        return CuPy

    @pytest.fixture
    def lines(self, ndim: int, xp: TestNamespace) -> RealArray:
        if ndim == 2:
            data = [[1.2, 2.1, 8.4, 6.3], [8.4, 6.3, 1.2, 2.1]]
        else:
            data = [[1.2, 2.1, 1.4, 8.4, 6.3, 4.8],
                    [8.4, 6.3, 4.8, 1.2, 2.1, 1.4]]
        return xp.asarray(data, dtype=xp.float64)

    @pytest.fixture
    def indices(self, xp: TestNamespace) -> IntArray:
        return xp.asarray([0, 1], dtype=xp.int64)

    def test_reconstructs_drawn_image(self, lines: RealArray, indices: IntArray, shape: Shape,
                                      xp: TestNamespace):
        pixel_indices, _, values = write_lines(lines, shape, indices, width=1.7,
                                                kernel='triangular')
        footprint = xp.zeros((prod(shape),), dtype=values.dtype)
        footprint = add_at(footprint, pixel_indices, values)
        image = draw_lines(xp.zeros(shape), lines, indices, width=1.7,
                           kernel='triangular')

        check_close(footprint.reshape(shape), image)

    def test_preserves_overlapping_pixels(self, lines: RealArray, shape: Shape,
                                          ndim: int, xp: TestNamespace):
        duplicate = xp.stack((lines[0], lines[0]))
        pixel_indices, line_indices, values = write_lines(
            duplicate, shape[-ndim:], width=1.7, kernel='rectangular')
        first = line_indices == 0
        second = line_indices == 1

        assert xp.any(first)
        assert xp.all(pixel_indices[first] == pixel_indices[second])
        assert xp.all(values[first] == values[second])

    def test_empty_lines(self, ndim: int, shape: Shape, xp: TestNamespace):
        result = write_lines(xp.zeros((0, 2 * ndim)), shape[-ndim:], width=1.0)

        assert all(array.size == 0 for array in result)

    def test_index_size(self, lines: RealArray, indices: IntArray, shape: Shape):
        with pytest.raises(ValueError, match="idxs has an invalid size"):
            _ = write_lines(lines, shape, indices[:1], width=1.0)

    def test_index_range(self, lines: RealArray, indices: IntArray, shape: Shape,
                         xp: TestNamespace):
        for value in (-1, shape[0]):
            with pytest.raises(IndexError, match="idxs range"):
                _ = write_lines(lines, shape, xp.full_like(indices, value), width=1.0)

    def test_non_finite_lines_are_ignored(self, ndim: int, shape: Shape,
                                          xp: TestNamespace):
        valid = xp.arange(2 * ndim, dtype=xp.float64)
        lines = xp.stack((valid, xp.full_like(valid, float('nan'))))
        _, line_indices, _ = write_lines(lines, shape[-ndim:], width=1.0,
                                         kernel='rectangular')

        assert xp.any(line_indices == 0)
        assert not xp.any(line_indices == 1)

class TestAccumulateCurves():
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
    def out(self, xp: TestNamespace) -> RealArray:
        return xp.zeros((1, 12, 12, 12))

    @pytest.fixture
    def frames(self, xp: TestNamespace) -> IntArray:
        return xp.array([0])

    def test_internal_max(self, out: RealArray, frames: IntArray, xp: TestNamespace):
        curve = xp.array([[[5.0, 5.0, 5.0],
                           [6.0, 5.0, 5.0],
                           [7.0, 5.0, 5.0]]])
        terms = xp.array([0])

        image = accumulate_lines(out, curve, terms, frames, width=0.6, kernel='rectangular',
                                 in_overlap='sum')

        assert image[0, 5, 5, 6] == 1.0

    def test_curves_overlap(self, out: RealArray, frames: IntArray, xp: TestNamespace):
        curves = xp.array([[[5.0, 5.0, 5.0],
                            [7.0, 5.0, 5.0]],
                           [[5.0, 5.0, 5.0],
                            [7.0, 5.0, 5.0]]])
        terms = xp.array([0, 0])

        image_sum = accumulate_lines(out.copy(), curves, terms, frames, width=0.6,
                                     kernel='rectangular', in_overlap='sum')
        image_max = accumulate_lines(out.copy(), curves, terms, frames, width=0.6,
                                     kernel='rectangular', in_overlap='max')

        assert image_sum[0, 5, 5, 6] == 2.0
        assert image_max[0, 5, 5, 6] == 1.0

    def test_curve_fail(self, out: RealArray, frames: IntArray, xp: TestNamespace):
        curve = xp.array([[[5.0, 5.0, 5.0]]])
        terms = xp.array([0])

        with pytest.raises(ValueError, match="at least two points"):
            _ = accumulate_lines(out, curve, terms, frames, width=0.6)

    def test_curve_matches_line(self, out: RealArray, frames: IntArray, xp: TestNamespace):
        curve = xp.array([[[5.0, 5.0, 5.0],
                           [7.0, 5.0, 5.0]]])
        line = xp.array([[5.0, 5.0, 5.0, 7.0, 5.0, 5.0]])
        terms = xp.array([0])

        curve_image = accumulate_lines(out.copy(), curve, terms, frames, width=0.6,
                                       kernel='rectangular')
        line_image = accumulate_lines(out.copy(), line, terms, frames, width=0.6,
                                      kernel='rectangular')

        check_close(curve_image, line_image)

    def test_oblique_curve_matches_line_regression(self, xp: TestNamespace):
        out = xp.zeros((1, 16, 16, 16))
        curve = xp.array([[[9.479237896005714, 7.303016452804371, 8.176423218945516],
                           [9.22149370307207, 8.238012375216442, 7.043417998706573]]])
        line = xp.reshape(curve, (1, 6))
        terms = xp.array([0])
        frames = xp.array([0])

        curve_image = accumulate_lines(out.copy(), curve, terms, frames, width=1.5,
                                       kernel='rectangular')
        line_image = accumulate_lines(out.copy(), line, terms, frames, width=1.5,
                                      kernel='rectangular')

        check_close(curve_image, line_image)
        assert float(line_image[0, 8, 8, 8]) == 1.0

    @pytest.mark.parametrize("ndim,shape,curves", [
        (2, (1, 32, 32), [[[5.25, 5.75], [9.75, 7.25]],
                          [[13.5, 4.25], [15.25, 12.75]],
                          [[4.5, 20.5], [18.25, 18.75]],
                          [[22.0, 8.0], [27.5, 14.5]]]),
        (3, (1, 24, 24, 24), [[[5.25, 5.75, 6.5], [9.75, 7.25, 8.25]],
                              [[13.5, 4.25, 7.0], [15.25, 12.75, 9.5]],
                              [[4.5, 20.5, 5.5], [18.25, 18.75, 7.25]],
                              [[16.0, 8.0, 18.0], [20.5, 14.5, 12.5]]]),
    ])
    def test_generated_curves_match_lines(self, ndim: int, shape: Shape, curves: RealArray,
                                          xp: TestNamespace):
        out = xp.zeros(shape)
        curve_array = xp.asarray(curves)
        line_array = xp.reshape(curve_array, (curve_array.shape[0], 2 * ndim))
        terms = xp.arange(curve_array.shape[0])
        frames = xp.zeros(curve_array.shape[0], dtype=terms.dtype)

        curve_image = accumulate_lines(out.copy(), curve_array, terms, frames, width=1.5,
                                       kernel='rectangular')
        line_image = accumulate_lines(out.copy(), line_array, terms, frames, width=1.5,
                                      kernel='rectangular')

        check_close(curve_image, line_image)

    def test_grouped_segments(self, out: RealArray, frames: IntArray, xp: TestNamespace):
        lines = xp.array([[[5.0, 5.0, 5.0, 6.0, 5.0, 5.0],
                           [6.0, 5.0, 5.0, 7.0, 5.0, 5.0]]])
        terms = xp.array([0])

        image = accumulate_lines(out, lines, terms, frames, width=0.6, kernel='rectangular')

        assert float(image[0, 5, 5, 6]) == 2.0
