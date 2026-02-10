import sys
from math import prod
from typing import Callable, Dict, Tuple
import pytest
from cbclib_v2 import device
from cbclib_v2 import Lines, add_at
from cbclib_v2.annotations import CuPy, Device, IntArray, RealArray, Shape
from cbclib_v2.ndimage import accumulate_lines, draw_lines, write_lines
from cbclib_v2.test_util import check_close, TestGenerator, TestNamespace

Kernel = Callable[[RealArray, RealArray], RealArray]
WriteResult = Tuple[IntArray, IntArray, RealArray]

@pytest.mark.parametrize("ndim,shape", [(2, (4, 3, 16, 22)), (3, (2, 20, 16, 22))])
class TestDrawLine():
    @pytest.fixture
    def xp(self, test_xp: TestNamespace) -> TestNamespace:
        return test_xp

    @pytest.fixture
    def rng(self, test_rng: TestGenerator) -> TestGenerator:
        return test_rng

    def kernel_dict(self, xp: TestNamespace) -> Dict[str, Kernel]:
        def biweight(x, sigma):
            return 0.9375 * xp.clip(1 - (x / sigma)**2, 0, xp.inf)**2
        def gaussian(x, sigma):
            return xp.where(xp.abs(x) < sigma,
                            xp.exp(-(3 * x / sigma)**2 / 2) / xp.sqrt(2 * xp.pi), 0)
        def parabolic(x, sigma):
            return 0.75 * xp.clip(1 - (x / sigma)**2, 0, xp.inf)
        def rectangular(x, sigma):
            return xp.where(xp.abs(x) < sigma, 1, 0)
        def triangular(x, sigma):
            return xp.clip(1 - xp.abs(x / sigma), 0, xp.inf)

        return {'biweight': biweight, 'gaussian': gaussian, 'parabolic': parabolic,
                'rectangular': rectangular, 'triangular': triangular}

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

    @pytest.fixture(params=[0, 3])
    def kernel(self, rng: TestGenerator, request: pytest.FixtureRequest, xp: TestNamespace
               ) -> str:
        keys = list(self.kernel_dict(xp).keys())
        index = int(rng.integers(0, len(keys)) + request.param) % len(keys)
        return keys[index]

    @pytest.fixture(params=[1.0, 10.0])
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
              kernel: str, test_device: Device, xp: TestNamespace) -> RealArray:
        with device.context(test_device):
            image = draw_lines(out, lines.to_lines(width), indices, max_val=max_val, kernel=kernel)
        return xp.asarray(image)

    @pytest.fixture
    def accumulated(self, out: RealArray, lines: Lines, width: float, terms: IntArray,
                    frames: IntArray, max_val: float, kernel: str, test_device: Device
                    ) -> RealArray:
        with device.context(test_device):
            image = accumulate_lines(out, lines.to_lines(width), terms, frames, max_val, kernel)
        return image

    @pytest.fixture
    def arrays(self, lines: Lines, width: float, indices: IntArray, shape: Shape, max_val: float,
              kernel: str, cpu_device: Device, xp: TestNamespace) -> WriteResult:
        if xp is CuPy:
            pytest.skip("Skipping write_lines test for CuPy backend")
        with device.context(cpu_device):
            idxs, ids, values = write_lines(lines.to_lines(width), shape, indices, max_val=max_val,
                                            kernel=kernel)
        return xp.asarray(idxs), xp.asarray(ids), xp.asarray(values)

    def test_ref_count(self, lines: Lines, width: float, shape: Shape, ndim: int,
                       test_device: Device, xp: TestNamespace):
        out = xp.zeros(shape[-ndim:])
        lines_array = lines.to_lines(width)
        out_refcount = sys.getrefcount(out)
        lines_refcount = sys.getrefcount(lines_array)

        with device.context(test_device):
            draw_lines(out, lines_array)

        assert out_refcount == sys.getrefcount(out)
        assert lines_refcount == sys.getrefcount(lines_array)

    def test_empty_lines(self, shape: Shape, ndim: int, test_device: Device, xp: TestNamespace):
        with device.context(test_device):
            image = draw_lines(xp.zeros(shape[-ndim:]), xp.zeros((0, 2 * ndim + 1)))

        assert xp.sum(image) == 0.0

    @pytest.mark.xfail(raises=ValueError)
    def test_image_wrong_size_lines(self, out: RealArray, lines: Lines, width: float,
                                    indices: IntArray, test_device: Device):
        with device.context(test_device):
            image = draw_lines(out, lines.to_lines(width)[::2], indices)

    @pytest.mark.xfail(raises=ValueError)
    def test_table_wrong_size_lines(self, lines: Lines, width: float, indices: IntArray,
                                    shape: Shape, cpu_device: Device):
        with device.context(cpu_device):
            idxs, ids, values = write_lines(lines.to_lines(width)[::2], shape, indices)

    def test_zero_width(self, out: RealArray, lines: Lines, indices: IntArray, kernel: str,
                        test_device: Device, xp: TestNamespace):
        zero_lines = lines.to_lines(0.0)
        with device.context(test_device):
            image = draw_lines(out, zero_lines, indices, kernel=kernel)

        assert xp.sum(image) == 0

    def test_negative_width(self, out: RealArray, lines: Lines, indices: IntArray, kernel: str,
                            test_device: Device, xp: TestNamespace):
        neg_lines = lines.to_lines(-1.0)
        with device.context(test_device):
            image = draw_lines(out, neg_lines, indices, kernel=kernel)

        assert xp.sum(image) == 0

    @pytest.mark.slow
    def test_max_val(self, image: RealArray, arrays: WriteResult, n_lines: int, max_val: float,
                     xp: TestNamespace):
        idxs, ids, values = arrays
        assert xp.min(image) == 0
        assert xp.all(xp.max(image, axis=(-2, -1)) <= n_lines * max_val)
        assert xp.min(idxs) >= 0 and xp.max(idxs) < image.size
        assert xp.min(ids) >= 0 and xp.max(ids) < n_lines
        assert xp.min(values) >= 0.0 and xp.max(values) <= max_val

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

    def test_draw_line_table(self, arrays: WriteResult, image_numpy: RealArray, xp: TestNamespace):
        image = xp.zeros(image_numpy.shape)
        idxs, _, values = arrays
        image = add_at(image, xp.unravel_index(idxs, image.shape), values)
        check_close(image, image_numpy)

    def test_accumulate_lines(self, accumulated, image_numpy: RealArray):
        check_close(accumulated, image_numpy)
