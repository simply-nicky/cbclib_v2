from typing import Callable, Dict, Tuple
import numpy as np
import pytest
from cbclib_v2.annotations import ArrayNamespace, IntArray, NumPy, RealArray, Shape
from cbclib_v2.ndimage import draw_lines, write_lines
from cbclib_v2.test_util import check_close
from cbclib_v2.jax import add_at, project_to_streak

Kernel = Callable[[RealArray, RealArray], RealArray]
WriteResult = Tuple[IntArray, IntArray, RealArray]

class TestDrawLine():
    @pytest.fixture
    def xp(self) -> ArrayNamespace:
        return NumPy

    def kernel_dict(self, xp: ArrayNamespace) -> Dict[str, Kernel]:
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

    @pytest.fixture(params=[(30, 80),])
    def n_lines(self, request: pytest.FixtureRequest, rng: np.random.Generator) -> int:
        vmin, vmax = request.param
        return rng.integers(vmin, vmax)

    @pytest.fixture(params=[(10, 50, 4),])
    def shape(self, request: pytest.FixtureRequest, rng: np.random.Generator) -> Shape:
        vmin, vmax, size = request.param
        return tuple(rng.integers(vmin, vmax, size=size))

    @pytest.fixture(params=[2, 3])
    def ndim(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[10.0])
    def length(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[2.0])
    def width(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[0, 3])
    def kernel(self, rng: np.random.Generator, request: pytest.FixtureRequest,
               xp: ArrayNamespace) -> str:
        keys = list(self.kernel_dict(xp).keys())
        index = (rng.integers(0, len(keys)) + request.param) % len(keys)
        return keys[index]

    @pytest.fixture(params=[1.0, 10.0])
    def max_val(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def indices(self, rng: np.random.Generator, shape: Shape, ndim: int, n_lines: int,
                xp: ArrayNamespace) -> IntArray:
        return xp.asarray(rng.integers(0, np.prod(shape[:-ndim]) - 1, size=n_lines))

    @pytest.fixture
    def lines(self, rng: np.random.Generator, shape: Shape, ndim: int, n_lines: int,
              length: float, width: float, xp: ArrayNamespace) -> RealArray:
        lengths = length * rng.random((n_lines,))
        pt0 = xp.array(shape[:-ndim - 1:-1]) * rng.random((n_lines, ndim))
        vec = rng.normal(xp.zeros(ndim), size=(n_lines, ndim))
        pt1 = pt0 + vec * (lengths / xp.sqrt(xp.sum(vec**2, axis=-1)))[:, None]
        return xp.concatenate((pt0, pt1, xp.full((n_lines, 1), width)), axis=-1)

    @pytest.fixture
    def image(self, lines: RealArray, indices: IntArray, shape: Shape, max_val: float,
              kernel: str, xp: ArrayNamespace) -> RealArray:
        image = draw_lines(lines, shape, indices, max_val=max_val, kernel=kernel)
        return xp.asarray(image)

    @pytest.fixture
    def arrays(self, lines: RealArray, indices: IntArray, shape: Shape, max_val: float,
              kernel: str, xp: ArrayNamespace) -> WriteResult:
        idxs, ids, values = write_lines(lines, shape, indices, max_val=max_val, kernel=kernel)
        return xp.asarray(idxs), xp.asarray(ids), xp.asarray(values)

    def test_empty_lines(self, shape: Shape, xp: ArrayNamespace):
        image = draw_lines(xp.zeros((0, 5)), shape[-2:])
        idxs, ids, values = write_lines(xp.zeros((0, 5)), shape[-2:])
        assert xp.sum(image) == 0.0
        assert idxs.size == ids.size == values.size == 0

    @pytest.mark.xfail(raises=ValueError)
    def test_image_wrong_size_lines(self, lines: RealArray, indices: IntArray, shape: Shape):
        image = draw_lines(lines[::2], shape, indices)

    @pytest.mark.xfail(raises=ValueError)
    def test_table_wrong_size_lines(self, lines: RealArray, indices: IntArray, shape: Shape):
        idxs, ids, values = draw_lines(lines[::2], shape, indices)

    def test_zero_width(self, lines: RealArray, indices: IntArray, shape: Shape, kernel: str,
                        xp: ArrayNamespace):
        zero_lines = xp.concatenate((lines[..., :4], xp.zeros(lines.shape[:-1] + (1,))), axis=-1)
        image = draw_lines(zero_lines, shape, indices, kernel=kernel)
        idxs, ids, values = write_lines(zero_lines, shape, indices, kernel=kernel)
        assert xp.sum(image) == 0
        assert idxs.size == ids.size == values.size == 0

    def test_negative_width(self, lines: RealArray, indices: IntArray, shape: Shape, kernel: str,
                            xp: ArrayNamespace):
        neg_lines = xp.concatenate((lines[..., :4], xp.full(lines.shape[:-1] + (1,), -1)), axis=-1)
        image = draw_lines(neg_lines, shape, indices, kernel=kernel)
        idxs, ids, values = write_lines(neg_lines, shape, indices, kernel=kernel)
        assert xp.sum(image) == 0
        assert idxs.size == ids.size == values.size == 0

    @pytest.mark.slow
    def test_max_val(self, image: RealArray, arrays: WriteResult, n_lines: int, max_val: float,
                     xp: ArrayNamespace):
        idxs, ids, values = arrays
        assert xp.min(image) == 0
        assert xp.all(xp.max(image, axis=(-2, -1)) <= n_lines * max_val)
        assert xp.min(idxs) >= 0 and xp.max(idxs) < image.size
        assert xp.min(ids) >= 0 and xp.max(ids) < n_lines
        assert xp.min(values) >= 0.0 and xp.max(values) <= max_val

    @pytest.fixture
    def image_numpy(self, lines: RealArray, indices: IntArray, shape: Shape, ndim: int,
                    max_val: float, kernel: str, xp: ArrayNamespace) -> RealArray:
        kernel_func = self.kernel_dict(xp)[kernel]
        pts = xp.meshgrid(*(xp.arange(length) for length in shape[-ndim:]), indexing='ij')
        pts = xp.stack(pts[::-1], axis=-1)[..., None, :]

        frames = []
        for fnum in range(np.prod(shape[:-ndim])):
            lns = lines[indices == fnum]
            dist = pts - project_to_streak(pts, lns[:, :ndim], lns[:, ndim:2 * ndim], xp)
            frame = max_val * kernel_func(xp.sqrt(xp.sum(dist**2, axis=-1)), lns[:, 2 * ndim])
            frames.append(xp.sum(frame, axis=-1))
        return xp.stack(frames).reshape(shape)

    def test_draw_line_image(self, image: RealArray, image_numpy: RealArray):
        check_close(image, image_numpy)

    def test_draw_line_table(self, arrays: WriteResult, image_numpy: RealArray, xp: ArrayNamespace):
        image = xp.zeros(image_numpy.shape)
        idxs, _, values = arrays
        image = add_at(image, xp.unravel_index(idxs, image.shape), values, xp)
        check_close(image, image_numpy)
