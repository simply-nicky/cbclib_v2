from typing import Callable, Dict
import numpy as np
import pytest
from cbclib_v2.annotations import IntArray, RealArray, Shape, Table
from cbclib_v2.ndimage import draw_line_image, draw_line_table
from cbclib_v2.test_util import check_close
from cbclib_v2.jax import project_to_streak

class TestDrawLine():
    def kernel_dict(self) -> Dict[str, Callable[[RealArray, RealArray], RealArray]]:
        def biweight(x, sigma):
            return 0.9375 * np.clip(1 - (x / sigma)**2, 0, np.infty)**2
        def gaussian(x, sigma):
            return np.where(np.abs(x) < sigma,
                            np.exp(-(3 * x / sigma)**2 / 2) / np.sqrt(2 * np.pi), 0)
        def parabolic(x, sigma):
            return 0.75 * np.clip(1 - (x / sigma)**2, 0, np.infty)
        def rectangular(x, sigma):
            return np.where(np.abs(x) < sigma, 1, 0)
        def triangular(x, sigma):
            return np.clip(1 - np.abs(x / sigma), 0, np.infty)

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
    def kernel(self, rng: np.random.Generator, request: pytest.FixtureRequest) -> str:
        keys = list(self.kernel_dict().keys())
        index = (rng.integers(0, len(keys)) + request.param) % len(keys)
        return keys[index]

    @pytest.fixture(params=[1.0, 10.0])
    def max_val(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def idxs(self, rng: np.random.Generator, shape: Shape, ndim: int, n_lines: int) -> IntArray:
        return rng.integers(0, np.prod(shape[:-ndim]) - 1, size=n_lines)

    @pytest.fixture
    def lines(self, rng: np.random.Generator, shape: Shape, ndim: int, n_lines: int,
              length: float, width: float) -> RealArray:
        lengths = length * rng.random((n_lines,))
        pt0 = np.array(shape[-ndim:]) * rng.random((n_lines, ndim))
        vec = rng.normal(np.zeros(ndim), size=(n_lines, ndim))
        pt1 = pt0 + vec * (lengths / np.sqrt(np.sum(vec**2, axis=-1)))[:, None]
        return np.concatenate((pt0, pt1, np.full((n_lines, 1), width)), axis=-1)

    @pytest.fixture
    def image(self, lines: RealArray, idxs: IntArray, shape: Shape, max_val: float,
              kernel: str) -> RealArray:
        return draw_line_image(lines, shape, idxs, max_val=max_val, kernel=kernel)

    @pytest.fixture
    def table(self, lines: RealArray, idxs: IntArray, shape: Shape, max_val: float,
              kernel: str) -> Table:
        return draw_line_table(lines, shape, idxs, max_val=max_val, kernel=kernel)

    def test_empty_lines(self, shape: Shape):
        image = draw_line_image(np.zeros((0, 5)), shape[-2:])
        table = draw_line_table(np.zeros((0, 5)), shape[-2:])
        assert np.sum(image) == 0.0
        assert len(table) == 0

    @pytest.mark.xfail(raises=ValueError)
    def test_image_wrong_size_lines(self, lines: RealArray, idxs: IntArray, shape: Shape):
        image = draw_line_image(lines[::2], shape, idxs)

    @pytest.mark.xfail(raises=ValueError)
    def test_table_wrong_size_lines(self, lines: RealArray, idxs: IntArray, shape: Shape):
        table = draw_line_image(lines[::2], shape, idxs)

    def test_zero_width(self, lines: RealArray, idxs: IntArray, shape: Shape, kernel: str):
        zero_lines = np.concatenate((lines[..., :4], np.zeros(lines.shape[:-1] + (1,))), axis=-1)
        image = draw_line_image(zero_lines, shape, idxs, kernel=kernel)
        table = draw_line_table(zero_lines, shape, idxs, kernel=kernel)
        assert np.sum(image) == 0
        assert len(table) == 0

    def test_negative_width(self, lines: RealArray, idxs: IntArray, shape: Shape, kernel: str):
        neg_lines = np.concatenate((lines[..., :4], np.full(lines.shape[:-1] + (1,), -1)), axis=-1)
        image = draw_line_image(neg_lines, shape, idxs, kernel=kernel)
        table = draw_line_table(neg_lines, shape, idxs, kernel=kernel)
        assert np.sum(image) == 0
        assert len(table) == 0

    @pytest.mark.slow
    def test_max_val(self, image: RealArray, table: Table, n_lines: int, max_val: float):
        assert np.min(image) == 0
        assert np.all(np.max(image, axis=(-2, -1)) <= n_lines * max_val)
        assert np.min(list(table.values())) >= 0
        assert np.all(np.array(list(table.values())) <= max_val)

    @pytest.fixture
    def image_numpy(self, lines: RealArray, idxs: IntArray, shape: Shape, ndim: int, max_val: float,
                    kernel: str) -> np.ndarray:
        kernel_func = self.kernel_dict()[kernel]
        pts = np.meshgrid(*(np.arange(length) for length in shape[-ndim:]), indexing='ij')
        pts = np.stack(pts[::-1], axis=-1)[..., None, :]

        frames = []
        for fnum in range(np.prod(shape[:-ndim])):
            lns = lines[idxs == fnum]
            dist = pts - project_to_streak(pts, lns[:, :ndim], lns[:, ndim:2 * ndim])
            frame = max_val * kernel_func(np.sqrt(np.sum(dist**2, axis=-1)), lns[:, 2 * ndim])
            frames.append(np.sum(frame, axis=-1))
        return np.stack(frames).reshape(shape)

    def test_draw_line_image(self, image: RealArray, image_numpy: RealArray):
        check_close(image, image_numpy)

    def test_draw_line_table(self, table: Table, image_numpy: RealArray):
        image = np.zeros(image_numpy.shape)
        for (_, idx), val in table.items():
            image[np.unravel_index(idx, image_numpy.shape)] += val
        check_close(image, image_numpy)
