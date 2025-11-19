from typing import Callable, Tuple, cast
import numpy as np
import pytest
import scipy.ndimage
import scipy.signal
from cbclib_v2.fft import fftn, ifftn, fft_convolve
from cbclib_v2.ndimage import gaussian_filter, gaussian_gradient_magnitude, gaussian_kernel
from cbclib_v2.annotations import Norm, NDArray, NDRealArray, Mode
from cbclib_v2.test_util import check_close

class TestPyBind11Functions:
    ArrayGenerator = Callable[[Tuple[int, ...]], NDArray]
    ShapeGenerator = Callable[[], Tuple[int, ...]]

    @pytest.fixture(params=[np.float32, np.float64])
    def float_type(self, request: pytest.FixtureRequest) -> np.dtype:
        return request.param

    @pytest.fixture(params=['backward', 'forward', 'ortho'])
    def norm(self, request: pytest.FixtureRequest) -> Norm:
        return request.param

    @pytest.fixture(params=[2])
    def num_threads(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def random_float(self, rng: np.random.Generator, float_type: np.dtype) -> ArrayGenerator:
        return lambda shape: rng.random(shape, dtype=float_type)

    @pytest.fixture
    def random_complex(self, random_float: ArrayGenerator) -> ArrayGenerator:
        return lambda shape: random_float(shape) + 1j * random_float(shape)

    @pytest.fixture(params=[(10, 50, 2),])
    def random_shape(self, request: pytest.FixtureRequest,
                     rng: np.random.Generator) -> ShapeGenerator:
        vmin, vmax, size = request.param
        return lambda : tuple(rng.integers(vmin, vmax, size=size))

    @pytest.fixture(params=[2.0])
    def sigma(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=['constant', 'nearest', 'mirror', 'reflect', 'wrap'])
    def mode(self, request: pytest.FixtureRequest) -> Mode:
        return request.param

    @pytest.fixture(params=[0, 2])
    def order(self, request: pytest.FixtureRequest) -> int:
        return request.param

    def test_fftn(self, random_float: ArrayGenerator, random_complex: ArrayGenerator,
                  random_shape: ShapeGenerator, norm: Norm, num_threads: int):
        shape = random_shape()
        out_shape = tuple(2 * ax for ax in shape)
        inp = random_complex(shape)
        check_close(fftn(inp, shape=out_shape[1:], norm=norm, num_threads=num_threads),
                    np.fft.fftn(inp, s=out_shape[1:], norm=norm))
        check_close(ifftn(inp, shape=out_shape[1:], norm=norm, num_threads=num_threads),
                    np.fft.ifftn(inp, s=out_shape[1:], norm=norm))

        inp = random_float(shape)
        check_close(fftn(inp, shape=out_shape[1:], norm=norm, num_threads=num_threads),
                         np.fft.fftn(inp, s=out_shape[1:], norm=norm))
        check_close(ifftn(inp, shape=out_shape[1:], norm=norm, num_threads=num_threads),
                         np.fft.ifftn(inp, s=out_shape[1:], norm=norm))

    def test_fft_convolve(self, random_float: ArrayGenerator, random_complex: ArrayGenerator,
                          random_shape: ShapeGenerator, num_threads: int):
        ishape, kshape = random_shape(), random_shape()
        inp = random_float(ishape)
        kernel = random_float(kshape)
        check_close(fft_convolve(inp, kernel, num_threads=num_threads),
                    cast(NDRealArray, scipy.signal.fftconvolve(inp, kernel, mode='same')))
        kernel = random_float(kshape[1:])
        axes = np.arange(-kernel.ndim, 0)
        check_close(fft_convolve(inp, kernel, axis=axes, num_threads=num_threads),
                    cast(NDRealArray, scipy.signal.fftconvolve(inp, kernel[None, ...],
                                                               mode='same', axes=axes)))
        inp = random_complex(ishape)
        kernel = random_complex(kshape)
        check_close(fft_convolve(inp, kernel, num_threads=num_threads),
                    cast(NDRealArray, scipy.signal.fftconvolve(inp, kernel, mode='same')))
        kernel = random_complex(kshape[1:])
        axes = np.arange(-kernel.ndim, 0)
        check_close(fft_convolve(inp, kernel, axis=axes, num_threads=num_threads),
                    cast(NDRealArray, scipy.signal.fftconvolve(inp, kernel[None, ...],
                                                               mode='same', axes=axes)))

    @pytest.mark.xfail(raises=ValueError)
    def test_gaussian_kernel_zero(self):
        gaussian_kernel(0.0)

    @pytest.mark.xfail(raises=ValueError)
    def test_gaussian_kernel_neg(self):
        gaussian_kernel(-1.0)

    def test_gaussian_filter(self, random_float: ArrayGenerator, random_complex: ArrayGenerator,
                             random_shape: ShapeGenerator, sigma: float, order: int, mode: Mode,
                             num_threads: int):
        inp = random_float(random_shape())
        check_close(gaussian_filter(inp, sigma, order=order, mode=mode, num_threads=num_threads),
                    scipy.ndimage.gaussian_filter(inp, sigma, order=order, mode=mode))
        inp = random_complex(random_shape())
        check_close(gaussian_filter(inp, sigma, order=order, mode=mode, num_threads=num_threads),
                    scipy.ndimage.gaussian_filter(inp, sigma, order=order, mode=mode))

    def test_gaussian_gradient(self, random_float: ArrayGenerator, random_complex: ArrayGenerator,
                               random_shape: ShapeGenerator, sigma: float, mode: Mode,
                               num_threads: int):
        inp = random_float(random_shape())
        check_close(gaussian_gradient_magnitude(inp, sigma, mode=mode, num_threads=num_threads),
                    scipy.ndimage.gaussian_gradient_magnitude(inp, sigma, mode=mode),
                    atol=1e-4)
        inp = random_complex(random_shape())
        check_close(np.abs(gaussian_gradient_magnitude(inp, sigma, mode=mode,
                                                       num_threads=num_threads)),
                    np.abs(scipy.ndimage.gaussian_gradient_magnitude(inp, sigma, mode=mode)),
                    atol=1e-4)
