import pytest
import scipy.ndimage
import scipy.signal
from cbclib_v2.fft import fftn, ifftn, fft_convolve
from cbclib_v2.ndimage import gaussian_filter, gaussian_gradient_magnitude, gaussian_kernel
from cbclib_v2.annotations import (DType, Generator, Norm, NDArray, NDComplexArray, NDRealArray,
                                   NumPy, NumPyNamespace, Mode, Shape)
from cbclib_v2.test_util import check_close

class TestFFTFunctions:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def rng(self, cpu_rng: Generator[NDArray]) -> Generator[NDArray]:
        return cpu_rng

    @pytest.fixture(params=['float32', 'float64'])
    def float_type(self, request: pytest.FixtureRequest, xp: NumPyNamespace) -> DType:
        return xp.dtype(request.param)

    @pytest.fixture(params=['backward', 'forward', 'ortho'])
    def norm(self, request: pytest.FixtureRequest) -> Norm:
        return request.param

    @pytest.fixture(params=[2])
    def num_threads(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[(34, 42)])
    def ishape(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture
    def float_input(self, rng: Generator[NDArray], float_type: DType, ishape: Shape) -> NDRealArray:
        return rng.random(ishape, dtype=float_type)

    @pytest.fixture
    def complex_input(self, rng: Generator[NDArray], float_type: DType, ishape: Shape
                      ) -> NDComplexArray:
        return rng.random(ishape, dtype=float_type) + 1j * rng.random(ishape, dtype=float_type)

    @pytest.fixture(params=[(23, 36)])
    def kshape(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture
    def float_kernel(self, rng: Generator[NDArray], float_type: DType, kshape: Shape
                     ) -> NDRealArray:
        return rng.random(kshape, dtype=float_type)

    @pytest.fixture
    def complex_kernel(self, rng: Generator[NDArray], float_type: DType, kshape: Shape
                       ) -> NDComplexArray:
        return rng.random(kshape, dtype=float_type) + 1j * rng.random(kshape, dtype=float_type)

    @pytest.fixture(params=[2.0])
    def sigma(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=['constant', 'nearest', 'mirror', 'reflect', 'wrap'])
    def mode(self, request: pytest.FixtureRequest) -> Mode:
        return request.param

    @pytest.fixture(params=[0, 2])
    def order(self, request: pytest.FixtureRequest) -> int:
        return request.param

    def test_fftn(self, float_input: NDRealArray, complex_input: NDComplexArray,
                  ishape: Shape, norm: Norm, xp: NumPyNamespace):
        out_shape = tuple(2 * ax for ax in ishape)
        inp = complex_input
        axes = tuple(range(-len(out_shape) + 1, 0))
        check_close(fftn(inp, shape=out_shape[1:], norm=norm),
                    xp.fft.fftn(inp, s=out_shape[1:], axes=axes, norm=norm))
        check_close(ifftn(inp, shape=out_shape[1:], norm=norm),
                    xp.fft.ifftn(inp, s=out_shape[1:], axes=axes, norm=norm))

        inp = float_input
        check_close(fftn(inp, shape=out_shape[1:], norm=norm),
                    xp.fft.fftn(inp, s=out_shape[1:], axes=axes, norm=norm))
        check_close(ifftn(inp, shape=out_shape[1:], norm=norm),
                    xp.fft.ifftn(inp, s=out_shape[1:], axes=axes, norm=norm))

    def test_fft_convolve(self, float_input: NDRealArray, complex_input: NDComplexArray,
                          float_kernel: NDRealArray, complex_kernel: NDComplexArray,
                          xp: NumPyNamespace):
        inp = float_input
        kernel = float_kernel
        check_close(fft_convolve(inp, kernel),
                    scipy.signal.fftconvolve(inp, kernel, mode='same'))

        kernel = float_kernel[0]
        axes = xp.arange(-kernel.ndim, 0)
        check_close(fft_convolve(inp, kernel, axis=axes),
                    scipy.signal.fftconvolve(inp, kernel[None, ...], mode='same', axes=axes))

        inp = complex_input
        kernel = complex_kernel
        check_close(fft_convolve(inp, kernel),
                    scipy.signal.fftconvolve(inp, kernel, mode='same'))

        kernel = complex_kernel[0]
        axes = xp.arange(-kernel.ndim, 0)
        check_close(fft_convolve(inp, kernel, axis=axes),
                    scipy.signal.fftconvolve(inp, kernel[None, ...], mode='same', axes=axes))

    @pytest.mark.xfail(raises=ValueError)
    def test_gaussian_kernel_zero(self):
        gaussian_kernel(0.0)

    @pytest.mark.xfail(raises=ValueError)
    def test_gaussian_kernel_neg(self):
        gaussian_kernel(-1.0)

    def test_gaussian_filter(self, float_input: NDRealArray, complex_input: NDComplexArray,
                             sigma: float, order: int, mode: Mode):
        inp = float_input
        check_close(gaussian_filter(inp, sigma, order=order, mode=mode),
                    scipy.ndimage.gaussian_filter(inp, sigma, order=order, mode=mode))
        inp = complex_input
        check_close(gaussian_filter(inp, sigma, order=order, mode=mode),
                    scipy.ndimage.gaussian_filter(inp, sigma, order=order, mode=mode))

    def test_gaussian_gradient(self, float_input: NDRealArray, complex_input: NDComplexArray,
                               sigma: float, mode: Mode, xp: NumPyNamespace):
        inp = float_input
        check_close(gaussian_gradient_magnitude(inp, sigma, mode=mode),
                    scipy.ndimage.gaussian_gradient_magnitude(inp, sigma, mode=mode),
                    atol=1e-4)

        inp = complex_input
        check_close(xp.abs(gaussian_gradient_magnitude(inp, sigma, mode=mode)),
                    xp.abs(scipy.ndimage.gaussian_gradient_magnitude(inp, sigma, mode=mode)),
                    atol=1e-4)
