from math import prod
from typing import Tuple
import pytest
from cbclib_v2 import default_rng
from cbclib_v2.annotations import (CPArray, CuPy, CuPyNamespace, Generator, NDArray, NumPy,
                                   NumPyNamespace, RealArray, Shape)
from cbclib_v2.ndimage import median, robust_mean, robust_lsq
from cbclib_v2.test_util import check_close

TestNamespace = NumPyNamespace | CuPyNamespace
TestGenerator = Generator[NDArray] | Generator[CPArray]

class TestImageProcessing():
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

    @pytest.fixture(params=[(4, 11, 15), (15, 20)])
    def shape(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture()
    def array(self, rng: TestGenerator, shape: Shape) -> RealArray:
        return rng.random(shape)

    @pytest.fixture(params=[(3, 2, 3)])
    def size(self, request: pytest.FixtureRequest, shape: Shape) -> Shape:
        return request.param[-len(shape):]

    def test_median_empty(self, xp: TestNamespace):
        if xp is CuPy:
            pytest.xfail("CuPy's median over an empty array raises an error")

        out = median(xp.zeros((0,)))
        out2 = xp.median(xp.zeros((0,)))

        assert xp.all((out == out2) | (xp.isnan(out) & xp.isnan(out2)))

    def test_median(self, array: RealArray, xp: TestNamespace):
        axes = list(range(array.ndim))
        out = median(array, axis=axes)
        out2 = xp.median(array, axis=axes)

        assert xp.all(out == out2)

        out = median(array, axis=(axes[0], axes[-1]))
        out2 = xp.median(array, axis=(axes[0], axes[-1]))

        assert xp.all(out == out2)

        for axis in range(array.ndim):
            out = median(array, axis=axis)
            out2 = xp.median(array, axis=axis)

            assert xp.all(out == out2)

        out = median(array, axis=axes)
        out2 = median(array.reshape(-1))

        assert xp.all(out == out2)

    def shift_axis(self, inp: RealArray, axis: Tuple[int, ...]) -> RealArray:
        reduce_axis = []
        out_axis = []
        out_shape = []
        for i in range(inp.ndim):
            if i in axis or i - inp.ndim in axis:
                reduce_axis.append(i)
            else:
                out_axis.append(i)
                out_shape.append(inp.shape[i])

        inp = inp.transpose(out_axis + reduce_axis)
        inp = inp.reshape((*out_shape, -1))
        return inp

    @pytest.fixture(params=[1.0])
    def v_inliers(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[50.0])
    def v_outliers(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[(10, 1000)])
    def dshape(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture
    def dataset(self, rng: TestGenerator, v_inliers: float, v_outliers: float, dshape: Shape,
                xp: TestNamespace) -> RealArray:
        inliers = rng.random(dshape) * v_inliers
        outliers = rng.random(dshape[:-1] + (dshape[-1] // 100,)) * v_outliers
        return xp.concat((inliers, outliers), axis=-1)

    def robust_mean(self, inp: RealArray, axis: int | Tuple[int, ...], r0: float, r1: float,
                    n_iter: int, lm: float, xp: TestNamespace) -> RealArray:
        if isinstance(axis, int):
            axis = (axis,)

        inp = self.shift_axis(inp, axis)

        mean = xp.median(inp, axis=-1, keepdims=True)

        shape = mean.shape[:-1]
        n_reduce = inp.shape[-1]
        j0, j1 = int(r0 * n_reduce), int(r1 * n_reduce)

        for _ in range(n_iter):
            error = (inp - mean)**2
            idxs = xp.argsort(error, axis=-1)
            mean = xp.mean(xp.take_along_axis(inp, idxs[..., j0:j1], axis=-1),
                           axis=-1, keepdims=True)

        errors = (inp - mean)**2
        idxs = xp.argsort(errors, axis=-1)
        errors = xp.take_along_axis(errors, idxs, axis=-1)

        cumsum = xp.cumulative_sum(errors, axis=-1)
        threshold = xp.arange(n_reduce) * errors

        mask = lm * cumsum < threshold
        cutoff = xp.where(xp.any(mask, axis=-1), xp.argmax(mask, axis=-1), n_reduce)
        mask = xp.broadcast_to(xp.arange(n_reduce), (*shape, n_reduce)) < cutoff[..., None]

        # Array API doesn't support 'where' in mean, use manual masked mean
        sum_val = xp.sum(xp.where(mask, xp.take_along_axis(inp, idxs, axis=-1), 0), axis=-1)
        count = xp.sum(mask.astype(inp.dtype), axis=-1)
        mean = xp.where(count > 0, sum_val / count, 0)
        return mean

    @pytest.fixture(params=[0.5])
    def r0(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[0.9])
    def r1(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture(params=[9.0])
    def lm(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.mark.parametrize('axis,n_iter', [(-1, 5), ((0, 1), 0)])
    def test_robust_mean(self, dataset: RealArray, axis: int, r0: float, r1: float,
                         n_iter: int, lm: float, xp: TestNamespace):
        expected = self.robust_mean(dataset, axis=axis, r0=r0, r1=r1, n_iter=n_iter, lm=lm, xp=xp)
        result = robust_mean(dataset, axis=axis, r0=r0, r1=r1, n_iter=n_iter, lm=lm)
        check_close(result, expected)

    def robust_lsq(self, W: RealArray, y: RealArray, axis: int | Tuple[int, ...], r0: float,
                   r1: float, n_iter: int, lm: float, xp: TestNamespace) -> RealArray:
        if isinstance(axis, int):
            axis = (axis,)

        y = self.shift_axis(y, axis)
        W = xp.reshape(W, (prod(W.shape[:-len(axis)]), -1))

        fits = xp.sum(y[..., None, :] * W, axis=-1) / xp.sum(W * W, axis=-1)

        shape = y.shape[:-1]
        n_reduce = y.shape[-1]
        j0, j1 = int(r0 * n_reduce), int(r1 * n_reduce)

        for _ in range(n_iter):
            errors = (y - xp.tensordot(fits, W, axes=(-1, 0)))**2
            idxs = xp.argsort(errors, axis=-1)
            YW = xp.take_along_axis(y[..., None, :] * W, idxs[..., None, j0:j1], axis=-1)
            WW = xp.reshape(W * W, tuple([1,] * len(shape)) + W.shape)
            WW = xp.take_along_axis(WW, idxs[..., None, j0:j1], axis=-1)
            fits = xp.sum(YW, axis=-1) / xp.sum(WW, axis=-1)

        errors = (y - xp.tensordot(fits, W, axes=(-1, 0)))**2
        idxs = xp.argsort(errors, axis=-1)
        errors = xp.take_along_axis(errors, idxs, axis=-1)

        cumsum = xp.cumulative_sum(errors, axis=-1)
        threshold = xp.arange(n_reduce) * errors

        mask = lm * cumsum < threshold
        cutoff = xp.where(xp.any(mask, axis=-1), xp.argmax(mask, axis=-1), n_reduce)
        mask = xp.broadcast_to(xp.arange(n_reduce), (*shape, n_reduce)) < cutoff[..., None]

        # Array API doesn't support 'where' in mean, use manual masked mean
        counts = xp.sum(mask[..., None, :], axis=-1)

        YW = xp.take_along_axis(y[..., None, :] * W, idxs[..., None, :], axis=-1)
        YW_sum = xp.sum(xp.where(mask[..., None, :], YW, 0), axis=-1)
        YW = xp.where(counts > 0, YW_sum / counts, 0)

        WW = xp.reshape(W * W, tuple([1,] * len(shape)) + W.shape)
        WW = xp.take_along_axis(WW, idxs[..., None, :], axis=-1)
        WW_sum = xp.sum(xp.where(mask[..., None, :], WW, 0), axis=-1)
        WW = xp.where(counts > 0, WW_sum / counts, 0)

        fits = xp.where(WW > 0, YW / WW, 0)
        return fits

    @pytest.fixture(params=[10,])
    def num_features(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def W(self, rng: TestGenerator, dataset: RealArray, axis: int | Tuple[int, ...],
          num_features: int) -> RealArray:
        if isinstance(axis, int):
            axis = (axis,)
        return rng.random((num_features,) + tuple(dataset.shape[i] for i in axis))

    @pytest.mark.parametrize('axis,n_iter', [(-1, 5), ((0, 1), 0)])
    def test_robust_lsq(self, dataset: RealArray, W: RealArray, axis: int | Tuple[int, ...],
                        r0: float, r1: float, n_iter: int, lm: float, xp: TestNamespace):
        expected = self.robust_lsq(W, dataset, axis=axis, r0=r0, r1=r1, n_iter=n_iter, lm=lm, xp=xp)
        result = robust_lsq(W, dataset, axis=axis, r0=r0, r1=r1, n_iter=n_iter, lm=lm)
        check_close(result, expected)
