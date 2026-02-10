from typing import ClassVar, Sequence, Tuple
from scipy import ndimage
import pytest
from cbclib_v2 import to_list
from cbclib_v2.annotations import (Generator, IntArray, IntSequence, RealArray, Mode, NDArray,
                                   NDBoolArray, NDRealArray, NumPy, NumPyNamespace, Shape)
from cbclib_v2.ndimage import (binterpolate, local_maxima, median, maximum_filter, median_filter,
                               robust_mean)
from cbclib_v2.streak_finder import PeaksList, detect_peaks
from cbclib_v2.test_util import check_close, compute_index

class TestImageProcessing():
    neighbours : ClassVar[NDBoolArray] = NumPy.asarray([[False, True , False],
                                                        [True , True , True ],
                                                        [False, True , False]], dtype=bool)

    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def rng(self, cpu_rng: Generator[NDArray]) -> Generator[NDArray]:
        return cpu_rng

    def binterpolate(self, inp: RealArray, grid: Sequence[IntArray | RealArray],
                     coords: RealArray, axes: IntSequence, xp: NumPyNamespace) -> NDRealArray:
        ndim = len(to_list(axes))

        lbound, ubound = [], []
        for i, side in list(enumerate(grid)):
            lbound.append(xp.clip(xp.searchsorted(side, coords[..., i], side='left') - 1,
                                  0, side.size - 1))
            ubound.append(xp.clip(xp.searchsorted(side, coords[..., i], side='right'), 0,
                                  side.size - 1))

        lbound, ubound = xp.stack(lbound, axis=-1), xp.stack(ubound, axis=-1)
        lcoords = xp.stack([side[lbound[..., i]] for i, side in enumerate(grid)], axis=-1)
        ucoords = xp.stack([side[ubound[..., i]] for i, side in enumerate(grid)], axis=-1)
        dx = xp.where(ucoords != lcoords, (coords - lcoords) / (ucoords - lcoords), 0.0)
        bounds = xp.stack((lbound, ubound))

        offsets = (xp.arange(2 ** ndim)[:, None] >> xp.arange(ndim) & 1)[..., :]
        indices = xp.stack([bounds[..., i][offsets[..., i]] for i in range(ndim)])
        factors = xp.stack([xp.where(offsets[..., i, None],
                                     dx[..., i].reshape(-1), 1.0 - dx[..., i].reshape(-1))
                            for i in range(ndim)], axis=-1)
        factors = xp.reshape(xp.prod(factors, axis=-1), indices.shape[1:])

        inp_flat = xp.moveaxis(inp, axes, -1 - xp.arange(ndim)[::-1])
        inp_flat = xp.reshape(inp_flat, inp_flat.shape[:-ndim] + (-1,))
        flat_idxs = xp.ravel_multi_index(list(indices),
                                         tuple(inp.shape[axis] for axis in to_list(axes)))
        result = xp.sum(inp_flat[..., flat_idxs] * factors, axis=-coords.ndim)
        return xp.moveaxis(result, -1 - xp.arange(ndim)[::-1], axes)

    @pytest.fixture(params=[(4, 11, 15), (15, 20)])
    def shape(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture()
    def array(self, rng: Generator[NDArray], shape: Shape) -> NDRealArray:
        return rng.random(shape)

    @pytest.fixture()
    def mask(self, rng: Generator[NDArray], shape: Shape, xp: NumPyNamespace) -> NDBoolArray:
        return xp.asarray(rng.integers(0, 2, size=shape), dtype=bool)

    @pytest.fixture(params=["constant", "nearest", "mirror", "reflect", "wrap"])
    def mode(self, request: pytest.FixtureRequest) -> Mode:
        return request.param

    @pytest.fixture(params=[(3, 2, 3)])
    def size(self, request: pytest.FixtureRequest, shape: Shape) -> Shape:
        return request.param[-len(shape):]

    @pytest.fixture(params=[[[False, True , True ],
                             [True , True , False]]])
    def footprint(self, request: pytest.FixtureRequest, size: Shape, xp: NumPyNamespace
                  ) -> NDBoolArray:
        return xp.broadcast_to(xp.asarray(request.param, dtype=bool), size)

    def test_median_empty(self, xp: NumPyNamespace):
        out = median(xp.zeros((0,)))
        out2 = xp.median(xp.zeros((0,)))

        assert xp.all((out == out2) | (xp.isnan(out) & xp.isnan(out2)))

    def test_median(self, array: NDRealArray, mask: NDBoolArray, xp: NumPyNamespace):
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

        out = median(array, mask, axis=axes)
        out2 = median(array.reshape(-1), mask.reshape(-1))

        assert xp.all(out == out2)

    def test_median_filter(self, array: NDRealArray, size: Shape, footprint: NDBoolArray,
                           mode: Mode, xp: NumPyNamespace):
        out = median_filter(array, size=size, mode=mode)
        out2 = ndimage.median_filter(array, size=size, mode=mode)

        assert xp.all(out == out2)

        out = median_filter(array, footprint=footprint, mode=mode)
        out2 = ndimage.median_filter(array, footprint=footprint, mode=mode)

        assert xp.all(out == out2)

    def test_maximum_filter(self, array: NDRealArray, size: Shape, footprint: NDBoolArray,
                            mode: Mode, xp: NumPyNamespace):
        out = maximum_filter(array, size=size, mode=mode)
        out2 = ndimage.maximum_filter(array, size=size, mode=mode)

        assert xp.all(out == out2)

        out = maximum_filter(array, footprint=footprint, mode=mode)
        out2 = ndimage.maximum_filter(array, footprint=footprint, mode=mode)

        assert xp.all(out == out2)

    @pytest.mark.parametrize('axis,lm', [(-1, 9.0)])
    def test_robust_mean(self, array: NDRealArray, axis: int, lm: float, xp: NumPyNamespace):
        mean = xp.median(array, axis=axis, keepdims=True)
        errors = (array - mean)**2
        indices = xp.lexsort((xp.indices(array.shape)[axis], errors), axis=axis)

        errors = xp.take_along_axis(errors, indices, axis=axis)
        cumsum = xp.cumulative_sum(errors, axis=axis)

        zero_shape = list(cumsum.shape)
        zero_shape[axis] = 1
        cumsum = xp.concat([xp.zeros(tuple(zero_shape), dtype=cumsum.dtype),
                            xp.take(cumsum, xp.arange(cumsum.shape[axis] - 1), axis=axis)],
                            axis=axis)

        threshold = xp.arange(array.shape[axis]) * errors
        mask = lm * cumsum > threshold
        mean = xp.mean(xp.take_along_axis(array, indices, axis=axis), where=mask, axis=axis)
        mean = xp.nan_to_num(mean)
        check_close(mean, robust_mean(array, axis=axis, n_iter=0, lm=lm))

    @pytest.mark.parametrize('axis,lm', [(1, 9.0)])
    def test_robust_mean_with_mask(self, array: NDRealArray, axis: int, lm: float, xp: NumPyNamespace):
        out = robust_mean(array, array > 0.5, axis=axis, lm=lm)

        permutation = tuple(i for i in range(array.ndim) if i != axis) + (axis,)
        array = xp.permute_dims(array, permutation)
        mean_values = []
        for index in range(array.size // array.shape[-1]):
            batch = array[xp.unravel_index(index, array.shape[:-1])]
            mean_values.append(robust_mean(batch[batch > 0.5], lm=lm))

        check_close(out, xp.array(mean_values).reshape(out.shape))

    @pytest.fixture(params=[(-2, -1),])
    def axes(self, request: pytest.FixtureRequest, shape: Shape) -> Tuple[int, int]:
        return (compute_index(request.param[0], len(shape)),
                compute_index(request.param[1], len(shape)))

    @pytest.fixture(params=[(7, 12)])
    def out_shape(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture
    def coords(self, rng: Generator[NDArray], shape: Shape, out_shape: Shape,
               axes: Tuple[int, int]) -> RealArray:
        array = rng.random(out_shape + (2,))
        array[..., 0] *= shape[axes[0]] - 1
        array[..., 1] *= shape[axes[1]] - 1
        return array

    @pytest.fixture
    def grid(self, shape: Shape, axes: Tuple[int, int], xp: NumPyNamespace
             ) -> Tuple[IntArray, IntArray]:
        return (xp.arange(shape[axes[0]]), xp.arange(shape[axes[1]]))

    def test_binterpolate(self, array: NDRealArray, grid: Tuple[IntArray, IntArray],
                          coords: RealArray, axes: Tuple[int, int], xp: NumPyNamespace):
        out = binterpolate(array, grid, coords, axes)
        out2 = self.binterpolate(array, grid, coords, axes, xp)
        check_close(out, out2)

    @pytest.fixture
    def vicinity(self, array: NDRealArray, axes: Tuple[int, int], xp: NumPyNamespace
                 ) -> NDBoolArray:
        return xp.expand_dims(self.neighbours, tuple(i for i in range(array.ndim) if i not in axes))

    def test_local_maxima(self, array: NDRealArray, vicinity: NDBoolArray, axes: Tuple[int, int],
                          xp: NumPyNamespace):
        out = local_maxima(array, axis=axes)
        filtered = maximum_filter(array, footprint=vicinity, mode='constant', cval=xp.inf)
        out2 = xp.stack(xp.asarray(array == filtered).nonzero(), axis=-1)

        assert xp.all(xp.sort(out, axis=0) == xp.sort(out2, axis=0))

    @pytest.fixture
    def peaks_list(self, array: NDRealArray, xp: NumPyNamespace) -> PeaksList:
        return detect_peaks(array, xp.ones_like(array, dtype=bool), radius=1, vmin=0.0)

    def test_detect_peaks(self, array: NDRealArray, peaks_list: PeaksList, xp: NumPyNamespace):
        for image, peaks in zip(xp.reshape(array, (-1,) + array.shape[-2:]), peaks_list):
            filtered = maximum_filter(image, footprint=self.neighbours, mode='constant',
                                      cval=xp.inf)
            expected_peaks = xp.stack(xp.asarray(image == filtered).nonzero(), axis=-1)
            peaks_array = xp.stack((peaks.y, peaks.x), axis=-1)
            assert xp.all(xp.sort(peaks_array, axis=0) == xp.sort(expected_peaks, axis=0))
