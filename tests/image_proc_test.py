from typing import ClassVar, Sequence, Tuple
import numpy as np
from scipy import ndimage
import pytest
from cbclib_v2 import to_list
from cbclib_v2.annotations import IntArray, IntSequence, RealArray, Mode, NDBoolArray, NDRealArray, Shape
from cbclib_v2.ndimage import (binterpolate, local_maxima, median, maximum_filter, median_filter,
                               robust_mean)
from cbclib_v2.streak_finder import PeaksList, detect_peaks
from cbclib_v2.test_util import check_close, compute_index

class TestImageProcessing():
    neighbours : ClassVar[NDBoolArray] = np.array([[False, True , False],
                                                   [True , True , True ],
                                                   [False, True , False]], dtype=bool)

    def binterpolate(self, inp: RealArray, grid: Sequence[IntArray | RealArray],
                     coords: RealArray, axes: IntSequence):
        ndim = len(to_list(axes))

        lbound, ubound = [], []
        for i, side in list(enumerate(grid)):
            lbound.append(np.clip(np.searchsorted(side, coords[..., i], side='left') - 1,
                                  0, side.size - 1))
            ubound.append(np.clip(np.searchsorted(side, coords[..., i], side='right'), 0,
                                  side.size - 1))

        lbound, ubound = np.stack(lbound, axis=-1), np.stack(ubound, axis=-1)
        lcoords = np.stack([side[lbound[..., i]] for i, side in enumerate(grid)], axis=-1)
        ucoords = np.stack([side[ubound[..., i]] for i, side in enumerate(grid)], axis=-1)
        dx = np.where(ucoords != lcoords, (coords - lcoords) / (ucoords - lcoords), 0.0)
        bounds = np.stack((lbound, ubound))

        offsets = (np.arange(2 ** ndim)[:, None] >> np.arange(ndim) & 1)[..., :]
        indices = np.stack([bounds[..., i][offsets[..., i]] for i in range(ndim)])
        factors = np.stack([np.where(offsets[..., i, None],
                                     dx[..., i].ravel(), 1.0 - dx[..., i].ravel())
                            for i in range(ndim)], axis=-1)
        factors = np.reshape(np.prod(factors, axis=-1), indices.shape[1:])

        inp_flat = np.moveaxis(inp, axes, -1 - np.arange(ndim)[::-1])
        inp_flat = np.reshape(inp_flat, inp_flat.shape[:-ndim] + (-1,))
        flat_idxs = np.ravel_multi_index(list(indices),
                                         [inp.shape[axis]for axis in to_list(axes)])
        result = np.sum(inp_flat[..., flat_idxs] * factors, axis=-coords.ndim)
        return np.moveaxis(result, -1 - np.arange(ndim)[::-1], axes)

    @pytest.fixture(params=[(4, 11, 15), (15, 20)])
    def shape(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture()
    def array(self, rng: np.random.Generator, shape: Shape) -> NDRealArray:
        return rng.random(shape)

    @pytest.fixture()
    def mask(self, rng: np.random.Generator, shape: Shape) -> NDBoolArray:
        return np.asarray(rng.integers(0, 2, size=shape), dtype=bool)

    @pytest.fixture(params=["constant", "nearest", "mirror", "reflect", "wrap"])
    def mode(self, request: pytest.FixtureRequest) -> Mode:
        return request.param

    @pytest.fixture(params=[(3, 2, 3)])
    def size(self, request: pytest.FixtureRequest, shape: Shape) -> Shape:
        return request.param[-len(shape):]

    @pytest.fixture(params=[[[False, True , True ],
                             [True , True , False]]])
    def footprint(self, request: pytest.FixtureRequest, size: Shape) -> NDBoolArray:
        return np.broadcast_to(np.asarray(request.param, dtype=bool), size)

    def test_median_empty(self):
        out = median(np.zeros((0,)))
        out2 = np.median(np.zeros((0,)))

        assert np.all((out == out2) | (np.isnan(out) & np.isnan(out2)))

    def test_median(self, array: NDRealArray, mask: NDBoolArray):
        axes = list(range(array.ndim))
        out = median(array, axis=axes)
        out2 = np.median(array, axis=axes)

        assert np.all(out == out2)

        out = median(array, axis=(axes[0], axes[-1]))
        out2 = np.median(array, axis=(axes[0], axes[-1]))

        assert np.all(out == out2)

        for axis in range(array.ndim):
            out = median(array, axis=axis)
            out2 = np.median(array, axis=axis)

            assert np.all(out == out2)

        out = median(array, mask, axis=axes)
        out2 = median(array.ravel(), mask.ravel())

        assert np.all(out == out2)

    def test_median_filter(self, array: NDRealArray, size: Shape, footprint: NDBoolArray,
                           mode: Mode):
        out = median_filter(array, size=size, mode=mode)
        out2 = ndimage.median_filter(array, size=size, mode=mode)

        assert np.all(out == out2)

        out = median_filter(array, footprint=footprint, mode=mode)
        out2 = ndimage.median_filter(array, footprint=footprint, mode=mode)

        assert np.all(out == out2)

    def test_maximum_filter(self, array: NDRealArray, size: Shape, footprint: NDBoolArray,
                            mode: Mode):
        out = maximum_filter(array, size=size, mode=mode)
        out2 = ndimage.maximum_filter(array, size=size, mode=mode)

        assert np.all(out == out2)

        out = maximum_filter(array, footprint=footprint, mode=mode)
        out2 = ndimage.maximum_filter(array, footprint=footprint, mode=mode)

        assert np.all(out == out2)

    @pytest.mark.parametrize('axis,lm', [(-1, 9.0)])
    def test_robust_mean(self, array: NDRealArray, axis: int, lm: float):
        mean = np.median(array, axis=axis, keepdims=True)
        errors = (array - mean)**2
        indices = np.lexsort((np.indices(array.shape)[axis], errors), axis=axis)

        errors = np.take_along_axis(errors, indices, axis=axis)
        cumsum = np.cumsum(errors, axis=axis)
        cumsum = np.delete(np.insert(cumsum, 0, 0, axis=axis), -1, axis=axis)
        threshold = np.arange(array.shape[axis]) * errors
        mask = lm * cumsum > threshold
        mean = np.mean(np.take_along_axis(array, indices, axis=axis), where=mask, axis=axis)
        mean = np.nan_to_num(mean)
        check_close(mean, robust_mean(array, axis=axis, n_iter=0, lm=lm))

    @pytest.mark.parametrize('axis,lm', [(1, 9.0)])
    def test_robust_mean_with_mask(self, array: NDRealArray, axis: int, lm: float):
        out = robust_mean(array, array > 0.5, axis=axis, lm=lm)

        permutation = tuple(i for i in range(array.ndim) if i != axis) + (axis,)
        array = np.transpose(array, permutation)
        mean_values = []
        for index in range(array.size // array.shape[-1]):
            batch = array[np.unravel_index(index, array.shape[:-1])]
            mean_values.append(robust_mean(batch[batch > 0.5], lm=lm))

        check_close(out, np.array(mean_values).reshape(out.shape))

    @pytest.fixture(params=[(-2, -1),])
    def axes(self, request: pytest.FixtureRequest, shape: Shape) -> Tuple[int, int]:
        return (compute_index(request.param[0], len(shape)),
                compute_index(request.param[1], len(shape)))

    @pytest.fixture(params=[(7, 12)])
    def out_shape(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture
    def coords(self, rng: np.random.Generator, shape: Shape, out_shape: Shape,
               axes: Tuple[int, int]) -> RealArray:
        array = rng.random(out_shape + (2,))
        array[..., 0] *= shape[axes[0]] - 1
        array[..., 1] *= shape[axes[1]] - 1
        return array

    @pytest.fixture
    def grid(self, shape: Shape, axes: Tuple[int, int]) -> Tuple[IntArray, IntArray]:
        return (np.arange(shape[axes[0]]), np.arange(shape[axes[1]]))

    def test_binterpolate(self, array: NDRealArray, grid: Tuple[IntArray, IntArray],
                          coords: RealArray, axes: Tuple[int, int]):
        out = binterpolate(array, grid, coords, axes)
        out2 = self.binterpolate(array, grid, coords, axes)
        check_close(out, out2)

    @pytest.fixture
    def vicinity(self, array: NDRealArray, axes: Tuple[int, int]) -> NDBoolArray:
        return np.expand_dims(self.neighbours, tuple(i for i in range(array.ndim) if i not in axes))

    def test_local_maxima(self, array: NDRealArray, vicinity: NDBoolArray, axes: Tuple[int, int]):
        out = local_maxima(array, axis=axes)
        filtered = maximum_filter(array, footprint=vicinity, mode='constant', cval=np.inf)
        out2 = np.stack(np.where(array == filtered), axis=-1)

        assert np.all(np.sort(out, axis=0) == np.sort(out2, axis=0))

    @pytest.fixture
    def peaks_list(self, array: NDRealArray) -> PeaksList:
        return detect_peaks(array, np.ones_like(array, dtype=bool), radius=1, vmin=0.0,
                            num_threads=8)

    def test_detect_peaks(self, array: NDRealArray, peaks_list: PeaksList):
        for image, peaks in zip(np.reshape(array, (-1,) + array.shape[-2:]), peaks_list):
            filtered = maximum_filter(image, footprint=self.neighbours, mode='constant',
                                      cval=np.inf)
            expected_peaks = np.stack(np.where(image == filtered), axis=-1)
            peaks_array = np.stack((peaks.y, peaks.x), axis=-1)
            assert np.all(np.sort(peaks_array, axis=0) == np.sort(expected_peaks, axis=0))
