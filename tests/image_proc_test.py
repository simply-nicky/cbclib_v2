from typing import Sequence, Tuple
import numpy as np
from scipy import ndimage
import pytest
from cbclib_v2 import to_list
from cbclib_v2.annotations import IntArray, IntSequence, RealArray
from cbclib_v2.ndimage import binterpolate, median, median_filter, robust_mean
from cbclib_v2.annotations import Mode, NDBoolArray, NDRealArray, Shape
from cbclib_v2.test_util import check_close

class TestImageProcessing():
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
    def input(self, rng: np.random.Generator, shape: Shape) -> NDRealArray:
        return rng.random(shape)

    @pytest.fixture()
    def mask(self, rng: np.random.Generator, shape: Shape) -> NDBoolArray:
        return np.asarray(rng.integers(0, 1, size=shape), dtype=bool)

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

    def test_median(self, input: NDRealArray, mask: NDBoolArray):
        axes = list(range(input.ndim))
        out = median(input, axis=axes)
        out2 = np.median(input, axis=axes)

        assert np.all(out == out2)

        for axis in range(input.ndim):
            out = median(input, axis=axis)
            out2 = np.median(input, axis=axis)

            assert np.all(out == out2)

        out = median(input, mask, axis=axes)
        out2 = median(input.ravel(), mask.ravel())

        assert np.all(out == out2)

    def test_median_filter(self, input: NDRealArray, size: Shape,
                           footprint: NDBoolArray, mode: Mode):
        out = median_filter(input, size=size, mode=mode)
        out2 = ndimage.median_filter(input, size=size, mode=mode)

        assert np.all(out == out2)

        out = median_filter(input, footprint=footprint, mode=mode)
        out2 = ndimage.median_filter(input, footprint=footprint, mode=mode)

        assert np.all(out == out2)

    @pytest.mark.parametrize('axis,lm', [(-1, 9.0)])
    def test_robust_mean(self, input: NDRealArray, axis: int, lm: float):
        mean = np.median(input, axis=axis, keepdims=True)
        errors = (input - mean)**2
        indices = np.lexsort((input, errors), axis=axis)
        errors = np.take_along_axis(errors, indices, axis=axis)
        cumsum = np.cumsum(errors, axis=axis)
        cumsum = np.delete(np.insert(cumsum, 0, 0, axis=axis), -1, axis=axis)
        threshold = np.arange(input.shape[axis]) * errors
        mask = lm * cumsum > threshold
        mean = np.mean(np.take_along_axis(input, indices, axis=axis), where=mask, axis=axis)
        check_close(mean, robust_mean(input, axis=axis, n_iter=0, lm=lm))

    @pytest.fixture(params=[(-2, -1),])
    def axes(self, request: pytest.FixtureRequest) -> Tuple[int, int]:
        return request.param

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

    def test_binterpolate(self, input: NDRealArray, grid: Tuple[IntArray, IntArray],
                          coords: RealArray, axes: Tuple[int, int]):
        out = binterpolate(input, grid, coords, axes)
        out2 = self.binterpolate(input, grid, coords, axes)
        check_close(out, out2)
