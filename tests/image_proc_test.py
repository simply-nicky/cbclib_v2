import numpy as np
from scipy import ndimage
import pytest
from cbclib_v2.ndimage import median, median_filter, robust_mean
from cbclib_v2.annotations import Mode, NDBoolArray, NDRealArray, Shape
from cbclib_v2.test_util import check_close

class TestImageProcessing():
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
