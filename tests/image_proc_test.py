import numpy as np
from scipy import ndimage
from scipy.spatial import KDTree
import pytest
from cbclib_v2.src import median, median_filter, build_tree
from cbclib_v2.annotations import Mode, NDBoolArray, NDRealArray, Shape

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

    @pytest.fixture(params=[100,])
    def num_points(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[(10, 10), (3, 2, 5)])
    def num_queries(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture(params=[1,])
    def num_neighbours(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[1, 3])
    def ndim(self, request: pytest.FixtureRequest) -> int:
        return request.param

    def test_median(self, input: NDRealArray, mask: NDBoolArray):
        axes = list(range(input.ndim))
        out = median(input, axis=axes)
        out2 = np.median(input, axis=axes)

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

    def test_kd_tree(self, rng: np.random.Generator, num_points: int, num_queries: Shape,
                     num_neighbours: int, ndim: int):
        points = rng.random((num_points, ndim))
        query = rng.random(num_queries + (ndim,))

        dist, out = build_tree(points).find_nearest(query, num_neighbours)
        dist2, out2 = KDTree(points).query(query, num_neighbours)

        np.testing.assert_allclose(np.squeeze(dist), dist2)
        assert np.all(np.squeeze(out) == out2)
