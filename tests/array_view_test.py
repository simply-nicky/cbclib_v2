from math import prod
from typing import List
import pytest
from cbclib_v2 import default_rng
from cbclib_v2.annotations import Generator, NDArray, NDIntArray, NDRealArray, NumPy, NumPyNamespace, Shape
from cbclib_v2.test_util import ArrayView, RectangleRange

@pytest.mark.parametrize('shape,permutation', [((10, 15, 20), [2, 0, 1]),
                                               ((8, 12), [1, 0])])
class TestArrayView:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def rng(self, xp: NumPyNamespace) -> Generator[NDArray]:
        return default_rng(42, xp)

    @pytest.fixture
    def indices(self, rng: Generator[NDArray], shape: Shape) -> NDIntArray:
        return rng.integers(0, prod(shape), size=(10,))

    @pytest.fixture
    def array(self, rng: Generator[NDArray], shape: Shape, permutation: List[int]) -> NDRealArray:
        return rng.random(shape).transpose(permutation)

    @pytest.fixture
    def array_view(self, array: NDRealArray) -> ArrayView:
        return ArrayView(array)

    def test_array_properties(self, array: NDRealArray, array_view: ArrayView):
        assert array.itemsize == array_view.itemsize
        assert array.ndim == array_view.ndim
        assert array.shape == tuple(array_view.shape)
        assert array.size == array_view.size
        assert array.strides == tuple(array_view.strides)

    def test_array_indexing(self, array: NDRealArray, array_view: ArrayView,
                            indices: NDIntArray, xp: NumPyNamespace):
        for index in indices:
            coord = array_view.coord_at(index)
            assert tuple(coord) == xp.unravel_index(index, array.shape)
            assert index == array_view.index_at(*coord) == array_view.index_at(coord)
            assert array.reshape(-1)[index] == xp.asarray(array_view[index])
            assert array.reshape(-1)[index] == array[tuple(coord)]
            assert array[tuple(coord)] == array_view.at(*coord) == array_view.at(coord)

class TestRectangleRange:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture(params=[(3, 4, 5), (8, 12)])
    def shape(self, request: pytest.FixtureRequest) -> Shape:
        return request.param

    @pytest.fixture
    def rectangle_range(self, shape: Shape) -> RectangleRange:
        return RectangleRange(list(shape))

    def test_rectangle_range(self, shape: Shape, rectangle_range: RectangleRange,
                             xp: NumPyNamespace):
        expected_size = prod(shape)
        assert rectangle_range.size() == expected_size

        points = list(rectangle_range)
        assert len(points) == expected_size

        for idx, point in enumerate(points):
            assert tuple(point) == xp.unravel_index(idx, shape)
