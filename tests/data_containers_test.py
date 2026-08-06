from dataclasses import dataclass
import pytest
from cbclib_v2 import ArrayContainer, IndexedContainer
from cbclib_v2.annotations import IntArray, NumPy, NumPyNamespace, RealArray

@dataclass
class ArrayLeaf(ArrayContainer):
    values: RealArray

@dataclass
class CompoundArray(ArrayContainer):
    array: RealArray
    container: ArrayLeaf

@dataclass
class IndexedLeaf(IndexedContainer):
    index: IntArray
    values: RealArray

@dataclass
class CompoundIndexed(IndexedContainer):
    index: IntArray
    array: RealArray
    array_container: ArrayLeaf
    indexed_container: IndexedLeaf

class TestArrayContainer:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def container(self, xp: NumPyNamespace) -> CompoundArray:
        return CompoundArray(
            array=xp.reshape(xp.arange(12), (4, 3)),
            container=ArrayLeaf(values=xp.reshape(xp.arange(8) + 20, (4, 2))),
        )

    @pytest.fixture
    def other(self, xp: NumPyNamespace) -> CompoundArray:
        return CompoundArray(
            array=xp.reshape(xp.arange(12) + 100, (4, 3)),
            container=ArrayLeaf(values=xp.reshape(xp.arange(8) + 200, (4, 2))),
        )

    def test_concat(self, container: CompoundArray, other: CompoundArray,
                    xp: NumPyNamespace) -> None:
        result = CompoundArray.concat((container, other))

        assert xp.all(result.array == xp.concat((container.array, other.array)))
        assert xp.all(result.container.values == xp.concat(
            (container.container.values, other.container.values)))

    def test_stack(self, container: CompoundArray, other: CompoundArray,
                   xp: NumPyNamespace) -> None:
        result = CompoundArray.stack((container, other), axis=1)

        assert xp.all(result.array == xp.stack((container.array, other.array), axis=1))
        assert xp.all(result.container.values == xp.stack(
            (container.container.values, other.container.values), axis=1))

    def test_shape(self, container: CompoundArray) -> None:
        assert container.shape == (4,)
        assert container.container.shape == (4, 2)

    def test_getitem(self, container: CompoundArray, xp: NumPyNamespace) -> None:
        indices = xp.asarray([1, 3])
        result = container[indices]

        assert xp.all(result.array == container.array[indices])
        assert xp.all(result.container.values == container.container.values[indices])

    def test_reshape(self, container: CompoundArray, xp: NumPyNamespace) -> None:
        result = container.reshape((2, 2))

        assert result.shape == (2, 2)
        assert xp.all(result.array == xp.reshape(container.array, (2, 2, 3)))
        assert xp.all(result.container.values == xp.reshape(
            container.container.values, (2, 2, 2)))

class TestIndexedContainer:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def container(self, xp: NumPyNamespace) -> CompoundIndexed:
        return CompoundIndexed(
            index=xp.asarray([3, 3, 7, 7]),
            array=xp.reshape(xp.arange(12), (4, 3)),
            array_container=ArrayLeaf(values=xp.reshape(xp.arange(8) + 20, (4, 2))),
            indexed_container=IndexedLeaf(
                index=xp.asarray([10, 10, 20, 20]),
                values=xp.reshape(xp.arange(16) + 40, (4, 4)),
            ),
        )

    @pytest.fixture
    def other(self, xp: NumPyNamespace) -> CompoundIndexed:
        return CompoundIndexed(
            index=xp.asarray([3, 3, 7, 7]),
            array=xp.reshape(xp.arange(12) + 100, (4, 3)),
            array_container=ArrayLeaf(values=xp.reshape(xp.arange(8) + 200, (4, 2))),
            indexed_container=IndexedLeaf(
                index=xp.asarray([10, 10, 20, 20]),
                values=xp.reshape(xp.arange(16) + 300, (4, 4)),
            ),
        )

    def test_getitem(self, container: CompoundIndexed, xp: NumPyNamespace) -> None:
        indices = xp.asarray([1, 3])
        result = container[indices]

        assert xp.all(result.index == container.index[indices])
        assert xp.all(result.array == container.array[indices])
        assert xp.all(result.array_container.values == container.array_container.values[indices])
        assert xp.all(result.indexed_container.index == container.indexed_container.index[indices])
        assert xp.all(
            result.indexed_container.values == container.indexed_container.values[indices])

    def test_reshape(self, container: CompoundIndexed, xp: NumPyNamespace) -> None:
        result = container.reshape((2, 2))

        assert xp.all(result.index == xp.asarray([3, 7]))
        assert xp.all(result.array == xp.reshape(container.array, (2, 2, 3)))
        assert xp.all(result.array_container.values == xp.reshape(
            container.array_container.values, (2, 2, 2)))
        assert xp.all(result.indexed_container.index == xp.asarray([10, 20]))
        assert xp.all(result.indexed_container.values == xp.reshape(
            container.indexed_container.values, (2, 2, 4)))

    def test_concat(self, container: CompoundIndexed, other: CompoundIndexed,
                    xp: NumPyNamespace) -> None:
        result = CompoundIndexed.concat((container, other))

        assert xp.all(result.index == xp.asarray([3, 3, 7, 7, 8, 8, 12, 12]))
        assert xp.all(result.array == xp.concat((container.array, other.array)))
        assert xp.all(result.array_container.values == xp.concat(
            (container.array_container.values, other.array_container.values)))
        assert xp.all(result.indexed_container.index == xp.asarray(
            [10, 10, 20, 20, 21, 21, 31, 31]))
        assert xp.all(result.indexed_container.values == xp.concat(
            (container.indexed_container.values, other.indexed_container.values)))

    def test_stack(self, container: CompoundIndexed, other: CompoundIndexed,
                   xp: NumPyNamespace) -> None:
        result = CompoundIndexed.stack((container, other), axis=1)

        assert xp.all(result.index == container.index)
        assert xp.all(result.array == xp.stack((container.array, other.array), axis=1))
        assert xp.all(result.array_container.values == xp.stack(
            (container.array_container.values, other.array_container.values), axis=1))
        assert xp.all(result.indexed_container.index == container.indexed_container.index)
        assert xp.all(result.indexed_container.values == xp.stack(
            (container.indexed_container.values, other.indexed_container.values), axis=1))

    def test_loc(self, container: CompoundIndexed, xp: NumPyNamespace) -> None:
        result = container.loc[7]

        assert xp.all(result.index == xp.asarray([0, 0]))
        assert xp.all(result.array == container.array[2:])
        assert xp.all(result.array_container.values == container.array_container.values[2:])
        assert xp.all(result.indexed_container.index == xp.asarray([20, 20]))
        assert xp.all(result.indexed_container.values == container.indexed_container.values[2:])

    def test_iloc(self, container: CompoundIndexed, xp: NumPyNamespace) -> None:
        result = container.iloc[1]

        assert xp.all(result.index == xp.asarray([0, 0]))
        assert xp.all(result.array == container.array[2:])
        assert xp.all(result.array_container.values == container.array_container.values[2:])
        assert xp.all(result.indexed_container.index == xp.asarray([20, 20]))
        assert xp.all(result.indexed_container.values == container.indexed_container.values[2:])
