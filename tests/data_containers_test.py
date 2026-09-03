from dataclasses import dataclass
import pytest
from cbclib_v2 import ArrayContainer, IndexedContainer
from cbclib_v2.annotations import IntArray, NumPy, NumPyNamespace, RealArray

@dataclass
class ArrayLeaf(ArrayContainer):
    values: RealArray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

@dataclass
class CompoundArray(ArrayContainer):
    array: RealArray
    container: ArrayLeaf

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape[:-1]

@dataclass
class IndexedLeaf(IndexedContainer):
    index: IntArray
    values: RealArray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape[:-1]

@dataclass
class CompoundIndexed(IndexedContainer):
    index: IntArray
    array: RealArray
    array_container: ArrayLeaf
    indexed_container: IndexedLeaf

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape[:-1]

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

    def test_getitem_ellipsis(self, container: CompoundArray, xp: NumPyNamespace) -> None:
        reshaped = container.reshape((2, 2))
        result = reshaped[..., 1]

        assert result.shape == (2,)
        assert xp.all(result.array == reshaped.array[:, 1, :])
        assert xp.all(result.container.values == reshaped.container.values[:, 1, :])

    def test_getitem_boolean_mask(self, container: CompoundArray,
                                  xp: NumPyNamespace) -> None:
        reshaped = container.reshape((2, 2))
        mask = xp.asarray([[True, False], [False, True]])
        result = reshaped[mask]

        assert result.shape == (2,)
        assert xp.all(result.array == reshaped.array[mask])
        assert xp.all(result.container.values == reshaped.container.values[mask])

    def test_getitem_scalar_preserves_array(self, xp: NumPyNamespace) -> None:
        container = ArrayLeaf(values=xp.arange(4))
        result = container[1]

        assert result.values.shape == ()
        assert 'values' in result.contents()

    def test_getitem_rejects_multiple_ellipses(self, container: CompoundArray) -> None:
        with pytest.raises(IndexError, match="one ellipsis"):
            _ = container[..., ...]

    def test_shape_validation(self, xp: NumPyNamespace) -> None:
        leaf = ArrayLeaf(values=xp.zeros((5, 2)))

        with pytest.raises(ValueError, match="incompatible with leading shape"):
            _ = CompoundArray(array=xp.zeros((4, 3)), container=leaf)

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

        assert xp.all(result.index == xp.reshape(container.index, (2, 2)))
        assert xp.all(result.array == xp.reshape(container.array, (2, 2, 3)))
        assert xp.all(result.array_container.values == xp.reshape(
            container.array_container.values, (2, 2, 2)))
        assert xp.all(result.indexed_container.index == xp.reshape(
            container.indexed_container.index, (2, 2)))
        assert xp.all(result.indexed_container.values == xp.reshape(
            container.indexed_container.values, (2, 2, 4)))

    def test_concat(self, container: CompoundIndexed, other: CompoundIndexed,
                    xp: NumPyNamespace) -> None:
        result = CompoundIndexed.concat((container, other))

        assert xp.all(result.index == xp.concat((container.index, other.index)))
        assert xp.all(result.array == xp.concat((container.array, other.array)))
        assert xp.all(result.array_container.values == xp.concat(
            (container.array_container.values, other.array_container.values)))
        assert xp.all(result.indexed_container.index == xp.concat(
            (container.indexed_container.index, other.indexed_container.index)))
        assert xp.all(result.indexed_container.values == xp.concat(
            (container.indexed_container.values, other.indexed_container.values)))

    def test_stack(self, container: CompoundIndexed, other: CompoundIndexed,
                   xp: NumPyNamespace) -> None:
        result = CompoundIndexed.stack((container, other), axis=1)

        assert xp.all(result.index == xp.stack((container.index, other.index), axis=1))
        assert xp.all(result.array == xp.stack((container.array, other.array), axis=1))
        assert xp.all(result.array_container.values == xp.stack(
            (container.array_container.values, other.array_container.values), axis=1))
        assert xp.all(result.indexed_container.index == xp.stack(
            (container.indexed_container.index, other.indexed_container.index), axis=1))
        assert xp.all(result.indexed_container.values == xp.stack(
            (container.indexed_container.values, other.indexed_container.values), axis=1))

    def test_loc(self, container: CompoundIndexed, xp: NumPyNamespace) -> None:
        result = container.loc[7]

        assert xp.all(result.index == xp.asarray([7, 7]))
        assert xp.all(result.array == container.array[2:])
        assert xp.all(result.array_container.values == container.array_container.values[2:])
        assert xp.all(result.indexed_container.index == xp.asarray([20, 20]))
        assert xp.all(result.indexed_container.values == container.indexed_container.values[2:])

    def test_iloc(self, container: CompoundIndexed, xp: NumPyNamespace) -> None:
        result = container.iloc[1]

        assert xp.all(result.index == xp.asarray([7, 7]))
        assert xp.all(result.array == container.array[2:])
        assert xp.all(result.array_container.values == container.array_container.values[2:])
        assert xp.all(result.indexed_container.index == xp.asarray([20, 20]))
        assert xp.all(result.indexed_container.values == container.indexed_container.values[2:])

    def test_take_unsorted_repeated_index(self, xp: NumPyNamespace) -> None:
        container = IndexedLeaf(
            index=xp.asarray([7, 3, 7, 5, 3]),
            values=xp.reshape(xp.arange(10), (5, 2)),
        )
        result = container.take([7, 3])

        positions = xp.asarray([0, 2, 1, 4])
        assert xp.all(result.index == xp.asarray([7, 7, 3, 3]))
        assert xp.all(result.values == container.values[positions])

    def test_take_repeated_labels_with_reset(self, xp: NumPyNamespace) -> None:
        container = IndexedLeaf(
            index=xp.asarray([7, 3, 7]),
            values=xp.reshape(xp.arange(6), (3, 2)),
        )
        result = container.take([7, 7], reset_index=True)

        assert xp.all(result.index == xp.asarray([0, 0, 1, 1]))
        assert xp.all(result.values == container.values[xp.asarray([0, 2, 0, 2])])

    def test_reset_preserves_physical_order(self, xp: NumPyNamespace) -> None:
        container = IndexedLeaf(
            index=xp.asarray([7, 3, 7, 5, 3]),
            values=xp.reshape(xp.arange(10), (5, 2)),
        )
        result = container.reset()

        assert xp.all(result.index == xp.asarray([2, 0, 2, 1, 0]))
        assert xp.all(result.values == container.values)

    def test_take_multidimensional_index(self, xp: NumPyNamespace) -> None:
        container = IndexedLeaf(
            index=xp.asarray([[7, 3, 7], [5, 3, 5]]),
            values=xp.reshape(xp.arange(12), (2, 3, 2)),
        )
        result = container.take([5, 7])

        flat_values = xp.reshape(container.values, (6, 2))
        positions = xp.asarray([3, 5, 0, 2])
        assert result.shape == (4,)
        assert xp.all(result.index == xp.asarray([5, 5, 7, 7]))
        assert xp.all(result.values == flat_values[positions])

    def test_reset_multidimensional_index(self, xp: NumPyNamespace) -> None:
        container = IndexedLeaf(
            index=xp.asarray([[7, 3, 7], [5, 3, 5]]),
            values=xp.reshape(xp.arange(12), (2, 3, 2)),
        )
        result = container.reset()

        assert result.index.shape == container.shape
        assert xp.all(result.index == xp.asarray([[2, 0, 2], [1, 0, 1]]))
        assert xp.all(result.values == container.values)

    def test_take_empty(self, xp: NumPyNamespace) -> None:
        container = IndexedLeaf(
            index=xp.asarray([], dtype=int),
            values=xp.empty((0, 2)),
        )
        result = container.take([])

        assert result.shape == (0,)
        assert result.index.size == 0
        assert result.values.shape == (0, 2)

    def test_take_missing_label(self, container: CompoundIndexed) -> None:
        with pytest.raises(KeyError, match="not present"):
            _ = container.take([11])

    def test_broadcast_index_lookup(self, xp: NumPyNamespace) -> None:
        container = IndexedLeaf(
            index=xp.asarray([[7], [3]]),
            values=xp.reshape(xp.arange(12), (2, 3, 2)),
        )
        result = container.take([7])

        assert result.shape == (3,)
        assert xp.all(result.index == xp.asarray([7, 7, 7]))
        assert xp.all(result.values == container.values[0])
