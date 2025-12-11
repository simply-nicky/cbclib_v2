from typing import Dict, List, Tuple
from dataclasses import dataclass
import pytest
from cbclib_v2 import Container, CrystData, StackedStreaks
from cbclib_v2.annotations import ArrayNamespace, NumPy

@dataclass
class SimpleContainer(Container):
    id : int

@dataclass
class ComplexContainer(Container):
    a : List[int | str]
    b : SimpleContainer
    c : Dict[str, float]
    d : SimpleContainer | None
    e : List[SimpleContainer]
    f : Tuple[SimpleContainer, ...]

class TestContainerSerialization:
    """Test serialization and deserialization of data containers."""
    def test_simple_container(self):
        """Test serialization/deserialization of a simple container."""
        original = SimpleContainer(id=42)
        data_dict = original.to_dict()
        restored = SimpleContainer.from_dict(**data_dict)

        assert original == restored, "Restored SimpleContainer does not match the original"

    def test_complex_container(self):
        """Test serialization/deserialization of a complex container."""
        original = ComplexContainer(
            a=[1, 2, 3],
            b=SimpleContainer(id=7),
            c={"pi": 3.14, "e": 2.71},
            d=None,
            e=[SimpleContainer(id=10), SimpleContainer(id=20)],
            f=(SimpleContainer(id=30), SimpleContainer(id=40))
        )
        data_dict = original.to_dict()
        restored = ComplexContainer.from_dict(**data_dict)

        assert original == restored, "Restored ComplexContainer does not match the original"

class TestDataContainers:
    """Test data container functionality."""
    @pytest.fixture
    def xp(self) -> ArrayNamespace:
        return NumPy

    @pytest.fixture
    def streaks1(self, xp: ArrayNamespace) -> StackedStreaks:
        """First StackedStreaks with 10 elements."""
        return StackedStreaks(index=xp.zeros(12), module_id=xp.zeros(12, dtype=int),
                              lines=xp.ones((12, 4)), num_modules=2)

    @pytest.fixture
    def streaks2(self, xp: ArrayNamespace) -> StackedStreaks:
        """Second StackedStreaks with 10 elements."""
        return StackedStreaks(index=xp.arange(10, 20), module_id=xp.ones(10, dtype=int),
                              lines=2.0 * xp.ones((10, 4)), num_modules=2)

    @pytest.fixture
    def streaks3(self, xp: ArrayNamespace) -> StackedStreaks:
        """Third StackedStreaks with 0 elements (empty)."""
        return StackedStreaks(index=xp.array([]), module_id=xp.array([], dtype=int),
                              lines=xp.empty((0, 4)), num_modules=2)

    @pytest.fixture
    def data(self, xp: ArrayNamespace) -> CrystData:
        """Sample CrystData fixture."""
        return CrystData(data=xp.ones((10, 10, 10)), whitefield=xp.array([]))

    def test_concatenate(self, streaks1: StackedStreaks, streaks2: StackedStreaks,
                         streaks3: StackedStreaks, xp: ArrayNamespace):
        """Test concatenation of three StackedStreaks including an empty one.
        """
        # Concatenate the three StackedStreaks
        result = StackedStreaks.concatenate((streaks1, streaks2, streaks3))
        size = sum((streaks1.shape[0], streaks2.shape[0], streaks3.shape[0]))

        # Verify the concatenated result has the correct total length
        assert result.index.shape[0] == size
        assert result.module_id.shape[0] == size
        assert result.lines.shape == (size, 4)

        # Verify num_modules is preserved
        assert result.num_modules == 2

        # Verify index values are correct
        index = xp.concatenate([streaks1.index, streaks2.index, streaks3.index])
        assert xp.all(result.index == index), "Index values don't match expected"

        # Verify module_id values are correct
        module_id = xp.concatenate([streaks1.module_id, streaks2.module_id, streaks3.module_id])
        assert xp.all(result.module_id == module_id), "Module IDs don't match expected"

        # Verify lines values are correct
        lines = xp.concatenate([streaks1.lines, streaks2.lines, streaks3.lines])
        assert xp.all(result.lines == lines), "Line values don't match expected"

    def test_concatenate_empty_only(self, streaks3: StackedStreaks):
        """Test concatenating a single empty StackedStreaks."""
        with pytest.raises(ValueError):
            StackedStreaks.concatenate((streaks3,))

    def test_contents(self, streaks1: StackedStreaks, streaks3: StackedStreaks, data: CrystData):
        """Test contents method of data containers."""
        assert set(streaks1.contents().keys()) == {'module_id', 'lines'}
        assert set(streaks3.contents().keys()) == {'module_id', 'lines'}
        assert set(data.contents().keys()) == {'data', 'frames', 'mask'}

    def test_shape(self, streaks1: StackedStreaks, streaks3: StackedStreaks, data: CrystData):
        """Test shape property of data containers."""
        assert streaks1.shape == (streaks1.index.shape[0],)
        assert streaks3.shape == (streaks3.index.shape[0],)
        assert data.shape == data.data.shape

    def test_reshape_zero(self, streaks3: StackedStreaks):
        """Test reshape method on empty StackedStreaks."""
        reshaped = streaks3.reshape((0, 1))

        assert reshaped.index.shape == (0,)
        assert reshaped.module_id.shape == (0, 1)
        assert reshaped.lines.shape == (0, 1, 4)

        with pytest.raises(ValueError):
            streaks3.reshape((1, 0))  # Invalid reshape for empty data

    def test_reshape(self, streaks1: StackedStreaks, xp: ArrayNamespace):
        """Test reshape method of StackedStreaks."""
        new_shape = (3, 2, 2)
        reshaped = streaks1.reshape(new_shape)

        assert reshaped.index.shape == new_shape[:1]
        assert reshaped.module_id.shape == new_shape
        assert reshaped.lines.shape == (3, 2, 2, 4)

        # Verify data integrity after reshape
        expected_index = xp.reshape(streaks1.index, new_shape)[:, 0, 0]
        expected_module_id = xp.reshape(streaks1.module_id, new_shape)
        expected_lines = xp.reshape(streaks1.lines, (3, 2, 2, 4))

        assert xp.all(reshaped.index == expected_index)
        assert xp.all(reshaped.module_id == expected_module_id)
        assert xp.all(reshaped.lines == expected_lines)

    def test_reshape_invalid(self, streaks2: StackedStreaks):
        """Test reshape method with invalid shape."""
        with pytest.raises(ValueError):
            streaks2.reshape((5, 2))  # streaks2.index is incompatible with this shape
