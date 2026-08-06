import pytest
from cbclib_v2.annotations import JaxNumPy, NumPy, NumPyNamespace
from cbclib_v2.indexer import Miller
from cbclib_v2.scaler import (PointGroup, ReflectionsMap, StreakPoints, SymmetryGroup,
                              SymmetryOperator)

POINT_GROUP_ORDERS = (
    ("1", 2), ("-1", 2), ("2", 4), ("m", 4), ("2/m", 4),
    ("222", 8), ("mm2", 8), ("mmm", 8),
    ("4", 8), ("-4", 8), ("4/m", 8), ("422", 16), ("4mm", 16),
    ("-42m", 16), ("4/mmm", 16),
    ("3", 6), ("-3", 6), ("32", 12), ("3m", 12), ("-3m", 12),
    ("6", 12), ("-6", 12), ("6/m", 12), ("622", 24), ("6mm", 24),
    ("-6m2", 24), ("6/mmm", 24),
    ("23", 24), ("m-3", 24), ("432", 48), ("-43m", 48), ("m-3m", 48),
)

class TestSymmetryOperator:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def c4(self) -> SymmetryOperator:
        return SymmetryOperator(((0, -1, 0), (1, 0, 0), (0, 0, 1)))

    def test_composition(self, c4: SymmetryOperator, xp: NumPyNamespace) -> None:
        hkl = xp.asarray([1, 2, 3])

        transformed = (c4 @ c4).apply(hkl, xp)

        assert xp.all(transformed == xp.asarray([-1, -2, 3]))

    def test_friedel(self, c4: SymmetryOperator, xp: NumPyNamespace) -> None:
        hkl = xp.asarray([1, 2, 3])

        transformed = (-c4).apply(hkl, xp)

        assert xp.all(transformed == -c4.apply(hkl, xp))

    def test_invalid_determinant(self) -> None:
        with pytest.raises(ValueError, match="determinant"):
            SymmetryOperator(((2, 0, 0), (0, 1, 0), (0, 0, 1)))

class TestSymmetryGroup:
    @pytest.fixture
    def c4(self) -> SymmetryOperator:
        return SymmetryOperator(((0, -1, 0), (1, 0, 0), (0, 0, 1)))

    def test_from_generators(self, c4: SymmetryOperator) -> None:
        group = SymmetryGroup.from_generators((c4,), expected_order=4)

        assert len(group) == 4
        assert c4 in group.operators

    def test_with_friedel(self, c4: SymmetryOperator) -> None:
        group = SymmetryGroup.from_generators((c4,), expected_order=4)

        completed = group.with_friedel()

        assert len(completed) == 8
        assert all(-operator in completed.operators for operator in group.operators)

    def test_expected_order(self, c4: SymmetryOperator) -> None:
        with pytest.raises(ValueError, match="expected order 3"):
            SymmetryGroup.from_generators((c4,), expected_order=3)

    def test_requires_closure(self, c4: SymmetryOperator) -> None:
        with pytest.raises(ValueError, match="closed under composition"):
            SymmetryGroup((SymmetryOperator.identity(), c4))

class TestPointGroup:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.mark.parametrize(("symbol", "order"), POINT_GROUP_ORDERS)
    def test_general_reflection_order(self, symbol: str, order: int,
                                      xp: NumPyNamespace) -> None:
        equivalents = PointGroup(symbol).equivalents(xp.asarray([1, 2, 3]), xp)

        assert equivalents.shape == (order, 3)

    def test_friedel_equivalents(self, xp: NumPyNamespace) -> None:
        symmetry = PointGroup("1")

        equivalents = symmetry.equivalents(xp.asarray([1, 2, 3]), xp)

        assert {tuple(hkl.tolist()) for hkl in equivalents} == {(1, 2, 3), (-1, -2, -3)}

    def test_tetragonal_equivalents(self, xp: NumPyNamespace) -> None:
        symmetry = PointGroup("4mm")

        equivalents = symmetry.equivalents(xp.asarray([1, 2, 3]), xp)

        assert equivalents.shape == (16, 3)
        assert len({tuple(hkl.tolist()) for hkl in equivalents}) == 16
        assert (-2, 1, 3) in {tuple(hkl.tolist()) for hkl in equivalents}
        assert (2, -1, -3) in {tuple(hkl.tolist()) for hkl in equivalents}

    def test_special_reflection_has_unique_equivalents(self, xp: NumPyNamespace) -> None:
        symmetry = PointGroup("4mm")

        equivalents = symmetry.equivalents(xp.asarray([1, 1, 0]), xp)

        assert equivalents.shape == (4, 3)

    def test_canonical(self, xp: NumPyNamespace) -> None:
        symmetry = PointGroup("4mm")
        hkl = xp.asarray([[1, 2, 3], [-2, 1, 3], [2, -1, -3], [4, 0, 1]])

        canonical = symmetry.canonical(hkl, xp)

        assert canonical.shape == hkl.shape
        assert xp.all(canonical[0] == canonical[1])
        assert xp.all(canonical[0] == canonical[2])
        assert xp.all(canonical[0] == xp.asarray([2, 1, 3]))
        assert xp.any(canonical[0] != canonical[3])

    def test_invalid_hkl_shape(self, xp: NumPyNamespace) -> None:
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            PointGroup("1").equivalents(xp.ones((2, 3), dtype=int), xp)

    def test_symbol_normalization(self) -> None:
        assert PointGroup(" 4 / mmm ").symbol == "4/mmm"
        assert PointGroup("m-3m").symbol == "m-3m"

    def test_invalid_symbol(self) -> None:
        with pytest.raises(ValueError, match="Unsupported point-group symbol"):
            PointGroup("not-a-point-group")

class TestReflectionsMap:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def miller(self, xp: NumPyNamespace) -> Miller:
        return Miller(index=xp.asarray([0, 0, 0, 1, 1]),
                      hkl=xp.asarray([[1, 2, 3], [-2, 1, 3], [4, 0, 1],
                                      [1, 2, 3], [-1, -2, -3]]))

    @pytest.fixture
    def reflections(self, miller: Miller, xp: NumPyNamespace) -> ReflectionsMap:
        return ReflectionsMap.from_miller(miller, PointGroup("4mm"), xp)

    def test_pattern_local_mapping(self, reflections: ReflectionsMap) -> None:
        reflection_id = reflections.reflection_id

        assert reflection_id[0] == reflection_id[1]
        assert reflection_id[0] != reflection_id[2]
        assert reflection_id[0] != reflection_id[3]
        assert reflection_id[3] == reflection_id[4]
        assert len(reflections) == 3

    def test_at_points(self, reflections: ReflectionsMap, xp: NumPyNamespace) -> None:
        points = StreakPoints(index=xp.asarray([0, 0, 1]),
                              streak_id=xp.asarray([0, 2, 4]),
                              points=xp.zeros((3, 2)))

        reflection_id = reflections.at(points)

        assert xp.all(reflection_id == reflections.reflection_id[points.streak_id])

    def test_canonical(self, reflections: ReflectionsMap, miller: Miller,
                       xp: NumPyNamespace) -> None:
        canonical = reflections.canonical(miller, PointGroup("4mm"), xp)

        assert xp.all(canonical.index == xp.asarray([0, 0, 1]))
        assert xp.all(canonical.hkl == xp.asarray([[2, 1, 3], [4, 0, 1],
                                                   [2, 1, 3]]))

    def test_jax_namespace(self) -> None:
        xp = JaxNumPy
        miller = Miller(index=xp.asarray([0, 0]),
                        hkl=xp.asarray([[1, 2, 3], [-2, 1, 3]]))

        reflections = ReflectionsMap.from_miller(miller, PointGroup("4mm"), xp)

        assert len(reflections) == 1
        assert xp.all(reflections.reflection_id == xp.asarray([0, 0]))
