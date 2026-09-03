import pytest
from cbclib_v2.annotations import IntArray, JaxNumPy, NumPy, NumPyNamespace
from cbclib_v2.indexer import Miller
from cbclib_v2.scaler import (PointGroup, ReflectionsMap, StreakPoints, SymmetryGroup,
                              SymmetryOperator)

REFLECTION_GROUP_ORDERS = (
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

    @pytest.fixture
    def hkl(self, xp: NumPyNamespace) -> IntArray:
        return xp.asarray([1, 2, 3])

    def test_composition(self, c4: SymmetryOperator, hkl: IntArray,
                         xp: NumPyNamespace) -> None:
        transformed = (c4 @ c4).apply(hkl, xp)
        expected = c4.apply(c4.apply(hkl, xp), xp)

        # Composed operators act identically to sequential transformations.
        assert xp.all(transformed == expected)

    def test_friedel(self, c4: SymmetryOperator, hkl: IntArray,
                     xp: NumPyNamespace) -> None:
        transformed = (-c4).apply(hkl, xp)

        # Friedel inversion negates every transformed Miller index.
        assert xp.all(transformed == -c4.apply(hkl, xp))

    def test_invalid_determinant(self) -> None:
        with pytest.raises(ValueError, match="determinant"):
            SymmetryOperator(((2, 0, 0), (0, 1, 0), (0, 0, 1)))

class TestSymmetryGroup:
    @pytest.fixture
    def c4(self) -> SymmetryOperator:
        return SymmetryOperator(((0, -1, 0), (1, 0, 0), (0, 0, 1)))

    @pytest.fixture
    def group(self, c4: SymmetryOperator) -> SymmetryGroup:
        return SymmetryGroup.from_generators((c4,), expected_order=4)

    def test_from_generators(self, c4: SymmetryOperator, group: SymmetryGroup) -> None:
        operator = SymmetryOperator.identity()

        # A generated cyclic group contains every power through the return to identity.
        for _ in range(len(group)):
            assert operator in group.operators
            operator = operator @ c4
        assert operator == SymmetryOperator.identity()

    def test_with_friedel(self, group: SymmetryGroup) -> None:
        completed = group.with_friedel()
        expected = set(group.operators) | {-operator for operator in group.operators}

        # Friedel completion is exactly the union of a group and its inverted operators.
        assert set(completed.operators) == expected

    def test_expected_order(self, c4: SymmetryOperator, group: SymmetryGroup) -> None:
        expected_order = len(group) - 1

        with pytest.raises(ValueError, match=f"expected order {expected_order}"):
            SymmetryGroup.from_generators((c4,), expected_order=expected_order)

    def test_requires_closure(self, c4: SymmetryOperator) -> None:
        with pytest.raises(ValueError, match="closed under composition"):
            SymmetryGroup((SymmetryOperator.identity(), c4))

class TestPointGroup:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def symmetry(self) -> PointGroup:
        return PointGroup("4mm")

    @pytest.fixture
    def hkl(self, xp: NumPyNamespace) -> IntArray:
        return xp.asarray([1, 2, 3])

    @pytest.fixture
    def special_hkl(self, xp: NumPyNamespace) -> IntArray:
        return xp.asarray([1, 1, 0])

    @pytest.fixture
    def equivalent_hkl(self, xp: NumPyNamespace) -> IntArray:
        return xp.asarray([[1, 2, 3], [-2, 1, 3], [2, -1, -3], [4, 0, 1]])

    @pytest.mark.parametrize(("symbol", "reflection_order"), REFLECTION_GROUP_ORDERS)
    def test_general_reflection_order(self, symbol: str, reflection_order: int,
                                      xp: NumPyNamespace) -> None:
        equivalents = PointGroup(symbol).equivalents(xp.asarray([1, 2, 3]), xp)

        # A general reflection has the tabulated crystallographic orbit order.
        assert equivalents.shape == (reflection_order, 3)

    def test_friedel_equivalents(self, hkl: IntArray, xp: NumPyNamespace) -> None:
        symmetry = PointGroup("1")

        equivalents = symmetry.equivalents(hkl, xp)
        actual = {tuple(value.tolist()) for value in equivalents}
        expected = {tuple(hkl.tolist()), tuple((-hkl).tolist())}

        # Non-anomalous reflection equivalence always includes the Friedel pair.
        assert actual == expected

    def test_tetragonal_equivalents(self, symmetry: PointGroup, hkl: IntArray,
                                    xp: NumPyNamespace) -> None:
        equivalents = symmetry.equivalents(hkl, xp)
        actual = {tuple(value.tolist()) for value in equivalents}
        group = symmetry.reflection_symmetry()

        # A general orbit contains every group action once and is closed under inversion.
        assert len(actual) == len(group)
        assert all(tuple(operator.apply(hkl, xp).tolist()) in actual
                   for operator in group.operators)
        assert all(tuple((-xp.asarray(value)).tolist()) in actual for value in actual)

    def test_special_reflection_has_unique_equivalents(self, symmetry: PointGroup,
                                                       special_hkl: IntArray,
                                                       xp: NumPyNamespace) -> None:
        equivalents = symmetry.equivalents(special_hkl, xp)
        transformed = symmetry.reflection_symmetry().apply(special_hkl, xp)
        unique = xp.unique(transformed, axis=0)

        # Stabilizing operations collapse a special reflection's orbit to unique members.
        assert xp.all(equivalents == unique)
        assert equivalents.shape[0] < transformed.shape[0]

    def test_canonical(self, symmetry: PointGroup, equivalent_hkl: IntArray,
                       xp: NumPyNamespace) -> None:
        canonical = symmetry.canonical(equivalent_hkl, xp)
        expected = xp.asarray([
            max(tuple(value.tolist()) for value in symmetry.equivalents(hkl, xp))
            for hkl in equivalent_hkl
        ])

        # Canonical indices are the lexicographically maximal members of their orbits.
        assert xp.all(canonical == expected)

    def test_invalid_hkl_shape(self, xp: NumPyNamespace) -> None:
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            PointGroup("1").equivalents(xp.ones((2, 3), dtype=int), xp)

    @pytest.mark.parametrize(("symbol", "normalized"),
                             ((" 4 / mmm ", "4/mmm"), ("m-3m", "m-3m")))
    def test_symbol_normalization(self, symbol: str, normalized: str) -> None:
        # Point-group parsing removes spacing without changing the crystallographic symbol.
        assert PointGroup(symbol).symbol == normalized

    def test_invalid_symbol(self) -> None:
        with pytest.raises(ValueError, match="Unsupported point-group symbol"):
            PointGroup("not-a-point-group")

class TestReflectionsMap:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def point_group(self) -> PointGroup:
        return PointGroup("4mm")

    @pytest.fixture
    def miller(self, xp: NumPyNamespace) -> Miller:
        return Miller(index=xp.asarray([0, 0, 0, 1, 1]),
                      hkl=xp.asarray([[1, 2, 3], [-2, 1, 3], [4, 0, 1],
                                      [1, 2, 3], [-1, -2, -3]]))

    @pytest.fixture
    def reflections(self, miller: Miller, point_group: PointGroup,
                    xp: NumPyNamespace) -> ReflectionsMap:
        return ReflectionsMap.from_miller(miller, point_group, xp)

    @pytest.fixture
    def points(self, xp: NumPyNamespace) -> StreakPoints:
        return StreakPoints(index=xp.asarray([0, 0, 1]),
                            streak_id=xp.asarray([0, 2, 4]),
                            points=xp.zeros((3, 2)))

    @pytest.fixture
    def jax_miller(self) -> Miller:
        return Miller(index=JaxNumPy.asarray([0, 0]),
                      hkl=JaxNumPy.asarray([[1, 2, 3], [-2, 1, 3]]))

    @pytest.fixture
    def jax_reflections(self, jax_miller: Miller,
                        point_group: PointGroup) -> ReflectionsMap:
        return ReflectionsMap.from_miller(jax_miller, point_group, JaxNumPy)

    def test_pattern_local_mapping(self, reflections: ReflectionsMap, miller: Miller,
                                   point_group: PointGroup, xp: NumPyNamespace) -> None:
        reflection_id = reflections.reflection_id
        canonical = point_group.canonical(miller.hkl_indices, xp)
        same_pattern = miller.index[:, None] == miller.index[None, :]
        same_orbit = xp.all(canonical[:, None, :] == canonical[None, :, :], axis=-1)
        same_reflection = reflection_id[:, None] == reflection_id[None, :]

        # Reflections merge exactly when they share both a pattern and a symmetry orbit.
        assert xp.all(same_reflection == (same_pattern & same_orbit))
        assert len(reflections) == xp.unique_values(reflection_id).size

    def test_at_points(self, reflections: ReflectionsMap, points: StreakPoints,
                       xp: NumPyNamespace) -> None:
        reflection_id = reflections.at(points)

        # Point lookup follows each point's source streak into the reflection map.
        assert xp.all(reflection_id == reflections.reflection_id[points.streak_id])

    def test_canonical(self, reflections: ReflectionsMap, miller: Miller,
                       point_group: PointGroup, xp: NumPyNamespace) -> None:
        canonical = reflections.canonical(miller, xp)

        # Each output row represents every member of one pattern-local equivalence class.
        for reflection_id in range(len(reflections)):
            members = xp.where(reflections.reflection_id == reflection_id)[0]
            member_hkl = point_group.canonical(miller.hkl_indices[members], xp)
            assert xp.all(miller.index[members] == canonical.index[reflection_id])
            assert xp.all(member_hkl == canonical.hkl[reflection_id])

    def test_jax_namespace(self, jax_miller: Miller, jax_reflections: ReflectionsMap,
                           point_group: PointGroup) -> None:
        canonical = point_group.canonical(jax_miller.hkl_indices, JaxNumPy)
        same_orbit = JaxNumPy.all(canonical[:, None, :] == canonical[None, :, :], axis=-1)
        reflection_id = jax_reflections.reflection_id
        same_reflection = reflection_id[:, None] == reflection_id[None, :]

        # JAX construction preserves the same orbit relation and originating namespace.
        assert JaxNumPy.all(same_reflection == same_orbit)
        assert jax_reflections.__array_namespace__() is JaxNumPy
