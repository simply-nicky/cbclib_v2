from dataclasses import dataclass
from numbers import Integral
from typing import cast, Dict, Tuple
from .._src.annotations import AnyNamespace, IntArray

MatrixRow = Tuple[int, int, int]
StaticMatrix = Tuple[MatrixRow, MatrixRow, MatrixRow]

def _determinant(matrix: StaticMatrix) -> int:
    return (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))

@dataclass(frozen=True, order=True)
class SymmetryOperator:
    """Represent one integer reciprocal-space symmetry operation.

    Args:
        matrix: Unimodular integer matrix acting on Miller-index column vectors.
    """

    matrix: StaticMatrix

    def __post_init__(self) -> None:
        if len(self.matrix) != 3 or any(len(row) != 3 for row in self.matrix):
            raise ValueError("Symmetry operator matrix must have shape (3, 3)")
        if any(not isinstance(value, Integral) for row in self.matrix for value in row):
            raise ValueError("Symmetry operator matrix must contain integers")
        matrix = tuple(tuple(int(value) for value in row) for row in self.matrix)
        matrix = cast(StaticMatrix, matrix)
        if abs(_determinant(matrix)) != 1:
            raise ValueError("Symmetry operator matrix must have determinant +1 or -1")
        object.__setattr__(self, "matrix", matrix)

    def __matmul__(self, other: 'SymmetryOperator') -> 'SymmetryOperator':
        """Compose this operation with another reciprocal-space operation."""
        matrix = tuple(tuple(sum(self.matrix[i][k] * other.matrix[k][j]
                                 for k in range(3)) for j in range(3))
                       for i in range(3))
        return SymmetryOperator(cast(StaticMatrix, matrix))

    def __neg__(self) -> 'SymmetryOperator':
        """Compose this operation with Friedel inversion."""
        matrix = tuple(tuple(-value for value in row) for row in self.matrix)
        return SymmetryOperator(cast(StaticMatrix, matrix))

    @classmethod
    def identity(cls) -> 'SymmetryOperator':
        """Return the identity reciprocal-space operation."""
        return cls(((1, 0, 0), (0, 1, 0), (0, 0, 1)))

    def apply(self, hkl: IntArray, xp: AnyNamespace) -> IntArray:
        """Apply this reciprocal-space operation to Miller indices.

        Args:
            hkl: Miller indices with shape ``(..., 3)``.
            xp: Array namespace used for the calculation.

        Returns:
            Transformed Miller indices with the same shape as *hkl*.

        Raises:
            ValueError: If the final axis of *hkl* does not have length three.
        """
        hkl = xp.asarray(hkl)
        if hkl.ndim == 0 or hkl.shape[-1] != 3:
            raise ValueError(f"hkl must have shape (..., 3), but got {hkl.shape}")
        matrix = xp.asarray(self.matrix, dtype=hkl.dtype)
        return xp.asarray(xp.sum(matrix * hkl[..., None, :], axis=-1))

@dataclass(frozen=True)
class SymmetryGroup:
    """Represent a closed collection of reciprocal-space symmetry operations."""

    operators: Tuple[SymmetryOperator, ...]

    def __post_init__(self) -> None:
        operators = tuple(sorted(set(self.operators)))
        if not operators:
            raise ValueError("Symmetry group must contain at least one operator")
        identity = SymmetryOperator.identity()
        if identity not in operators:
            raise ValueError("Symmetry group must contain the identity operator")
        operator_set = set(operators)
        if any(left @ right not in operator_set
               for left in operators for right in operators):
            raise ValueError("Symmetry group operators must be closed under composition")
        object.__setattr__(self, "operators", operators)

    @classmethod
    def from_generators(cls, generators: Tuple[SymmetryOperator, ...],
                        expected_order: int) -> 'SymmetryGroup':
        """Generate the closed symmetry group containing a set of generators.

        Args:
            generators: Operations that generate the group under composition.
            expected_order: Expected number of operations used to validate the
                generated group and bound closure generation.

        Returns:
            Closed symmetry group containing the supplied generators.

        Raises:
            ValueError: If the generated group does not have *expected_order* operations.
        """
        if expected_order < 1:
            raise ValueError("expected_order must be positive")
        operators = {SymmetryOperator.identity()}
        frontier = [SymmetryOperator.identity()]
        while frontier:
            operator = frontier.pop()
            for generator in generators:
                candidate = operator @ generator
                if candidate not in operators:
                    operators.add(candidate)
                    frontier.append(candidate)
                    if len(operators) > expected_order:
                        raise ValueError(f"Generated symmetry group exceeds expected order "
                                         f"{expected_order}")
        if len(operators) != expected_order:
            raise ValueError(f"Generated symmetry group has {len(operators)} operations, "
                             f"expected {expected_order}")
        return cls(tuple(operators))

    @classmethod
    def identity(cls) -> 'SymmetryGroup':
        """Return the group containing only the identity operation."""
        return cls((SymmetryOperator.identity(),))

    def __len__(self) -> int:
        return len(self.operators)

    def with_friedel(self) -> 'SymmetryGroup':
        """Return this group completed with Friedel-inverted operations."""
        operators = set(self.operators) | {-operator for operator in self.operators}
        return SymmetryGroup(tuple(operators))

    def apply(self, hkl: IntArray, xp: AnyNamespace) -> IntArray:
        """Apply every group operation to Miller indices.

        The operation axis is inserted before the Miller-index axis, producing
        an array with shape ``(..., n_operators, 3)``.
        """
        return xp.asarray(xp.stack(tuple(operator.apply(hkl, xp)
                                         for operator in self.operators), axis=-2))

INVERSION = SymmetryOperator(((-1, 0, 0), (0, -1, 0), (0, 0, -1)))
C2_X = SymmetryOperator(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
C2_Y = SymmetryOperator(((-1, 0, 0), (0, 1, 0), (0, 0, -1)))
C2_Z = SymmetryOperator(((-1, 0, 0), (0, -1, 0), (0, 0, 1)))
MIRROR_Y = SymmetryOperator(((1, 0, 0), (0, -1, 0), (0, 0, 1)))
C4_Z = SymmetryOperator(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
MIRROR_DIAGONAL = SymmetryOperator(((0, 1, 0), (1, 0, 0), (0, 0, 1)))
C2_DIAGONAL = SymmetryOperator(((0, 1, 0), (1, 0, 0), (0, 0, -1)))
C6_Z = SymmetryOperator(((0, -1, 0), (1, 1, 0), (0, 0, 1)))
C3_Z = SymmetryOperator(((-1, -1, 0), (1, 0, 0), (0, 0, 1)))
C3_CUBIC = SymmetryOperator(((0, 0, 1), (1, 0, 0), (0, 1, 0)))
MIRROR_CUBIC = SymmetryOperator(((0, 1, 0), (1, 0, 0), (0, 0, 1)))
S4_Z = INVERSION @ C4_Z
S6_Z = INVERSION @ C6_Z

@dataclass(frozen=True)
class PointGroupSpec:
    generators: Tuple[SymmetryOperator, ...]
    order: int

POINT_GROUPS: Dict[str, PointGroupSpec] = {
    "1": PointGroupSpec((), 1),
    "-1": PointGroupSpec((INVERSION,), 2),
    "2": PointGroupSpec((C2_Y,), 2),
    "m": PointGroupSpec((MIRROR_Y,), 2),
    "2/m": PointGroupSpec((C2_Y, INVERSION), 4),
    "222": PointGroupSpec((C2_X, C2_Y), 4),
    "mm2": PointGroupSpec((C2_Z, MIRROR_Y), 4),
    "mmm": PointGroupSpec((C2_X, C2_Y, INVERSION), 8),
    "4": PointGroupSpec((C4_Z,), 4),
    "-4": PointGroupSpec((S4_Z,), 4),
    "4/m": PointGroupSpec((C4_Z, INVERSION), 8),
    "422": PointGroupSpec((C4_Z, C2_DIAGONAL), 8),
    "4mm": PointGroupSpec((C4_Z, MIRROR_Y), 8),
    "-42m": PointGroupSpec((S4_Z, C2_X), 8),
    "4/mmm": PointGroupSpec((C4_Z, MIRROR_Y, INVERSION), 16),
    "3": PointGroupSpec((C3_Z,), 3),
    "-3": PointGroupSpec((C3_Z, INVERSION), 6),
    "32": PointGroupSpec((C3_Z, C2_DIAGONAL), 6),
    "3m": PointGroupSpec((C3_Z, MIRROR_DIAGONAL), 6),
    "-3m": PointGroupSpec((C3_Z, MIRROR_DIAGONAL, INVERSION), 12),
    "6": PointGroupSpec((C6_Z,), 6),
    "-6": PointGroupSpec((S6_Z,), 6),
    "6/m": PointGroupSpec((C6_Z, INVERSION), 12),
    "622": PointGroupSpec((C6_Z, C2_DIAGONAL), 12),
    "6mm": PointGroupSpec((C6_Z, MIRROR_DIAGONAL), 12),
    "-6m2": PointGroupSpec((S6_Z, MIRROR_DIAGONAL), 12),
    "6/mmm": PointGroupSpec((C6_Z, MIRROR_DIAGONAL, INVERSION), 24),
    "23": PointGroupSpec((C3_CUBIC, C2_Z), 12),
    "m-3": PointGroupSpec((C3_CUBIC, C2_Z, INVERSION), 24),
    "432": PointGroupSpec((C3_CUBIC, C4_Z), 24),
    "-43m": PointGroupSpec((C3_CUBIC, C2_Z, MIRROR_CUBIC), 24),
    "m-3m": PointGroupSpec((C3_CUBIC, C4_Z, INVERSION), 48),
}

@dataclass(frozen=True)
class PointGroup:
    """Represent a crystallographic point group for reflection merging.

    Reflection equivalence includes the supplied point-group operations and
    Friedel inversion, as appropriate for non-anomalous kinematic diffraction.

    Args:
        symbol: Hermann--Mauguin point-group symbol in the standard setting.
    """

    symbol: str = "1"

    def __post_init__(self) -> None:
        symbol = self.symbol.replace("\N{MINUS SIGN}", "-").replace(" ", "")
        if symbol not in POINT_GROUPS:
            raise ValueError(f"Unsupported point-group symbol: {self.symbol!r}")
        object.__setattr__(self, "symbol", symbol)

    def reflection_symmetry(self) -> SymmetryGroup:
        specification = POINT_GROUPS[self.symbol]
        point_group = SymmetryGroup.from_generators(specification.generators,
                                                    specification.order)
        return point_group.with_friedel()

    def equivalents(self, hkl: IntArray, xp: AnyNamespace) -> IntArray:
        """Return the unique intensity-equivalent indices of one reflection.

        Args:
            hkl: One Miller index with shape ``(3,)``.
            xp: Array namespace used for the calculation.

        Returns:
            Unique equivalent Miller indices with shape ``(n_equivalents, 3)``.

        Raises:
            ValueError: If *hkl* does not have shape ``(3,)``.
        """
        hkl = xp.asarray(hkl)
        if hkl.shape != (3,):
            raise ValueError(f"hkl must have shape (3,), but got {hkl.shape}")
        transformed = self.reflection_symmetry().apply(hkl, xp)
        return xp.asarray(xp.unique(transformed, axis=0))

    def canonical(self, hkl: IntArray, xp: AnyNamespace) -> IntArray:
        """Return deterministic representatives of intensity-equivalent reflections.

        Args:
            hkl: Miller indices with shape ``(..., 3)``.
            xp: Array namespace used for the calculation.

        Returns:
            Lexicographically maximal orbit representatives with the same shape as *hkl*.
            Because Friedel equivalents are included, the first nonzero component of each
            representative is positive.

        Raises:
            ValueError: If the final axis of *hkl* does not have length three.
        """
        hkl = xp.asarray(hkl)
        if hkl.ndim == 0 or hkl.shape[-1] != 3:
            raise ValueError(f"hkl must have shape (..., 3), but got {hkl.shape}")
        transformed = self.reflection_symmetry().apply(hkl, xp)
        canonical = transformed[..., 0, :]
        for index in range(1, transformed.shape[-2]):
            candidate = transformed[..., index, :]
            is_greater = candidate[..., 0] > canonical[..., 0]
            is_greater |= ((candidate[..., 0] == canonical[..., 0]) &
                           (candidate[..., 1] > canonical[..., 1]))
            is_greater |= ((candidate[..., 0] == canonical[..., 0]) &
                           (candidate[..., 1] == canonical[..., 1]) &
                           (candidate[..., 2] > canonical[..., 2]))
            canonical = xp.where(is_greater[..., None], candidate, canonical)
        return xp.asarray(canonical)
