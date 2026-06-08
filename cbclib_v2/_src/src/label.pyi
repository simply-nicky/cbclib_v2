from typing import Iterator, List, Sequence, Tuple
from ..annotations import (Array, BoolArray, IntArray, NDBoolArray, NDIntArray, NDRealArray,
                           RealArray)

class Structure:
    """Pixel connectivity structure class. Defines a two-dimensional connectivity kernel.
    Used in peaks and streaks detection algorithms.

    Args:
        radius : Radius of connectivity kernel. The size of the kernel is (2 * radius + 1,
            2 * radius + 1).
        rank : Rank determines which elements belong to the connectivity kernel, i.e. are
            considered as neighbors of the central element. Elements up to a squared distance
            of rank from the center are considered neighbors. Rank may range from 1 (no adjacent
            elements are neighbours) to radius (all elements in (2 * radius + 1, 2 * radius + 1)
            square are neighbours).

    Attributes:
        size : Number of elements in the connectivity kernel.
        x : x indices of the connectivity kernel.
        y : y indices of the connectivity kernel.
    """
    connectivity : int
    rank : int
    shape : List[int]

    def __init__(self, radii: List[int], connectivity: int): ...

    def __iter__(self) -> Iterator[List[int]]: ...

    def __len__(self) -> int: ...

    def squeeze(self) -> 'Structure': ...

    def expand_dims(self, axis: int | Sequence[int]) -> 'Structure': ...

    def to_array(self, out: NDBoolArray | None=None) -> NDBoolArray: ...

LabelResult = Tuple[NDIntArray, NDIntArray]

def binary_dilation(inp: BoolArray, structure: Structure, iterations: int=1,
                    mask: BoolArray | None=None, num_threads: int=1) -> NDBoolArray: ...

def label(inp: BoolArray | IntArray, structure: Structure, npts: int=1, num_threads: int=1
          ) -> LabelResult: ...

def total_mass(labels: LabelResult, data: Array) -> NDRealArray: ...

def mean(labels: LabelResult, data: Array) -> NDRealArray: ...

def center_of_mass(labels: LabelResult, data: Array) -> NDRealArray: ...

def moment_of_inertia(labels: LabelResult, data: Array) -> NDRealArray: ...

def covariance_matrix(labels: LabelResult, data: Array) -> NDRealArray: ...

def line_fit(labels: LabelResult, data: Array) -> NDRealArray: ...

def p_values(labels: LabelResult, lines: RealArray, data: Array, p0: float, vmin: float,
             xtol: float) -> NDRealArray: ...

def line_score(labels: LabelResult, lines: RealArray, data: Array, vmin: float, xtol: float,
               tau: float=1.0, kernel: str='gaussian') -> NDRealArray: ...
