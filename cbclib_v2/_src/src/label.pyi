from __future__ import annotations
from typing import Iterable, Iterator, List, overload
from ..annotations import Array, BoolArray, IntSequence, NDIntArray, NDRealArray, RealSequence

class PointSet2D:
    x : List[int]
    y : List[int]

    def __init__(self, x: IntSequence, y: IntSequence): ...

    def __contains__(self, point: Iterable[int]) -> bool: ...

    def __iter__(self) -> Iterator[List[int]]: ...

    def __len__(self) -> int: ...

class PointSet3D:
    x : List[int]
    y : List[int]
    z : List[int]

    def __init__(self, x: IntSequence, y: IntSequence, z: IntSequence): ...

    def __contains__(self, point: Iterable[int]) -> bool: ...

    def __iter__(self) -> Iterator[List[int]]: ...

    def __len__(self) -> int: ...

class Structure2D:
    """Pixel connectivity structure class. Defines a two-dimensional connectivity kernel.
    Used in peaks and streaks detection algorithms.

    Args:
        radius : Radius of connectivity kernel. The size of the kernel is (2 * radius + 1,
            2 * radius + 1).
        rank : Rank determines which elements belong to the connectivity kernel, i.e. are
            considered as neighbors of the central element. Elements up to a squared distance
            of raml from the center are considered neighbors. Rank may range from 1 (no adjacent
            elements are neighbours) to radius (all elements in (2 * radius + 1, 2 * radius + 1)
            square are neighbours).

    Attributes:
        size : Number of elements in the connectivity kernel.
        x : x indices of the connectivity kernel.
        y : y indices of the connectivity kernel.
    """
    radius : int
    rank : int
    x : List[int]
    y : List[int]

    def __init__(self, radius: int, rank: int): ...

    def __iter__(self) -> Iterator[List[int]]: ...

    def __len__(self) -> int: ...

class Structure3D:
    """Pixel connectivity structure class. Defines a three-dimensional connectivity kernel.
    Used in peaks and streaks detection algorithms.

    Args:
        radius : Radius of connectivity kernel. The size of the kernel is (2 * radius + 1,
            2 * radius + 1, 2 * radius + 1).
        rank : Rank determines which elements belong to the connectivity kernel, i.e. are
            considered as neighbors of the central element. Elements up to a squared distance
            of raml from the center are considered neighbors. Rank may range from 1 (no adjacent
            elements are neighbours) to radius (all elements in (2 * radius + 1, 2 * radius + 1,
            2 * radius + 1) square are neighbours).

    Attributes:
        size : Number of elements in the connectivity kernel.
        x : x indices of the connectivity kernel.
        y : y indices of the connectivity kernel.
        z : z indices of the connectivity kernel.
    """
    radius : int
    rank : int
    x : List[int]
    y : List[int]
    z : List[int]

    def __init__(self, radius: int, rank: int): ...

    def __iter__(self) -> Iterator[List[int]]: ...

    def __len__(self) -> int: ...

class Regions2D:
    x : List[int]
    y : List[int]

    def __init__(self): ...

    def __delitem__(self, idxs: int | slice): ...

    @overload
    def __getitem__(self, idxs: int) -> PointSet2D: ...

    @overload
    def __getitem__(self, idxs: slice) -> Regions2D: ...

    def __iter__(self) -> Iterator[PointSet2D]: ...

    def __len__(self) -> int: ...

    @overload
    def __setitem__(self, idxs: int, value: PointSet2D): ...

    @overload
    def __setitem__(self, idxs: slice, value: Regions2D): ...

    def append(self, value: PointSet2D): ...

    def extend(self, value: Regions2D): ...

class Regions3D:
    x : List[int]
    y : List[int]
    z : List[int]

    def __init__(self): ...

    def __delitem__(self, idxs: int | slice): ...

    @overload
    def __getitem__(self, idxs: int) -> PointSet3D: ...

    @overload
    def __getitem__(self, idxs: slice) -> Regions3D: ...

    def __iter__(self) -> Iterator[PointSet3D]: ...

    def __len__(self) -> int: ...

    @overload
    def __setitem__(self, idxs: int, value: PointSet3D): ...

    @overload
    def __setitem__(self, idxs: slice, value: Regions3D): ...

    def append(self, value: PointSet3D): ...

    def extend(self, value: Regions3D): ...

class RegionsList2D:
    def __init__(self): ...

    def __delitem__(self, index: int | slice): ...

    @overload
    def __getitem__(self, index: int) -> Regions2D: ...

    @overload
    def __getitem__(self, index: slice) -> 'RegionsList2D': ...

    @overload
    def __setitem__(self, index: int, value: Regions2D): ...

    @overload
    def __setitem__(self, index: slice, value: 'RegionsList2D'): ...

    def __iter__(self) -> Iterator[Regions2D]: ...

    def __len__(self) -> int: ...

    def append(self, elem: Regions2D) -> None: ...

    def extend(self, elem: 'RegionsList2D') -> None: ...

    def frames(self) -> NDIntArray: ...

    def index(self) -> NDIntArray: ...

    def x(self) -> NDIntArray: ...

    def y(self) -> NDIntArray: ...

class RegionsList3D:
    def __init__(self): ...

    def __delitem__(self, index: int | slice): ...

    @overload
    def __getitem__(self, index: int) -> Regions3D: ...

    @overload
    def __getitem__(self, index: slice) -> 'RegionsList3D': ...

    @overload
    def __setitem__(self, index: int, value: Regions3D): ...

    @overload
    def __setitem__(self, index: slice, value: 'RegionsList3D'): ...

    def __iter__(self) -> Iterator[Regions3D]: ...

    def __len__(self) -> int: ...

    def append(self, elem: Regions3D) -> None: ...

    def extend(self, elem: 'RegionsList3D') -> None: ...

    def frames(self) -> NDIntArray: ...

    def index(self) -> NDIntArray: ...

    def x(self) -> NDIntArray: ...

    def y(self) -> NDIntArray: ...

    def z(self) -> NDIntArray: ...

class Pixels2DFloat:
    x : List[int]
    y : List[int]
    value : List[float]

    def __init__(self, x: RealSequence = [], y: RealSequence = [],
                 value: RealSequence = []): ...

    def merge(self, source: Pixels2DFloat) -> Pixels2DFloat: ...

    def line(self) -> List[float]: ...

    def total_mass(self) -> float: ...

    def mean(self) -> List[float]: ...

    def center_of_mass(self) -> List[float]: ...

    def moment_of_inertia(self) -> List[float]: ...

    def covariance_matrix(self) -> List[float]: ...

class Pixels2DDouble:
    x : List[int]
    y : List[int]
    value : List[float]

    def __init__(self, x: RealSequence = [], y: RealSequence = [],
                 value: RealSequence = []): ...

    def merge(self, source: Pixels2DDouble) -> Pixels2DDouble: ...

    def line(self) -> List[float]: ...

    def total_mass(self) -> float: ...

    def mean(self) -> List[float]: ...

    def center_of_mass(self) -> List[float]: ...

    def moment_of_inertia(self) -> List[float]: ...

    def covariance_matrix(self) -> List[float]: ...

Structure = Structure2D | Structure3D
PointSet = PointSet2D | PointSet3D
Regions = Regions2D | Regions3D
RegionsList = RegionsList2D | RegionsList3D

def binary_dilation(input: BoolArray, structure: Structure,
                    seeds: Regions | PointSet | None=None, iterations: int=1,
                    mask: BoolArray | None=None, axes: List[int] | None=None,
                    num_threads: int=1) -> BoolArray:
    ...

@overload
def label(mask: Array, structure: Structure2D, seeds: Regions2D | PointSet2D | None=None, npts: int=1,
          axes: List[int] | None=None, num_threads: int=1) -> RegionsList2D:
    ...

@overload
def label(mask: Array, structure: Structure3D, seeds: Regions3D | PointSet3D | None=None, npts: int=1,
          axes: List[int] | None=None, num_threads: int=1) -> RegionsList3D:
    ...

def label(mask: Array, structure: Structure, seeds: Regions | PointSet | None=None,
          npts: int=1, axes: List[int] | None=None, num_threads: int=1) -> RegionsList:
    ...

def total_mass(regions: RegionsList, data: Array, axes: List[int] | None=None) -> NDRealArray:
    ...

def mean(regions: RegionsList, data: Array, axes: List[int] | None=None) -> NDRealArray:
    ...

def center_of_mass(regions: RegionsList, data: Array, axes: List[int] | None=None) -> NDRealArray:
    ...

def moment_of_inertia(regions: RegionsList, data: Array, axes: List[int] | None=None) -> NDRealArray:
    ...

def covariance_matrix(regions: RegionsList, data: Array, axes: List[int] | None=None) -> NDRealArray:
    ...
