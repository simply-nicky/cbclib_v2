from __future__ import annotations
from typing import Iterator, List, Optional, Union, overload
from ..annotations import CPPIntSequence, CPPRealSequence, NDRealArray, NDBoolArray

class PointsSet:
    x : List[int]
    y : List[int]

    def __init__(self, x: CPPIntSequence, y: CPPIntSequence):
        ...

class Structure:
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

    def __init__(self, radius: int, rank: int):
        ...

class Regions:
    x : List[int]
    y : List[int]

    def __init__(self, regions: List[PointsSet]=[]):
        ...

    def __delitem__(self, idxs: Union[int, slice]):
        ...

    @overload
    def __getitem__(self, idxs: int) -> PointsSet:
        ...

    @overload
    def __getitem__(self, idxs: slice) -> Regions:
        ...

    def __getitem__(self, idxs: Union[int, slice]) -> Union[PointsSet, Regions]:
        ...

    def __iter__(self) -> Iterator[PointsSet]:
        ...

    def __len__(self) -> int:
        ...

    @overload
    def __setitem__(self, idxs: int, value: PointsSet):
        ...

    @overload
    def __setitem__(self, idxs: slice, value: Regions):
        ...

    def __setitem__(self, idxs: Union[int, slice], value: Union[PointsSet, Regions]):
        ...

    def append(self, value: PointsSet):
        ...

class PixelsFloat:
    x : List[int]
    y : List[int]
    value : List[float]

    def __init__(self, x: CPPRealSequence = [], y: CPPRealSequence = [],
                 value: CPPRealSequence = []):
        ...

    def merge(self, source: PixelsFloat) -> PixelsFloat:
        ...

    def line(self) -> List[float]:
        ...

    def total_mass(self) -> float:
        ...

    def mean(self) -> List[float]:
        ...

    def center_of_mass(self) -> List[float]:
        ...

    def moment_of_inertia(self) -> List[float]:
        ...

    def covariance_matrix(self) -> List[float]:
        ...

class PixelsDouble:
    x : List[int]
    y : List[int]
    value : List[float]

    def __init__(self, x: CPPRealSequence = [], y: CPPRealSequence = [],
                 value: CPPRealSequence = []):
        ...

    def merge(self, source: PixelsDouble) -> PixelsDouble:
        ...

    def line(self) -> List[float]:
        ...

    def total_mass(self) -> float:
        ...

    def mean(self) -> List[float]:
        ...

    def center_of_mass(self) -> List[float]:
        ...

    def moment_of_inertia(self) -> List[float]:
        ...

    def covariance_matrix(self) -> List[float]:
        ...

def label(mask: NDBoolArray, structure: Structure, npts: int=1, seeds: Optional[List[PointsSet]]=None,
          axes: Optional[List[int]]=None, num_threads: int=1) -> List[Regions]:
    ...

@overload
def total_mass(regions: Regions, data: NDRealArray, axes: Optional[List[int]]=None) -> NDRealArray:
    ...

@overload
def total_mass(regions: List[Regions], data: NDRealArray, axes: Optional[List[int]]=None) -> List[NDRealArray]:
    ...

def total_mass(regions: Union[Regions, List[Regions]], data: NDRealArray,
               axes: Optional[List[int]]=None) -> Union[NDRealArray, List[NDRealArray]]:
    ...

@overload
def mean(regions: Regions, data: NDRealArray, axes: Optional[List[int]]=None) -> NDRealArray:
    ...

@overload
def mean(regions: List[Regions], data: NDRealArray, axes: Optional[List[int]]=None) -> List[NDRealArray]:
    ...

def mean(regions: Union[Regions, List[Regions]], data: NDRealArray
         ) -> Union[NDRealArray, List[NDRealArray]]:
    ...

@overload
def center_of_mass(regions: Regions, data: NDRealArray) -> NDRealArray:
    ...

@overload
def center_of_mass(regions: List[Regions], data: NDRealArray) -> List[NDRealArray]:
    ...

def center_of_mass(regions: Union[Regions, List[Regions]], data: NDRealArray,
                   axes: Optional[List[int]]=None) -> Union[NDRealArray, List[NDRealArray]]:
    ...

@overload
def moment_of_inertia(regions: Regions, data: NDRealArray, axes: Optional[List[int]]=None
                      ) -> NDRealArray:
    ...

@overload
def moment_of_inertia(regions: List[Regions], data: NDRealArray, axes: Optional[List[int]]=None
                      ) -> List[NDRealArray]:
    ...

def moment_of_inertia(regions: Union[Regions, List[Regions]], data: NDRealArray,
                      axes: Optional[List[int]]=None) -> Union[NDRealArray, List[NDRealArray]]:
    ...

@overload
def covariance_matrix(regions: Regions, data: NDRealArray, axes: Optional[List[int]]=None
                      ) -> NDRealArray:
    ...

@overload
def covariance_matrix(regions: List[Regions], data: NDRealArray, axes: Optional[List[int]]=None
                      ) -> List[NDRealArray]:
    ...

def covariance_matrix(regions: Union[Regions, List[Regions]], data: NDRealArray,
                      axes: Optional[List[int]]=None) -> Union[NDRealArray, List[NDRealArray]]:
    ...
