from __future__ import annotations
from typing import Iterator, List, Optional, Tuple, Union, overload
from ..annotations import CPPIntSequence, NDRealArray, NDBoolArray

class PointsSet:
    x : List[int]
    y : List[int]
    size : int

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
    size : int
    x : List[int]
    y : List[int]

    def __init__(self, radius: int, rank: int):
        ...

class Regions:
    x: List[int]
    y: List[int]
    shape : List[int]

    def __init__(self, shape: CPPIntSequence, regions: List[PointsSet]=[]):
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

    def filter(self, structure: Structure, npts: int) -> Regions:
        ...

    def mask(self) -> NDBoolArray:
        ...

def label(mask: NDBoolArray, structure: Structure, npts: int=1,
          axes: Optional[Tuple[int, int]]=None, num_threads: int=1) -> List[Regions]:
    ...

@overload
def center_of_mass(regions: Regions, data: NDRealArray) -> NDRealArray:
    ...

@overload
def center_of_mass(regions: List[Regions], data: NDRealArray) -> List[NDRealArray]:
    ...

def center_of_mass(regions: Union[Regions, List[Regions]], data: NDRealArray
                   ) -> Tuple[NDRealArray, List[NDRealArray]]:
    ...

@overload
def central_moments(regions: Regions, data: NDRealArray) -> NDRealArray:
    ...

@overload
def central_moments(regions: List[Regions], data: NDRealArray) -> List[NDRealArray]:
    ...

def central_moments(regions: Union[Regions, List[Regions]], data: NDRealArray
                    ) -> Tuple[NDRealArray, List[NDRealArray]]:
    ...

@overload
def gauss_fit(regions: Regions, data: NDRealArray) -> NDRealArray:
    ...

@overload
def gauss_fit(regions: List[Regions], data: NDRealArray) -> List[NDRealArray]:
    ...

def gauss_fit(regions: Union[Regions, List[Regions]], data: NDRealArray
              ) -> Tuple[NDRealArray, List[NDRealArray]]:
    ...

@overload
def ellipse_fit(regions: Regions, data: NDRealArray) -> NDRealArray:
    ...

@overload
def ellipse_fit(regions: List[Regions], data: NDRealArray) -> List[NDRealArray]:
    ...

def ellipse_fit(regions: Union[Regions, List[Regions]], data: NDRealArray
                ) -> Tuple[NDRealArray, List[NDRealArray]]:
    ...

@overload
def line_fit(regions: Regions, data: NDRealArray) -> NDRealArray:
    ...

@overload
def line_fit(regions: List[Regions], data: NDRealArray) -> List[NDRealArray]:
    ...

def line_fit(regions: Union[Regions, List[Regions]], data: NDRealArray
             ) -> Tuple[NDRealArray, List[NDRealArray]]:
    ...

@overload
def moments(regions: Regions, data: NDRealArray) -> NDRealArray:
    ...

@overload
def moments(regions: List[Regions], data: NDRealArray) -> List[NDRealArray]:
    ...

def moments(regions: Union[Regions, List[Regions]], data: NDRealArray
            ) -> Tuple[NDRealArray, List[NDRealArray]]:
    ...
