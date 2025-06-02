from __future__ import annotations
from typing import Dict, Iterator, List, Optional, Tuple, overload
from ..annotations import IntSequence, NDBoolArray, NDIntArray, NDRealArray, RealSequence
from .label import Structure2D

class Peaks:
    """Peak finding algorithm. Finds sparse peaks in a two-dimensional image.

    Args:
        data : A rasterised 2D image.
        mask : Mask of bad pixels. mask is False if the pixel is bad. Bad pixels are
            skipped in the peak finding algorithm.
        radius : The minimal distance between peaks. At maximum one peak belongs
            to a single square in a radius x radius 2d grid.
        vmin : Peak is discarded if it's value is lower than ``vmin``.

    Attributes:
        size : Number of found peaks.
        x : x coordinates of peak locations.
        y : y coordinates of peak locations.
    """
    x : List[int]
    y : List[int]

    def __init__(self, x: IntSequence, y: IntSequence):
        ...

    def __iter__(self) -> Iterator[List[int]]:
        ...

    def __len__(self) -> int:
        ...

    def filter(self, data: NDRealArray, mask: NDBoolArray, structure: Structure2D,
               vmin: float, npts: int) -> Peaks:
        """Discard all the peaks the support structure of which is too small. The support
        structure is a connected set of pixels which value is above the threshold ``vmin``.
        A peak is discarded is the size of support set is lower than ``npts``.

        Args:
            data : A rasterised 2D image.
            mask : Mask of bad pixels. mask is False if the pixel is bad. Bad pixels are
                skipped in the peak finding algorithm.
            vmin : Threshold value.
            npts : Minimal size of support structure.

        Returns:
            A new filtered set of peaks.
        """
        ...

class StreakDouble:
    centers : List[List[int]]
    ends : List[List[float]]
    x : List[int]
    y : List[int]
    value : List[float]

    def __init__(self, x: int, y: int, structure: Structure2D, data: NDRealArray):
        ...

    def center(self) -> List[float]:
        ...

    def central_line(self) -> List[float]:
        ...

    def line(self) -> List[float]:
        ...

    def merge(self, source: StreakDouble) -> StreakDouble:
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

class StreakFloat:
    centers : List[List[int]]
    ends : List[List[float]]
    x : List[int]
    y : List[int]
    value : List[float]

    def __init__(self, x: int, y: int, structure: Structure2D, data: NDRealArray):
        ...

    def center(self) -> List[float]:
        ...

    def central_line(self) -> List[float]:
        ...

    def line(self) -> List[float]:
        ...

    def merge(self, source: StreakFloat) -> StreakFloat:
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

class StreakFinderResultDouble:
    mask : NDIntArray
    streaks : Dict[int, StreakDouble]

    def __init__(self, data: NDRealArray, mask: NDBoolArray):
        ...

    def probability(self, data: NDRealArray, vmin: float) -> float:
        ...

    def p_value(self, index: int, xtol: float, vmin: float, probability: float) -> float:
        ...

    def to_lines(self, width: Optional[RealSequence]=None) -> NDRealArray:
        ...

class StreakFinderResultFloat:
    mask : NDIntArray
    streaks : Dict[int, StreakFloat]

    def __init__(self, data: NDRealArray, mask: NDBoolArray):
        ...

    def probability(self, data: NDRealArray, vmin: float) -> float:
        ...

    def p_value(self, index: int, xtol: float, vmin: float, probability: float) -> float:
        ...

    def to_lines(self, width: Optional[RealSequence]) -> NDRealArray:
        ...

StreakFinderResult = StreakFinderResultDouble | StreakFinderResultFloat

class StreakFinder:
    structure   : Structure2D
    min_size    : int
    lookahead   : int
    nfa         : int

    def __init__(self, structure: Structure2D, min_size: int, lookahead: int=0, nfa: int=0):
        ...

    def detect_streaks(self, data: NDRealArray, mask: NDBoolArray, peaks: Peaks,
                       xtol: float, vmin: float) -> StreakFinderResult:
        ...

def detect_peaks(data: NDRealArray, mask: NDBoolArray, radius: int, vmin: float,
                 axes: Optional[Tuple[int, int]]=None, num_threads: int=1) -> List[Peaks]:
    ...

@overload
def filter_peaks(peaks: Peaks, data: NDRealArray, mask: NDBoolArray,
                 structure: Structure2D, vmin: float, npts: int,
                 axes: Optional[Tuple[int, int]]=None, num_threads: int=1) -> Peaks: ...

@overload
def filter_peaks(peaks: List[Peaks], data: NDRealArray, mask: NDBoolArray,
                 structure: Structure2D, vmin: float, npts: int,
                 axes: Optional[Tuple[int, int]]=None, num_threads: int=1) -> List[Peaks]: ...

def filter_peaks(peaks: Peaks | List[Peaks], data: NDRealArray, mask: NDBoolArray,
                 structure: Structure2D, vmin: float, npts: int,
                 axes: Optional[Tuple[int, int]]=None, num_threads: int=1) -> Peaks | List[Peaks]:
    ...

@overload
def detect_steraks(peaks: Peaks, data: NDRealArray, mask: NDBoolArray,
                   structure: Structure2D, xtol: float, vmin: float, min_size: int,
                   lookahead: int=0, nfa: int=0, axes: Optional[Tuple[int, int]]=None,
                   num_threads: int=1) -> StreakFinderResult: ...

@overload
def detect_steraks(peaks: List[Peaks], data: NDRealArray, mask: NDBoolArray,
                   structure: Structure2D, xtol: float, vmin: float, min_size: int,
                   lookahead: int=0, nfa: int=0, axes: Optional[Tuple[int, int]]=None,
                   num_threads: int=1) -> List[StreakFinderResult]: ...

def detect_streaks(peaks: Peaks | List[Peaks], data: NDRealArray, mask: NDBoolArray,
                   structure: Structure2D, xtol: float, vmin: float, min_size: int,
                   lookahead: int=0, nfa: int=0, axes: Optional[Tuple[int, int]]=None,
                   num_threads: int=1) -> StreakFinderResult | List[StreakFinderResult]:
    """Streak finding algorithm. Starting from the set of seed peaks, the lines are iteratively
    extended with a connectivity structure.

    Args:
        peaks : A set of peaks used as seed locations for the streak growing algorithm.
        data : A 2D rasterised image.
        mask : Mask of bad pixels. mask is False if the pixel is bad. Bad pixels are skipped in the
            streak detection algorithm.
        structure : A connectivity structure.
        xtol : Distance threshold. A new linelet is added to a streak if it's distance to the
            streak is no more than ``xtol``.
        vmin : Value threshold. A new linelet is added to a streak if it's value at the center of
            mass is above ``vmin``.
        log_eps : Detection threshold. A streak is added to the final list if it's p-value under
            null hypothesis is below ``np.exp(log_eps)``.
        lookahead : Number of linelets considered at the ends of a streak to be added to the streak.
        nfa : Number of false alarms, allowed number of unaligned points in a streak.

    Returns:
        A list of detected streaks.
    """
    ...
