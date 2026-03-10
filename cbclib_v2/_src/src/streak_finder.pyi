from typing import Iterable, Iterator, List, Sequence, Tuple, overload
from ..annotations import IntArray, NDIntArray, NDRealArray, RealArray, Shape
from .label import Region, Regions, Structure

@overload
def local_maxima(inp: RealArray, structure: Structure, num_threads: int=1) -> NDRealArray: ...

@overload
def local_maxima(inp: IntArray, structure: Structure, num_threads: int=1) -> NDIntArray: ...

def local_maxima(inp: RealArray | IntArray, structure: Structure, num_threads: int=1
                 ) -> NDRealArray | NDIntArray:
    """
    Find local maxima in a multidimensional array along a set of axes. This function returns
    the indices of the maxima.

    Args:
        x : The array to search for local maxima.
        structure : The structuring element used to determine the neighborhood of each element.

    Returns:
        A list of indices of the local maxima. Each index is a list of coordinates corresponding
        to the position of the maximum in the input array.
    """
    ...

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
    radius  : int
    shape   : Tuple[int, int]

    @overload
    def __init__(self, shape: Sequence[int], radius: int): ...
    @overload
    def __init__(self, indices: List[int], shape: Sequence[int], radius: int): ...
    @overload
    def __init__(self, indices: NDIntArray, shape: Sequence[int], radius: int): ...

    def __iter__(self) -> Iterator[int]: ...

    def __len__(self) -> int: ...

    def clear(self): ...

    def append(self, index: int) -> None: ...

    @overload
    def extend(self, indices: List[int]) -> None: ...
    @overload
    def extend(self, indices: NDIntArray) -> None: ...

    def find_range(self, index: int, range: int) -> int: ...

    def remove(self, index: int): ...

class PeaksList:
    @overload
    def __init__(self): ...
    @overload
    def __init__(self, elements: Iterable[Peaks]): ...

    def __delitem__(self, index: int | slice): ...

    @overload
    def __getitem__(self, index: int) -> Peaks: ...
    @overload
    def __getitem__(self, index: slice) -> 'PeaksList': ...

    @overload
    def __setitem__(self, index: int, value: Peaks): ...
    @overload
    def __setitem__(self, index: slice, value: 'PeaksList'): ...

    def __iter__(self) -> Iterator[Peaks]: ...

    def __len__(self) -> int: ...

    def append(self, elem: Peaks) -> None: ...

    def extend(self, elem: 'PeaksList') -> None: ...

    def index(self) -> NDIntArray: ...

    def to_array(self) -> NDIntArray: ...

class Streak:
    id      : int
    centers : List[List[int]]
    ends    : List[List[float]]
    region  : Region

    def __init__(self, seed: int, structure: Structure, data: RealArray): ...

    def center(self) -> List[float]: ...

    def central_line(self) -> List[float]: ...

    def line(self) -> List[float]: ...

class Pattern:
    @overload
    def __init__(self): ...
    @overload
    def __init__(self, elements: Iterable[Streak]): ...

    def __delitem__(self, index: int | slice): ...

    @overload
    def __getitem__(self, index: int) -> Streak: ...
    @overload
    def __getitem__(self, index: slice) -> 'Pattern': ...

    @overload
    def __setitem__(self, index: int, value: Streak): ...
    @overload
    def __setitem__(self, index: slice, value: 'Pattern'): ...

    def __iter__(self) -> Iterator[Streak]: ...

    def __len__(self) -> int: ...

    def append(self, elem: Streak) -> None: ...

    def extend(self, elem: 'Pattern') -> None: ...

    def to_lines(self, out: NDRealArray) -> NDRealArray: ...

    def to_regions(self) -> Regions: ...

class PatternList:
    @overload
    def __init__(self): ...
    @overload
    def __init__(self, elements: Iterable[Pattern]): ...

    def __delitem__(self, index: int | slice): ...

    @overload
    def __getitem__(self, index: int) -> Pattern: ...
    @overload
    def __getitem__(self, index: slice) -> 'PatternList': ...

    @overload
    def __setitem__(self, index: int, value: Pattern): ...
    @overload
    def __setitem__(self, index: slice, value: 'PatternList'): ...

    def __iter__(self) -> Iterator[Pattern]: ...

    def __len__(self) -> int: ...

    def index(self) -> NDIntArray: ...

    def total(self) -> int: ...

    def to_lines(self, out: NDRealArray) -> NDRealArray: ...

def detect_peaks(data: RealArray, structure: Structure, radius: int, vmin: float,
                 axes: Tuple[int, int] | None=None, num_threads: int=1) -> PeaksList: ...

def filter_peaks(peaks: PeaksList, data: RealArray, structure: Structure, vmin: float,
                 npts: int, axes: Tuple[int, int] | None=None, num_threads: int=1): ...

def p0_values(data: RealArray, vmin: float, axes: Tuple[int, int] | None=None, num_threads: int=1
              ) -> NDRealArray: ...

def detect_streaks(peaks: PeaksList, p0: RealArray, data: RealArray, structure: Structure, xtol: float,
                   vmin: float, min_size: float, lookahead: int=0, nfa: int=0,
                   axes: Tuple[int, int] | None=None, num_threads: int=1) -> PatternList:
    """Streak finding algorithm. Starting from the set of seed peaks, the lines are iteratively
    extended with a connectivity structure.

    Args:
        peaks : A set of peaks used as seed locations for the streak growing algorithm.
        p0 : Null hypothesis probabilities.
        data : A 2D rasterised image.
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

def p_value(streaks: Pattern, data: RealArray, p0: float, xtol: float, vmin: float) -> NDRealArray: ...
