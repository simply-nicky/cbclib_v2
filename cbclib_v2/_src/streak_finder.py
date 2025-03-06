from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import List, Union
import numpy as np
from .data_container import DataContainer
from .annotations import Indices, NDBoolArray, NDRealArray, Shape
from .src.label import Structure
from .src.streak_finder import (detect_peaks, detect_streaks, filter_peaks, StreakFinder,
                                StreakFinderResultDouble, StreakFinderResultFloat, Peaks)
from .src.streak_finder import StreakDouble, StreakFloat

Streak = Union[StreakDouble, StreakFloat]
StreakFinderResult = Union[StreakFinderResultDouble, StreakFinderResultFloat]

class PatternStreakFinder:
    def __init__(self, data: NDRealArray, mask: NDBoolArray, structure: Structure,
                 min_size: int, lookahead: int=0, nfa: int=0):
        self.finder = StreakFinder(structure, min_size, lookahead, nfa)
        self.mask, self.data = mask, data

    @property
    def structure(self) -> Structure:
        return self.finder.structure

    def detect_peaks(self, vmin: float, npts: int,
                     connectivity: Structure=Structure(1, 1)) -> Peaks:
        peaks = detect_peaks(self.data, self.mask, self.finder.structure.rank, vmin)
        return filter_peaks(peaks, self.data, self.mask, connectivity, vmin, npts)[0]

    def detect_streaks(self, peaks: Peaks, xtol: float, vmin: float) -> StreakFinderResult:
        return self.finder.detect_streaks(self.data, self.mask, peaks, xtol, vmin)

@dataclass
class PatternsStreakFinder(DataContainer):
    data        : NDRealArray
    structure   : Structure
    mask        : NDBoolArray = field(default_factory=lambda: np.array([], dtype=bool))
    num_threads : int = field(default_factory=cpu_count)

    def __post_init__(self):
        if self.data.ndim < 2:
            raise ValueError(f"Invalid number of dimensions: {self.data.ndim} != 2")
        if self.data.shape[-2:] != self.mask.shape:
            self.mask = np.ones(self.data.shape[-2:], dtype=bool)

    def get_frames(self, idxs: Indices) -> 'PatternsStreakFinder':
        return self.replace(data=self.data[idxs])

    @property
    def shape(self) -> Shape:
        return self.data.shape

    def detect_peaks(self, vmin: float, npts: int,
                     connectivity: Structure=Structure(1, 1)) -> List[Peaks]:
        """Find peaks in a pattern. Returns a sparse set of peaks which values are above a threshold
        ``vmin`` that have a supporing set of a size larger than ``npts``. The minimal distance
        between peaks is ``2 * structure.radius``.

        Args:
            vmin : Peak threshold. All peaks with values lower than ``vmin`` are discarded.
            npts : Support size threshold. The support structure is a connected set of pixels which
                value is above the threshold ``vmin``. A peak is discarded is the size of support
                set is lower than ``npts``.
            connectivity : Connectivity structure used in finding a supporting set.

        Returns:
            Set of detected peaks.
        """
        peaks = detect_peaks(self.data, self.mask, self.structure.rank, vmin,
                             num_threads=self.num_threads)
        return filter_peaks(peaks, self.data, self.mask, connectivity, vmin, npts,
                            num_threads=self.num_threads)

    def detect_streaks(self, peaks: List[Peaks], xtol: float, vmin: float, min_size: int,
                       lookahead: int=0, nfa: int=0) -> List[NDRealArray]:
        """Streak finding algorithm. Starting from the set of seed peaks, the lines are iteratively
        extended with a connectivity structure.

        Args:
            peaks : A set of peaks used as seed locations for the streak growing algorithm.
            xtol : Distance threshold. A new linelet is added to a streak if it's distance to the
                streak is no more than ``xtol``.
            vmin : Value threshold. A new linelet is added to a streak if it's value at the center
                of mass is above ``vmin``.
            min_size : Minimum number of linelets required in a detected streak.
            lookahead : Number of linelets considered at the ends of a streak to be added to the
                streak.

        Returns:
            A list of detected streaks.
        """
        result = detect_streaks(peaks, self.data, self.mask, self.structure, xtol, vmin, min_size,
                                lookahead, nfa, num_threads=self.num_threads)
        return [np.array(list(pattern.streaks.values())) for pattern in result]
