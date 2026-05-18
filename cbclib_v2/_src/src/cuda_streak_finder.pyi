from typing import Iterator, List
from ..annotations import CPIntArray, CPRealArray, IntArray, RealArray
from .label import Structure

class PeakLabels:
    labels : CPIntArray
    n_seeds : int
    n_labels : int
    n_good : int
    radius : int

    def __init__(self, labels: CPIntArray, n_seeds: int, n_labels: int, n_good: int, radius: int
                 ): ...

    def keep_best(self, quantile: float = 0.5) -> 'PeakLabels': ...

class Streak:
    indices : List[int]

    def line(self, labels: PeakLabels, linelets: RealArray) -> List[float]: ...

class Streaks:
    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Streak]: ...

    def __getitem__(self, index: int) -> 'Streak': ...

    def to_lines(self, out: CPRealArray, labels: PeakLabels, linelets: RealArray
                 ) -> CPRealArray: ...

def detect_peaks(peaks: CPIntArray, labels: CPIntArray, data: RealArray, structure: Structure,
                 radius: int, vmin: float) -> CPIntArray: ...

def line_fit(out: CPRealArray, labels: PeakLabels, peaks: IntArray, data: RealArray,
             structure: Structure, vmin: float) -> CPRealArray: ...

def detect_streaks(labels: PeakLabels, peaks: IntArray, linelets: RealArray, data: RealArray,
                   structure: Structure, vmin: float, xtol: float, nfa: float, keep_last: int=11
                   ) -> Streaks: ...

def n_signal(out: CPIntArray, streaks: Streaks, labels: PeakLabels, peaks: IntArray,
             data: RealArray, structure: Structure, vmin: float) -> CPIntArray: ...

def streak_labels(out: CPIntArray, streaks: Streaks, ranks: IntArray, labels: PeakLabels,
                  parray: IntArray, structure: Structure) -> CPIntArray: ...
