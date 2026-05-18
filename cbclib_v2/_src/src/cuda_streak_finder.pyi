from typing import Iterator, List, Tuple

from cbclib_v2._src.src.streak_finder import LabelsTuple
from ..annotations import CPIntArray, CPRealArray, IntArray, RealArray
from .label import Structure

class Streak:
    indices : List[int]

    def line(self, labels: CPIntArray, linelets: RealArray) -> List[float]: ...

class Streaks:
    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Streak]: ...

    def __getitem__(self, index: int) -> 'Streak': ...

    def to_lines(self, out: CPRealArray, labels: CPIntArray, linelets: RealArray
                 ) -> CPRealArray: ...

LabelsTuple = Tuple[IntArray, int, int, int, int]

def detect_peaks(peaks: CPIntArray, labels: CPIntArray, data: RealArray, structure: Structure,
                 radius: int, vmin: float) -> CPIntArray: ...

def line_fit(out: CPRealArray, labels: LabelsTuple, peaks: IntArray, data: RealArray,
             structure: Structure, vmin: float) -> Tuple[CPRealArray, LabelsTuple]: ...

def detect_streaks(labels: LabelsTuple, peaks: IntArray, linelets: RealArray, data: RealArray,
                   structure: Structure, vmin: float, xtol: float, nfa: float, keep_last: int=11
                   ) -> Streaks: ...

def n_signal(out: CPIntArray, streaks: Streaks, labels: LabelsTuple, peaks: IntArray,
             data: RealArray, structure: Structure, vmin: float) -> CPIntArray: ...

def streak_labels(out: CPIntArray, streaks: Streaks, ranks: IntArray, labels: LabelsTuple,
                  parray: IntArray, structure: Structure) -> CPIntArray: ...
