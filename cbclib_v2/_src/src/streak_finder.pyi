from typing import Iterable, Iterator, List, Tuple, overload
from ..annotations import IntArray, NDBoolArray, NDIntArray, NDRealArray, RealArray
from .label import Structure

class Streak:
    indices : List[int]

    def line(self, labels: NDIntArray, linelets: RealArray) -> List[float]: ...

class Streaks:
    """List of detected streaks returned by :func:`detect_streaks`.

    Behaves as a mutable sequence of :class:`Streak` objects and supports
    integer indexing, slicing, boolean-mask indexing, ``append``, and
    ``extend``.  Use :meth:`to_lines` to convert the collection to a
    line-endpoint array suitable for further processing.
    """
    @overload
    def __init__(self): ...
    @overload
    def __init__(self, elements: Iterable[Streak]): ...

    def __delitem__(self, index: int | slice): ...

    @overload
    def __getitem__(self, index: int) -> Streak: ...
    @overload
    def __getitem__(self, index: slice | NDIntArray | NDBoolArray) -> 'Streaks': ...

    @overload
    def __setitem__(self, index: int, value: Streak): ...
    @overload
    def __setitem__(self, index: slice, value: 'Streaks'): ...

    def __iter__(self) -> Iterator[Streak]: ...

    def __len__(self) -> int: ...

    def append(self, elem: Streak) -> None: ...

    def extend(self, elem: 'Streaks') -> None: ...

    def to_lines(self, labels: NDIntArray, lines: RealArray) -> NDRealArray:
        """Return the endpoint coordinates of each streak as a 2-D array.

        Args:
            labels: Flat bin-label array returned by :func:`peak_labels`.
            lines: Linelet endpoint array returned by :func:`fit_linelets`.

        Returns:
            Array of shape ``(N, 4)`` with ``(x0, y0, x1, y1)`` endpoints for
            each streak, where the endpoints span the outermost linelets across
            all bins in the streak.
        """
        ...

LabelsTuple = Tuple[IntArray, int, int, int, int]

def detect_peaks(labels: IntArray, data: RealArray, structure: Structure, radius: int, vmin: float,
                 num_threads: int=1) -> NDIntArray: ...

def line_fit(labels: LabelsTuple, peaks: IntArray, data: RealArray, structure: Structure,
             vmin: float, num_threads: int=1) -> Tuple[NDRealArray, LabelsTuple]: ...

def detect_streaks(labels: LabelsTuple, peaks: IntArray, linelets: RealArray, data: RealArray,
                   structure: Structure, vmin: float, xtol: float, nfa: int=0, num_threads: int=1
                   ) -> Streaks: ...

def p_values(streaks: Streaks, labels: LabelsTuple, peaks: IntArray, data: RealArray,
             structure: Structure, p0: float, vmin: float, xtol: float, num_threads: int=1
             ) -> NDRealArray: ...

def n_signal(streaks: Streaks, labels: LabelsTuple, peaks: IntArray, data: RealArray,
             structure: Structure, vmin: float, num_threads: int=1) -> NDIntArray: ...

def streak_labels(out: NDIntArray, streaks: Streaks, indices: IntArray, labels: LabelsTuple,
                  peaks: IntArray, structure: Structure, num_threads: int=1) -> NDIntArray: ...
