from .annotations import NDRealArray
from .functions import Structure, detect_peaks, detect_streaks, filter_peaks
from .src.streak_finder import PatternList, PeaksList

class PatternStreakFinder:
    def __init__(self, data: NDRealArray, structure: Structure, min_size: float, lookahead: int=0,
                 nfa: int=0):
        self.data, self.structure = data, structure
        self.min_size, self.lookahead, self.nfa = min_size, lookahead, nfa

    def detect_peaks(self, vmin: float, npts: int, connectivity: Structure=Structure([1, 1], 1),
                     ) -> PeaksList:
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
        peaks = detect_peaks(self.data, self.structure.connectivity, vmin)
        return filter_peaks(peaks, self.data, connectivity, vmin, npts)

    def detect_streaks(self, peaks: PeaksList, xtol: float, vmin: float) -> PatternList:
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
        return detect_streaks(peaks, self.data, self.structure, xtol, vmin, self.min_size,
                              self.lookahead, self.nfa)
