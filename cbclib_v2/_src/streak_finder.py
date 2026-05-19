from typing import Tuple
from math import log
from .annotations import IntArray, RealArray
from .array_api import array_namespace
from .functions import (LabelResult, PeakLabels, Streaks, Structure, detect_peaks,
                        detect_streaks, fit_linelets, label, line_fit, n_signal, p_values,
                        peak_labels, streak_labels, to_lines)

class PatternStreakFinder:
    def __init__(self, data: RealArray, structure: Structure, vmin: float):
        self.data, self.structure, self.vmin = data, structure, vmin
        self._p0 = None

    @property
    def p0(self) -> float:
        if self._p0 is None:
            xp = array_namespace(self.data)
            self._p0 = float(xp.sum(self.data >= self.vmin) / self.data.size)
        return self._p0

    def detect_regions(self, npts: int, connectivity: Structure | None=None) -> LabelResult:
        if connectivity is None:
            connectivity = Structure([0] * (self.data.ndim - 2) + [1, 1], 1)
        return label(self.data >= self.vmin, structure=connectivity, npts=npts)

    def detect_peaks(self, regions: LabelResult) -> Tuple[PeakLabels, IntArray]:
        peaks = detect_peaks(self.data, regions, self.structure.connectivity, self.vmin)
        return peak_labels(peaks, self.data, self.structure.connectivity)

    def fit_linelets(self, labels: PeakLabels, peaks: IntArray) -> Tuple[RealArray, PeakLabels]:
        return fit_linelets(labels, peaks, self.data, self.structure, self.vmin)

    def detect_streaks(self, labels: PeakLabels, peaks: IntArray, linelets: RealArray,
                       xtol: float, nfa: int = 0) -> Streaks:
        return detect_streaks(labels, peaks, linelets, self.data, self.structure,
                                  self.vmin, xtol, nfa)

    def n_signal(self, streaks: Streaks, labels: PeakLabels, peaks: IntArray) -> IntArray:
        return n_signal(streaks, labels, peaks, self.data, self.structure, self.vmin)

    def ranking(self, streaks: Streaks, labels: PeakLabels, peaks: IntArray) -> IntArray:
        counts = self.n_signal(streaks, labels, peaks)
        xp = array_namespace(counts)
        indices = xp.arange(counts.size)
        order = xp.lexsort(xp.stack((indices, -counts)))
        ranks = xp.empty(order.shape, dtype=peaks.dtype)
        ranks[order] = indices
        return ranks

    def streak_labels(self, streaks: Streaks, ranks: IntArray, labels: PeakLabels, peaks: IntArray
                      ) -> LabelResult:
        xp = array_namespace(ranks)
        out = xp.zeros(self.data.shape, dtype=peaks.dtype)
        labeled = streak_labels(out, streaks, ranks, labels, peaks, self.structure)
        radii = [0,] * (self.data.ndim - 2) + [1, 1]
        return label(labeled, Structure(radii, 1))

    def line_fit(self, labeled: LabelResult) -> RealArray:
        return line_fit(labeled, self.data)

    def min_support(self, labeled: LabelResult, lines: RealArray, xtol: float) -> RealArray:
        return p_values(labeled, lines, self.data, self.p0, self.vmin, xtol) / log(self.p0)
