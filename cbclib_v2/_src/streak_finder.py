from typing import Tuple
from math import log
from .annotations import IntArray, RealArray
from .array_api import array_namespace
from .functions import (LabelResult, PeakLabels, Streaks, Structure, detect_peaks,
                        detect_streaks, fit_linelets, label, line_fit, n_signal, p_values,
                        peak_labels, streak_labels, to_lines)

class PatternStreakFinder:
    """Detect diffraction streaks in CBC SNR frames.

    Implements a seed-and-grow algorithm that detects the central spine of
    line features in monochrome intensity frames.  Unlike generic computer
    vision line detectors, it is tuned for the narrow, high-contrast streaks
    produced by convergent beam diffraction and designed to work reliably on
    noisy, low-exposure diffraction patterns.

    The frame is first partitioned into a grid of bins whose side length
    equals ``structure.connectivity`` (the *radius*).  The pipeline then
    proceeds in six stages:

    1. **Region detection** (:meth:`detect_regions`) — label every connected
       group of foreground pixels (SNR ≥ *vmin*) to mask out background bins.

    2. **Peak detection** (:meth:`detect_peaks`) — within each grid bin, find
       the brightest local maximum (a pixel whose value exceeds all its
       neighbours defined by *structure*).  A bin is marked *good* if it
       overlaps a labeled region (even without a strict local maximum), and
       *bad* if it has no signal at all.

    3. **Linelet fitting** (:meth:`fit_linelets`) — for each peak bin, fit a
       short line segment (*linelet*) using intensity-weighted image moments of
       the local neighbourhood.  Starting from every peak, propagate forward
       and backward along the local linelet orientation — stepping one bin at
       a time by intersecting the ray with the next bin boundary — and fit
       linelets to all reachable bins.  The result is a table of one linelet
       per reachable bin.

    4. **Streak growing** (:meth:`detect_streaks`) — for each seed peak,
       initialise a streak containing just that bin, then extend it one bin
       at a time in both directions.  A candidate bin is accepted if every
       linelet endpoint in the extended streak lies within *xtol* pixels of
       the current total streak line (self-consistency check), with at most
       *nfa* outlier endpoints tolerated.

    5. **Ranking** (:meth:`ranking`, :meth:`streak_labels`) — score each
       streak by counting above-threshold pixels in its footprint
       (:meth:`n_signal`).  Paint ranked streaks onto a label image;
       higher-ranked (stronger) streaks overwrite weaker ones where they
       overlap.  Re-label the result to obtain clean connected regions.

    6. **Line fit and statistical filtering** (:meth:`line_fit`,
       :meth:`min_support`) — fit a final sub-pixel line to each labeled
       region using image moments, then compute a minimal-support score
       (:meth:`min_support`) to discard spurious detections.

    Attributes:
        data: SNR frame stack, shape ``(n_frames, *frame_shape)``.
        structure: Structuring element controlling neighbourhood size and
            grid bin width (``structure.connectivity`` is the bin radius).
        vmin: SNR threshold separating foreground from background pixels.

    Example:

        Full pipeline on a stack of SNR frames:

        >>> structure = Structure([0, 0, 3, 3], 4)
        >>> finder = PatternStreakFinder(data, structure, vmin=3.0)
        >>> regions = finder.detect_regions(20)
        >>> labels, peaks = finder.detect_peaks(regions)
        >>> linelets, labels = finder.fit_linelets(labels, peaks)
        >>> streaks = finder.detect_streaks(labels.keep_best(0.5), peaks, linelets, xtol=2.25, nfa=1)
        >>> ranks = finder.ranking(streaks, labels, peaks)
        >>> labeled = finder.streak_labels(streaks, ranks, labels, peaks)
        >>> lines = finder.line_fit(labeled)
        >>> support = finder.min_support(labeled, lines, xtol=2.25)
        >>> lines = lines[support > 6.8]
    """

    def __init__(self, data: RealArray, structure: Structure, vmin: float):
        self.data, self.structure, self.vmin = data, structure, vmin
        self._p0 = None

    @property
    def p0(self) -> float:
        """Fraction of pixels at or above *vmin* across the full frame stack.

        Used as the background pixel probability in the log-binomial test
        inside :meth:`min_support`.  Computed lazily on first access.
        """
        if self._p0 is None:
            xp = array_namespace(self.data)
            self._p0 = float(xp.sum(self.data >= self.vmin) / self.data.size)
        return self._p0

    def detect_regions(self, npts: int, connectivity: Structure | None=None) -> LabelResult:
        """Label connected foreground regions in the SNR frame stack.

        All pixels with ``data >= vmin`` are treated as foreground.
        Connected foreground pixels receive the same positive integer label;
        background pixels are labeled 0.  Regions with fewer than *npts*
        pixels are discarded.  This step provides the binary mask that
        tells the peak-detection step which grid bins contain any signal.

        Args:
            npts: Minimum region size in pixels.  Smaller regions are
                removed from the result.
            connectivity: Structuring element defining pixel connectivity for
                region labeling.  If ``None``, a 3x3 square neighbourhood is
                used (radius 1 in the last two spatial dimensions).

        Returns:
            Labeled foreground regions.
        """
        if connectivity is None:
            connectivity = Structure([0] * (self.data.ndim - 2) + [1, 1], 1)
        return label(self.data >= self.vmin, structure=connectivity, npts=npts)

    def detect_peaks(self, regions: LabelResult) -> Tuple[PeakLabels, IntArray]:
        """Find local maxima within each grid bin.

        The frame is divided into a grid of NxN bins where
        ``N = structure.connectivity``.  Within each bin that overlaps at
        least one labeled foreground region, the algorithm searches for the
        brightest pixel that is also a strict local maximum (its value
        exceeds every neighbour defined by *structure*).

        Each bin is assigned one of three states:

        - **Peak** — contains a local maximum above *vmin*; stored as its
          flat pixel index in :attr:`data`.
        - **Good** — overlaps a labeled region but has no strict local
          maximum; can still participate in linelet propagation.
        - **Bad** — no labeled foreground pixels; ignored in all later steps.

        Peaks are sorted by descending intensity so that the strongest seeds
        are tried first during streak growing.

        Args:
            regions: Labeled foreground regions returned by
                :meth:`detect_regions`.

        Returns:
            A tuple ``(labels, peaks)`` where *labels* is a
            :class:`~cbclib_v2.PeakLabels` describing the bin state of every
            grid bin, and *peaks* is a 1-D array of flat pixel indices of the
            detected peaks, sorted by descending intensity.
        """
        peaks = detect_peaks(self.data, regions, self.structure.connectivity, self.vmin)
        return peak_labels(peaks, self.data, self.structure.connectivity)

    def fit_linelets(self, labels: PeakLabels, peaks: IntArray) -> Tuple[RealArray, PeakLabels]:
        """Fit a linelet to each reachable grid bin.

        For every peak bin, a linelet is fitted to the intensity distribution
        in the local neighbourhood (pixels within *structure*) using
        intensity-weighted image moments.  The linelet is a short line segment
        ``(x0, y0, x1, y1)`` whose direction is the major axis of the
        covariance ellipse.

        Starting from each peak, the algorithm then propagates forward and
        backward: it steps one bin at a time by intersecting the current
        linelet direction with the nearest bin boundary, fits a linelet at
        the new location, and writes it into the linelet table.  Propagation
        continues until all reachable bins have been assigned a linelet.
        Bins without a strict local maximum but with sufficient signal (good
        bins) acquire a linelet this way and become usable for streak growing.

        Args:
            labels: Bin state descriptor returned by :meth:`detect_peaks`.
            peaks: Peak pixel indices returned by :meth:`detect_peaks`.

        Returns:
            A tuple ``(linelets, labels)`` where *linelets* is an array of
            shape ``(n_peaks, 4)`` containing the endpoint coordinates
            ``(x0, y0, x1, y1)`` of each linelet, and *labels* is an updated
            :class:`~cbclib_v2.PeakLabels` whose ``n_labels`` has grown to
            include newly filled bins.
        """
        return fit_linelets(labels, peaks, self.data, self.structure, self.vmin)

    def detect_streaks(self, labels: PeakLabels, peaks: IntArray, linelets: RealArray,
                       xtol: float, nfa: int = 0) -> Streaks:
        """Grow streaks from seed peaks by aggregating aligned linelet bins.

        For each seed peak (the first ``labels.n_seeds`` peaks in *labels*),
        a streak is initialised from that single bin and then extended one bin
        at a time in both directions by following the current streak line.

        At each extension step the algorithm:

        1. Projects the current streak line to locate the next bin in the
           forward or backward direction.
        2. Checks self-consistency: the candidate linelet endpoints must lie
           within *xtol* pixels of the total streak line.  At most *nfa*
           endpoints across the whole streak are allowed to exceed this
           tolerance.
        3. If the check passes, adds the bin to the streak and updates its
           span to cover the new extent.

        Growth stops when no aligned neighbour can be found in either
        direction.

        Use :meth:`~cbclib_v2.PeakLabels.keep_best` on *labels* to restrict
        seeds to the top-*q* fraction of peaks by intensity, which reduces
        the number of spurious short streaks and run time.

        Args:
            labels: Bin state descriptor, usually filtered with
                ``labels.keep_best(q)`` to limit seed peaks.
            peaks: Peak pixel indices returned by :meth:`detect_peaks`.
            linelets: Linelet endpoint array returned by
                :meth:`fit_linelets`.
            xtol: Distance tolerance in pixels.  A candidate bin is accepted
                only if all linelet endpoints of the updated streak lie within
                *xtol* of the total streak line.
            nfa: Maximum number of false alarms — linelet endpoints that may
                exceed *xtol* and still be accepted.  ``nfa=0`` enforces
                strict collinearity; ``nfa=1`` allows one outlier endpoint.

        Returns:
            List of detected :class:`~cbclib_v2.streak_finder.Streaks`.
        """
        return detect_streaks(labels, peaks, linelets, self.data, self.structure,
                                  self.vmin, xtol, nfa)

    def n_signal(self, streaks: Streaks, labels: PeakLabels, peaks: IntArray) -> IntArray:
        """Count above-threshold pixels in each streak's footprint.

        The footprint of a streak is the union of the *structure*
        neighbourhood around every peak pixel belonging to that streak.
        Pixels that appear in more than one peak's neighbourhood are counted
        only once.

        Args:
            streaks: Detected streaks returned by :meth:`detect_streaks`.
            labels: Bin state descriptor returned by :meth:`detect_peaks`.
            peaks: Peak pixel indices returned by :meth:`detect_peaks`.

        Returns:
            Integer array of length ``len(streaks)`` with the number of
            pixels ≥ *vmin* in each streak's footprint.
        """
        return n_signal(streaks, labels, peaks, self.data, self.structure, self.vmin)

    def ranking(self, streaks: Streaks, labels: PeakLabels, peaks: IntArray) -> IntArray:
        """Rank streaks by descending signal count.

        Streaks are scored by :meth:`n_signal` and ranked so that rank 0
        is the streak with the most above-threshold pixels.  Ties are broken
        by streak index (earlier seed wins).

        Args:
            streaks: Detected streaks returned by :meth:`detect_streaks`.
            labels: Bin state descriptor returned by :meth:`detect_peaks`.
            peaks: Peak pixel indices returned by :meth:`detect_peaks`.

        Returns:
            Integer array of length ``len(streaks)`` where entry *i* is the
            rank of streak *i* (0 = strongest).
        """
        counts = self.n_signal(streaks, labels, peaks)
        xp = array_namespace(counts)
        indices = xp.arange(counts.size)
        order = xp.lexsort(xp.stack((indices, -counts)))
        ranks = xp.empty(order.shape, dtype=peaks.dtype)
        ranks[order] = indices
        return ranks

    def streak_labels(self, streaks: Streaks, ranks: IntArray, labels: PeakLabels, peaks: IntArray
                      ) -> LabelResult:
        """Paint ranked streaks onto a label image and re-label.

        Each streak's footprint (the *structure* neighbourhood of its peak
        pixels) is written to an integer array.  Where two streaks overlap,
        the one with the lower rank (higher signal count) overwrites the
        weaker one.  The resulting image is re-labeled with
        :func:`~cbclib_v2.label` to produce clean connected regions suitable
        for line fitting.

        Args:
            streaks: Detected streaks returned by :meth:`detect_streaks`.
            ranks: Rank array returned by :meth:`ranking`.
            labels: Bin state descriptor returned by :meth:`detect_peaks`.
            peaks: Peak pixel indices returned by :meth:`detect_peaks`.

        Returns:
            Labeled regions, one per surviving streak, as a
            :class:`~cbclib_v2.LabelResult`.
        """
        xp = array_namespace(ranks)
        out = xp.zeros(self.data.shape, dtype=peaks.dtype)
        labeled = streak_labels(out, streaks, ranks, labels, peaks, self.structure)
        radii = [0,] * (self.data.ndim - 2) + [1, 1]
        return label(labeled, Structure(radii, 1))

    def line_fit(self, labeled: LabelResult) -> RealArray:
        """Fit a line to each labeled region using image moments.

        Delegates to :func:`~cbclib_v2.line_fit`.  See that function for the
        moment-based fitting details.

        Args:
            labeled: Labeled regions returned by :meth:`streak_labels`.

        Returns:
            Array of shape ``(N, 2 * ndim)`` with the endpoint coordinates
            of the fitted line segment for each of the *N* labeled regions.
        """
        return line_fit(labeled, self.data)

    def min_support(self, labeled: LabelResult, lines: RealArray, xtol: float) -> RealArray:
        """Compute a minimal-support score for each fitted line.

        The score quantifies how much information a detected streak carries
        relative to the background pixel rate :attr:`p0`.  For each labeled
        region:

        1. Build its *footprint*: the union of *structure* neighbourhoods
           around the region's peak pixels.
        2. Among footprint pixels within *xtol* of the fitted line, count
           *n* (total) and *k* (above *vmin*).
        3. Evaluate the log tail probability of the observation under the
           null hypothesis of a random uniform image:
           ``log P(X ≥ k)`` where ``X ~ Binomial(n, p0)``.
        4. Find the footprint size *m* of the smallest *fully saturated*
           streak (all *m* pixels above *vmin*) whose detection probability
           equals that of the observed streak.  A fully saturated footprint
           of *m* pixels has log probability ``m · log(p0)``, so setting it
           equal to step 3 gives ``m = log P(X ≥ k) / log(p0)``.

        The returned score *m* is the size of the minimal noiseless streak
        that would be as hard to detect by chance as the observed one.
        Larger *m* means stronger statistical evidence.  This criterion is
        particularly effective for noisy, low-exposure patterns where per-pixel
        SNR is low but streak length provides discriminating power.

        Args:
            labeled: Labeled regions returned by :meth:`streak_labels`.
            lines: Line endpoint array returned by :meth:`line_fit`.
            xtol: Distance tolerance in pixels, matching the value used in
                :meth:`detect_streaks`.

        Returns:
            Float array of length *N* (one score per labeled region).
            Filter with e.g. ``lines = lines[support > 6.8]`` to retain only
            well-supported streaks.
        """
        return p_values(labeled, lines, self.data, self.p0, self.vmin, xtol) / log(self.p0)
