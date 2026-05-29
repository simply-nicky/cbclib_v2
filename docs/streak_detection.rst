Streak detection
================

Convergent beam diffraction patterns contain *streaks* — elongated,
high-SNR features whose orientation encodes the crystal unit-cell geometry.
Detecting and fitting these streaks accurately is a prerequisite for indexing.

:class:`~cbclib_v2.streak_finder.PatternStreakFinder` implements a seed-and-grow
algorithm that detects the central spine of each streak rather than its edges.
Unlike generic computer-vision line detectors, it is tuned specifically for CBC
data and is designed to work reliably on **noisy, low-exposure** diffraction
patterns where per-pixel SNR is modest but streak length provides the
discriminating signal.

.. _streak-algorithm:

How the algorithm works
-----------------------

The detector frame is partitioned into a regular grid of bins whose side length
equals ``structure.connectivity`` (the *radius* **r**). The algorithm then runs
six stages:

.. code-block:: text

        SNR frames
            │
            ▼
   ┌──────────────────┐
   │  detect_regions  │  label connected foreground blobs (SNR ≥ vmin)
   └────────┬─────────┘
            │  LabelResult
            ▼
   ┌──────────────────┐
   │  detect_peaks    │  one local maximum per r x r bin
   └────────┬─────────┘
            │  PeakLabels, peaks
            ▼
   ┌──────────────────┐
   │  fit_linelets    │  local line fit per bin; propagate to neighbours
   └────────┬─────────┘
            │  linelets, PeakLabels
            ▼
   ┌──────────────────┐
   │  detect_streaks  │  grow streaks from seeds; self-consistency check
   └────────┬─────────┘
            │  Streaks
            ▼
   ┌───────────────────────────────────────┐
   │  ranking → streak_labels → line_fit   │  rank, paint, refit
   └────────┬──────────────────────────────┘
            │  lines
            ▼
   ┌──────────────────┐
   │  min_support     │  discard spurious lines
   └──────────────────┘

Stage 1 — region detection
^^^^^^^^^^^^^^^^^^^^^^^^^^

All pixels above *vmin* are treated as foreground; connected foreground pixels
are grouped into labeled blobs. Blobs with fewer than *npts* pixels are dropped.
The result is a binary mask over grid bins: a bin is *active* if at least one of
its pixels belongs to a blob.

Stage 2 — peak detection
^^^^^^^^^^^^^^^^^^^^^^^^^

The frame is divided into a regular grid of *r* x *r* bins, where *r* is
``structure.connectivity``. Within each active bin the algorithm searches for
the brightest pixel that is also a **strict local maximum** — its value exceeds
every neighbour defined by *structure*. Three outcomes are possible:

* **Peak** — a local maximum was found; stored as its flat pixel index.
* **Good** — the bin overlaps a blob but contains no strict local maximum;
  it can still receive a linelet by propagation (Stage 3).
* **Bad** — no foreground pixels; ignored in all later stages.

Peaks are sorted by descending intensity so that the strongest seeds are
explored first.

Stage 3 — linelet fitting and propagation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each peak bin a *linelet* is fitted — a short line segment
:math:`(x_0, y_0, x_1, y_1)` representing the local orientation of the streak.
The fit uses intensity-weighted image moments: the linelet direction is the
eigenvector of the covariance matrix of pixel coordinates (weighted by SNR)
corresponding to the larger eigenvalue, and the half-length equals the FWHM of
the distribution along that eigenvector.

Starting from every peak, the algorithm **propagates** the linelet table forward
and backward. At each step it traces the current linelet direction to the nearest
bin boundary (by ray-boundary intersection), fits a linelet at the new location,
and writes it into the table. This fills *good* bins along the streak that did
not contain a local maximum.

Stage 4 — streak growing
^^^^^^^^^^^^^^^^^^^^^^^^^

For each *seed* peak a streak is initialised from that single bin and extended
one bin at a time in both directions. The current extent of a streak is
represented by the line connecting the two outermost linelet endpoints across
all bins in the streak. A candidate neighbouring bin is accepted when its linelet
satisfies a **self-consistency check**: both endpoints of the candidate linelet
must lie within *xtol* pixels of this total streak line. At most *nfa* endpoint
violations are tolerated across the whole streak. Growth stops when no aligned
neighbour can be found in either direction.

Stage 5 — ranking, painting, and line fitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each streak is scored by the number of above-threshold pixels in its *footprint*
(the union of *structure* neighbourhoods around its peak pixels, deduplicated).
Streaks are sorted in descending score order and painted onto an integer label
image so that stronger streaks overwrite weaker ones where they overlap. The
label image is re-labeled to produce clean connected regions, and a final
sub-pixel line is fitted to each region using image moments.

Stage 6 — minimal-support filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each fitted line is scored by a **minimal-support** statistic that quantifies
how much information the streak carries relative to the background pixel rate
:math:`p_0` (fraction of pixels above *vmin* across the whole stack).

For a streak with *k* above-threshold pixels in a footprint of *n* pixels
within *xtol* of the fitted line, the log tail probability under the null
hypothesis of a random uniform image is:

.. math::

   \log P(X \geq k), \quad X \sim \mathrm{Binomial}(n,\, p_0)

The minimal support is the footprint size :math:`m` of the smallest **fully
saturated** streak (all *m* pixels above *vmin*) whose detection probability
equals that of the observed streak. Since a fully saturated footprint of size
:math:`m` has :math:`\log P = m \log p_0`, setting the two expressions equal
gives:

.. math::

   m = \frac{\log P(X \geq k)}{\log p_0}

Larger :math:`m` means stronger statistical evidence. This criterion is
particularly effective for low-exposure patterns because it rewards streak
*length* over per-pixel intensity.

.. _streak-parameters:

Parameters
----------

``PatternStreakFinder(data, structure, vmin)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **data** — SNR frame stack of shape ``(n_frames, *frame_shape)``. Produce
  *data* from raw detector frames with :meth:`~cbclib_v2.CrystData.update_snr`.

* **structure** — a :class:`~cbclib_v2.label.Structure` with two roles:

  * ``structure.connectivity`` sets the **bin radius** *r*: the frame is
    divided into *r* x *r* bins. The radius should be chosen to match the
    expected streak width. A larger radius averages more pixels when fitting
    each linelet, which improves robustness to noise; however, if the radius
    is too large the algorithm loses the ability to resolve closely spaced
    streaks and to detect short streaks.
  * The full structure defines the **local neighbourhood** used for linelet
    fitting, the local-maximum test, and streak footprints.

* **vmin** — SNR threshold. Pixels below *vmin* are invisible to the
  algorithm. A value of 2-5 is typical; too low includes noise, too high
  misses faint streaks.

``detect_regions(npts, connectivity=None)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **npts** — minimum blob size in pixels. Blobs smaller than *npts* are
  removed before peak detection, eliminating isolated hot pixels and small
  noise clusters. A value of 5-30 is typical.

* **connectivity** — structuring element for blob labeling. Defaults to a
  3 x 3 square (all 8 neighbours).

``detect_streaks(labels, peaks, linelets, xtol, nfa=0)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **xtol** — collinearity tolerance in pixels. Every linelet endpoint in
  the streak must lie within *xtol* of the total streak line. *xtol* should
  be proportional to the bin radius *r*: a value of 0.5-0.75 x *r* works
  well in practice.

* **nfa** — maximum number of false alarms: linelet endpoints allowed to
  exceed *xtol* while still being accepted. ``nfa=0`` enforces strict
  collinearity; ``nfa=1`` or ``nfa=2`` adds robustness to locally bent
  or interrupted streaks.

* ``labels.keep_best(q)`` — pass ``labels.keep_best(q)`` instead of *labels*
  to restrict growing to the top fraction *q* of peaks by intensity. This
  suppresses spurious short streaks from weak local maxima.

``min_support(labeled, lines, xtol)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* The threshold applied to the returned score should be chosen based on the
  expected footprint size of the smallest streaks to detect. Streaks with a
  minimal support below the threshold are discarded. A threshold of **6-8**
  works well for typical CBC data; lower values recover faint streaks at the
  cost of more false positives.

.. _streak-example:

Example
-------

The example below generates a synthetic stack of two frames containing 40
randomly placed Gaussian streaks on a noisy background, runs the full detection
pipeline, and prints the number of detected lines.

``init_lines`` generates random line parameters ``(x0, y0, x1, y1, width)``
suitable for :func:`~cbclib_v2.ndimage.draw_lines`:

.. code-block:: python

   import cbclib_v2 as cbc
   from cbclib_v2.annotations import AnyGenerator, AnyNamespace, NumPy, RealArray, Shape
   from cbclib_v2.label import Structure
   from cbclib_v2.streak_finder import PatternStreakFinder

   def init_lines(
       rng: AnyGenerator,
       shape: Shape,
       n_lines: int,
       length: float,
       width: tuple[float, float],
       xp: AnyNamespace = NumPy,
   ) -> RealArray:
       """Return random line parameters (x0, y0, x1, y1, width)."""
       ndim = len(shape)
       lengths = length * rng.random((n_lines,))
       pt0 = xp.array(shape[:-ndim - 1:-1]) * rng.random((n_lines, ndim))
       vec = rng.standard_normal(size=(n_lines, ndim))
       pt1 = pt0 + vec * (lengths / xp.sqrt(xp.sum(vec**2, axis=-1)))[:, None]
       widths = width[0] + (width[1] - width[0]) * rng.random((n_lines, 1))
       return xp.concat((pt0, pt1, widths), axis=-1)

   xp = NumPy
   rng = cbc.default_rng(42, xp)
   cbc.set_cpu_config(1)

   n_frames = 2
   shape = (100, 80)   # frame height x width
   vmin = 0.25         # SNR threshold

   # Draw 40 Gaussian streaks distributed across 2 frames
   lines = init_lines(rng, shape, n_lines=40, length=20.0, width=(2.0, 3.0), xp=xp)
   idxs = rng.integers(0, n_frames, size=(40,))
   images = cbc.ndimage.draw_lines(
       xp.zeros((n_frames,) + shape), lines, idxs, kernel='gaussian'
   )
   # Add uniform background noise at half the detection threshold
   images += 0.5 * vmin * rng.random((n_frames,) + shape)

   # Structure([0, r, r], r): 2-D neighbourhood of radius r; bin size r x r
   structure = Structure([0, 2, 2], 2)

   finder = PatternStreakFinder(images, structure, vmin)

   # Stage 1 — blob detection; discard blobs smaller than 5 pixels
   regions = finder.detect_regions(npts=5)

   # Stage 2 — one local maximum per 2 x 2 bin
   labels, peaks = finder.detect_peaks(regions)

   # Stage 3 — linelet fitting and propagation
   linelets, labels = finder.fit_linelets(labels, peaks)

   # Stage 4 — grow streaks
   # xtol = 0.625 x r (r = 2)
   streaks = finder.detect_streaks(
       labels, peaks, linelets, xtol=1.25, nfa=0
   )

   # Stage 5 — rank, paint, and refit
   ranks = finder.ranking(streaks, labels, peaks)
   labeled = finder.streak_labels(streaks, ranks, labels, peaks)
   lines_fit = finder.line_fit(labeled)

   # Stage 6 — keep lines with minimal support ≥ 1
   support = finder.min_support(labeled, lines_fit, xtol=1.25)
   lines_fit = lines_fit[support > 1.0]

   print(f"Detected {lines_fit.shape[0]} streaks across {n_frames} frames")


.. seealso::

   :class:`~cbclib_v2.streak_finder.PatternStreakFinder`
      Full API reference for the streak finder.

   :doc:`background_subtraction`
      How to produce the SNR frames that serve as input to streak detection.

   :class:`~cbclib_v2.StreakDetector`
      High-level wrapper that drives :class:`~cbclib_v2.streak_finder.PatternStreakFinder`
      from a :class:`~cbclib_v2.CrystData` container.
