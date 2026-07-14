Image processing
================

.. currentmodule:: cbclib_v2.ndimage

**cbclib_v2.ndimage** provides array operations on frame stacks: rasterising line
segments onto pixel grids and computing robust statistical reductions along
arbitrary axes.  All functions dispatch automatically to a CPU (NumPy) or GPU
(CuPy) backend depending on the array type.

Drawing
-------

Rasterise parameterised line segments onto an existing array.  A line is
specified as ``(x0, y0, x1, y1, width)``; the kernel argument controls the
radial profile used for antialiasing.

.. autosummary::
   :toctree: generated
   :nosignatures:

   draw_lines
   accumulate_lines

Statistical filtering
---------------------

Robust reductions that are resistant to outliers and structured artefacts
(hot pixels, zingers).

.. autosummary::
   :toctree: generated
   :nosignatures:

   median
   robust_mean
   robust_lsq

Labeling and morphology
-----------------------

.. currentmodule:: cbclib_v2.label

**cbclib_v2.label** provides connected-component labeling, structuring elements,
morphological operations, and intensity-weighted image moments.  It is the
building block for :class:`~cbclib_v2.streak_finder.PatternStreakFinder` and the
:class:`~cbclib_v2.RegionDetector` high-level pipeline step.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Structure
   LabelResult
   label
   binary_dilation
   center_of_mass
   covariance_matrix
   ellipse_fit
   line_fit
   maximum_position
   p_values

.. note::

   Intensity-weighted image moments are computed per labeled region.  All moment
   functions follow the formulation from `Image moment
   <https://en.wikipedia.org/wiki/Image_moment>`_ (Wikipedia).

Radial profiles
---------------

Compact radial background profiles used by online detection and hit finding.
See :doc:`online_detection` for the high-level workflow.

.. autosummary::
   :toctree: generated
   :nosignatures:

   RadialProfiles
   radial_profiles

Seed-and-grow streak detection
------------------------------

**cbclib_v2.streak_finder** provides a seed-and-grow streak detection pipeline that
is more selective for narrow, line-like features than connected-region detection.
:class:`~cbclib_v2.StreakDetector` uses the functions in this module to implement a
high-level streak detection pipeline.

.. currentmodule:: cbclib_v2.streak_finder

.. autosummary::
   :toctree: generated
   :nosignatures:

   PatternStreakFinder
   PeakLabels
   Streaks
   detect_peaks
   peak_labels
   fit_linelets
   detect_streaks
   to_lines
   n_signal
   streak_labels
