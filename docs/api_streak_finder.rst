Streak detection
================

.. currentmodule:: cbclib_v2.streak_finder

This page collects the APIs used to detect streak-like features in SNR frames.
cbclib_v2 provides two routes described in :doc:`streak_detection`:

* connected-region detection, which labels all thresholded foreground and fits
  one line per region;
* the seed-and-grow streak finder, which is more selective for narrow,
  line-like features.

See :doc:`streak_detection` for a full narrative description of both workflows
and worked examples.

Connected-region detection
--------------------------

High-level interface
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: cbclib_v2

.. autosummary::
   :toctree: generated
   :nosignatures:

   RegionDetector

Low-level pipeline functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These functions are called internally by :class:`RegionDetector` but can be
used directly when finer control is needed.

.. currentmodule:: cbclib_v2.label

.. autosummary::
   :toctree: generated
   :nosignatures:

   label
   center_of_mass
   covariance_matrix
   ellipse_fit
   line_fit

Seed-and-grow streak detection
------------------------------

High-level interface
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: cbclib_v2

.. autosummary::
   :toctree: generated
   :nosignatures:

   StreakDetector
   ~streak_finder.PatternStreakFinder

Data types
^^^^^^^^^^

.. currentmodule:: cbclib_v2.streak_finder

.. autosummary::
   :toctree: generated
   :nosignatures:

   PeakLabels
   Streaks

Low-level pipeline functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These functions correspond to the individual stages of the streak detection
pipeline. They are called internally by :class:`PatternStreakFinder` but can
be used directly when finer control is needed.

.. autosummary::
   :toctree: generated
   :nosignatures:

   detect_peaks
   peak_labels
   fit_linelets
   detect_streaks
   to_lines
   n_signal
   streak_labels
