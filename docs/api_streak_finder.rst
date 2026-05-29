Streak detection
================

.. currentmodule:: cbclib_v2.streak_finder

**cbclib_v2.streak_finder** implements the seed-and-grow streak detection
algorithm described in :doc:`streak_detection`.  The high-level entry point
is :class:`PatternStreakFinder`; the low-level functions below it expose each
pipeline stage individually for custom workflows.

See :doc:`streak_detection` for a full narrative description of the algorithm
and a worked example.

High-level interface
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   PatternStreakFinder

Data types
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   PeakLabels
   Streaks

Low-level pipeline functions
----------------------------

These functions correspond to the individual stages of the streak detection
pipeline.  They are called internally by :class:`PatternStreakFinder` but can
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
