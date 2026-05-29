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

----

.. currentmodule:: cbclib_v2.label

**cbclib_v2.label** provides connected-component labeling, morphological
operations, and intensity-weighted image moments.  It is the building block
for :class:`~cbclib_v2.streak_finder.PatternStreakFinder` and the
:class:`~cbclib_v2.RegionDetector` high-level pipeline step.

Structuring elements
--------------------

A :class:`Structure` describes the pixel neighbourhood used for
connectivity tests and morphological operations.  ``structure.connectivity``
is the bin radius used throughout streak detection.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Structure

Label types
-----------

:func:`label` dispatches to a CPU or GPU implementation and returns one of
two backend-specific result types:

* **NPLabelResult** (``cbclib_v2.label.NPLabelResult``) — CPU / NumPy
  backend.  An alias for the C++ class documented below; stores regions
  in a compact sparse representation.
* **CPLabelResult** (``cbclib_v2.label.CPLabelResult``) — GPU / CuPy
  backend.  A :class:`~typing.NamedTuple` holding a dense label array and
  a region-index vector.

Use :func:`index` and :func:`labels` to extract region indices or the
dense per-pixel integer label map from either type.

.. currentmodule:: cbclib_v2.label

.. autosummary::
   :toctree: generated
   :nosignatures:

   LabelResult
   NPLabelResult
   CPLabelResult

Morphological operations
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   label
   binary_dilation

Accessors
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   index
   labels

Image moments
-------------

Intensity-weighted image moments computed per labeled region.  All moment
functions follow the formulation from `Image moment
<https://en.wikipedia.org/wiki/Image_moment>`_ (Wikipedia).

.. autosummary::
   :toctree: generated
   :nosignatures:

   center_of_mass
   covariance_matrix
   ellipse_fit
   line_fit
   p_values
