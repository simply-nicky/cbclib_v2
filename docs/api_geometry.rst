CrystFEL geometry
=================

.. currentmodule:: cbclib_v2

Classes and functions for parsing CrystFEL ``.geom`` files and working with
detector geometry.  :func:`read_crystfel` loads a geometry file into a
:class:`Detector` object, which provides coordinate transforms and image
assembly for single- and multi-module detectors.

See :doc:`geometry` for a narrative introduction.

.. autosummary::
   :toctree: generated
   :nosignatures:

   read_crystfel
   Detector
   Panel
