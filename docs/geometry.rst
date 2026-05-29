CrystFEL detector geometry
==========================

cbclib_v2 uses `CrystFEL`_ ``.geom`` files to describe the physical layout
of detector panels.  A geometry file records where each panel sits in the
lab frame and how its raw (ss, fs) pixel indices map to physical coordinates.
:func:`~cbclib_v2.read_crystfel` parses such a file into a
:class:`~cbclib_v2.Detector` object.

The full specification of the file format is given in the `CrystFEL geometry
reference
<https://gitlab.desy.de/thomas.white/crystfel/-/blob/master/doc/man/crystfel_geometry.5.md>`_.

.. _CrystFEL: https://www.desy.de/~twhite/crystfel/

.. contents:: On this page
   :local:
   :depth: 1

Coordinate system
-----------------

CrystFEL defines a right-handed lab frame:

* **+z** — along the beam, away from the source.
* **+y** — toward zenith (upward in the GUI).
* **+x** — completing the right-handed system (left to right in the GUI).

Panel positions and directions are expressed in units of pixels.

Panel geometry
--------------

Each detector module is described by a :class:`~cbclib_v2.Panel` with the
following key parameters (see the CrystFEL reference for the full list):

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``corner_x``, ``corner_y``
     - Position of the panel corner in lab-frame pixel units.
   * - ``fs``, ``ss``
     - Unit vectors of the fast-scan and slow-scan axes in lab-frame pixel
       units.
   * - ``res``
     - Detector resolution in pixels per metre.
   * - ``clen``, ``coffset``
     - Overall camera length and per-panel z offset, both in metres.
   * - ``min_fs``, ``max_fs``, ``min_ss``, ``max_ss``
     - Pixel bounding box selecting this panel from the raw data array.
   * - ``dim0``, ``dim1``, …
     - Dimension labels (``'ss'``, ``'fs'``, ``'%'``, or a fixed integer)
       mapping raw array axes to image coordinates.

Loading a geometry file
-----------------------

.. code-block:: python

   from cbclib_v2 import read_crystfel

   detector = read_crystfel('detector.geom')

   # Number of modules and pixel size
   print(detector.num_modules, detector.pixel_size)

   # Access a specific panel by zero-based index
   panel = detector.panel(0)
   print(panel.res, panel.corner)

Assembling stacked module data
------------------------------

:meth:`~cbclib_v2.Detector.assembler` builds a callable that places stacked
module data onto a single image in the lab coordinate system.

.. code-block:: python

   assembler = detector.assembler()

   # frames shape: (n_modules, ss_size, fs_size) or (N, n_modules, ss_size, fs_size)
   assembled = assembler(frames)   # shape: (H, W) or (N, H, W)

Coordinate transforms
---------------------

:meth:`~cbclib_v2.Detector.to_detector` converts raw array indices to
lab-frame ``(x, y, z)`` coordinates, optionally in metres.

.. code-block:: python

   import numpy as np

   ss = np.arange(detector.shape[-2])
   fs = np.arange(detector.shape[-1])
   ss_grid, fs_grid = np.meshgrid(ss, fs, indexing='ij')

   # Coordinates in pixel units (default)
   x, y, z = detector.to_detector(ss_grid, fs_grid)

   # Coordinates in metres
   x_m, y_m, z_m = detector.to_detector(ss_grid, fs_grid, units='meter')

For multi-module detectors, prepend the module index:

.. code-block:: python

   module_ids = np.zeros_like(ss_grid)   # module 0
   x, y, z = detector.to_detector(module_ids, ss_grid, fs_grid)

Streak coordinate transforms
----------------------------

:meth:`~cbclib_v2.Detector.to_streaks` converts pixel-space
:class:`~cbclib_v2.Streaks` to lab-frame coordinates.
:meth:`~cbclib_v2.Detector.to_patterns` then scales to metres for indexing:

.. code-block:: python

   from cbclib_v2 import read_crystfel

   detector = read_crystfel('detector.geom')

   # pixel_streaks: Streaks or StackedStreaks in raw array coordinates
   lab_streaks = detector.to_streaks(pixel_streaks)
   patterns    = detector.to_patterns(lab_streaks)

See also
--------

For full API documentation see :doc:`api_geometry`.
