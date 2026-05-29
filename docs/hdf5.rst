Working with HDF5 files
=======================

HDF5 files from different FEL facilities store the same physical data
(detector frames, masks, pulse IDs, …) at different dataset paths.
:class:`~cbclib_v2.H5Protocol` solves this by acting as a lookup table:
it maps each *attribute name* (a logical name such as ``'data'`` or
``'mask'``) to a ranked list of candidate HDF5 paths and to a
:class:`~cbclib_v2.Kinds` value that controls how the data is indexed and
loaded.  Pass the protocol to :class:`~cbclib_v2.H5Handler` to read and
write those attributes from HDF5 files without caring about the
facility-specific layout.

.. contents:: On this page
   :local:
   :depth: 1

The HDF5 protocol
-----------------

A protocol stores two mappings:

``paths``
    Maps each attribute name to an ordered list of candidate HDF5 dataset
    paths.  When a file is opened, :class:`~cbclib_v2.H5Handler` searches
    the paths in order and uses the first one that exists.  This lets a
    single protocol cover several file-layout conventions.

``kinds``
    Maps each attribute name to its :class:`~cbclib_v2.Kinds` value, which
    describes the dimensionality of the stored data.

.. list-table:: Data kinds
   :header-rows: 1
   :widths: 15 55 30

   * - Kind
     - Meaning
     - Typical use
   * - ``scalar``
     - Single value per file; no index subsetting.
     - Global metadata (e.g. wavelength).
   * - ``sequence``
     - 1-D array per file; elements selected by frame index.
     - Per-frame scalars (e.g. pulse ID, energy).
   * - ``frame``
     - Single 2-D image; pixel ROI via ss/fs indices.
     - Calibration images (mask, flat-field, whitefield).
   * - ``stack``
     - 3-D array of frames; supports both frame-index and pixel ROI.
     - Detector data.

Writing a protocol file
-----------------------

Both JSON and INI formats are supported.  The format is detected from
the file extension.  :meth:`~cbclib_v2.H5Protocol.read` loads either;
:meth:`~cbclib_v2.H5Protocol.write` writes either.

**JSON** (``protocol.json``)

The top-level object has two keys, ``"paths"`` and ``"kinds"``.  Each
path list is a JSON array of strings; leading slashes are optional.

.. code-block:: json

   {
      "paths": {
          "data":     ["data/JF07T32V02/data"],
          "mask":     ["data/JF07T32V02/meta/pixel_mask"],
          "pulse_id": ["data/JF07T32V02/pulse_id"]
      },
      "kinds": {
          "data":     "stack",
          "mask":     "frame",
          "pulse_id": "sequence"
      }
   }

**INI** (``protocol.ini``)

The INI file has a ``[paths]`` section and a ``[kinds]`` section.  Path
lists are written in bracket notation; trailing commas are allowed.

.. code-block:: ini

   [paths]
   data       = [/entry/data/data, /entry_1/data/data, /data/data]
   frames     = [/entry/crystallography/frames, /frame_selector/frames]
   mask       = [/entry/crystallography/mask, /entry/instrument/detector/mask]
   whitefield = [/entry/crystallography/whitefield,]
   std        = [/entry/crystallography/std,]

   [kinds]
   data       = stack
   frames     = sequence
   mask       = frame
   whitefield = frame
   std        = frame

Loading data
------------

The typical workflow is:

1. Load the protocol with :meth:`~cbclib_v2.H5Protocol.read`.
2. Create an :class:`~cbclib_v2.H5Handler` from the protocol.
3. Call :meth:`~cbclib_v2.H5Handler.indices` to build a frame-index table
   (:class:`~cbclib_v2.LoadIndices`) over one or more HDF5 files.
4. Subset the index table (``indices[:100]``) and pass it to
   :meth:`~cbclib_v2.H5Handler.load`.

.. code-block:: python

   from cbclib_v2 import H5Protocol, H5Handler

   protocol = H5Protocol.read("protocol.json")
   handler  = H5Handler(protocol)

   # Build a frame-index table for the 'data' attribute
   indices = handler.indices("run_001.h5", "data")

   # Load the first 200 frames
   frames = handler.load(indices[:200])          # shape: (200, H, W)

   # Load with a pixel ROI (rows 100–300, all columns)
   frames = handler.load(indices[:200], ss_idxs=slice(100, 300))

The :func:`~cbclib_v2.read_hdf` convenience function loads several
attributes at once and returns a dictionary:

.. code-block:: python

   from cbclib_v2 import read_hdf

   arrays = read_hdf("run_001.h5", handler, "data", "mask",
                     indices=slice(0, 200))
   frames = arrays["data"]   # (200, H, W)
   mask   = arrays["mask"]   # (H, W)

See also
--------

For full API documentation see :doc:`api_hdf5`.
