Modular I/O layer
=================

Different FEL facilities store detector data in different HDF5 structures and
metadata layouts.  **cbclib_v2** handles this through a modular I/O layer: a
common :class:`~cbclib_v2.BaseRun` interface provides uniform access to
detector frames and per-frame metadata, while facility-specific details live in
separate :class:`~cbclib_v2.RunConfig` subclasses.

The run object is created with a single call to :func:`~cbclib_v2.open_run`,
which dispatches on the ``"facility"`` field of the configuration.  The same
analysis code therefore runs unchanged on datasets from different facilities.

.. contents:: On this page
   :local:
   :depth: 1

Supported facilities
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Facility
     - Config class
     - Notes
   * - European XFEL
     - :class:`~cbclib_v2.XFELRunConfig`
     - Multi-module detectors; per-module HDF5 files matched by a
       ``(run_id, module_id)`` pattern.
   * - SwissFEL
     - :class:`~cbclib_v2.SwissFELConfig`
     - Single detector; HDF5 files matched by a run-level regex pattern.
   * - LCLS
     - —
     - Planned.

Configuration files
-------------------

Both facilities are configured through a JSON file.  The ``"facility"`` key
selects the config class; the remaining keys map to the dataclass fields.

**European XFEL** (``xfel_config.json``)

.. code-block:: json

   {
       "facility":       "XFEL",
       "data_dir":       "/gpfs/exfel/exp/SPB/202302/p004456/proc/r{0:04d}",
       "hdf5_protocol":  "/path/to/agipd_protocol.json",
       "file_pattern":   "CORR-R{0:04d}-JNGFR{1:02d}-S(\\d{{5}})\\.h5",
       "geometry_file":  "/path/to/detector.geom",
       "num_modules":    8,
       "starts_at":      1
   }

.. list-table:: EuXFEL config fields
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Description
   * - ``data_dir``
     - Directory format string; ``{0}`` is replaced by the run ID (e.g.
       ``r{0:04d}`` → ``r0042``).
   * - ``hdf5_protocol``
     - Path to the :class:`~cbclib_v2.H5Protocol` JSON or INI file
       (see :doc:`hdf5`).
   * - ``file_pattern``
     - Python ``str.format`` pattern converted to a regex; positional fields
       are filled with ``(run_id, module_id)``.  Regex special characters must
       be escaped (``\\.``), and regex curly braces must be double-escaped
       (``\\d{{5}}``).
   * - ``geometry_file``
     - Path to the CrystFEL ``.geom`` file (see :doc:`geometry`).
   * - ``num_modules``
     - Number of detector modules to read (default ``1``).
   * - ``starts_at``
     - Index of the first module; modules ``[starts_at, starts_at + num_modules)``
       are loaded (default ``0``).

**SwissFEL** (``swissfel_config.json``)

.. code-block:: json

   {
       "facility":       "SwissFEL",
       "data_dir":       "/sf/bernina/data/p19000/raw/r{0:04d}/data",
       "hdf5_protocol":  "/path/to/jungfrau_protocol.json",
       "file_pattern":   "acq(\\d{4})\\.JF07T32V02\\.h5",
       "geometry_file":  "/path/to/detector.geom"
   }

.. list-table:: SwissFEL config fields
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Description
   * - ``data_dir``
     - Directory format string; ``{0}`` is replaced by the run ID.
   * - ``hdf5_protocol``
     - Path to the :class:`~cbclib_v2.H5Protocol` JSON or INI file.
   * - ``file_pattern``
     - Regex matched against filenames in ``data_dir`` to select HDF5 files.
       The acquisition index captured in the first group determines file order.
   * - ``geometry_file``
     - Path to the CrystFEL ``.geom`` file.

Opening a run
-------------

Load the config with :meth:`~cbclib_v2.RunConfig.read` and pass it to
:func:`~cbclib_v2.open_run`:

.. code-block:: python

   from cbclib_v2 import open_run, XFELRunConfig

   # Load facility config
   config = XFELRunConfig.read('xfel_config.json')

   # Open run 100
   run = open_run(100, config)

   # List available metadata attributes (pulse ID, energy, …)
   print(run.attributes())

   # Build the full frame-index table, then take the first 50 frames
   indices = run.indices()
   frames = run.data(indices[:50], geometry=True)
   print(frames.shape)   # e.g. (50, 1480, 1552) with geometry applied

The same code works for SwissFEL by substituting
:class:`~cbclib_v2.SwissFELConfig` for :class:`~cbclib_v2.XFELRunConfig` —
the :func:`~cbclib_v2.open_run` call and the rest of the pipeline are
identical.

See also
--------

For full API documentation see :doc:`api_run`.
