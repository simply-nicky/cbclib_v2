Getting started
===============

.. contents:: On this page
   :local:
   :depth: 1

Installation
------------

CPU-only (default)::

   pip install cbclib_v2

With CUDA support::

   python setup.py build_ext -i
   pip install cbclib_v2

To skip CUDA compilation when CUDA headers are present but unwanted::

   CBCLIB_SKIP_CUDA=1 pip install cbclib_v2

Quickstart
----------

The usual starting point is to connect detector geometry with experimental
HDF5 data, then read a subset of detector frames for processing. The pages
below contain the maintained code examples for each step.

**1. Load a detector geometry**

Use :func:`~cbclib_v2.read_crystfel` to read a CrystFEL ``.geom`` file into a
:class:`~cbclib_v2.Detector`. See :doc:`geometry` for examples of loading a
geometry file, assembling module data, and converting coordinates.

.. code-block:: python

   from cbclib_v2 import read_crystfel

   detector = read_crystfel("detector.geom")
   assembler = detector.assembler()

**2. Open an HDF5 file**

Use :class:`~cbclib_v2.H5Protocol` and :class:`~cbclib_v2.H5Handler` when you
want direct access to one or more HDF5 files. See :doc:`hdf5` for examples of
building frame indices and loading named datasets.

.. code-block:: python

   from cbclib_v2 import H5Protocol, H5Handler

   protocol = H5Protocol.read("protocol.json")
   handler = H5Handler(protocol)

   indices = handler.indices("run_001.h5", "data")
   frames = handler.load(indices[:200])

**3. Open a run**

Use a facility-specific :class:`~cbclib_v2.RunConfig` and
:func:`~cbclib_v2.open_run` when working with a beamtime run rather than a
single file. See :doc:`io_layer` for EuXFEL and SwissFEL run configuration
examples.

.. code-block:: python

   from cbclib_v2 import XFELRunConfig, open_run

   config = XFELRunConfig.read("xfel_config.json")
   run = open_run(100, config)

**4. Load detector data**

Use :meth:`~cbclib_v2.BaseRun.indices` to select frames and
:meth:`~cbclib_v2.BaseRun.data` to load detector data, optionally assembled
with detector geometry. The full frame-loading examples are in :doc:`io_layer`;
background-ready loading into :class:`~cbclib_v2.CrystData` is shown in
:doc:`background_subtraction`.

.. code-block:: python

   indices = run.indices()
   frames = run.data(indices[:50], geometry=True)

   print(frames.shape)

Next step: workflows
--------------------

For the most user-facing example of how these pieces fit together, continue to
:doc:`workflows`. It shows the same library components in interactive Python,
command-line processing, and SLURM batch workflows.
