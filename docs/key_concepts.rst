Key concepts
============

The design of **cbclib_v2** is organised around three practical concerns:
modularity, performance, and workflows.

Modularity
----------

Modularity keeps the processing pipeline adaptable. Data access, geometry
handling, background subtraction, streak detection, and indexing are separated
into independent layers so that a workflow can be adjusted for different FEL
facilities, detector layouts, and indexing strategies without rewriting the
whole stack.

.. toctree::
   :maxdepth: 1

   io_layer
   hdf5
   geometry
   background_subtraction
   online_detection
   streak_detection

Performance
-----------

Performance matters because CBC processing usually starts from large detector
frame stacks. The library uses array-backend-aware code paths for NumPy, CuPy,
and JAX arrays, and selected image-processing kernels have CPU/OpenMP and
CUDA implementations. The Array API page explains backend selection and array
conversion utilities; the low-level labeling and streak-detection APIs are
documented in :doc:`api_image`.

.. toctree::
   :maxdepth: 1

   array_api

Workflows
---------

Workflows connect these components into usable processing modes. **cbclib_v2**
can be used interactively from Python or Jupyter notebooks, through
configuration-driven command-line scripts, or as automated batch pipelines on
the MAXWELL cluster using SLURM.

.. toctree::
   :maxdepth: 1

   workflows
