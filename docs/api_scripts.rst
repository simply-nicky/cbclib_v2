Scripts
=======

.. currentmodule:: cbclib_v2.scripts

**cbclib_v2.scripts** provides the configuration classes and processing
functions that underlie all three workflows described in :doc:`workflows`.
Configuration objects are JSON-serialisable dataclasses; processing functions
operate on :class:`~cbclib_v2.CrystData` / :class:`~cbclib_v2.CrystMetadata`
and dispatch to CPU or GPU depending on the active array namespace.

Scan configuration
------------------

:class:`ScanConfig` is the top-level experiment descriptor shared by all
three workflows (see :doc:`workflows`).  It is loaded from ``scan.json``
and aggregates six sub-configuration objects — one per pipeline step —
each of which corresponds to a named section in the JSON file.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ScanConfig

The sub-configuration classes below map directly to ``scan.json`` sections
and are also accessible as attributes of :class:`ScanConfig`:

.. autosummary::
   :toctree: generated
   :nosignatures:

   DetectConfig
   MetadataConfig
   MetaListConfig
   SetupConfig
   SystemConfig

Configuration
-------------

Base class shared by all parameter containers.  Every subclass can be
serialised to/from JSON via :meth:`~cbclib_v2.Container.to_dict` /
:meth:`~cbclib_v2.Container.from_dict` and read directly from a file with
:meth:`~cbclib_v2.Container.read`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseParameters

Common sub-configurations used across multiple processing steps:

.. autosummary::
   :toctree: generated
   :nosignatures:

   ROIParameters
   MaskParameters
   StructureParameters

Background and metadata configuration:

.. autosummary::
   :toctree: generated
   :nosignatures:

   BackgroundParameters
   MetadataParameters
   ScalingParameters

Detection configuration:

.. autosummary::
   :toctree: generated
   :nosignatures:

   RegionParameters
   RegionFinderConfig
   PeakParameters
   StreakParameters
   StreakFinderConfig

Indexing configuration:

.. autosummary::
   :toctree: generated
   :nosignatures:

   IndexingConfig

Background and SNR
------------------

Functions for computing background whitefields, assembling
:class:`~cbclib_v2.CrystMetadata`, and applying background subtraction to
produce SNR frames.

.. autosummary::
   :toctree: generated
   :nosignatures:

   create_background
   create_metadata
   scale_background

Detection
---------

Frame-level detection functions and multi-process pool runners for streak and
region detection.

.. autosummary::
   :toctree: generated
   :nosignatures:

   detect_regions
   detect_streaks
   concentric_only
   run_detection
   pool_detection

Indexing
--------

Pattern indexing functions and multi-process pool runners.

.. autosummary::
   :toctree: generated
   :nosignatures:

   index_patterns
   indexing_candidates
   run_indexing
   pool_indexing

----

.. currentmodule:: cbclib_v2.slurm

SLURM job management
--------------------

Classes for building ``sbatch`` scripts and submitting them to a SLURM
cluster.  Used by the :ref:`SLURM batch pipeline <workflows-slurm>`
workflow.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Scripts
   SLURMJobManager

SLURM data types
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SLURMScript
   ScriptSpec
   SLURMConfig
   JobID
   JobOutput
   JobStatus
