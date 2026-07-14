Data containers
===============

.. currentmodule:: cbclib_v2

Frozen dataclass containers that carry arrays through the processing pipeline.
All containers inherit from a common hierarchy (:class:`Container` →
:class:`DataContainer` → :class:`ArrayContainer`) and are array-namespace
portable — they work with NumPy, JAX, and CuPy arrays interchangeably.

Container hierarchy
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Container
   DataContainer
   ArrayContainer
   IndexedContainer
   split
   to_list

Data processing
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   CrystData
   CrystMetadata

Streak detectors
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   OnlineDetector
   RegionDetector
   StreakDetector

Streaks
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Streaks
   StackedStreaks
   Lines
