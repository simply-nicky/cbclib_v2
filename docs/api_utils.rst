Utilities
=========

.. currentmodule:: cbclib_v2

Array API utilities
-------------------

These helpers implement the array-namespace pattern described in :doc:`array_api`.
They provide device-portable in-place operations and explicit conversions between
backends that are not part of the `Python Array API standard`_.

.. _Python Array API standard: https://data-apis.org/array-api/latest/

.. autosummary::
   :toctree: generated
   :nosignatures:

   array_namespace
   default_api
   asnumpy
   asjax
   ascupy
   add_at
   set_at
   min_at
   default_rng

CPU configuration
-----------------

The image processing routines in **cbclib_v2** are backed by C++ extensions
compiled with OpenMP.  When arrays are on the CPU (NumPy backend), these
extensions pick up the thread count from a thread-local :class:`CPUConfig`
object.  JAX and CuPy manage their own parallelism independently and are
unaffected by this setting.

Use :func:`set_cpu_config` to set the thread count globally for the current
thread, or use :class:`CPUConfig` as a context manager to change it temporarily:

.. code-block:: python

   from cbclib_v2 import CPUConfig, set_cpu_config
   import cbclib_v2.ndimage as ndimage

   # Permanent change for this thread
   set_cpu_config(8)

   # Temporary change — restored on exit
   with CPUConfig(num_threads=4):
       whitefield = ndimage.median(frames, axis=0)   # uses 4 OpenMP threads

Non-main threads are automatically limited to a single thread to prevent nested
parallelism.

.. autosummary::
   :toctree: generated
   :nosignatures:

   CPUConfig
   get_cpu_config
   set_cpu_config
   reset_cpu_config
