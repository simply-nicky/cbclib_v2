Array API
=========

**cbclib_v2** follows the principle of the `Python Array API standard`_:
*array type in equals array type out*.  Every routine inspects its arguments
at call time and dispatches through the matching array namespace — no
configuration flags, no manual switching.

Two main workflows benefit from this design:

- **Image processing** (:class:`~cbclib_v2.CrystData`, streak detection, data
  transforms) runs on CPU with NumPy and on NVIDIA GPUs with CuPy, with
  identical code paths.
- **Crystallographic indexing** supports JAX arrays so that JAX's automatic
  differentiation can drive gradient-descent optimisers over indexing
  parameters.

Namespace detection is powered by `array-api-compat`_, a thin compatibility
shim that implements the standard interface over NumPy, JAX, and CuPy.

.. _Python Array API standard: https://data-apis.org/array-api/latest/
.. _array-api-compat: https://data-apis.org/array-api-compat/

.. contents:: On this page
   :local:
   :depth: 1

Supported backends
------------------

Three backends are supported out of the box:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Backend
     - Array type
     - Notes
   * - NumPy
     - :class:`numpy.ndarray`
     - CPU; default when no GPU is available.  Supported across all features.
   * - JAX
     - :class:`jax.Array`
     - CPU or GPU via XLA.  Enables JIT compilation and automatic
       differentiation; used by the indexing module for gradient-descent
       optimisation.
   * - CuPy
     - :class:`~cbclib_v2.annotations.CuPyArray`
     - NVIDIA GPU; optional dependency — available when ``import cupy``
       succeeds.  Supported by all image processing routines.

When arrays from different backends are mixed, the precedence order is
**CuPy > JAX > NumPy**: passing both a JAX array and a NumPy array to the same
function yields a JAX result.

Getting the active namespace
----------------------------

:func:`~cbclib_v2.array_namespace` inspects the arrays it receives and returns
the matching namespace object:

.. code-block:: python

   from cbclib_v2 import array_namespace

   xp = array_namespace(data)   # resolves to NumPy, JaxNumPy, or CuPy
   out = xp.zeros_like(data)    # calls the right backend

All :class:`~cbclib_v2.DataContainer` subclasses expose a ``namespace``
property that returns the namespace of their first array field, so explicit
calls to :func:`~cbclib_v2.array_namespace` are rarely needed when working
with containers.

To get the default namespace for a platform without any arrays at hand use
:func:`~cbclib_v2.default_api`:

.. code-block:: python

   from cbclib_v2 import default_api

   xp = default_api('cpu')   # NumPy
   xp = default_api('gpu')   # CuPy (requires CuPy)

Writing array-portable code
---------------------------

The key rule is: **never import** ``numpy`` **or** ``jax.numpy`` **directly
inside a function that must stay device-agnostic.**  Resolve the namespace from
the input arrays and call every operation through it:

.. code-block:: python

   from cbclib_v2 import array_namespace

   def normalise(data):
       xp = array_namespace(data)
       return (data - xp.mean(data)) / xp.std(data)

The namespace exposes the same interface regardless of the backend —
``xp.zeros``, ``xp.concat``, ``xp.stack``, ``xp.where``, and so on — so the
function above runs unchanged on NumPy arrays, JAX arrays, or CuPy arrays.

When calling compiled code (C++ extensions or CUDA kernels) that only accepts
NumPy arrays, convert at the boundary and restore the original type on the way
out:

.. code-block:: python

   from cbclib_v2 import array_namespace, asnumpy

   def call_native(data):
       xp = array_namespace(data)
       result_np = _native_extension(asnumpy(data))   # C++ / CUDA call
       return xp.asarray(result_np)                   # restore original type

Converting between backends
---------------------------

Three explicit converters move data between backends:

.. code-block:: python

   from cbclib_v2 import asnumpy, asjax, ascupy

   np_data  = asnumpy(jax_data)    # jax.Array  → numpy.ndarray
   jax_data = asjax(np_data)       # numpy.ndarray → jax.Array
   cp_data  = ascupy(np_data)      # numpy.ndarray → cupy.ndarray  (requires CuPy)

:class:`~cbclib_v2.DataContainer` subclasses expose the same conversions as
methods: ``container.to_numpy()``, ``container.to_jax()``,
``container.to_cupy()``.

JAX considerations
------------------

**JIT compilation.**  Any function that follows the array-namespace pattern
(no direct NumPy calls) is immediately compatible with :func:`jax.jit`:

.. code-block:: python

   import jax
   from cbclib_v2 import array_namespace

   @jax.jit
   def normalise(data):
       xp = array_namespace(data)
       return (data - xp.mean(data)) / xp.std(data)

**Pytrees.**  :class:`~cbclib_v2.State` subclasses are registered as JAX
pytrees, so they can be passed directly to :func:`jax.jit`, :func:`jax.grad`,
and :func:`jax.vmap`.  Geometry and configuration attributes that must not be
traced or differentiated should be declared with
:func:`~cbclib_v2.field` ``(static=True)``:

.. code-block:: python

   from cbclib_v2 import State, field

   class MyState(State):
       data:     RealArray
       n_frames: int = field(static=True)   # excluded from JAX tracing

**Lazy registration.**  JAX pytree registration for :class:`~cbclib_v2.State`
subclasses happens on *first instantiation*, not at import time.  This avoids
the ~12 GB GPU-context pre-allocation that ``import jax`` triggers when a GPU
is present.  Set the environment variable ``XLA_PYTHON_CLIENT_PREALLOCATE=false``
or ``XLA_PYTHON_CLIENT_MEM_FRACTION`` to control JAX memory usage independently
of the import.

**Random number generation.**  Use :func:`~cbclib_v2.default_rng` instead of
``numpy.random.default_rng`` or a raw JAX PRNG key — it returns a
backend-appropriate generator:

.. code-block:: python

   from cbclib_v2 import default_rng, array_namespace

   xp  = array_namespace(data)
   rng = default_rng(seed=42, xp=xp)
   noise = rng.normal(size=data.shape)   # NumPy, JAX, or CuPy result

Type aliases
------------

All public functions and classes use a small set of type aliases for
array arguments. They are defined in :mod:`cbclib_v2.annotations` and
listed below.

.. currentmodule:: cbclib_v2.annotations

**Generic arrays** (device-portable):

.. py:class:: Array

   ``jax.Array | numpy.ndarray | CuPyArray`` — any device-portable array.

.. py:class:: RealArray

   ``jax.Array | NDRealArray | CuPyArray`` — floating-point array on any backend.

.. py:class:: IntArray

   ``jax.Array | NDIntArray | CuPyArray`` — integer array on any backend.

.. py:class:: BoolArray

   ``jax.Array | NDBoolArray | CuPyArray`` — boolean array on any backend.

.. py:class:: ArrayLike

   ``Array | Scalar | Sequence`` — anything that can be converted to an array.

**NumPy-specific**:

.. py:class:: NDArray

   Alias for :class:`numpy.ndarray`.

.. py:class:: NDRealArray

   NumPy array of real (floating-point) values;
   alias for :class:`numpy.typing.NDArray`\[:class:`numpy.floating`\].

**Shape and dtype helpers**:

.. py:class:: ShapeLike

   ``int | Sequence[int] | IntArray`` — anything that describes an array shape.

.. py:class:: DTypeLike

   ``str | type | numpy.dtype | SupportsDType`` — anything that describes a dtype.

.. py:class:: Indices

   ``int | slice | IntArray | Sequence[int]`` — anything used to index an array axis.

See also
--------

For full API documentation of the array utility functions see
:doc:`api_utils`.
