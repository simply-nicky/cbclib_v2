State
=====

.. currentmodule:: cbclib_v2

JAX-compatible dataclasses for carrying algorithm state through
differentiable or JIT-compiled computations.  :class:`State` subclasses
register as JAX pytrees; fields marked with :func:`field` and
``static=True`` are treated as static metadata rather than traced arrays.

.. autosummary::
   :toctree: generated
   :nosignatures:

   State
   DynamicField
   field
   dynamic_fields
   static_fields
