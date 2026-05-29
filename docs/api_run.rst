Run configuration
=================

.. currentmodule:: cbclib_v2

Classes and functions for opening experimental runs from different FEL
facilities through a common interface.  :func:`open_run` dispatches on the
``"facility"`` field of a :class:`RunConfig` to return the appropriate
:class:`BaseRun` subclass, so analysis code stays facility-independent.

See :doc:`io_layer` for a narrative introduction.

.. autosummary::
   :toctree: generated
   :nosignatures:

   open_run
   RunConfig
   BaseRun
   XFELRunConfig
   XFELRun
   SwissFELConfig
   SwissFELRun
