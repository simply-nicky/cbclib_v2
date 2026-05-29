HDF5 protocol
=============

.. currentmodule:: cbclib_v2

Classes and functions for reading and writing HDF5 files using a
facility-agnostic protocol.  :class:`H5Protocol` maps logical attribute
names to ranked lists of HDF5 dataset paths; :class:`H5Handler` uses that
map to load and store data without hard-coding facility-specific paths.

See :doc:`hdf5` for a narrative introduction.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Kinds
   H5Protocol
   H5Files
   LoadIndices
   H5Handler
   read_hdf
   write_hdf
