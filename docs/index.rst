cbclib_v2
=========

**cbclib_v2** is a Python library for processing serial crystallography datasets
measured at free-electron lasers (FELs) such as EuXFEL, SwissFEL, and LCLS.
It targets convergent beam crystallography (CBC) experiments on protein crystals
and small-molecule crystals, and covers the main stages of an X-ray
crystallography data-processing pipeline.

The library is inspired by `CrystFEL`_ and extends its concepts to the specific
geometry of convergent beam diffraction. See [Li2026]_ for the scientific
background.

.. _CrystFEL: https://www.desy.de/~twhite/crystfel/

Processing pipeline
-------------------

- **Pattern pre-processing** — background estimation, variance analysis, and
  PCA-based whitefield correction (``CrystData``, ``CrystMetadata``).
- **Streak detection** — connected-component labelling and line-fitting in
  C++/CUDA (``cbclib_v2.label``, ``cbclib_v2.streak_finder``).
- **Indexing** — JAX-based optimisation of crystal orientation and unit-cell
  parameters (``cbclib_v2.indexer``).
- **Intensity scaling** — planned.

Installation
------------

CPU-only (default)::

   pip install cbclib_v2

With CUDA support::

   python setup.py build_ext -i
   pip install cbclib_v2

To skip CUDA compilation when CUDA headers are present but unwanted::

   CBCLIB_SKIP_CUDA=1 pip install cbclib_v2

.. toctree::
   :maxdepth: 1
   :hidden:

.. Note: API reference pages will be added here once the page layout is decided.

References
----------

.. [Li2026] C. Li *et al.*, "Convergent-Beam X-ray Crystallography,"
            arXiv:2602.14402 (2026). https://arxiv.org/abs/2602.14402
