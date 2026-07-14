cbclib documentation
====================

| **Version**: |release|
| **Useful links**: `Source code in GitHub <https://github.com/simply-nicky/cbclib_v2>`__ | `Mail to the maintainer <mailto:nikolay.ivanov@desy.de>`__

----

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

The data processing pipeline of crystallography datasets the following stages that are
implemented in **cbclib_v2**:

1. **Data I/O** — loads experimental frames and metadata from facility HDF5
   files (:class:`~cbclib_v2.H5Protocol`, :class:`~cbclib_v2.H5Handler`,
   :func:`~cbclib_v2.open_run`).

2. **Background estimation** — separates the crystal diffraction signal from
   diffuse scatter using variance analysis and PCA-based whitefield correction
   (:class:`~cbclib_v2.CrystData`, :class:`~cbclib_v2.CrystMetadata`).

3. **Hit finding and streak detection** — identifies frames that contain
   diffraction features and detects the streaks characteristic of convergent
   beam diffraction (:class:`~cbclib_v2.StreakDetector`,
   :doc:`usage example<streak_detection>`, and :doc:`API reference<api_image>`).

4. **Preliminary indexing** — provides an initial estimate of the crystal
   orientation from the detected streaks (not documented yet).

5. **Full indexing and geometry refinement** — refines the orientation and
   detector geometry to match the observed diffraction patterns using
   JAX-based optimisation (not documented yet).

6. **Intensity scaling** — combines integrated signal from detected hits into
   a common table of structure factors (not implemented yet).

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting started

   getting_started

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Key concepts

   key_concepts

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   api

References
----------

.. [Li2026] C. Li *et al.*, "Convergent-Beam X-ray Crystallography,"
            arXiv:2602.14402 (2026). https://arxiv.org/abs/2602.14402
