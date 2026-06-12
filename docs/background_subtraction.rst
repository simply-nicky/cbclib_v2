Background subtraction
======================

The first step of processing a diffraction pattern is separating the crystal
diffraction signal from the diffuse scattering background produced by the
incident X-ray beam, the sample environment, and the air path. This diffuse
component — called the *whitefield* — typically dominates the detector signal
and must be estimated and subtracted before any streak detection or indexing
can take place. The result of background subtraction is an *SNR frame*: a
per-pixel signal-to-noise ratio image used in streak detection.

**cbclib_v2** provides two container classes for this stage:

* :class:`~cbclib_v2.CrystData` — a single stack of raw detector frames
  together with the mask, whitefield, noise standard deviation, and the
  resulting SNR.
* :class:`~cbclib_v2.CrystMetadata` — a background *model* built from one or
  more :class:`~cbclib_v2.CrystData` containers; stores the flatfield,
  per-pixel noise, and optionally a PCA decomposition of background
  variability.

The three background subtraction options are:

.. code-block:: text

   Option A — in-place background subtraction
   ───────────────────────────────────────────
            raw frames ──► CrystData
                                │
                          update_mask
                                │
                          update_metadata ──► update_snr (SNR frames)

   Option B — static background subtraction
   ─────────────────────────────────────────
   background frames ──► CrystMetadata
                                │
                         to_data(frames)
                                │
                         CrystData ──► update_snr (SNR frames)

   Option C — dynamic background subtraction
   ──────────────────────────────────────────
   background frames ──► CrystMetadata
                                │
                               pca
                                │
                          projection(frames) ──► whitefields (per-frame background)
                                │
                      to_data(frames, whitefields)
                                │
                            CrystData ──► update_snr (SNR frames)


.. _background-whitefield:

The whitefield
--------------

In a CBC experiment the beam profile changes slowly between shots but does not
vanish entirely. The *whitefield* (also called the *flatfield* or *background
model*) is the per-pixel expected photon count in the absence of crystal
diffraction. It is estimated from a stack of frames under the assumption that
diffraction contributes to only a small fraction of the pixels in any single
frame.

The noise standard deviation **std** complements the whitefield: it quantifies
the per-pixel spread of the background across the stack and is used to
normalise the residual signal into an SNR.

Once whitefield and std are known, the SNR for frame :math:`i` is:

.. math::

   \mathrm{SNR}_i = \frac{D_i \cdot m - W}{\mathrm{max}(\sigma,\, \sigma_{min})}

where :math:`D_i` is the raw detector data, :math:`m` is the bad-pixel mask
(:math:`1` = good, :math:`0` = bad), :math:`W` is the whitefield,
:math:`\sigma` is the standard deviation, and :math:`\sigma_{min}` is a noise
floor controlled by the ``std_min`` argument of
:meth:`~cbclib_v2.CrystData.update_snr`.


Loading raw frames
------------------

Data loading is handled by the run-configuration layer (see :doc:`io_layer`).
A :class:`~cbclib_v2.BaseRun` object provides per-frame access; the resulting
array is wrapped directly into :class:`~cbclib_v2.CrystData`:

.. code-block:: python

    import cbclib_v2 as cbc

    cbc.set_cpu_config(num_threads=32)

    config = cbc.XFELConfig(
        data_dir="/gpfs/exfel/exp/SPB/202302/p004456/proc/r{0:04d}",
        hdf5_protocol="/path/to/agipd_protocol.json",
        file_pattern=r"CORR-R{0:04d}-JNGFR{1:02d}-S(\d{5})\.h5",
        geometry_file="/path/to/detector.geom",
        num_modules=8,
        starts_at=1,
    )

    run = cbc.open_run(373, config)
    indices = run.indices()          # all available frame indices

    frames = run.data(indices[:20])  # load first 20 frames
    data = cbc.CrystData(frames)     # wrap in CrystData


.. _background-mask:

Bad-pixel masking
-----------------

Before estimating the background, hot pixels and dead pixels should be
excluded. :meth:`~cbclib_v2.CrystData.update_mask` generates a boolean mask
(:math:`\text{True}` = good pixel) using one of four strategies:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Description
   * - ``'no-bad'``
     - All pixels are considered good (default).
   * - ``'all-bad'``
     - All pixels are masked — useful as a starting point for manual mask
       construction.
   * - ``'range'``
     - Pixels whose value lies outside ``[vmin, vmax)`` in **all** frames
       are masked. Catches permanently saturated or dead pixels.
   * - ``'snr'``
     - Pixels whose mean absolute SNR exceeds ``snr_max`` are masked.
       Requires an SNR array to already be present in the container.

The most common choice for an initial mask is ``'range'``, which rejects
pixels that are always outside the linear regime of the detector:

.. code-block:: python

    # Mask pixels that are always outside [0, 10 000 000) across all frames
    data = data.update_mask(method='range', vmin=0, vmax=10_000_000)

.. note::

   Masks from different sources can be added together by calling
   :meth:`~cbclib_v2.CrystData.import_mask` with ``update='multiply'``, which
   multiplies the existing and incoming masks element-wise.  For example, a
   static bad-pixel map from the detector calibration can be combined with a
   run-time ``'range'`` mask to reject both permanently dead pixels and
   transiently saturated ones.  See [Sadri2022]_ for an automatic bad-pixel
   mask generation approach for X-ray pixel detectors.

   .. code-block:: python

       import numpy as np

       # Load a pre-computed static bad-pixel map
       static_mask = np.load('static_bad_pixels.npy')

       # Combine with the range mask already on the container
       data = data.import_mask(static_mask, update='multiply')


.. _background-single:

Scenario 1 — single background
-------------------------------

The simplest scenario: the beam profile is sufficiently stable that a single
whitefield estimated from one batch of frames can be applied to all subsequent
frames in the run.

.. code-block:: python

    frames = run.data(indices[:20])
    data = cbc.CrystData(frames)
    data = data.update_mask(method='range', vmin=0, vmax=10_000_000)

    # Estimate whitefield and std jointly
    data = data.update_metadata(method='robust-mean-scale',
                                r0=0.5, r1=0.95, n_iter=2, lm=9.0)

    # Persist to disk for reuse
    cbc.write_hdf(data, 'results/metadata.h5',
                  cbc.H5Handler(data.protocol),
                  'mask', 'whitefield', 'std')

:meth:`~cbclib_v2.CrystData.update_metadata` with ``method='robust-mean-scale'``
applies a robust mean estimator across the frame stack: it iteratively fits a
Gaussian to the stack residuals and rejects outlier pixels. The result is
robust to the sparse diffraction peaks present in individual frames.

The keyword arguments ``r0``, ``r1``, ``n_iter``, and ``lm`` control the
FLkOS (Fast Least k-th Order Statistics) estimator:

* ``r0`` / ``r1`` — lower and upper bounds on the expected fraction of
  inliers.  Set ``r0`` conservatively (e.g. ``0.5``) and ``r1`` close to
  ``1.0`` for frames that are mostly background.
* ``n_iter`` — number of Gaussian-fitting iterations.
* ``lm`` — outlier threshold in units of the estimated standard deviation.

.. note::

   Alternative methods are available through ``update_metadata``:

   * ``'median-poisson'`` — pixel-wise median as the whitefield with Poisson
     noise (``std = √whitefield``).  Faster but less robust to bright outlier
     pixels.
   * ``'robust-mean-poisson'`` — robust mean whitefield combined with Poisson
     noise.

   Whitefield and std can also be computed independently via
   :meth:`~cbclib_v2.CrystData.update_whitefield` and
   :meth:`~cbclib_v2.CrystData.update_std`.


.. _background-multi:

Scenario 2 — multiple backgrounds
----------------------------------

When the beam profile drifts over the course of a run, a single whitefield is
not sufficient. Multiple background estimates are computed from different
frame intervals and merged into a :class:`~cbclib_v2.CrystMetadata` object
that retains the full whitefield stack.

.. code-block:: python

    data_containers = []
    for start, end in zip([0, 10, 20], [10, 20, 30]):
        frames = run.data(indices[start:end])
        d = cbc.CrystData(frames)
        d = d.update_mask(method='range', vmin=0, vmax=10_000_000)
        d = d.update_metadata(method='robust-mean-scale',
                              r0=0.5, r1=0.95, n_iter=2, lm=9.0)
        data_containers.append(d)

    # Merge into one CrystMetadata object
    metadata = cbc.CrystMetadata.from_data(*data_containers)

    cbc.write_hdf(metadata, 'results/metadata_multi.h5',
                  cbc.H5Handler(metadata.protocol))

:meth:`~cbclib_v2.CrystMetadata.from_data` combines the per-container
estimates as follows:

* The bad-pixel **mask** is the logical AND of all individual masks — a pixel
  must be good in every interval to be considered good overall.
* The **std** is the quadratic mean of the per-interval standard deviations:
  :math:`\sigma = \sqrt{\frac{1}{N}\sum_k \sigma_k^2}`.
* The individual **whitefields** are stacked into a ``(N, ...)`` array stored
  under the ``whitefields`` attribute.  The ``flatfield`` is initialised as
  their mean.

.. note::

   The resulting HDF5 file contains both the per-interval whitefield stack
   and the combined mask and std:

   .. code-block:: text

       /entry/crystallography/mask        {n_modules, ss, fs}
       /entry/crystallography/std         {n_modules, ss, fs}
       /entry/metadata/flatfield          {n_modules, ss, fs}
       /entry/metadata/whitefields        {N, n_modules, ss, fs}


.. _background-pca:

Scenario 3 — PCA decomposition of background variability
---------------------------------------------------------

When several background images are available, PCA captures the dominant modes
of background variation. Each new frame can then be matched to a linear
combination of these modes, producing a per-frame background estimate that
tracks the actual beam profile rather than using a fixed average.

Starting from a :class:`~cbclib_v2.CrystMetadata` object that contains a
``whitefields`` stack (from :ref:`Scenario 2 <background-multi>`), call
:meth:`~cbclib_v2.CrystMetadata.pca`:

.. code-block:: python

    handler = cbc.H5Handler(cbc.CrystMetadata.default_protocol())
    metadata = cbc.CrystMetadata(
        **cbc.read_hdf('results/metadata_multi.h5', handler,
                       'flatfield', 'mask', 'std', 'whitefields')
    )

    # Decompose whitefield variability into principal components
    metadata = metadata.pca()

    cbc.write_hdf(metadata, 'results/metadata_pca.h5',
                  cbc.H5Handler(metadata.protocol))

:meth:`~cbclib_v2.CrystMetadata.pca` computes the eigendecomposition of the
covariance matrix of zero-mean whitefield fluctuations
:math:`\Delta W_k = W_k - \bar{W}`.  It adds two new arrays to the container:

* ``eigen_fields`` — principal-component images, shape ``(N, ...)``.
* ``eigen_values`` — normalised eigenvalues (sum to 1) indicating the
  fraction of background variance explained by each component.

.. note::

   At least two whitefields are required to perform PCA.  For a reliable
   decomposition, the number of background estimates should exceed the
   expected number of significant background modes (typically 2–5 for FEL
   experiments).

   After PCA the HDF5 file contains:

   .. code-block:: text

       /entry/crystallography/mask        {n_modules, ss, fs}
       /entry/crystallography/std         {n_modules, ss, fs}
       /entry/metadata/eigen_fields       {N, n_modules, ss, fs}
       /entry/metadata/eigen_values       {N}
       /entry/metadata/flatfield          {n_modules, ss, fs}
       /entry/metadata/whitefields        {N, n_modules, ss, fs}


.. _background-snr:

Scenario 4 — scale background to data and compute SNR
------------------------------------------------------

With the background model in hand, each new batch of frames is attached to
the model and background-subtracted to yield SNR frames ready for streak
detection. There are two approaches depending on whether the beam profile is
assumed constant or is fitted per frame.

**Static subtraction** uses the flatfield (mean background image) as the
whitefield for every frame:

.. code-block:: python

    handler = cbc.H5Handler(cbc.CrystMetadata.default_protocol())
    metadata = cbc.CrystMetadata(
        **cbc.read_hdf('results/metadata_pca.h5', handler,
                       'eigen_field', 'eigen_value', 'flatfield', 'mask', 'std')
    )

    frames = run.data(indices[:20])

    # Subtract the static background (metadata.flatfield) from the data
    data = metadata.to_data(frames)
    data = data.update_snr(std_min=0.5)

**Dynamic subtraction** projects each frame onto the PCA basis to obtain a
per-frame whitefield before subtraction.  This compensates for shot-to-shot
intensity fluctuations and slow beam-profile drift:

.. code-block:: python

    # Fit each frame to the PCA basis using least squares
    projection = metadata.projection(frames, method='lsq')

    # Reconstruct the per-frame background: flatfield + linear combination of eigen_fields
    whitefields = metadata.project(projection)

    # Attach the per-frame background to the data
    data = metadata.to_data(frames, whitefield=whitefields)
    data = data.update_snr(std_min=0.5)

:meth:`~cbclib_v2.CrystMetadata.projection` solves a least-squares problem to
find the coefficients of the PCA components that best explain the residual
:math:`D_i - \bar{W}` for each frame :math:`i`.  The ``method`` argument
accepts:

* ``'lsq'`` — ordinary least squares.
* ``'robust-lsq'`` — least squares with FLkOS outlier rejection, more
  resistant to frames that contain bright diffraction streaks.

:meth:`~cbclib_v2.CrystMetadata.project` reconstructs the per-frame
background as :math:`\bar{W} + \sum_k c_{ik}\, e_k`, where :math:`c_{ik}`
are the projection coefficients and :math:`e_k` are the eigen fields.

:meth:`~cbclib_v2.CrystData.update_snr` then computes the SNR and stores it
in ``data.snr``.  The ``std_min`` floor prevents division by near-zero noise
estimates; a value of ``0.5`` is typical for photon-counting detectors.

.. note::

   Dynamic subtraction requires that ``eigen_fields`` and ``eigen_values`` are
   present in the :class:`~cbclib_v2.CrystMetadata` container, i.e. that
   :meth:`~cbclib_v2.CrystMetadata.pca` has been called first
   (:ref:`Scenario 3 <background-pca>`).  Static subtraction only needs
   ``flatfield``, ``mask``, and ``std``.

The resulting ``data.snr`` array is passed to the streak detector in the next
stage of the pipeline (see :doc:`key_concepts`).


Summary
-------

.. list-table:: Background subtraction workflow
   :header-rows: 1
   :widths: 5 35 60

   * - Step
     - Method
     - Purpose
   * - 1
     - :class:`~cbclib_v2.CrystData`
     - Wrap raw detector array.
   * - 2
     - :meth:`~cbclib_v2.CrystData.update_mask`
     - Build bad-pixel mask (``'range'``, ``'no-bad'``, …).
   * - 3
     - :meth:`~cbclib_v2.CrystData.update_metadata`
     - Estimate whitefield and std from a single batch of frames.
   * - 4
     - :meth:`~cbclib_v2.CrystMetadata.from_data`
     - Merge multiple background estimates into one model.
   * - 5
     - :meth:`~cbclib_v2.CrystMetadata.pca`
     - Decompose background variability into principal components.
   * - 6a
     - :meth:`~cbclib_v2.CrystMetadata.to_data`
     - Attach static background model to new data frames.
   * - 6b
     - :meth:`~cbclib_v2.CrystMetadata.projection` + :meth:`~cbclib_v2.CrystMetadata.project`
     - Fit and reconstruct a per-frame dynamic background.
   * - 7
     - :meth:`~cbclib_v2.CrystData.update_snr`
     - Compute per-frame SNR ready for streak detection.

References
----------

.. [Sadri2022] A. Sadri *et al.*, "Automatic bad-pixel mask maker for X-ray
               pixel detectors with application to serial crystallography,"
               *J. Appl. Cryst.* **55**, 1549–1561 (2022).
               https://doi.org/10.1107/S1600576722009815

See also
--------

* :doc:`io_layer` — loading raw frames from XFEL and SwissFEL runs.
* :doc:`geometry` — assembling multi-module detector data into a single image.
* For full API documentation see :doc:`api_containers`.
