Online detection
================

Online detection is the fast path for deciding, during an experiment, which raw
detector frames are worth saving. Modern FEL facilities can deliver diffraction
patterns at MHz-class repetition rates on high-resolution detectors such as a
16M Jungfrau. At that data rate, detector output can exceed the practical disk
capacity of an experiment: saving every frame is often impossible or wasteful.
A quick hit finder that keeps likely crystal hits and rejects empty frames is
therefore part of the experiment strategy, not only a convenience for later
analysis.

The goal is speed and recall. Online detection should be fast enough to run
near the detector data stream and permissive enough not to throw away rare
crystal hits. It is not intended to provide the final background subtraction
used for intensity scaling, nor does it enforce the CBC-specific requirement
that diffraction signal should have a streaky shape.

The detector estimates a radial background directly from the frames being
searched. Pixels at similar scattering radius are expected to share a comparable
diffuse background level, so the frame can be reduced to compact radial profiles
of mean intensity and standard deviation. Signal pixels are then pixels whose
residual above that radial background exceeds an SNR threshold.

This strategy is inspired by Cheetah's Peakfinder8 hit finder, which uses
iterative radial mean/std estimation, rejects bright outliers from the radial
background, then searches for connected above-threshold pixel regions. See the
`Cheetah SFX hit-finding notes`_ and the referenced `peakfinder8.cpp`_
implementation for the original SFX peak-finding context.

.. _Cheetah SFX hit-finding notes: https://www.desy.de/~barty/cheetah/Cheetah/SFX_hitfinding.html
.. _peakfinder8.cpp:
   https://github.com/omdevteam/om/blob/features/mfx101210926/src/cython/peakfinder8.cpp


When to use it
--------------

Use :class:`~cbclib_v2.OnlineDetector` for immediate data reduction during
acquisition or early inspection:

* Selecting frames that look like hits.
* Saving likely hits when the full detector stream is too large.
* Locating bright candidate regions for quick inspection.
* Checking that the geometry, mask, and radial background parameters are
  sensible.
* Pre-filtering frames before a more expensive streak-detection or indexing
  pass.

Do not treat online profiles as a calibrated reusable flatfield. For careful
background subtraction, build a persistent :class:`~cbclib_v2.CrystMetadata`
object as described in :doc:`background_subtraction`. Online detection estimates
the background from the same data it thresholds, which is exactly what makes it
fast and useful for hit finding, but also what makes it less stable than a
dedicated background model.

Also expect over-detection. The online detector labels connected radial-SNR
outliers; it does not check whether those outliers form the elongated CBC
streaks needed for indexing and intensity analysis. Non-streak artefacts,
compact Bragg-like spots, detector effects, or sample-environment scatter can
therefore pass the online hit criterion. That is usually acceptable for a
save-the-hit decision, where false positives cost disk space, while false
negatives can permanently discard useful diffraction data.


Data flow
---------

The online path has four pieces:

.. code-block:: text

   CrystFEL geometry
          │
          ▼
   Detector.radial_index(center, n_bins)
          │  integer radius bin per detector pixel
          ▼
   CrystData.online_detector(structure, radial_index, n_bins)
          │
          ├── profiles(clip_snr, n_iter)
          │       radial whitefield/std/counts per frame
          │
          └── detect_regions(profiles, min_snr, npts)
                  connected signal regions

The radial index map is geometry-dependent but data-independent, so it can be
created once per detector geometry and beam-centre choice. The radial profiles
are data-dependent and are recomputed for the frames being searched.


Geometry and radial bins
------------------------

The geometry step starts from a CrystFEL ``.geom`` file:

.. code-block:: python

   import cbclib_v2 as cbc
   from cbclib_v2.label import Structure

   geometry = cbc.read_crystfel("detector.geom")
   radial_index = geometry.radial_index(center=(512.0, 512.0), n_bins=1024)

``center`` is the direct-beam position in CrystFEL lab-frame pixel
coordinates. It is not a module-local array coordinate. The helper methods
:meth:`~cbclib_v2.Detector.pixel_map`, :meth:`~cbclib_v2.Detector.radii`, and
:meth:`~cbclib_v2.Detector.radial_index` all use the same detector-plane
coordinate convention, so the assembled image, radius map, and radial bins stay
aligned.

Native CPU/CUDA geometry kernels mark non-panel pixels as ``-1`` in the radial
index map. Those pixels are ignored by the profile kernels. Valid panel pixels
fall in the range ``0`` to ``n_bins - 1``.

The number of bins controls the tradeoff between radial resolution and
statistical support. More bins preserve sharper radial structure, such as
powder rings, but each bin receives fewer pixels. Fewer bins produce smoother
and more stable estimates but can wash out sharp background features.


Profile estimation
------------------

Create the detector from a :class:`~cbclib_v2.CrystData` container containing
raw detector counts:

.. code-block:: python

   structure = Structure([1, 1], 1)
   online = data.online_detector(structure, radial_index, n_bins=1024)
   profiles = online.profiles(clip_snr=4.0, n_iter=5, std_min=1.0)

For each frame, :meth:`~cbclib_v2.OnlineDetector.profiles` computes compact
arrays:

* ``whitefield``: radial mean intensity, shape ``(n_frames, n_bins)``.
* ``std``: radial standard deviation, shape ``(n_frames, n_bins)``.
* ``counts``: number of valid pixels contributing to each bin.

The estimator is iterative. After a first radial mean/std pass, pixels above
``mean + clip_snr * std`` are excluded and the profile is recomputed. Repeating
this a few times keeps sparse diffraction peaks, hot regions, and jet artefacts
from inflating the background model.

``std_min`` is a noise floor. It prevents very quiet radial bins from producing
unreasonably large SNR values from tiny absolute residuals.


Region detection
----------------

Once the profiles are available, signal detection is a residual-SNR test:

.. math::

   \mathrm{SNR}(p) =
   \frac{D(p) - W_i(r(p))}
        {\max(\sigma_i(r(p)),\, \sigma_{\min})}

where :math:`D(p)` is the raw value of pixel :math:`p`, :math:`r(p)` is its
radial bin, and :math:`W_i` and :math:`\sigma_i` are the per-frame radial
whitefield and standard deviation.

Pixels with residual SNR at least ``min_snr`` become foreground. Connected
foreground pixels are labeled with the supplied :class:`~cbclib_v2.label.Structure`:

.. code-block:: python

   regions = online.detect_regions(profiles, min_snr=5.0, npts=3, std_min=1.0)

``npts`` removes small connected components. In hit-finding terms it is the
minimum number of connected above-threshold pixels required for a candidate
region. Larger values reject isolated noise but can miss weak or fragmented
features.


Relation to streak detection
----------------------------

Online detection answers a different question from the dedicated CBC streak
detector in :doc:`streak_detection`.

Online detection identifies:

* Frames with connected pixels that stand out from the radial background.
* Candidate signal regions in the raw detector data.

Streak detection identifies:

* Line-like CBC streaks among those candidate regions.
* Fitted line parameters used for indexing.
* Detections that survive because they match the expected streaky diffraction
  morphology.

A typical early-stage workflow is therefore:

.. code-block:: python

   profiles = online.profiles(clip_snr=4.0, n_iter=5, std_min=1.0)
   regions = online.detect_regions(profiles, min_snr=5.0, npts=3)

   hit_frames = regions.index

Then, for frames that pass the online hit criterion, build or apply a more
stable background model and run :class:`~cbclib_v2.StreakDetector` or
:class:`~cbclib_v2.streak_finder.PatternStreakFinder`.

The second stage is where morphology matters. A frame can be worth saving even
when the online detector over-labels it, but it should not contribute to
indexing or intensity scaling until a streak-aware detector and a better
background model have filtered the candidate signal.


Tuning notes
------------

Start with a moderate ``min_snr`` and ``npts`` and inspect the labeled regions.
If many isolated noise pixels survive, increase ``min_snr``, increase ``npts``,
or raise ``std_min``. If weak diffraction is missed, lower ``min_snr`` or use
more profile iterations so bright outliers are removed from the radial
background estimate.

Geometry quality matters. Radial background subtraction assumes that pixels in
the same radial bin correspond to similar scattering angle. A stale beam centre
or geometry can smear sharp radial features across bins and make the threshold
too high in some regions and too low in others.
