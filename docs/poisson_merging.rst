Poisson merging
===============

Repeated observations of symmetry-equivalent reflections constrain a common
intensity model across a scan. :class:`~cbclib_v2.scaler.PoissonMerger` fits that
model directly to detector photon counts, jointly refining signed reflection
coefficients, positive pattern scales, and a smooth pupil illumination shape.
Geometry, profile widths and estimated background remain fixed from the preceding
scaling procedure.

Count model
-----------

For contribution :math:`j` to measured pixel :math:`p`, the signal predictor is

.. math::

   s_{pj} = a_{f(j)} I_{h(j)} P_{f(j)}(u_{pj}, v_{pj}) G_{pj}.

Here :math:`a_f` is the positive pattern scale, :math:`I_h` is a signed shared
reflection coefficient, and :math:`G_{pj}` is the fixed geometric streak profile.
``MergeState`` stores ``asinh_hkl`` as its fitted intensity parameter and exposes
physical coefficients through ``I_hkl = sinh(asinh_hkl)``, as in local streak scaling.
The pupil coordinates :math:`(u,v)` are the affine coordinates of the unprojected
incident-vector candidate :math:`(k_x,k_y)` within its rectangular pupil.
For pupil illumination, coordinates outside the aperture are clipped to the
unit square, extending the intensity by its nearest boundary value. The geometric
profile uses the original incident-vector candidates and supplies attenuation
away from the Bragg condition and pupil support.

Overlapping contributions are added before the existing softplus count
transformation:

.. math::

   \lambda_p = \epsilon + \operatorname{softplus}
       \left(b_p + \sum_j s_{pj}\right),
   \qquad
   L = \operatorname{mean}_p\left[\lambda_p-y_p\log\lambda_p\right].

Minimising :math:`L` maximises the Poisson likelihood, omitting factorial terms
independent of the fitted parameters. Each unique measured pixel contributes
once. The background :math:`b_p` is used in the same additive predictor convention
as local scaling, and :math:`\epsilon` is ``ScalerModel.rate_floor``.

Pupil shape and scale
---------------------

The pupil amplitude uses nonconstant orthonormal shifted Legendre products
:math:`B_k` on the unit square:

.. math::

   P_f(u,v) =
   \frac{\left[1 + \sum_{k>0} c_{fk} B_k(u,v)\right]^2}
        {1 + \sum_{k>0} c_{fk}^2},
   \qquad
   \int_0^1\int_0^1 P_f(u,v)\,du\,dv = 1.

Normalisation is algebraic and uses no numerical quadrature. In physical
:math:`(k_x,k_y)` coordinates the integral equals the rectangular pupil area,
which stays fixed during fitting. Adjusting pupil shape therefore preserves
integrated illumination separately from the pattern scale. Squaring gives a
nonnegative, smooth intensity with possible zeros; an amplitude degree of three
produces an intensity degree up to six.

One coefficient row shares a pupil across the scan. One row per pattern fits
independent pupil shapes, each with the same integral constraint. Zero
coefficients give uniform illumination. ``ScanState.normalise()`` fixes the
remaining scale/reflection ambiguity by setting the geometric mean pattern
scale to one. Disconnected observation components require separate scale
conventions, as with ``MergeState``.

Clipping bounds the basis values outside the aperture without changing the
analytic integral inside it. Geometry is fixed during merging, so clipping
introduces no nonsmoothness with respect to the fitted intensity, scale or pupil
coefficients. Joint geometry refinement would require reconsidering the
coordinate derivative at the pupil boundary.

Fit a scan
----------

Start from the detector data, predicted streak-pixel associations, Miller
indices, resolved setup and fitted widths used in local scaling. They must cover
the complete scan and use the same compact pattern indices. Use ``OverlapData``
when multiple streaks can contribute to the same measured pixel; ``ScalerData``
is suitable for non-overlapping observations.

.. code-block:: python

   import jax
   import optax
   from cbclib_v2 import scaler
   from cbclib_v2.annotations import JaxNumPy as xp, RealArray
   from cbclib_v2.scripts import optimisation_loop

   model = scaler.ScalerModel()
   scaling = scaler.OverlapData.import_data(
       cryst_data, signal_ids, miller, detector, xp,
   )
   streaks = scaler.StreakData(scaling, model.init_model(scaling, resolved, xp))
   basis = scaler.LegendreFactory.create(degree=3, xp=xp)
   data = scaler.ScanData.from_streaks(
       streaks, scaled_state.log_sigma, scaler.PointGroup('mmm'), basis,
   )

   pupil = scaler.PupilState.default(data.basis, n_patterns=1, xp=xp)
   # Use n_patterns=data.n_patterns for independent pattern pupils.
   initial = scaler.ScanState.from_data(data, pupil)
   merger = scaler.PoissonMerger(model=model)

   loss_grad_fn = jax.jit(jax.value_and_grad(merger, argnums=1))

   def gradient(state: scaler.ScanState) -> tuple[RealArray, scaler.ScanState]:
       return loss_grad_fn(data, state)

   num_steps = 1500
   schedule = optax.cosine_decay_schedule(0.04, num_steps, alpha=0.01)
   optimiser = optax.adam(schedule)
   fitted, stats = optimisation_loop(
       gradient, initial, optimiser, schedule, num_steps, trace_every=100,
   )
   fitted = fitted.normalise()

   hkl = data.hkl
   coefficients = fitted.merge.I_hkl
   pattern_scales = xp.exp(fitted.merge.log_scale)
   expected_counts = merger.expected(data, fitted)
   deviance = merger.poisson_deviance(data, fitted, xp)

``ScanState.from_data`` uses measured counts minus background, assuming a uniform
unit pupil. All pattern scales start at one, with ``log_scale`` filled with zeros.
Each reflection is initialised from
``sum(signal * profile) / sum(profile)``, grouped by canonical HKL without
pattern-scale correction. These measured-signal means are approximate predictor amplitudes;
they do not undo geometric attenuation or the softplus transformation. The full
Poisson fit refines both scales and intensities. An existing ``MergeState`` can
instead be supplied directly with
``ScanState(merge=merge, pupil=pupil)``, provided its reflection order matches
``data.hkl`` and its scale array covers ``data.n_patterns``.

The returned shared coefficients are signed amplitudes inside the softplus
predictor. They are distinct from the profile-normalised rate contrasts exported
by local scaling. ``expected_counts`` and ``deviance`` follow unique measured-pixel
order; ``coefficients`` follows ``data.hkl``. The learning rate and step count
above are starting values to tune for the scan, rather than a convergence rule.

Ownership
---------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Object
     - Responsibility
   * - ``RectangularPupil`` / ``IntensityModel``
     - Pupil coordinates and incident-vector geometry.
   * - ``ScanData``
     - Detector observations, symmetry binding, fixed widths and the Legendre pupil basis.
   * - ``MergeState``
     - Shared reflection coefficients, positive pattern scales and their scale convention.
   * - ``PupilState``
     - Fitted shape coefficients and analytically normalised pupil evaluation.
   * - ``ScanState``
     - Parameter composition, initialisation and state-dependent point lookup.
   * - ``PoissonMerger``
     - Composition of the signal model and evaluation of the shared Poisson likelihood.
   * - ``optimisation_loop``
     - Parameter updates, schedules and progress statistics.
