"""Polar / tangential streak detection for convergent-beam crystallography.

This module hosts a third, physics-grounded streak-detection route that is
developed independently of the existing :class:`~cbclib_v2.StreakDetector`
(seed-and-grow) and :class:`~cbclib_v2.RegionDetector` (connected regions).

Two things live here:

* :func:`calibrate_center` — a cheap, one-time estimator of the global beam
  centre used for radial binning / the polar remap.  It seeds provisional
  streaks locally (no connected-component labeling), then solves for the centre
  from the fact that every streak's normal passes through it.  This is the
  first piece to be built and tested.

* :class:`PolarStreakFinder` — a stub for the per-frame polar-domain detector.
  Its algorithm (polar remap → cross-streak matched integration → along-azimuth
  run detection → geometric endpoints) is the *next* goal; the class currently
  exposes the interface and clearly-marked ``TODO(physics)`` hooks so the real
  machinery can drop in without changing the public contract.

Coordinate convention throughout: ``x`` is the column (fast) axis and ``y`` is
the row (slow) axis; images are indexed ``img[y, x]`` and a centre is the pair
``(cx, cy)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter

from .streaks import Streaks

__all__ = [
    "CenterResult",
    "ProvisionalStreaks",
    "calibrate_center",
    "simulate_tangential_streaks",
    "PolarStreakFinder",
]

Array = np.ndarray
Center = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Results containers
# --------------------------------------------------------------------------- #
@dataclass
class ProvisionalStreaks:
    """Local streaks found by the center-free provisional pass.

    Attributes:
        midpoints: ``(M, 2)`` array of streak midpoints ``(x, y)`` — the
            intensity-weighted centroid along the best probe line, *not* the
            seed pixel.
        tangents: ``(M, 2)`` array of unit tangent directions ``(tx, ty)``.
        anisotropy: ``(M,)`` orientation contrast in ``[0, 1]`` (how much the
            best probe angle beats the median angle); higher is more line-like.
        peak: ``(M,)`` integrated photon weight along the best probe line.
        frame: ``(M,)`` index of the frame each streak came from.
    """

    midpoints   : Array
    tangents    : Array
    anisotropy  : Array
    peak        : Array
    frame       : Array

    def __len__(self) -> int:
        return int(self.midpoints.shape[0])


@dataclass
class CenterResult:
    """Output of :func:`calibrate_center`.

    Attributes:
        center: Refined beam centre ``(cx, cy)`` in pixels.
        seed: The seed centre the calibration started from.
        streaks: The provisional streaks used for the solve.
        weights: Final per-streak weights (resolution weight x robust weight).
        residuals: Signed normal residual ``(C - m) . t`` per streak, in pixels;
            zero means the streak's normal passes exactly through the centre.
        inliers: Boolean mask of streaks kept by the robust reweighting.
        condition: Condition number of the 2x2 normal matrix (large => the
            streak orientations lack spread and the solve is ill-conditioned).
        history: Centre estimate after each outer refinement iteration.
    """

    center      : Center
    seed        : Center
    streaks     : ProvisionalStreaks
    weights     : Array
    residuals   : Array
    inliers     : Array
    condition   : float
    history     : List[Center] = field(default_factory=list)

    @property
    def rms_residual(self) -> float:
        """Weighted RMS of the inlier normal residuals (pixels)."""
        m = self.inliers
        if not np.any(m):
            return float("nan")
        w = self.weights[m]
        r = self.residuals[m]
        return float(np.sqrt(np.sum(w * r * r) / np.sum(w)))


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _radius_map(shape: Tuple[int, int], center: Center) -> Array:
    ys, xs = np.indices(shape, dtype=np.float64)
    return np.hypot(xs - center[0], ys - center[1])


def _radial_flatten(img: Array, center: Center, n_bins: int) -> Array:
    """Re-whiten an image radially (peakfinder8-style background flattening).

    Returns ``(img - mean_r) / std_r`` where the per-radius mean and std are
    estimated in ``n_bins`` concentric rings about *center*.  This removes the
    residual radial trend so that seeding is not dominated by the bright,
    low-resolution region near the centre.
    """
    if n_bins <= 0:
        return img
    r = _radius_map(img.shape, center)
    rmax = float(r.max()) + 1e-6
    idx = np.minimum((r / rmax * n_bins).astype(np.intp), n_bins - 1).ravel()
    flat = img.ravel().astype(np.float64)

    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    mean = np.bincount(idx, weights=flat, minlength=n_bins) / counts
    meansq = np.bincount(idx, weights=flat * flat, minlength=n_bins) / counts
    std = np.sqrt(np.maximum(meansq - mean * mean, 0.0))
    std = np.maximum(std, 1e-6)

    return ((flat - mean[idx]) / std[idx]).reshape(img.shape)


def _find_seeds(flat: Array, center: Center, r_bounds: Tuple[float, float],
                n_seeds: int, min_distance: float, seed_percentile: float) -> Array:
    """Return up to *n_seeds* ``(x, y)`` local maxima inside the annulus.

    Local maxima of *flat* that fall within ``r_bounds`` about *center*, are at
    least *min_distance* apart (greedy non-max suppression), and exceed the
    *seed_percentile* of the in-annulus values.
    """
    r = _radius_map(flat.shape, center)
    annulus = (r >= r_bounds[0]) & (r <= r_bounds[1])
    if not np.any(annulus):
        return np.empty((0, 2), dtype=np.float64)

    size = max(int(round(min_distance)), 1)
    is_max = flat >= maximum_filter(flat, size=2 * size + 1, mode="nearest")
    thresh = np.percentile(flat[annulus], seed_percentile)
    cand = annulus & is_max & (flat >= thresh)

    ys, xs = np.nonzero(cand)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    vals = flat[ys, xs]
    order = np.argsort(vals)[::-1]
    xs, ys = xs[order], ys[order]

    kept: List[Tuple[float, float]] = []
    min_d2 = float(min_distance) ** 2
    for x, y in zip(xs, ys):
        if all((x - kx) ** 2 + (y - ky) ** 2 >= min_d2 for kx, ky in kept):
            kept.append((float(x), float(y)))
            if len(kept) >= n_seeds:
                break
    return np.asarray(kept, dtype=np.float64)


def _probe_streak(img: Array, seed: Sequence[float], na_length: float,
                  n_angles: int, n_samples: int
                  ) -> Tuple[Array, Array, float, float]:
    """Fit a local streak at *seed* with a rotating line probe.

    Rotates a line of length *na_length* through *n_angles* orientations and
    picks the one with the largest integrated (positive) photon weight — a
    local Radon / oriented matched filter.  The midpoint is the intensity
    weighted centroid *along* the winning line, not the seed pixel.

    Returns ``(midpoint, tangent, anisotropy, peak)``.
    """
    sx, sy = float(seed[0]), float(seed[1])
    s = np.linspace(-na_length / 2.0, na_length / 2.0, n_samples)
    thetas = np.linspace(0.0, np.pi, n_angles, endpoint=False)

    cos, sin = np.cos(thetas), np.sin(thetas)
    # sample grid: (n_angles, n_samples)
    rows = sy + np.outer(sin, s)
    cols = sx + np.outer(cos, s)
    samples = map_coordinates(img, [rows.ravel(), cols.ravel()], order=1,
                              mode="constant", cval=0.0).reshape(rows.shape)
    pos = np.clip(samples, 0.0, None)
    weight = pos.sum(axis=1)

    best = int(np.argmax(weight))
    peak = float(weight[best])
    med = float(np.median(weight))
    anisotropy = 0.0 if peak <= 0 else float((peak - med) / peak)

    # parabolic refinement of the orientation around the discrete maximum
    theta = thetas[best]
    wl, wc, wr = weight[(best - 1) % n_angles], weight[best], weight[(best + 1) % n_angles]
    denom = wl - 2.0 * wc + wr
    if denom < 0:
        theta = theta + 0.5 * (wl - wr) / denom * (np.pi / n_angles)
    tx, ty = float(np.cos(theta)), float(np.sin(theta))

    # midpoint = weighted centroid along the winning line
    wprofile = pos[best]
    wsum = wprofile.sum()
    s_bar = float((wprofile * s).sum() / wsum) if wsum > 0 else 0.0
    midpoint = np.array([sx + s_bar * tx, sy + s_bar * ty], dtype=np.float64)
    tangent = np.array([tx, ty], dtype=np.float64)
    return midpoint, tangent, anisotropy, peak


def _solve_center(mid: Array, tan: Array, w: Array) -> Tuple[Array, float]:
    """Weighted least-squares intersection of the streak normals.

    Solves ``min_C sum_i w_i [ (C - m_i) . t_i ]^2`` via the normal equations
    ``(sum w_i t_i t_i^T) C = sum w_i t_i (t_i . m_i)``.
    """
    A = np.einsum("i,ij,ik->jk", w, tan, tan)
    tm = np.einsum("ij,ij->i", tan, mid)
    b = np.einsum("i,i,ij->j", w, tm, tan)
    cond = float(np.linalg.cond(A))
    C, *_ = np.linalg.lstsq(A, b, rcond=None)
    return C, cond


# --------------------------------------------------------------------------- #
# Public: centre calibration
# --------------------------------------------------------------------------- #
def calibrate_center(
    snr: Array,
    center: Center,
    na_length: float,
    r_bounds: Tuple[float, float],
    *,
    n_seeds: int = 20,
    min_distance: Optional[float] = None,
    n_angles: int = 64,
    n_samples: Optional[int] = None,
    weight_exponent: float = 1.0,
    weight_fn: Optional[Callable[[Array], Array]] = None,
    min_anisotropy: float = 0.1,
    flatten_bins: int = 128,
    n_irls: int = 3,
    huber_delta: Optional[float] = None,
    n_refine: int = 1,
    seed_percentile: float = 99.0,
) -> CenterResult:
    """Estimate the global beam centre from tangential streaks.

    One-time, per-run calibration (the centre is assumed stable across the run).
    Seeded by the geometry-file centre *center* (treated as approximate).  The
    pipeline is:

    1. Radially flatten each SNR frame about *center* and seed the strongest,
       well-separated local maxima inside the resolution annulus *r_bounds*
       (removes the low-resolution / bright-artifact seeding bias).
    2. Fit a local streak at each seed with a rotating line probe of length
       *na_length* (proportional to the numerical aperture); keep streaks whose
       orientation contrast exceeds *min_anisotropy*.
    3. Weight each streak by its resolution ``r0 = |m - center|`` (high-res
       streaks constrain the centre best) and solve the weighted normal
       intersection, with Huber IRLS reweighting to reject outliers.
    4. Optionally refine ``center -> C`` and repeat *n_refine* times.

    Args:
        snr: SNR frame or stack, shape ``(H, W)`` or ``(N, H, W)`` in a single
            coordinate system (assemble multi-module data first).
        center: Seed centre ``(cx, cy)`` in pixels (e.g. from the .geom file).
        na_length: Probe line length in pixels, ``~`` the expected streak length
            (set by the convergence angle / numerical aperture).
        r_bounds: ``(r_min, r_max)`` seeding annulus in pixels about *center* —
            a mid/high-resolution band where rings are separated and streak
            normals are reliable.
        n_seeds: Maximum seeds per frame.
        min_distance: Non-max-suppression distance in pixels (default
            ``na_length`` so seeds are at least one streak-length apart).
        n_angles: Number of probe orientations tested per seed.
        n_samples: Samples along the probe line (default ``int(na_length) + 1``).
        weight_exponent: Resolution weight is ``(r0 / median(r0)) ** exponent``.
        weight_fn: Optional callable ``r0 -> weight`` overriding *weight_exponent*
            (use once the weight law is fit empirically from residual-vs-r0).
        min_anisotropy: Minimum orientation contrast to accept a seed as a
            streak (rejects isotropic blobs).
        flatten_bins: Radial bins for background flattening (``0`` disables).
        n_irls: Huber IRLS reweighting iterations for outlier rejection.
        huber_delta: Huber threshold in pixels (default ``1.4826 * MAD`` of the
            residuals, i.e. a robust sigma).
        n_refine: Outer centre-refinement iterations (``center -> C``).
        seed_percentile: Intensity percentile (within the annulus) below which
            candidate maxima are ignored.

    Returns:
        :class:`CenterResult` with the refined centre, the provisional streaks,
        per-streak weights / residuals / inlier mask, and the solve condition
        number.

    Raises:
        ValueError: If fewer than two usable streaks are found.
    """
    frames = snr[None] if snr.ndim == 2 else snr
    if min_distance is None:
        min_distance = na_length
    if n_samples is None:
        n_samples = int(na_length) + 1

    seed_center = (float(center[0]), float(center[1]))

    mids: List[Array] = []
    tans: List[Array] = []
    anis: List[float] = []
    peaks: List[float] = []
    fidx: List[int] = []

    for f, frame in enumerate(frames):
        frame = np.asarray(frame, dtype=np.float64)
        flat = _radial_flatten(frame, seed_center, flatten_bins)
        seeds = _find_seeds(flat, seed_center, r_bounds, n_seeds, min_distance,
                            seed_percentile)
        for seed in seeds:
            m, t, a, p = _probe_streak(frame, seed, na_length, n_angles, n_samples)
            if a < min_anisotropy:
                continue
            mids.append(m)
            tans.append(t)
            anis.append(a)
            peaks.append(p)
            fidx.append(f)

    if len(mids) < 2:
        raise ValueError(
            f"calibrate_center found only {len(mids)} usable streak(s); "
            "loosen min_anisotropy / seed_percentile, widen r_bounds, or pool "
            "more frames."
        )

    streaks = ProvisionalStreaks(
        midpoints=np.asarray(mids), tangents=np.asarray(tans),
        anisotropy=np.asarray(anis), peak=np.asarray(peaks),
        frame=np.asarray(fidx, dtype=np.intp),
    )

    C = np.asarray(seed_center, dtype=np.float64)
    history: List[Center] = []
    weights = np.ones(len(streaks))
    residuals = np.zeros(len(streaks))
    inliers = np.ones(len(streaks), dtype=bool)
    condition = float("inf")

    for _ in range(max(n_refine, 0) + 1):
        r0 = np.linalg.norm(streaks.midpoints - C, axis=1)
        if weight_fn is not None:
            w_res = np.asarray(weight_fn(r0), dtype=np.float64)
        else:
            med = np.median(r0) if np.median(r0) > 0 else 1.0
            w_res = (r0 / med) ** weight_exponent

        w = w_res.copy()
        for _it in range(max(n_irls, 1)):
            C, condition = _solve_center(streaks.midpoints, streaks.tangents, w)
            residuals = np.einsum("ij,ij->i", C - streaks.midpoints, streaks.tangents)
            if huber_delta is None:
                mad = np.median(np.abs(residuals - np.median(residuals)))
                delta = max(1.4826 * mad, 1e-3)
            else:
                delta = huber_delta
            robust = np.where(np.abs(residuals) <= delta, 1.0,
                              delta / np.maximum(np.abs(residuals), 1e-9))
            w = w_res * robust
            inliers = np.abs(residuals) <= delta
        history.append((float(C[0]), float(C[1])))

    return CenterResult(
        center=(float(C[0]), float(C[1])), seed=seed_center, streaks=streaks,
        weights=w, residuals=residuals, inliers=inliers, condition=condition,
        history=history,
    )


# --------------------------------------------------------------------------- #
# Synthetic data (for testing / notebooks)
# --------------------------------------------------------------------------- #
def simulate_tangential_streaks(
    center: Center,
    shape: Tuple[int, int] = (1024, 1024),
    n_streaks: int = 40,
    r_bounds: Tuple[float, float] = (200.0, 450.0),
    length: Tuple[float, float] = (12.0, 40.0),
    width: float = 1.5,
    amplitude: Tuple[float, float] = (3.0, 10.0),
    noise: float = 1.0,
    taper: float = 0.6,
    n_outliers: int = 0,
    seed: Optional[int] = None,
) -> Array:
    """Simulate an SNR-like frame of streaks tangential to circles about *center*.

    Each streak is placed tangent to a circle of random radius in *r_bounds* and
    given a **non-uniform along-length intensity** (controlled by *taper*) and a
    Gaussian cross-profile of half-width *width*, reproducing the real-data
    conditions (variable length, intensity taper) that stress the estimator.

    Args:
        center: True beam centre ``(cx, cy)`` to recover.
        shape: Frame shape ``(H, W)``.
        n_streaks: Number of tangential streaks.
        r_bounds: Radius range for the tangent circles (pixels).
        length: Min/max streak length (pixels).
        width: Gaussian cross-profile half-width (pixels).
        amplitude: Min/max peak SNR of a streak.
        noise: Std of additive Gaussian noise.
        taper: Along-length intensity asymmetry in ``[0, 1)``; ``0`` = uniform.
        n_outliers: Number of randomly oriented (non-tangential) decoy streaks.
        seed: RNG seed.

    Returns:
        Float image of shape *shape*.
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, noise, size=shape)
    H, W = shape
    pad = int(np.ceil(4.0 * width)) + 1

    def draw(mx: float, my: float, tx: float, ty: float, L: float, amp: float) -> None:
        s = np.linspace(-L / 2.0, L / 2.0, int(L) + 1)
        env = amp * (1.0 + taper * (s / (L / 2.0)))  # non-uniform along length
        for si, ei in zip(s, env):
            px, py = mx + si * tx, my + si * ty
            x0, x1 = max(int(px) - pad, 0), min(int(px) + pad + 1, W)
            y0, y1 = max(int(py) - pad, 0), min(int(py) + pad + 1, H)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            img[y0:y1, x0:x1] += ei * np.exp(
                -((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * width ** 2))

    for _ in range(n_streaks):
        r0 = rng.uniform(*r_bounds)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        px, py = center[0] + r0 * np.cos(phi), center[1] + r0 * np.sin(phi)
        tx, ty = -np.sin(phi), np.cos(phi)  # tangential direction
        draw(px, py, tx, ty, rng.uniform(*length), rng.uniform(*amplitude))

    for _ in range(n_outliers):
        r0 = rng.uniform(*r_bounds)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        px, py = center[0] + r0 * np.cos(phi), center[1] + r0 * np.sin(phi)
        ang = rng.uniform(0.0, np.pi)  # random (non-tangential) orientation
        draw(px, py, np.cos(ang), np.sin(ang), rng.uniform(*length),
             rng.uniform(*amplitude))

    return img


# --------------------------------------------------------------------------- #
# Per-frame polar detector — STUB (next goal)
# --------------------------------------------------------------------------- #
@dataclass
class PolarStreakFinder:
    """Per-frame polar-domain streak detector (interface + hooks; WIP).

    The detection algorithm — polar remap about the calibrated centre,
    cross-streak matched integration, along-azimuth run detection, and geometric
    endpoints — is the next milestone.  For now :meth:`detect` returns an empty
    :class:`~cbclib_v2.Streaks` so the scaffold and comparison notebook run
    end-to-end; each ``TODO(physics)`` hook below is where a stage plugs in.

    Attributes:
        snr: SNR frame stack, shape ``(N, H, W)`` (single coordinate system).
        center: Calibrated global beam centre ``(cx, cy)`` from
            :func:`calibrate_center`.
        num_modules: Number of detector modules (1 => flat single-module).
    """

    snr         : Array
    center      : Center
    num_modules : int = 1

    # -- TODO(physics) hooks -------------------------------------------------
    def to_polar(self, frame: Array) -> Array:
        """Remap a frame to ``(phi, r)`` about :attr:`center`."""
        raise NotImplementedError("polar remap: next milestone")

    def azimuthal_matched_filter(self, polar: Array) -> Array:
        """Cross-streak (radial) matched integration + along-phi run detection."""
        raise NotImplementedError("azimuthal matched filter: next milestone")

    def endpoints_from_run(self, *args, **kwargs) -> Array:
        """Geometric endpoints from the azimuthal run extent."""
        raise NotImplementedError("geometric endpoints: next milestone")

    # -- current contract ----------------------------------------------------
    def detect(self) -> Streaks:
        """Return detected streaks (currently an empty placeholder)."""
        frames = self.snr[None] if self.snr.ndim == 2 else self.snr
        _ = frames  # placeholder; real per-frame detection lands here
        return Streaks(index=np.empty((0,), dtype=np.intp),
                       lines=np.empty((0, 4), dtype=np.float64))
