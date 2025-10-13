from typing import Tuple
from math import prod
from .._src.annotations import (Array, ArrayNamespace, BoolArray, IntArray, JaxNumPy, RealArray,
                                Shape)

def arange(shape: Shape, xp: ArrayNamespace = JaxNumPy) -> IntArray:
    return xp.reshape(xp.arange(prod(shape), dtype=int), shape)

def safe_sqrt(x: Array, xp: ArrayNamespace = JaxNumPy) -> Array:
    _x = xp.where(x <= 0.0, 0.0, x)
    return xp.where(x <= 0.0, 0.0, xp.sqrt(_x))

def safe_divide(x: Array, y: Array, xp: ArrayNamespace = JaxNumPy) -> Array:
    _y = xp.where(y == 0.0, 1.0, y)
    return xp.where(y == 0.0, 0.0, x / _y)

def kxy_to_k(kxy: RealArray, xp: ArrayNamespace = JaxNumPy) -> RealArray:
    kz = safe_sqrt(1 - xp.sum(kxy[..., :2]**2, axis=-1), xp)
    return xp.stack((kxy[..., 0], kxy[..., 1], kz), axis=-1)

def euler_angles(rmats: RealArray, xp: ArrayNamespace = JaxNumPy) -> RealArray:
    r"""Calculate Euler angles with Bunge convention [EUL]_.

    Args:
        rmats : A set of rotation matrices.

    Returns:
        A set of Euler angles with Bunge convention :math:`\phi_1, \Phi, \phi_2`.

    References:
        .. [EUL] Depriester, Dorian. (2018), "Computing Euler angles with Bunge convention from
                rotation matrix", 10.13140/RG.2.2.34498.48321/5.
    """
    beta = xp.arccos(rmats[..., 2, 2])
    is_zero = xp.isclose(beta, 0)
    is_pi = xp.isclose(beta, xp.pi)
    alpha = xp.where(is_zero, xp.arctan2(-rmats[..., 1, 0], rmats[..., 0, 0]), 0.0)
    alpha = xp.where(is_pi, xp.arctan2(rmats[..., 1, 0], rmats[..., 0, 0]), alpha)
    alpha = xp.where(xp.invert(is_zero) & xp.invert(is_pi),
                     xp.arctan2(rmats[..., 2, 0], -rmats[..., 2, 1]), alpha)
    gamma = xp.where(xp.invert(is_zero) & xp.invert(is_pi),
                     xp.arctan2(rmats[..., 0, 2], rmats[..., 1, 2]), 0.0)
    alpha = xp.where(alpha < 0.0, alpha + 2 * xp.pi, alpha)
    gamma = xp.where(gamma < 0.0, gamma + 2 * xp.pi, gamma)
    return xp.stack((alpha, beta, gamma), axis=-1)

def euler_matrix(angles: RealArray, xp: ArrayNamespace = JaxNumPy) -> RealArray:
    r"""Calculate rotation matrices from Euler angles with Bunge convention [EUL]_.

    Args:
        angles : Euler angles :math:`\phi_1, \Phi, \phi_2`.
        num_threads : Number of threads used in the calculations.

    Returns:
        A set of rotation matrices.
    """
    cos = xp.cos(angles)
    sin = xp.sin(angles)
    row0 = xp.stack([ cos[..., 0] * cos[..., 2] - sin[..., 0] * sin[..., 2] * cos[..., 1],
                      sin[..., 0] * cos[..., 2] + cos[..., 0] * sin[..., 2] * cos[..., 1],
                      sin[..., 2] * sin[..., 1]], axis=-1)
    row1 = xp.stack([-cos[..., 0] * sin[..., 2] - sin[..., 0] * cos[..., 2] * cos[..., 1],
                     -sin[..., 0] * sin[..., 2] + cos[..., 0] * cos[..., 2] * cos[..., 1],
                      cos[..., 2] * sin[..., 1]], axis=-1)
    row2 = xp.stack([ sin[..., 0] * sin[..., 1],
                     -cos[..., 0] * sin[..., 1],
                      cos[..., 1]], axis=-1)
    return xp.stack((row0, row1, row2), axis=-2)

def tilt_angles(rmats: RealArray, xp: ArrayNamespace = JaxNumPy) -> RealArray:
    r"""Calculate an axis of rotation and a rotation angle for a rotation matrix.

    Args:
        rmats : A set of rotation matrices.

    Returns:
        A set of three angles :math:`\theta, \alpha, \beta`, a rotation angle :math:`\theta`, an
        angle between the axis of rotation and OZ axis :math:`\alpha`, and a polar angle of the
        axis of rotation :math:`\beta`.
    """
    # This transformation is accurate for proper rotations ONLY => det(rmats) == 1
    # from http://scipp.ucsc.edu/~haber/ph116A/rotation_11.pdf
    vec = xp.stack([rmats[..., 2, 1] - rmats[..., 1, 2],
                    rmats[..., 0, 2] - rmats[..., 2, 0],
                    rmats[..., 1, 0] - rmats[..., 0, 1]], axis=-1)
    rabs = xp.sqrt(xp.sum(vec**2, axis=-1))
    return xp.stack([xp.arctan2(rabs, xp.trace(rmats, axis1=-2, axis2=-1) - 1),
                     xp.arccos(vec[..., 2] / rabs),
                     xp.arctan2(vec[..., 1], vec[..., 0])], axis=-1)

def tilt_matrix(angles: RealArray, xp: ArrayNamespace = JaxNumPy) -> RealArray:
    r"""Calculate a rotation matrix for a set of three angles set of three angles :math:`\theta,
    \alpha, \beta`, a rotation angle :math:`\theta`, an angle between the axis of rotation and
    OZ axis :math:`\alpha`, and a polar angle of the axis of rotation :math:`\beta`.

    Args:
        angles : A set of angles :math:`\theta, \alpha, \beta`.

    Returns:
        A set of rotation matrices.
    """
    vec = xp.stack([ xp.cos(0.5 * angles[..., 0]),
                    -xp.sin(0.5 * angles[..., 0]) * xp.sin(angles[..., 1]) * xp.cos(angles[..., 2]),
                    -xp.sin(0.5 * angles[..., 0]) * xp.sin(angles[..., 1]) * xp.sin(angles[..., 2]),
                    -xp.sin(0.5 * angles[..., 0]) * xp.cos(angles[..., 1])], axis=-1)
    row0 = xp.stack([vec[..., 0]**2 + vec[..., 1]**2 - vec[..., 2]**2 - vec[..., 3]**2,
                     2 * (vec[..., 1] * vec[..., 2] + vec[..., 0] * vec[..., 3]),
                     2 * (vec[..., 1] * vec[..., 3] - vec[..., 0] * vec[..., 2])], axis=-1)
    row1 = xp.stack([2 * (vec[..., 1] * vec[..., 2] - vec[..., 0] * vec[..., 3]),
                     vec[..., 0]**2 + vec[..., 2]**2 - vec[..., 1]**2 - vec[..., 3]**2,
                     2 * (vec[..., 2] * vec[..., 3] + vec[..., 0] * vec[..., 1])], axis=-1)
    row2 = xp.stack([2 * (vec[..., 1] * vec[..., 3] + vec[..., 0] * vec[..., 2]),
                     2 * (vec[..., 2] * vec[..., 3] - vec[..., 0] * vec[..., 1]),
                     vec[..., 0]**2 + vec[..., 3]**2 - vec[..., 1]**2 - vec[..., 2]**2], axis=-1)
    return xp.stack((row0, row1, row2), axis=-2)

def det_to_k(pts: RealArray, src: RealArray, idxs: IntArray, xp: ArrayNamespace = JaxNumPy
             ) -> RealArray:
    """Convert coordinates on the detector ``x`, ``y`` to wave-vectors originating from
    the source points ``src``.

    Args:
        x : x coordinates in pixels.
        y : y coordinates in pixels.
        src : Source points in meters (relative to the detector).

    Returns:
        A set of wave-vectors.
    """
    src = xp.reshape(xp.reshape(src, (-1, 3))[xp.ravel(idxs)], idxs.shape + (3,))
    xy = pts - src[..., :2]
    norm = xp.sqrt(xp.sum(xy**2, axis=-1) + src[..., 2]**2)
    vec = xp.append(xy, xp.broadcast_to(-src[..., 2], pts.shape[:-1])[..., None], axis=-1)
    return vec / norm[..., None]

def k_to_det(k: RealArray, src: RealArray, idxs: IntArray, xp: ArrayNamespace = JaxNumPy
             ) -> RealArray:
    """Convert wave-vectors originating from the source points ``src`` to coordinates on the
    detector.

    Args:
        k : An array of wave-vectors.
        src : Source points in meters (relative to the detector).
        idxs : Source point indices.

    Returns:
        A tuple of x and y coordinates in meters.
    """
    src = xp.reshape(xp.reshape(src, (-1, 3))[xp.ravel(idxs)], idxs.shape + (3,))
    src = xp.broadcast_to(src, k.shape)
    kz = xp.where(k[..., 2] == 0, 1.0, k[..., 2])
    pos = xp.where((k[..., 2] == 0)[..., None], 0.0,
                   xp.stack((src[..., 0] - k[..., 0] / kz * src[..., 2],
                             src[..., 1] - k[..., 1] / kz * src[..., 2]), axis=-1))
    return pos

def k_to_smp(k: RealArray, z: RealArray, src: RealArray, xp: ArrayNamespace = JaxNumPy) -> RealArray:
    """Convert wave-vectors originating from the source point ``src`` to sample
    planes at the z coordinate ``z``.

    Args:
        k : An array of wave-vectors.
        src : Source point in meters (relative to the detector).
        z : Plane z coordinates in meters (relative to the detector).
        idxs : Plane indices.

    Returns:
        An array of points belonging to the ``z`` planes.
    """
    kz = xp.where(k[..., 2, None], k[..., 2, None], 1.0)
    theta = xp.where(k[..., 2, None], k[..., :2] / kz, 0)
    xy = src[:2] + theta * (z - src[2])[..., None]
    return xp.stack((xy[..., 0], xy[..., 1], xp.broadcast_to(z, xy.shape[:-1])), axis=-1)

def project_to_rect(point: RealArray, vmin: RealArray, vmax: RealArray,
                    xp: ArrayNamespace = JaxNumPy) -> RealArray:
    return xp.clip(point, vmin, vmax)

def circle(r: RealArray, center: RealArray, vec1: RealArray, vec2: RealArray, theta: RealArray,
           xp: ArrayNamespace = JaxNumPy) -> RealArray:
    return (r * xp.cos(theta))[..., None] * vec1 + (r * xp.sin(theta))[..., None] * vec2 + center

def source_lines(q: RealArray, edges: RealArray, atol: float=2e-6, xp: ArrayNamespace = JaxNumPy
                 ) -> Tuple[RealArray, BoolArray]:
    r"""Calculate the source lines for a set of reciprocal lattice points ``q``.

    Args:
        q : Array of reciprocal lattice points.
        kmin : Lower bound of the rectangular aperture function.
        kmax : Upper bound of the rectangular aperture function.

    Returns:
        A set of source lines in the aperture function.
    """
    tau = edges[:, 1] - edges[:, 0]
    point = edges[:, 0]
    q_mag = xp.sum(q**2, axis=-1)

    f1 = -0.5 * q_mag[..., None] - xp.sum(edges[:, 0] * q[..., None, :2], axis=-1)
    f2 = xp.sum(tau * q[..., None, :2], axis=-1)

    a = f2 * f2 + q[..., None, 2]**2 * xp.sum(tau**2, axis=-1)
    b = f1 * f2 - q[..., None, 2]**2 * xp.sum(point * tau, axis=-1)
    c = f1 * f1 - q[..., None, 2]**2 * (1 - xp.sum(point**2, axis=-1))

    def get_k(t):
        return kxy_to_k(point + t[..., None] * tau, xp)

    delta = xp.where(b * b > a * c, b * b - a * c, 0.0)

    a = xp.where(a == 0.0, 1.0, a)
    t0 = xp.where(a == 0.0, 0.0, (b - xp.sqrt(delta)) / a)
    t1 = xp.where(a == 0.0, 0.0, (b + xp.sqrt(delta)) / a)
    t = xp.stack((t0, t1), axis=-2)

    kin = get_k(t)
    prod = xp.abs(xp.sum(kin * q[..., None, None, :], axis=-1) + 0.5 * q_mag[..., None, None])
    prod = xp.where(xp.isclose(prod, 0.0, atol=atol), 0.0, prod)
    fitness = 0.5 * (xp.abs(t - xp.clip(t, 0.0, 1.0)) + xp.sqrt(prod / q_mag[..., None, None]))

    kin = xp.reshape(kin, (*kin.shape[:-3], -1, *kin.shape[-1:]))
    fitness = xp.reshape(fitness, (*fitness.shape[:-2], -1))

    kin = xp.take_along_axis(kin, xp.argsort(fitness[..., None], axis=-2)[..., :2, :], axis=-2)
    fitness = xp.mean(xp.sort(fitness, axis=-1)[..., :2], axis=-1)

    mask = fitness == 0.0
    return xp.where(mask[..., None, None], kin, 0.0), mask
