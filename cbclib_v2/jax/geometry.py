from typing import Tuple
import jax.numpy as jnp
from ..annotations import IntArray, RealArray

def euler_angles(rmats: RealArray) -> RealArray:
    r"""Calculate Euler angles with Bunge convention [EUL]_.

    Args:
        rmats : A set of rotation matrices.

    Returns:
        A set of Euler angles with Bunge convention :math:`\phi_1, \Phi, \phi_2`.

    References:
        .. [EUL] Depriester, Dorian. (2018), "Computing Euler angles with Bunge convention from
                rotation matrix", 10.13140/RG.2.2.34498.48321/5.
    """
    beta = jnp.arccos(rmats[..., 2, 2])
    is_zero = jnp.isclose(beta, 0)
    is_pi = jnp.isclose(beta, jnp.pi)
    alpha = jnp.where(is_zero, jnp.arctan2(-rmats[..., 1, 0], rmats[..., 0, 0]), 0.0)
    alpha = jnp.where(is_pi, jnp.arctan2(rmats[..., 1, 0], rmats[..., 0, 0]), alpha)
    alpha = jnp.where(jnp.invert(is_zero) & jnp.invert(is_pi),
                      jnp.arctan2(rmats[..., 2, 0], -rmats[..., 2, 1]), alpha)
    gamma = jnp.where(jnp.invert(is_zero) & jnp.invert(is_pi),
                      jnp.arctan2(rmats[..., 0, 2], rmats[..., 1, 2]), 0.0)
    alpha = jnp.where(alpha < 0.0, alpha + 2 * jnp.pi, alpha)
    gamma = jnp.where(gamma < 0.0, gamma + 2 * jnp.pi, gamma)
    return jnp.stack((alpha, beta, gamma), axis=-1)

def euler_matrix(angles: RealArray) -> RealArray:
    r"""Calculate rotation matrices from Euler angles with Bunge convention [EUL]_.

    Args:
        angles : Euler angles :math:`\phi_1, \Phi, \phi_2`.
        num_threads : Number of threads used in the calculations.

    Returns:
        A set of rotation matrices.
    """
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    row0 = jnp.stack([ cos[..., 0] * cos[..., 2] - sin[..., 0] * sin[..., 2] * cos[..., 1],
                       sin[..., 0] * cos[..., 2] + cos[..., 0] * sin[..., 2] * cos[..., 1],
                       sin[..., 2] * sin[..., 1]], axis=-1)
    row1 = jnp.stack([-cos[..., 0] * sin[..., 2] - sin[..., 0] * cos[..., 2] * cos[..., 1],
                      -sin[..., 0] * sin[..., 2] + cos[..., 0] * cos[..., 2] * cos[..., 1],
                       cos[..., 2] * sin[..., 1]], axis=-1)
    row2 = jnp.stack([ sin[..., 0] * sin[..., 1],
                      -cos[..., 0] * sin[..., 1],
                       cos[..., 1]], axis=-1)
    return jnp.stack((row0, row1, row2), axis=-2)

def tilt_angles(rmats: RealArray) -> RealArray:
    r"""Calculate an axis of rotation and a rotation angle for a rotation matrix.

    Args:
        rmats : A set of rotation matrices.

    Returns:
        A set of three angles :math:`\theta, \alpha, \beta`, a rotation angle :math:`\theta`, an
        angle between the axis of rotation and OZ axis :math:`\alpha`, and a polar angle of the
        axis of rotation :math:`\beta`.
        """
    vec = jnp.stack([rmats[..., 2, 1] - rmats[..., 1, 2],
                     rmats[..., 0, 2] - rmats[..., 2, 0],
                     rmats[..., 1, 0] - rmats[..., 0, 1]], axis=-1)
    mag = jnp.sum(vec**2, axis=-1)
    return jnp.stack([jnp.arccos(0.5 * (jnp.trace(rmats, axis1=-2, axis2=-1) - 1)),
                      jnp.arccos(vec[..., 2] / jnp.sqrt(mag)),
                      jnp.arctan2(vec[..., 1], vec[..., 0])], axis=-1)

def tilt_matrix(angles: RealArray) -> RealArray:
    r"""Calculate a rotation matrix for a set of three angles set of three angles :math:`\theta,
    \alpha, \beta`, a rotation angle :math:`\theta`, an angle between the axis of rotation and
    OZ axis :math:`\alpha`, and a polar angle of the axis of rotation :math:`\beta`.

    Args:
        angles : A set of angles :math:`\theta, \alpha, \beta`.

    Returns:
        A set of rotation matrices.
    """
    vec = jnp.stack([ jnp.cos(0.5 * angles[..., 0]),
                     -jnp.sin(0.5 * angles[..., 0]) * jnp.sin(angles[..., 1]) * jnp.cos(angles[..., 2]),
                     -jnp.sin(0.5 * angles[..., 0]) * jnp.sin(angles[..., 1]) * jnp.sin(angles[..., 2]),
                     -jnp.sin(0.5 * angles[..., 0]) * jnp.cos(angles[..., 1])], axis=-1)
    row0 = jnp.stack([vec[..., 0]**2 + vec[..., 1]**2 - vec[..., 2]**2 - vec[..., 3]**2,
                      2 * (vec[..., 1] * vec[..., 2] + vec[..., 0] * vec[..., 3]),
                      2 * (vec[..., 1] * vec[..., 3] - vec[..., 0] * vec[..., 2])], axis=-1)
    row1 = jnp.stack([2 * (vec[..., 1] * vec[..., 2] - vec[..., 0] * vec[..., 3]),
                      vec[..., 0]**2 + vec[..., 2]**2 - vec[..., 1]**2 - vec[..., 3]**2,
                      2 * (vec[..., 2] * vec[..., 3] + vec[..., 0] * vec[..., 1])], axis=-1)
    row2 = jnp.stack([2 * (vec[..., 1] * vec[..., 3] + vec[..., 0] * vec[..., 2]),
                      2 * (vec[..., 2] * vec[..., 3] - vec[..., 0] * vec[..., 1]),
                      vec[..., 0]**2 + vec[..., 3]**2 - vec[..., 1]**2 - vec[..., 2]**2], axis=-1)
    return jnp.stack((row0, row1, row2), axis=-2)

def det_to_k(x: RealArray, y: RealArray, src: RealArray, idxs: IntArray) -> RealArray:
    """Convert coordinates on the detector ``x`, ``y`` to wave-vectors originating from
    the source points ``src``.

    Args:
        x : x coordinates in pixels.
        y : y coordinates in pixels.
        src : Source points in meters (relative to the detector).

    Returns:
        A set of wave-vectors.
    """
    src = jnp.reshape(jnp.reshape(src, (-1, 3))[jnp.ravel(idxs)], idxs.shape + (3,))
    vec = jnp.stack((x, y, jnp.zeros(x.shape)), axis=-1) - src
    norm = jnp.sqrt(jnp.sum(vec**2, axis=-1))
    return vec / norm[..., None]

def k_to_det(k: RealArray, src: RealArray, idxs: IntArray) -> RealArray:
    """Convert wave-vectors originating from the source points ``src`` to coordinates on
    the detector.

    Args:
        k : An array of wave-vectors.
        src : Source points in meters (relative to the detector).
        idxs : Source point indices.

    Returns:
        A tuple of x and y coordinates in meters.
    """
    src = jnp.reshape(jnp.reshape(src, (-1, 3))[jnp.ravel(idxs)], idxs.shape + (3,))
    src = jnp.broadcast_to(src, k.shape)
    kz = jnp.where(k[..., 2] == 0, 1.0, k[..., 2])
    pos = jnp.where((k[..., 2] == 0)[..., None], 0.0,
                    jnp.stack((src[..., 0] - k[..., 0] / kz * src[..., 2],
                               src[..., 1] - k[..., 1] / kz * src[..., 2]), axis=-1))
    return pos

def k_to_smp(k: RealArray, z: RealArray, src: RealArray, idxs: IntArray
             ) -> RealArray:
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
    z = jnp.reshape(jnp.ravel(z)[jnp.ravel(idxs)], idxs.shape)
    kz = jnp.where(k[..., 2, None], k[..., 2, None], 1.0)
    theta = jnp.where(k[..., 2, None], k[..., :2] / kz, 0)
    xy = src[:2] + theta * (z - src[2])[..., None]
    return jnp.stack((xy[..., 0], xy[..., 1], jnp.broadcast_to(z, xy.shape[:-1])), axis=-1)

def project_to_streak(point: RealArray, pt0: RealArray, pt1: RealArray) -> RealArray:
    tau = pt1 - pt0
    r = point - pt0
    r_tau = jnp.sum(tau * r, axis=-1) / jnp.sum(tau**2, axis=-1)
    return tau * jnp.clip(r_tau[..., None], 0.0, 1.0) + pt0

def distance_to_streak(point: RealArray, pt0: RealArray, pt1: RealArray) -> RealArray:
    dist = point - project_to_streak(point, pt0, pt1)
    return jnp.sqrt(jnp.sum(dist**2, axis=-1))

def source_lines(q: RealArray, kmin: RealArray, kmax: RealArray) -> RealArray:
    r"""Calculate the source lines for a set of reciprocal lattice points ``q``.

    Args:
        q : Array of reciprocal lattice points.
        kmin : Lower bound of the rectangular aperture function.
        kmax : Upper bound of the rectangular aperture function.

    Returns:
        A set of source lines in the aperture function.
    """
    edges = jnp.array([[[kmin[0], kmin[1]], [kmin[0], kmax[1]]],
                       [[kmin[0], kmin[1]], [kmax[0], kmin[1]]],
                       [[kmax[0], kmin[1]], [kmax[0], kmax[1]]],
                       [[kmin[0], kmax[1]], [kmax[0], kmax[1]]]])
    tau = edges[:, 1] - edges[:, 0]
    point = edges[:, 0]
    q_mag = -0.5 * jnp.sum(q**2, axis=-1)

    f1 = q_mag[..., None] - jnp.sum(edges[:, 0] * q[..., None, :2], axis=-1)
    f2 = jnp.sum(tau * q[..., None, :2], axis=-1)

    a = f2 * f2 + q[..., None, 2]**2 * jnp.sum(tau**2, axis=-1)
    b = f1 * f2 - q[..., None, 2]**2 * jnp.sum(point * tau, axis=-1)
    c = f1 * f1 - q[..., None, 2]**2 * (1 - jnp.sum(point**2, axis=-1))

    def get_k(t):
        kxy = point + t[..., None] * tau
        kz_sq = 1 - jnp.sum(kxy**2, axis=-1)
        kz_sq = jnp.where(kz_sq <= 0.0, 0.0, kz_sq)
        kz = jnp.where(kz_sq <= 0.0, 0.0, jnp.sqrt(kz_sq))
        return jnp.stack((kxy[..., 0], kxy[..., 1], kz), axis=-1)

    is_intersect = b * b > a * c
    delta = jnp.where(is_intersect, b * b - a * c, 0)
    k0 = jnp.where(is_intersect[..., None], get_k((b - jnp.sqrt(delta)) / a), 0)
    k1 = jnp.where(is_intersect[..., None], get_k((b + jnp.sqrt(delta)) / a), 0)

    kin = jnp.stack((k0, k1), axis=-3)
    dist = distance_to_streak(kin[..., :2], edges[:, 0], edges[:, 1])
    prod = jnp.abs(jnp.sum(kin * q[..., None, None, :], axis=-1) - q_mag[..., None, None])
    fitness = dist + prod

    kin = jnp.reshape(kin, (*kin.shape[:-3], -1, *kin.shape[-1:]))
    fitness = jnp.reshape(fitness, (*fitness.shape[:-2], -1))

    kin = jnp.take_along_axis(kin, jnp.argsort(fitness[..., None], axis=-2)[..., :2, :], axis=-2)
    fitness = jnp.sort(fitness, axis=-1)[..., :2]
    kin = jnp.where(jnp.isclose(jnp.sum(fitness, axis=-1), 0.0)[..., None, None], kin, 0.0)
    return kin
