from typing import Callable, Literal, Protocol, Tuple, Union
from dataclasses import dataclass
import jax.numpy as jnp
from jax import random, vmap
from .cbc_data import (AnyPoints, CBData, CBDPoints, CircleState, LaueVectors, Miller,
                       MillerWithRLP, Patterns, PointsWithK, RLP, Rotograms, UCA)
from .cbc_setup import (EulerState, InternalState, LensState, RotationState, TiltState,
                        TiltOverAxisState, XtalState)
from .geometry import (add_at, arange, det_to_k, k_to_det, k_to_smp, kxy_to_k, project_to_rect,
                       safe_divide, source_lines)
from .state import State
from .._src.annotations import ArrayNamespace, KeyArray, IntArray, JaxNumPy, NumPy, RealArray
from .._src.data_container import array_namespace
from .._src.src.bresenham import accumulate_lines
from .._src.src.label import PointSet3D, Structure3D, binary_dilation, center_of_mass, label
from .._src.streaks import Streaks

def key_combine(key: KeyArray, hkl: IntArray, xp: ArrayNamespace = JaxNumPy) -> IntArray:
    array = xp.reshape(hkl, (-1, 3))

    def combine(key: IntArray, val: IntArray) -> IntArray:
        bits = xp.asarray(random.key_data(key))
        new_bits = xp.asarray(random.key_data(vmap(random.key, 0, 0)(val)))
        bits ^= new_bits + xp.uint32(0x9e3779b9) + (bits << 6) + (bits >> 2)
        return xp.asarray(random.wrap_key_data(jnp.asarray(bits)))

    keys = xp.broadcast_to(xp.asarray(key), (array.shape[0],))
    keys = combine(combine(combine(keys, array[:, 0]), array[:, 1]), array[:, 2])
    return xp.reshape(keys, hkl.shape[:-1])

def uniform(keys: IntArray, size: int, xp: ArrayNamespace = JaxNumPy) -> RealArray:
    map_func = vmap(random.uniform, in_axes=(0, None), out_axes=-1)
    res = xp.asarray(map_func(xp.ravel(keys), (size,)))
    return xp.reshape(res, (size,) + keys.shape)

@dataclass
class Detector():
    x_pixel_size : float
    y_pixel_size : float

    def to_indices(self, x: RealArray, y: RealArray) -> Tuple[RealArray, RealArray]:
        return x / self.x_pixel_size, y / self.y_pixel_size

    def to_coordinates(self, i: RealArray, j: RealArray) -> Tuple[RealArray, RealArray]:
        return i * self.x_pixel_size, j * self.y_pixel_size

    def to_patterns(self, streaks: Streaks, xp: ArrayNamespace = JaxNumPy) -> Patterns:
        pts = xp.stack(self.to_coordinates(xp.asarray(streaks.x), xp.asarray(streaks.y)), axis=-1)
        return Patterns(xp.asarray(streaks.index), xp.reshape(pts, pts.shape[:-2] + (4,)))

    def to_streaks(self, patterns: Patterns, xp: ArrayNamespace = NumPy) -> Streaks:
        pts = xp.stack(self.to_indices(xp.asarray(patterns.x), xp.asarray(patterns.y)), axis=-1)
        return Streaks(xp.asarray(patterns.index), xp.reshape(pts, pts.shape[:-2] + (4,)))

class Transform():
    def __call__(self, argument: RealArray, state: State) -> RealArray:
        raise NotImplementedError

class Rotation(Transform):
    def __call__(self, argument: RealArray, state: RotationState) -> RealArray:
        return argument @ state.matrix

    def of_xtal(self, argument: XtalState, state: RotationState) -> XtalState:
        return XtalState(self(argument.basis, state))

class EulerRotation(Rotation):
    def __call__(self, argument: RealArray, state: EulerState) -> RealArray:
        return super().__call__(argument, state.to_rotation())

    def of_xtal(self, argument: XtalState, state: EulerState) -> XtalState:
        return XtalState(self(argument.basis, state))

class Tilt(Rotation):
    def __call__(self, argument: RealArray, state: TiltState) -> RealArray:
        return super().__call__(argument, state.to_rotation())

    def of_xtal(self, argument: XtalState, state: TiltState) -> XtalState:
        return XtalState(self(argument.basis, state))

class TiltOverAxis(Tilt):
    def __call__(self, argument: RealArray, state: TiltOverAxisState) -> RealArray:
        return super().__call__(argument, state.to_tilt())

    def of_xtal(self, argument: XtalState, state: TiltOverAxisState) -> XtalState:
        return XtalState(self(argument.basis, state))

class ChainRotations(Transform):
    def __init__(self, transforms : Tuple[Rotation, ...]):
        self.transforms = transforms

    def __call__(self, argument: RealArray, state: Tuple[RotationState, ...]) -> RealArray:
        for s, transform in zip(state, self.transforms):
            argument = transform(argument, s)
        return argument

    def of_xtal(self, argument: XtalState, state: Tuple[RotationState, ...]) -> XtalState:
        for s, transform in zip(state, self.transforms):
            argument = transform.of_xtal(argument, s)
        return argument

    def combine(self, state: Tuple[RotationState, ...]) -> RotationState:
        xp = array_namespace(*state)
        return RotationState(self(xp.eye(3), state))

class Circle(Transform):
    def __call__(self, argument: RealArray, state: CircleState) -> RealArray:
        xp = state.__array_namespace__()
        return (state.radius * xp.cos(argument))[..., None] * state.axis1 \
             + (state.radius * xp.sin(argument))[..., None] * state.axis2 + state.center

class Xtal():
    def hkl_in_aperture(self, theta: Union[float, RealArray], hkl: IntArray, state: XtalState,
                        xp: ArrayNamespace) -> Miller:
        index = xp.broadcast_to(xp.expand_dims(xp.arange(len(state)),
                                               axis=tuple(range(1, hkl.ndim))),
                                (len(state),) + hkl.shape[:-1])
        hkl = xp.broadcast_to(hkl[None, ...], (len(state),) + hkl.shape)
        miller = Miller(hkl=hkl, index=index)

        miller = self.hkl_to_q(miller, state)
        rec_abs = xp.sqrt((miller.q**2).sum(axis=-1))
        rec_th = xp.arccos(-miller.q[..., 2] / rec_abs)
        src_th = rec_th - xp.arccos(0.5 * rec_abs)
        return miller[xp.where((xp.abs(src_th) < theta))]

    def hkl_in_ball(self, q_abs: Union[float, RealArray], state: XtalState, xp: ArrayNamespace
                    ) -> IntArray:
        lat_size = xp.asarray(xp.rint(q_abs / state.unit_cell.lengths), dtype=int)
        lat_size = xp.max(xp.reshape(lat_size, (-1, 3)), axis=0)
        h_idxs = xp.arange(-lat_size[0], lat_size[0] + 1)
        k_idxs = xp.arange(-lat_size[1], lat_size[1] + 1)
        l_idxs = xp.arange(-lat_size[2], lat_size[2] + 1)
        h_grid, k_grid, l_grid = xp.meshgrid(h_idxs, k_idxs, l_idxs)
        hkl = xp.stack((xp.ravel(h_grid), xp.ravel(k_grid), xp.ravel(l_grid)), axis=1)
        hkl = xp.compress(xp.any(hkl, axis=1), hkl, axis=0)

        rec_vec = xp.dot(hkl, state.basis)
        rec_abs = xp.sqrt(xp.sum(rec_vec**2, axis=-1))
        rec_abs = xp.reshape(rec_abs, (hkl.shape[0], -1))
        return hkl[xp.any(rec_abs < q_abs, axis=-1)]

    def hkl_to_q(self, miller: Miller, state: XtalState) -> MillerWithRLP:
        xp = array_namespace(miller)
        q = xp.sum(state.basis[miller.index] * miller.hkl_indices[..., None], axis=-2)
        return MillerWithRLP(index=miller.index, hkl=miller.hkl, q=q)

    def q_to_hkl(self, rlp: RLP, state: XtalState) -> MillerWithRLP:
        xp = array_namespace(rlp)
        hkl = xp.sum(xp.linalg.inv(state.basis)[rlp.index] * rlp.q[..., None], axis=-2)
        return MillerWithRLP(q=rlp.q, index=rlp.index, hkl=hkl)

    def hkl_bounds(self, rlp1: RLP, rlp2: RLP, state: XtalState) -> Tuple[Miller, Miller]:
        xp = array_namespace(rlp1, rlp2)
        miller1, miller2 = self.q_to_hkl(rlp1, state), self.q_to_hkl(rlp2, state)
        hkl_min, hkl_max = xp.sort(xp.stack((miller1.hkl, miller2.hkl)), axis=0)
        hkl_min, hkl_max = xp.floor(hkl_min), xp.ceil(hkl_max)
        return Miller(hkl_min, rlp1.index), Miller(hkl_max, rlp1.index)

    def hkl_offsets(self, hkl_min: Miller, hkl_max: Miller) -> IntArray:
        xp = array_namespace(hkl_min, hkl_max)
        dhkl = xp.max(xp.reshape(hkl_max.hkl_indices - hkl_min.hkl_indices, (-1, 3)), axis=-2) + 1
        offsets = xp.meshgrid(xp.arange(-dhkl[0] // 2 + 1, dhkl[0] // 2 + 1),
                              xp.arange(-dhkl[1] // 2 + 1, dhkl[1] // 2 + 1),
                              xp.arange(-dhkl[2] // 2 + 1, dhkl[2] // 2 + 1))
        return xp.reshape(xp.stack(offsets, axis=-1), (-1, 3))

    def miller_in_ball(self, q_abs: Union[float, RealArray], state: XtalState, xp: ArrayNamespace
                       ) -> Miller:
        indices, hkls = [], []
        for index, xtal in enumerate(state):
            hkl = self.hkl_in_ball(q_abs, xtal, xp)
            hkls.append(hkl)
            indices.append(xp.full(hkl.shape[0], index))
        return Miller(xp.concatenate(hkls), xp.concatenate(indices))

class Lens():
    def kin_center(self, state: LensState, xp: ArrayNamespace) -> RealArray:
        return det_to_k(xp.asarray(state.pupil_center), xp.asarray(state.foc_pos), xp.array(0), xp)

    def kin_max(self, state: LensState, xp: ArrayNamespace) -> RealArray:
        return det_to_k(xp.asarray(state.pupil_max), xp.asarray(state.foc_pos), xp.array(0), xp)

    def kin_min(self, state: LensState, xp: ArrayNamespace) -> RealArray:
        return det_to_k(xp.asarray(state.pupil_min), xp.asarray(state.foc_pos), xp.array(0), xp)

    def kin_edges(self, state: LensState, xp: ArrayNamespace) -> RealArray:
        kmin, kmax = self.kin_min(state, xp), self.kin_max(state, xp)
        return xp.array([[[kmin[0], kmax[1]], [kmax[0], kmax[1]]],
                         [[kmax[0], kmax[1]], [kmax[0], kmin[1]]],
                         [[kmax[0], kmin[1]], [kmin[0], kmin[1]]],
                         [[kmin[0], kmin[1]], [kmin[0], kmax[1]]]])

    def kin_to_sample(self, kin: RealArray, z: RealArray, idxs: IntArray, state: LensState
                      ) -> RealArray:
        """Project incident wave-vectors to the sample planes.

        Args:
            setup : Experimental setup.
            kin : An array of incident wave-vectors.
            idxs : Sample indices.

        Returns:
            Array of sample coordinates.
        """
        xp = array_namespace(kin, z, idxs)
        return k_to_smp(kin, z, xp.asarray(state.foc_pos), idxs, xp)

    def source_lines(self, miller: MillerWithRLP, state: LensState) -> LaueVectors:
        xp = array_namespace(miller)
        kin, is_good = source_lines(miller.q, self.kin_edges(state, xp), xp=xp)
        index = xp.broadcast_to(miller.index, miller.q.shape[:-1])
        laue = LaueVectors(index=index[..., None], q=miller.q[..., None, :], hkl=miller.hkl,
                           kin=kin, kout=kin + miller.q[..., None, :])
        return laue[is_good]

    def project_to_pupil(self, kin: RealArray, state: LensState) -> RealArray:
        xp = array_namespace(kin)
        kin = safe_divide(kin, xp.sqrt(xp.sum(kin**2, axis=-1))[..., None], xp)
        kxy = project_to_rect(kin[..., :2], self.kin_min(state, xp)[:2],
                              self.kin_max(state, xp)[:2], xp)
        return kxy_to_k(kxy, xp)

    def zero_order(self, state: LensState, xp: ArrayNamespace):
        return k_to_det(self.kin_center(state, xp), xp.asarray(state.foc_pos), xp.array(0), xp)

    def line_projector(self, laue: LaueVectors, state: LensState) -> RealArray:
        return self.project_to_pupil(laue.source_line, state)

    def pupil_projector(self, laue: LaueVectors, state: LensState) -> RealArray:
        return self.project_to_pupil(laue.kin, state)

class LaueSampler(Protocol):
    def __call__(self, state: InternalState) -> CBDPoints:
        ...

class Projector(Protocol):
    def __call__(self, laue: LaueVectors, state: LensState) -> RealArray:
        ...

Criterion = Callable[[State,], RealArray]
LossFn = Callable[[RealArray, RealArray], RealArray]
Loss = Literal['l1', 'l2', 'log_cosh']

def loss_function(loss: Loss, xp: ArrayNamespace = JaxNumPy) -> LossFn:
    def l1(predictions: RealArray, targets: RealArray) -> RealArray:
        return xp.abs(predictions - targets)
    def l2(predictions: RealArray, targets: RealArray) -> RealArray:
        return (predictions - targets)**2
    def log_cosh(predictions: RealArray, targets: RealArray) -> RealArray:
        return xp.log(xp.cosh(predictions - targets))

    loss_fns = {'l1': l1, 'l2': l2, 'log_cosh': log_cosh}
    return loss_fns[loss]

class CBDSetup():
    lens    : Lens = Lens()
    xtal    : Xtal = Xtal()

    def kin_to_sample(self, kin: RealArray, idxs: IntArray, state: InternalState
                      ) -> RealArray:
        """Project incident wave-vectors to the sample planes.

        Args:
            setup : Experimental setup.
            kin : An array of incident wave-vectors.
            idxs : Sample indices.

        Returns:
            Array of sample coordinates.
        """
        xp = state.__array_namespace__()
        return self.lens.kin_to_sample(kin, xp.asarray(state.z), idxs, state.lens)

    def hkl_in_aperture(self, q_abs: float, state: InternalState) -> Miller:
        xp = state.__array_namespace__()
        hkl = self.xtal.hkl_in_ball(q_abs, state.xtal, xp)
        kz = xp.asarray([self.lens.kin_min(state.lens, xp)[..., 2],
                         self.lens.kin_max(state.lens, xp)[..., 2]])
        return self.xtal.hkl_in_aperture(xp.arccos(xp.min(kz)), hkl, state.xtal, xp)

    def points_to_kout(self, points: AnyPoints, state: InternalState) -> PointsWithK:
        xp = state.__array_namespace__()
        if isinstance(points, CBDPoints):
            kin = points.kin
        else:
            kin = self.lens.kin_center(state.lens, xp)
        smp_pos = self.kin_to_sample(kin, points.index, state)
        kout = det_to_k(points.points, smp_pos, arange(smp_pos.shape[:-1]), xp)
        return PointsWithK(index=points.index, points=points.points, kout=kout)

    def kout_to_points(self, laue: LaueVectors, state: InternalState) -> CBDPoints:
        xp = state.__array_namespace__()
        smp_pos = self.kin_to_sample(laue.kin, laue.index, state)
        points = k_to_det(laue.kout, smp_pos, arange(smp_pos.shape[:-1], xp), xp)
        return CBDPoints(**laue.to_dict(), points=points)

    def patterns_to_kout(self, patterns: Patterns, state: InternalState
                         ) -> Tuple[RealArray, RealArray]:
        xp = state.__array_namespace__()
        pts = self.points_to_kout(patterns.to_points(), state)
        return xp.min(pts.kout[..., :2], axis=-2), xp.max(pts.kout[..., :2], axis=-2)

    def patterns_to_q(self, patterns: Patterns, state: InternalState) -> Tuple[RLP, RLP]:
        xp = state.__array_namespace__()
        kmin, kmax = self.lens.kin_min(state.lens, xp), self.lens.kin_max(state.lens, xp)
        kout_min, kout_max = self.patterns_to_kout(patterns, state)
        q1 = kxy_to_k(kout_min, xp) - kmin
        q2 = kxy_to_k(kout_max, xp) - kmax
        return RLP(q1, patterns.index), RLP(q2, patterns.index)

class CBDIndexer(CBDSetup):
    circle  : Circle = Circle()

    @classmethod
    def rho_map(cls, xp: ArrayNamespace) -> float:
        return 2 * xp.pi

    def patterns_to_uca(self, patterns: Patterns, points: PointsWithK, state: InternalState
                        ) -> UCA:
        xp = state.__array_namespace__()
        kmin, kmax = self.lens.kin_min(state.lens, xp), self.lens.kin_max(state.lens, xp)
        kout_min, kout_max = self.patterns_to_kout(patterns, state)
        xy = xp.stack((kout_min - kmin[:2], kout_max - kmax[:2]))
        xy_min, xy_max = xp.min(xy, axis=0), xp.max(xy, axis=0)
        return UCA(patterns.index, xp.arange(patterns.shape[0]), points.kout,
                   points.kout[:, :2] - xy_min, points.kout[:, :2] - xy_max)

    def candidates(self, rlp: MillerWithRLP, uca: UCA) -> Tuple[MillerWithRLP, UCA]:
        xp = array_namespace(rlp, uca)
        min_res, max_res = uca.min_resolution, uca.max_resolution
        rlp_res = xp.sum(rlp.q**2, axis=-1)
        rlp_idxs, idxs = xp.where(rlp.index[..., None] == uca.index)
        mask = (rlp_res[rlp_idxs] > min_res[idxs]) & (rlp_res[rlp_idxs] < max_res[idxs])
        return rlp[rlp_idxs[mask]], uca[idxs[mask]]

    def intersection(self, rlp: MillerWithRLP, uca: UCA) -> CircleState:
        xp = array_namespace(rlp, uca)
        rlp_res = xp.sum(rlp.q**2, axis=-1)
        radius = xp.sqrt(rlp_res - 0.25 * rlp_res**2)
        center = 0.5 * rlp_res[..., None] * uca.kout
        axis1 = xp.stack((uca.kout[..., 1], -uca.kout[..., 0], xp.zeros(uca.kout.shape[:-1])),
                          axis=-1)
        axis1 = axis1 / xp.sqrt(xp.sum(axis1**2, axis=-1, keepdims=True))
        axis2 = xp.cross(uca.kout, axis1)
        return CircleState(rlp.index, center, axis1, axis2, radius)

    def uca_endpoints(self, circle: CircleState, uca: UCA, xp: ArrayNamespace = NumPy) -> RealArray:
        xy_min, xy_max = uca.q_min[..., :2], uca.q_max[..., :2]

        a = xp.stack((circle.radius * circle.axis1[..., 0],
                      circle.radius * circle.axis1[..., 1],
                      circle.radius * circle.axis1[..., 0],
                      circle.radius * circle.axis1[..., 1]))
        b = xp.stack((circle.radius * circle.axis2[..., 0],
                      circle.radius * circle.axis2[..., 1],
                      circle.radius * circle.axis2[..., 0],
                      circle.radius * circle.axis2[..., 1]))
        c = xp.stack((xy_min[..., 0] - circle.center[..., 0],
                      xy_min[..., 1] - circle.center[..., 1],
                      xy_max[..., 0] - circle.center[..., 0],
                      xy_max[..., 1] - circle.center[..., 1]))
        theta = xp.concatenate((2.0 * xp.arctan((b - xp.sqrt(a**2 + b**2 - c**2)) / (a + c)),
                                2.0 * xp.arctan((b + xp.sqrt(a**2 + b**2 - c**2)) / (a + c))))

        points = self.circle(theta, circle)
        proj = project_to_rect(points[..., :2], xy_min, xy_max, xp)
        dist = xp.sqrt(xp.sum((points[..., :2] - proj)**2, axis=-1))
        return xp.take_along_axis(theta, xp.argsort(dist, axis=0)[:2], axis=0)

    def rotations(self, phi: RealArray, rlp: MillerWithRLP, midpoints: RealArray,
                  xp: ArrayNamespace = NumPy) -> TiltOverAxisState:
        source = rlp.q / xp.sqrt(xp.sum(rlp.q**2, axis=-1, keepdims=True))
        target = midpoints / xp.sqrt(xp.sum(midpoints**2, axis=-1, keepdims=True))

        bisector = source + target
        bisector = bisector / xp.sqrt(xp.sum(bisector**2, axis=-1, keepdims=True))
        S, C = xp.sin(0.5 * phi), xp.cos(0.5 * phi)
        prod = xp.sum(target * bisector, axis=-1)
        cross = xp.cross(bisector, target)

        theta = 2.0 * xp.arccos(-S[..., None] * prod)
        S2 = xp.sin(0.5 * theta)
        axis = (C[..., None, None] * bisector + S[..., None, None] * cross) / S2[..., None]
        return TiltOverAxisState(theta, axis)

    def rotograms(self, phi: RealArray, rlp: MillerWithRLP, circle: CircleState, uca: UCA
                  ) -> Rotograms:
        xp = array_namespace(phi, rlp, circle)
        endpoints = self.uca_endpoints(circle, uca, xp)
        midpoints = self.circle(xp.mean(endpoints, axis=0), circle)
        tilts = self.rotations(phi, rlp, midpoints, xp)
        return Rotograms.from_tilts(tilts, uca.index, uca.streak_id)

    def rotogrid(self, shape: Tuple[int, int, int], return_step: bool=False,
                 xp: ArrayNamespace=NumPy) -> RealArray | Tuple[RealArray, RealArray]:
        step = 2.0 * self.rho_map(xp) / (xp.asarray(shape) - 1)
        indices = xp.stack(xp.meshgrid(xp.arange(shape[0]), xp.arange(shape[1]),
                                       xp.arange(shape[2])))
        coords = indices * step - self.rho_map(xp)
        if return_step:
            return coords, step
        return coords

    def rotomap(self, shape: Tuple[int, int, int], rotograms: Rotograms,
                width: float, num_threads: int=1) -> RealArray:
        xp = rotograms.__array_namespace__()
        step = 2.0 * self.rho_map(xp) / (xp.asarray(shape) - 1)
        lines = xp.concatenate(((rotograms.lines + self.rho_map(xp)) / xp.tile(step, 2),
                                xp.full(rotograms.lines.shape[:-1] + (1,), width)),
                               axis=-1)
        in_idxs = xp.broadcast_to(rotograms.streak_id, lines.shape[:-1])
        out_idxs = xp.broadcast_to(rotograms.index, lines.shape[:-1])
        rmap = accumulate_lines(lines, (rotograms.num_frames,) + shape, in_idxs, out_idxs,
                               kernel='gaussian', in_overlap='max', out_overlap='sum',
                               num_threads=num_threads)
        return xp.asarray(rmap)

    def refine_peaks(self, x: IntArray, y: IntArray, z: IntArray, rotomap: RealArray,
                     vicinity: Structure3D, connectivity: Structure3D = Structure3D(1, 1)
                     ) -> RealArray:
        xp = array_namespace(rotomap, x, y, z)
        mask = xp.zeros(rotomap.shape, dtype=int)
        mask = add_at(mask, (z, y, x), 1, xp)
        mask = binary_dilation(mask, vicinity)
        regions = label(mask, connectivity, seeds=PointSet3D(x.ravel(), y.ravel(), z.ravel()))
        return center_of_mass(regions, rotomap)

class CBDModel(CBDSetup):
    def to_internal(self, state: State) -> InternalState:
        raise NotImplementedError

    def init_hkl(self, hkl_min: Miller, hkl_max: Miller, offsets: IntArray) -> Miller:
        return Miller(hkl=(hkl_min.hkl_indices + hkl_max.hkl_indices) // 2,
                      index=hkl_min.index).offset(offsets)

    def init_patterns(self, miller: MillerWithRLP, state: InternalState) -> Patterns:
        laue = self.lens.source_lines(miller, state.lens)
        laue = self.kout_to_points(laue, state)
        return Patterns.from_points(laue)

    def init_data(self, key: KeyArray, patterns: Patterns, num_points: int,
                  state: InternalState, combine_key: bool=False) -> 'CBData':
        q1, q2 = self.patterns_to_q(patterns, state)
        hkl_min, hkl_max = self.xtal.hkl_bounds(q1, q2, state.xtal)
        offsets = self.xtal.hkl_offsets(hkl_min, hkl_max)
        miller = self.init_hkl(hkl_min, hkl_max, offsets)

        xp = state.__array_namespace__()
        if combine_key:
            x = uniform(key_combine(key, miller.hkl_indices), num_points, xp)
        else:
            x = xp.asarray(random.uniform(key, (num_points,) + miller.hkl.shape[:-1]))
        return CBData(miller, patterns.sample(x))

    def line_loss(self, num_inliers: int, loss: Loss='l1', xp: ArrayNamespace = JaxNumPy
                  ) -> 'CBDLoss':
        return CBDLoss(self, self.lens.line_projector, loss_function(loss, xp), num_inliers)

    def pupil_loss(self, num_inliers: int, loss: Loss='l1', xp: ArrayNamespace = JaxNumPy
                   ) -> 'CBDLoss':
        return CBDLoss(self, self.lens.pupil_projector, loss_function(loss, xp), num_inliers)

@dataclass(frozen=True, unsafe_hash=True)
class CBDLoss():
    model           : CBDModel
    projector       : Projector
    loss_fn         : LossFn
    num_inliers     : int

    def project_data(self, data: CBData, state: InternalState) -> CBDPoints:
        pts = self.model.points_to_kout(data.points, state)
        rlp = self.model.xtal.hkl_to_q(data.miller, state.xtal)
        return CBDPoints(index=pts.index, points=pts.points, hkl=rlp.hkl,
                         q=rlp.q, kin=pts.kout - rlp.q, kout=pts.kout)

    def __call__(self, data: CBData, state: State) -> RealArray:
        xp = state.__array_namespace__()
        int_state = self.model.to_internal(state)
        points = self.project_data(data, int_state)
        projected = self.projector(points, int_state.lens)
        dist = xp.mean(xp.sum(self.loss_fn(projected, points.kin), axis=-1), axis=0)
        dist = xp.min(dist, axis=0)
        return xp.mean(xp.sort(dist)[:self.num_inliers])

    def index(self, data: CBData, state: State) -> Miller:
        xp = state.__array_namespace__()
        int_state = self.model.to_internal(state)
        points = self.project_data(data, int_state)
        projected = self.projector(points, int_state.lens)
        dist = xp.mean(xp.sum(self.loss_fn(projected, points.kin), axis=-1), axis=0)
        idxs = xp.argmin(dist, axis=0)
        hkl = xp.take_along_axis(points.hkl, idxs[None, ..., None], axis=0)[0]
        return Miller(hkl=hkl, index=points.index)

    def loss_per_pattern(self, data: CBData, state: State) -> RealArray:
        xp = state.__array_namespace__()
        int_state = self.model.to_internal(state)
        points = self.project_data(data, int_state)
        projected = self.projector(points, int_state.lens)
        dist = xp.mean(xp.sum(self.loss_fn(projected, points.kin), axis=-1), axis=0)
        dist = xp.min(dist, axis=0)
        indices = xp.argsort(dist)
        loss = xp.where(indices < self.num_inliers, dist, 0.0)
        return add_at(xp.zeros(len(int_state.xtal)), points.index, loss, xp) / self.num_inliers
