from typing import Callable, Protocol, Tuple, Union, cast
from dataclasses import dataclass
import jax.numpy as jnp
from jax import random, vmap
from .cbc_data import (AnyPoints, CBData, CBDPoints, Miller, MillerWithRLP, LaueVectors, RLP,
                       Patterns, PointsWithK)
from .cbc_setup import (EulerState, InternalState, LensState, RotationState, TiltState,
                        TiltOverAxisState, XtalState)
from .geometry import (arange, det_to_k, k_to_det, k_to_smp, kxy_to_k, project_to_rect, safe_divide,
                       source_lines)
from .state import State
from .._src.annotations import KeyArray, IntArray, RealArray

def key_combine(key: KeyArray, hkl: IntArray) -> KeyArray:
    array = jnp.reshape(hkl, (-1, 3))

    def combine(key: KeyArray, val: IntArray) -> KeyArray:
        bits = random.key_data(key)
        new_bits = random.key_data(vmap(random.key, 0, 0)(val))
        bits ^= new_bits + jnp.uint32(0x9e3779b9) + (bits << 6) + (bits >> 2)
        return cast(KeyArray, random.wrap_key_data(bits))

    keys = jnp.broadcast_to(key, (array.shape[0],))
    keys = combine(combine(combine(keys, array[:, 0]), array[:, 1]), array[:, 2])
    return jnp.reshape(keys, hkl.shape[:-1])

def uniform(keys: KeyArray, size: int) -> RealArray:
    res = vmap(random.uniform, in_axes=(0, None), out_axes=-1)(jnp.ravel(keys), (size,))
    return jnp.reshape(res, (size,) + keys.shape)

class Transform():
    def apply(self, xtal: XtalState, state: State) -> XtalState:
        raise NotImplementedError

class Rotation(Transform):
    def apply(self, xtal: XtalState, state: RotationState) -> XtalState:
        return XtalState(xtal.basis @ state.matrix)

class EulerRotation(Rotation):
    def apply(self, xtal: XtalState, state: EulerState) -> XtalState:
        return super().apply(xtal, state.to_rotation())

class Tilt(Rotation):
    def apply(self, xtal: XtalState, state: TiltState) -> XtalState:
        return super().apply(xtal, state.to_rotation())

class TiltOverAxis(Tilt):
    def apply(self, xtal: XtalState, state: TiltOverAxisState) -> XtalState:
        return super().apply(xtal, state.to_tilt())

class ChainRotations():
    def __init__(self, transforms : Tuple[Rotation, ...]):
        self.transforms = transforms

    def apply(self, xtal: XtalState, state: Tuple[RotationState, ...]) -> XtalState:
        for s, transform in zip(state, self.transforms):
            xtal = transform.apply(xtal, s)
        return xtal

    def combine(self, state: Tuple[RotationState, ...]) -> RotationState:
        result = self.apply(XtalState(jnp.eye(3)), state)
        return RotationState(result.basis)

class Xtal():
    def hkl_in_aperture(self, theta: Union[float, RealArray], hkl: IntArray, state: XtalState
                        ) -> Miller:
        index = jnp.broadcast_to(jnp.expand_dims(jnp.arange(len(state)), axis=range(1, hkl.ndim)),
                                 (len(state),) + hkl.shape[:-1])
        hkl = jnp.broadcast_to(hkl[None, ...], (len(state),) + hkl.shape)
        miller = Miller(hkl=hkl, index=index)

        miller = self.hkl_to_q(miller, state)
        rec_abs = jnp.sqrt((miller.q**2).sum(axis=-1))
        rec_th = jnp.arccos(-miller.q[..., 2] / rec_abs)
        src_th = rec_th - jnp.arccos(0.5 * rec_abs)
        return miller.filter(jnp.where((jnp.abs(src_th) < theta)))

    def hkl_in_ball(self, q_abs: Union[float, RealArray], state: XtalState) -> IntArray:
        constants = state.lattice_constants()
        lat_size = jnp.asarray(jnp.rint(q_abs / constants.lengths), dtype=int)
        lat_size = jnp.max(jnp.reshape(lat_size, (-1, 3)), axis=0)
        h_idxs = jnp.arange(-lat_size[0], lat_size[0] + 1)
        k_idxs = jnp.arange(-lat_size[1], lat_size[1] + 1)
        l_idxs = jnp.arange(-lat_size[2], lat_size[2] + 1)
        h_grid, k_grid, l_grid = jnp.meshgrid(h_idxs, k_idxs, l_idxs)
        hkl = jnp.stack((jnp.ravel(h_grid), jnp.ravel(k_grid), jnp.ravel(l_grid)), axis=1)
        hkl = jnp.compress(jnp.any(hkl, axis=1), hkl, axis=0)

        rec_vec = jnp.dot(hkl, state.basis)
        rec_abs = jnp.sqrt(jnp.sum(rec_vec**2, axis=-1))
        rec_abs = jnp.reshape(rec_abs, (hkl.shape[0], -1))
        return hkl[jnp.any(rec_abs < q_abs, axis=-1)]

    def hkl_to_q(self, miller: Miller, state: XtalState) -> MillerWithRLP:
        q = jnp.sum(state.basis[miller.index] * miller.hkl_indices[..., None], axis=-2)
        return MillerWithRLP(index=miller.index, hkl=miller.hkl, q=q)

    def q_to_hkl(self, rlp: RLP, state: XtalState) -> MillerWithRLP:
        hkl = jnp.sum(jnp.linalg.inv(state.basis)[rlp.index] * rlp.q[..., None], axis=-2)
        return MillerWithRLP(q=rlp.q, index=rlp.index, hkl=hkl)

    def hkl_bounds(self, rlp1: RLP, rlp2: RLP, state: XtalState) -> Tuple[Miller, Miller]:
        miller1, miller2 = self.q_to_hkl(rlp1, state), self.q_to_hkl(rlp2, state)
        hkl_min, hkl_max = jnp.sort(jnp.stack((miller1.hkl, miller2.hkl)), axis=0)
        hkl_min, hkl_max = jnp.floor(hkl_min), jnp.ceil(hkl_max)
        return Miller(hkl_min, rlp1.index), Miller(hkl_max, rlp1.index)

    def hkl_offsets(self, hkl_min: Miller, hkl_max: Miller) -> IntArray:
        dhkl = jnp.max(jnp.reshape(hkl_max.hkl_indices - hkl_min.hkl_indices, (-1, 3)), axis=-2) + 1
        offsets = jnp.meshgrid(jnp.arange(-dhkl[0] // 2 + 1, dhkl[0] // 2 + 1),
                               jnp.arange(-dhkl[1] // 2 + 1, dhkl[1] // 2 + 1),
                               jnp.arange(-dhkl[2] // 2 + 1, dhkl[2] // 2 + 1))
        return jnp.reshape(jnp.stack(offsets, axis=-1), (-1, 3))

class Lens():
    def kin_center(self, state: LensState) -> RealArray:
        point = 0.5 * (state.pupil_min + state.pupil_max)
        return det_to_k(point, state.foc_pos, jnp.array(0))

    def kin_max(self, state: LensState) -> RealArray:
        return det_to_k(state.pupil_max, state.foc_pos, jnp.array(0))

    def kin_min(self, state: LensState) -> RealArray:
        return det_to_k(state.pupil_min, state.foc_pos, jnp.array(0))

    def kin_edges(self, state: LensState) -> RealArray:
        kmin, kmax = self.kin_min(state), self.kin_max(state)
        return jnp.array([[[kmin[0], kmax[1]], [kmax[0], kmax[1]]],
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
        return k_to_smp(kin, z, state.foc_pos, idxs)

    def source_lines(self, miller: MillerWithRLP, state: LensState) -> LaueVectors:
        kin, is_good = source_lines(miller.q, self.kin_edges(state))
        index = jnp.broadcast_to(miller.index, miller.q.shape[:-1])
        laue = LaueVectors(index=index[..., None], q=miller.q[..., None, :], hkl=miller.hkl,
                           kin=kin, kout=kin + miller.q[..., None, :])
        return laue.filter(is_good)

    def project_to_pupil(self, kin: RealArray, state: LensState) -> RealArray:
        kin = safe_divide(kin, jnp.sqrt(jnp.sum(kin**2, axis=-1))[..., None])
        kxy = project_to_rect(kin[..., :2], self.kin_min(state)[:2], self.kin_max(state)[:2])
        return kxy_to_k(kxy)

    def zero_order(self, state: LensState):
        return k_to_det(self.kin_center(state), state.foc_pos, jnp.array(0))

    def line_projector(self, laue: LaueVectors, state: LensState) -> RealArray:
        return self.project_to_pupil(laue.kin_to_source_line(), state)

    def pupil_projector(self, laue: LaueVectors, state: LensState) -> RealArray:
        return self.project_to_pupil(laue.kin, state)

class LaueSampler(Protocol):
    def __call__(self, state: InternalState) -> CBDPoints:
        ...

class Projector(Protocol):
    def __call__(self, laue: LaueVectors, state: LensState) -> RealArray:
        ...

Criterion = Callable[[State,], RealArray]

class CBDModel():
    lens    : Lens = Lens()
    xtal    : Xtal = Xtal()

    def to_internal(self, state: State) -> InternalState:
        raise NotImplementedError

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
        return self.lens.kin_to_sample(kin, state.z, idxs, state.lens)

    def hkl_in_aperture(self, q_abs: float, state: InternalState) -> Miller:
        hkl = self.xtal.hkl_in_ball(q_abs, state.xtal)
        kz = jnp.asarray([self.lens.kin_min(state.lens)[..., 2],
                          self.lens.kin_max(state.lens)[..., 2]])
        return self.xtal.hkl_in_aperture(jnp.arccos(jnp.min(kz)), hkl, state.xtal)

    def points_to_kout(self, points: AnyPoints, state: InternalState) -> PointsWithK:
        if isinstance(points, CBDPoints):
            kin = points.kin
        else:
            kin = self.lens.kin_center(state.lens)
        smp_pos = self.kin_to_sample(kin, points.index, state)
        kout = det_to_k(points.points, smp_pos, arange(smp_pos.shape[:-1]))
        return PointsWithK(index=points.index, points=points.points, kout=kout)

    def kout_to_points(self, laue: LaueVectors, state: InternalState) -> CBDPoints:
        smp_pos = self.kin_to_sample(laue.kin, laue.index, state)
        points = k_to_det(laue.kout, smp_pos, arange(smp_pos.shape[:-1]))
        return CBDPoints(**laue.to_dict(), points=points)

    def init_hkl(self, hkl_min: Miller, hkl_max: Miller, offsets: IntArray) -> Miller:
        return Miller(hkl=(hkl_min.hkl_indices + hkl_max.hkl_indices) // 2,
                      index=hkl_min.index).offset(offsets)

    def init_patterns(self, miller: MillerWithRLP, state: InternalState) -> Patterns:
        laue = self.lens.source_lines(miller, state.lens)
        laue = self.kout_to_points(laue, state)
        return Patterns.from_points(laue)

    def patterns_to_kout(self, patterns: Patterns, state: InternalState
                         ) -> Tuple[RealArray, RealArray]:
        pts = self.points_to_kout(patterns.to_points(), state)
        return jnp.min(pts.kout[..., :2], axis=-2), jnp.max(pts.kout[..., :2], axis=-2)

    # def patterns_to_q(self, patterns: Patterns, state: InternalState) -> Tuple[RLP, RLP]:
    #     kmin, kmax = self.lens.kin_min(state.lens), self.lens.kin_max(state.lens)
    #     kout_min, kout_max = self.patterns_to_kout(patterns, state)
    #     q1 = kxy_to_k(kout_min) - kxy_to_k(kmax[:2] - (kout_max - kout_min))
    #     q2 = kxy_to_k(kout_max) - kmin
    #     return RLP(q1, patterns.index), RLP(q2, patterns.index)

    # def points_to_q(self, points: Points, patterns: Patterns, state: InternalState
    #                 ) -> Tuple[RLP, RLP]:
    #     kmin, kmax = self.lens.kin_min(state.lens), self.lens.kin_max(state.lens)
    #     kout_min, kout_max = self.patterns_to_kout(patterns, state)
    #     points = self.points_to_kout(points, state)
    #     q1, q2 = points.kout - kxy_to_k(kmax[:2] - (kout_max - kout_min)), points.kout - kmin
    #     return RLP(q1, points.index), RLP(q2, points.index)

    def patterns_to_q(self, patterns: Patterns, state: InternalState) -> Tuple[RLP, RLP]:
        kmin, kmax = self.lens.kin_min(state.lens), self.lens.kin_max(state.lens)
        kout_min, kout_max = self.patterns_to_kout(patterns, state)
        q1 = kxy_to_k(kout_min) - kmin
        q2 = kxy_to_k(kout_max) - kmax
        return RLP(q1, patterns.index), RLP(q2, patterns.index)

    def init_data(self, key: KeyArray, patterns: Patterns, num_points: int,
                  state: InternalState, combine_key: bool=False) -> 'CBData':
        q1, q2 = self.patterns_to_q(patterns, state)
        hkl_min, hkl_max = self.xtal.hkl_bounds(q1, q2, state.xtal)
        offsets = self.xtal.hkl_offsets(hkl_min, hkl_max)
        miller = self.init_hkl(hkl_min, hkl_max, offsets)

        if combine_key:
            x = uniform(key_combine(key, miller.hkl_indices), num_points)
        else:
            x = random.uniform(key, (num_points,) + miller.hkl.shape[:-1])
        return CBData(miller, patterns.sample(x))

    def line_loss(self, num_inliers: int) -> 'CBDLoss':
        return CBDLoss(self, self.lens.line_projector, num_inliers)

    def pupil_loss(self, num_inliers: int) -> 'CBDLoss':
        return CBDLoss(self, self.lens.pupil_projector, num_inliers)

@dataclass(frozen=True, unsafe_hash=True)
class CBDLoss():
    model       : CBDModel
    projector   : Projector
    num_inliers : int

    def project_data(self, data: CBData, state: InternalState) -> CBDPoints:
        pts = self.model.points_to_kout(data.points, state)
        rlp = self.model.xtal.hkl_to_q(data.miller, state.xtal)
        return CBDPoints(index=pts.index, points=pts.points, hkl=rlp.hkl,
                         q=rlp.q, kin=pts.kout - rlp.q, kout=pts.kout)

    def __call__(self, data: CBData, state: State) -> RealArray:
        int_state = self.model.to_internal(state)
        points = self.project_data(data, int_state)
        projected = self.projector(points, int_state.lens)
        dist = jnp.mean(jnp.sum((points.kin - projected)**2, axis=-1), axis=0)
        dist = jnp.min(dist, axis=0)
        return jnp.mean(jnp.sort(dist)[:self.num_inliers])

    def index(self, data: CBData, state: State) -> Miller:
        int_state = self.model.to_internal(state)
        points = self.project_data(data, int_state)
        projected = self.projector(points, int_state.lens)
        dist = jnp.mean(jnp.sum((points.kin - projected)**2, axis=-1), axis=0)
        idxs = jnp.argmin(dist, axis=0)
        hkl = jnp.take_along_axis(points.hkl, idxs[None, ..., None], axis=0)[0]
        return Miller(hkl=hkl, index=points.index)

    def loss_per_pattern(self, data: CBData, state: State) -> RealArray:
        int_state = self.model.to_internal(state)
        points = self.project_data(data, int_state)
        projected = self.projector(points, int_state.lens)
        dist = jnp.mean(jnp.sum((points.kin - projected)**2, axis=-1), axis=0)
        dist = jnp.min(dist, axis=0)
        indices = jnp.argsort(dist)
        loss = jnp.where(indices < self.num_inliers, dist, 0.0)
        return jnp.zeros(len(int_state.xtal)).at[points.index].add(loss) / self.num_inliers
