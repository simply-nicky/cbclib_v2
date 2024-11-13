from typing import Callable, Protocol, Union, cast
import jax.numpy as jnp
from jax import random, vmap
from .cbc_data import (AnyPoints, Miller, MillerWithRLP, LaueVectors, CBDPoints, RLP, Patterns,
                       Points, PointsWithK)
from .cbc_setup import LensState, XtalState, InternalState
from .dataclasses import jax_dataclass
from .geometry import (arange, det_to_k, k_to_det, k_to_smp, kxy_to_k, project_to_rect, safe_divide,
                       source_lines)
from .._src.annotations import KeyArray, IntArray, RealArray
from .._src.data_container import DataclassInstance

State = DataclassInstance

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

def generate_points(key: KeyArray, generator: Callable[[KeyArray,], RealArray],
                    pt0: AnyPoints, pt1: AnyPoints) -> Points:
    pts = pt0.points + generator(key)[..., None] * (pt1.points - pt0.points)
    return Points(points=pts, index=pt0.index)

class Xtal():
    def hkl_in_aperture(self, theta: Union[float, RealArray], hkl: IntArray, state: XtalState
                        ) -> Miller:
        index = jnp.broadcast_to(jnp.expand_dims(jnp.arange(state.num), axis=range(1, hkl.ndim)),
                                 (state.num,) + hkl.shape[:-1])
        hkl = jnp.broadcast_to(hkl[None, ...], (state.num,) + hkl.shape)
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

@jax_dataclass
class Pupil():
    vmin    : RealArray
    vmax    : RealArray

    def center(self) -> RealArray:
        return 0.5 * (self.vmin + self.vmax)

    def diagonals(self) -> RealArray:
        return jnp.array([[self.vmin, self.vmax],
                          [[self.vmin[0], self.vmax[1]], [self.vmax[0], self.vmin[1]]]])

    def edges(self, offset: float=0.0) -> RealArray:
        vmin = self.vmin - jnp.clip(offset, 0.0, 1.0) * (self.vmax - self.vmin)
        vmax = self.vmax + jnp.clip(offset, 0.0, 1.0) * (self.vmax - self.vmin)
        return jnp.array([[[vmin[0], vmax[1]], [vmax[0], vmax[1]]],
                          [[vmax[0], vmax[1]], [vmax[0], vmin[1]]],
                          [[vmax[0], vmin[1]], [vmin[0], vmin[1]]],
                          [[vmin[0], vmin[1]], [vmin[0], vmax[1]]]])

    def project(self, kxy: RealArray) -> RealArray:
        return project_to_rect(kxy, self.vmin, self.vmax)

class Lens():
    def kin_center(self, state: LensState) -> RealArray:
        pupil_roi = jnp.asarray(state.pupil_roi)
        x, y = jnp.mean(pupil_roi[2:]), jnp.mean(pupil_roi[:2])
        return det_to_k(x, y, state.foc_pos, jnp.zeros(x.shape, dtype=int))

    def kin_max(self, state: LensState) -> RealArray:
        return det_to_k(jnp.array(state.pupil_roi[3]), jnp.array(state.pupil_roi[1]),
                        state.foc_pos, jnp.array(0))

    def kin_min(self, state: LensState) -> RealArray:
        return det_to_k(jnp.array(state.pupil_roi[2]), jnp.array(state.pupil_roi[0]),
                        state.foc_pos, jnp.array(0))

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
        kin, is_good = source_lines(miller.q, self.pupil(state).edges())
        laue = LaueVectors(index=miller.index[..., None], q=miller.q[..., None, :],
                           hkl=miller.hkl, kin=kin, kout=kin + miller.q[..., None, :])
        return laue.filter(is_good)

    def project_to_pupil(self, kin: RealArray, state: LensState) -> RealArray:
        kin = safe_divide(kin, jnp.sqrt(jnp.sum(kin**2, axis=-1))[..., None])
        return kxy_to_k(self.pupil(state).project(kin[..., :2]))

    def pupil(self, state: LensState) -> Pupil:
        return Pupil(self.kin_min(state)[:2], self.kin_max(state)[:2])

    def zero_order(self, state: LensState):
        return k_to_det(self.kin_center(state), state.foc_pos, jnp.array(0))

class LaueSampler(Protocol):
    def __call__(self, state: InternalState) -> CBDPoints:
        ...

class Projector(Protocol):
    def __call__(self, laue: LaueVectors, state: InternalState) -> RealArray:
        ...

Criterion = Callable[[State,], RealArray]

class CBDModel():
    lens    : Lens = Lens()
    xtal    : Xtal = Xtal()

    def init(self, rng: KeyArray) -> State:
        raise NotImplementedError

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
        kout = det_to_k(points.x, points.y, smp_pos, arange(smp_pos.shape[:-1]))
        return PointsWithK(index=points.index, points=points.points, kout=kout)

    def kout_to_points(self, laue: LaueVectors, state: InternalState) -> CBDPoints:
        smp_pos = self.kin_to_sample(laue.kin, laue.index, state)
        points = k_to_det(laue.kout, smp_pos, arange(smp_pos.shape[:-1]))
        return CBDPoints(**laue.to_dict(), points=points)

    def init_hkl(self, patterns: Patterns, offsets: IntArray, state: InternalState
                 ) -> Miller:
        hkl = self.patterns_to_hkl(patterns, state)
        hkl = jnp.asarray(jnp.floor(jnp.mean(hkl, axis=-2)), dtype=int)
        return Miller(hkl=hkl, index=patterns.index).offset(offsets)

    def init_offsets(self, patterns: Patterns, state: InternalState) -> IntArray:
        hkl = self.patterns_to_hkl(patterns, state)
        dhkl = jnp.max(hkl[:, 1, :] - hkl[:, 0, :], axis=-2) + 1
        offsets = jnp.meshgrid(jnp.arange(-dhkl[0] // 2 + 1, dhkl[0] // 2 + 1),
                               jnp.arange(-dhkl[1] // 2 + 1, dhkl[1] // 2 + 1),
                               jnp.arange(-dhkl[2] // 2 + 1, dhkl[2] // 2 + 1))
        return jnp.reshape(jnp.stack(offsets, axis=-1), (-1, 3))

    def init_patterns(self, miller: MillerWithRLP, state: InternalState) -> Patterns:
        laue = self.lens.source_lines(miller, state.lens)
        laue = self.kout_to_points(laue, state)
        return Patterns.from_points(laue)

    def index(self, sampler: LaueSampler, projector: Projector, state: InternalState
              ) -> MillerWithRLP:
        laue = sampler(state)
        proj = projector(laue, state)
        dist = jnp.mean(jnp.sum((laue.kin - proj)**2, axis=-1), axis=0)
        idxs = jnp.argmin(dist, axis=0)
        hkl = jnp.take_along_axis(laue.hkl, idxs[None, ..., None], axis=0)[0]
        miller = Miller(hkl=hkl, index=laue.index)
        return self.xtal.hkl_to_q(miller, state.xtal)

    def patterns_to_hkl(self, patterns: Patterns, state: InternalState) -> RealArray:
        rlp = self.patterns_to_q(patterns, state)
        miller = self.xtal.q_to_hkl(rlp, state.xtal)
        hkl = jnp.sort(miller.hkl, axis=-2)
        hkl_min, hkl_max = jnp.floor(hkl[..., 0, :]), jnp.ceil(hkl[..., 1, :])
        return jnp.stack((hkl_min, hkl_max), axis=-2, dtype=int)

    def patterns_to_q(self, patterns: Patterns, state: InternalState) -> RLP:
        kmin, kmax = self.lens.kin_min(state.lens), self.lens.kin_max(state.lens)
        points = patterns.to_points()
        points = self.points_to_kout(points, state)
        kout_min = jnp.min(points.kout[..., :2], axis=-2)
        dkout = jnp.max(points.kout[..., :2], axis=-2) - kout_min
        q = jnp.stack((kxy_to_k(kout_min) - kmin,
                       kxy_to_k(kout_min) - kxy_to_k(kmax[:2] - dkout)), axis=-2)
        return RLP(q=q, index=patterns.index[..., None])

    def static_sampler(self, key: KeyArray, patterns: Patterns, offsets: IntArray,
                       num_points: int, state: InternalState) -> LaueSampler:
        miller = self.init_hkl(patterns, offsets, state)
        generator = lambda key: random.uniform(key, (num_points,) + miller.hkl.shape[:-1])
        points = generate_points(key, generator, patterns.pt0, patterns.pt1)

        def sampler(state: InternalState) -> CBDPoints:
            pts = self.points_to_kout(points, state)
            rlp = self.xtal.hkl_to_q(miller, state.xtal)
            return CBDPoints(index=pts.index, points=pts.points, hkl=rlp.hkl,
                             q=rlp.q, kin=pts.kout - rlp.q, kout=pts.kout)

        return sampler

    def dynamic_sampler(self, key: KeyArray, patterns: Patterns, offsets: IntArray,
                        num_points: int) -> LaueSampler:
        def sampler(state: InternalState) -> CBDPoints:
            miller = self.init_hkl(patterns, offsets, state)
            generator = lambda key: uniform(key_combine(key, miller.hkl_indices), num_points)
            pts = generate_points(key, generator, patterns.pt0, patterns.pt1)
            pts = self.points_to_kout(pts, state)
            rlp = self.xtal.hkl_to_q(miller, state.xtal)
            return CBDPoints(index=pts.index, points=pts.points, hkl=rlp.hkl,
                             q=rlp.q, kin=pts.kout - rlp.q, kout=pts.kout)

        return sampler

    def line_projector(self, laue: LaueVectors, state: InternalState) -> RealArray:
        return self.lens.project_to_pupil(laue.kin_to_source_line(), state.lens)

    def pupil_projector(self, laue: LaueVectors, state: InternalState) -> RealArray:
        return self.lens.project_to_pupil(laue.kin, state.lens)

    def criterion(self, sampler: LaueSampler, projector: Projector, num_inliers: int) -> Criterion:
        def criterion(state: State) -> RealArray:
            int_state = self.to_internal(state)
            laue = sampler(int_state)
            proj = projector(laue, int_state)
            dist = jnp.mean(jnp.sum((laue.kin - proj)**2, axis=-1), axis=0)
            dist = jnp.min(dist, axis=0)
            return jnp.mean(jnp.sort(dist)[:num_inliers])

        return criterion
