from typing import Callable, Protocol
import jax.numpy as jnp
from jax import random, vmap
from .cbc_setup import MillerIndices, LensState, Lens, Patterns, Pupil, XtalState, Xtal
from .dataclasses import jax_dataclass
from .geometry import arange, det_to_k, k_to_det, kxy_to_k, safe_divide, source_lines
from ..annotations import KeyArray, IntArray, RealArray
from ..data_container import DataclassInstance

State = DataclassInstance

def key_combine(key: KeyArray, hkl: IntArray) -> KeyArray:
    array = jnp.reshape(hkl, (-1, 3))

    def combine(key: KeyArray, val: IntArray) -> KeyArray:
        bits = random.key_data(key)
        bits ^= random.key_data(vmap(random.key, 0, 0)(val)) + jnp.uint32(0x9e3779b9) + (bits << 6) + (bits >> 2)
        return random.wrap_key_data(bits)

    keys = jnp.broadcast_to(key, (array.shape[0],))
    keys = combine(combine(combine(keys, array[:, 0]), array[:, 1]), array[:, 2])
    return jnp.reshape(keys, hkl.shape[:-1])

def uniform(keys: KeyArray, size: int) -> RealArray:
    res = vmap(random.uniform, in_axes=(0, None), out_axes=-1)(jnp.ravel(keys), (size,))
    return jnp.reshape(res, (size,) + keys.shape)

@jax_dataclass
class LaueVectors():
    miller  : MillerIndices
    kout    : RealArray
    q       : RealArray
    kin     : RealArray

    def kin_to_source_line(self) -> RealArray:
        q_mag = jnp.sum(self.q**2, axis=-1)
        t = safe_divide(jnp.sum(self.kin * self.q, axis=-1), q_mag) + 0.5
        kin = self.kin - t[..., None] * self.q
        tau = kin + 0.5 * self.q
        tau_mag = jnp.sum(tau**2, axis=-1)
        s = safe_divide(jnp.sum(kin**2, axis=-1) - 1.0,
                        jnp.sum(kin * tau, axis=-1) + jnp.sqrt(tau_mag))
        return kin - s[..., None] * tau

    def source_points(self) -> RealArray:
        rec_abs = jnp.sqrt(jnp.sum(self.q**2, axis=-1))
        theta = jnp.arccos(0.5 * rec_abs) - jnp.arccos(safe_divide(-self.q[..., 2], rec_abs))
        phi = jnp.arctan2(self.q[..., 1], self.q[..., 0])
        return jnp.stack((jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi),
                          jnp.cos(theta)), axis=-1)

@jax_dataclass
class InternalState():
    xtal    : XtalState
    lens    : LensState
    z       : RealArray

class LaueSampler(Protocol):
    def __call__(self, state: InternalState) -> LaueVectors:
        ...

class Projector(Protocol):
    def __call__(self, laue: LaueVectors, pupil: Pupil) -> RealArray:
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

    def detector_to_kout(self, x: RealArray, y: RealArray, idxs: IntArray, state: InternalState
                         ) -> RealArray:
        smp_pos = self.kin_to_sample(self.lens.kin_center(state.lens), idxs, state)
        return det_to_k(x, y, smp_pos, arange(smp_pos.shape[:-1]))

    def kout_to_detector(self, kout: RealArray, idxs: IntArray, rec_vec: RealArray,
                         state: InternalState) -> RealArray:
        smp_pos = self.kin_to_sample(kout - rec_vec, idxs, state)
        return k_to_det(kout, smp_pos, arange(smp_pos.shape[:-1]))

    def init_hkl(self, patterns: Patterns, offsets: IntArray, state: InternalState
                 ) -> MillerIndices:
        hkl = self.patterns_to_hkl(patterns, state)
        hkl = jnp.asarray(jnp.floor(jnp.mean(hkl, axis=-2)), dtype=int)
        return MillerIndices(hkl, patterns.frames).offset(offsets)

    def init_offsets(self, patterns: Patterns, state: InternalState) -> IntArray:
        hkl = self.patterns_to_hkl(patterns, state)
        dhkl = jnp.max(hkl[:, 1, :] - hkl[:, 0, :], axis=-2) + 1
        offsets = jnp.meshgrid(jnp.arange(-dhkl[0] // 2 + 1, dhkl[0] // 2 + 1),
                               jnp.arange(-dhkl[1] // 2 + 1, dhkl[1] // 2 + 1),
                               jnp.arange(-dhkl[2] // 2 + 1, dhkl[2] // 2 + 1))
        return jnp.reshape(jnp.stack(offsets, axis=-1), (-1, 3))

    def init_patterns(self, sampler: LaueSampler, projector: Projector, state: InternalState
                      ) -> Patterns:
        pupil = self.lens.pupil(state.lens)
        laue = sampler(state)
        proj = projector(laue, pupil)
        dist = jnp.mean(jnp.sum((laue.kin - proj)**2, axis=-1), axis=0)
        idxs = jnp.argmin(dist, axis=0)
        miller = MillerIndices(jnp.take_along_axis(laue.miller.hkl, idxs[None, ..., None], axis=0)[0],
                               laue.miller.index)

        rec_vec = self.xtal.hkl_to_q(miller, state.xtal)
        kin, is_good = source_lines(rec_vec, pupil.edges())

        sim = self.kout_to_detector(kin + rec_vec[..., None, :], miller.index[..., None],
                                    rec_vec[..., None, :], state)[is_good]
        return Patterns(frames=miller.index[is_good], lines=jnp.reshape(sim, (-1, 4)),
                        hkl=miller.hkl[is_good])

    def patterns_to_hkl(self, patterns: Patterns, state: InternalState) -> RealArray:
        q = self.patterns_to_q(patterns, state)
        hkl = self.xtal.q_to_hkl(q, patterns.frames[..., None], state.xtal).hkl
        hkl = jnp.sort(hkl, axis=-2)
        hkl_min, hkl_max = jnp.floor(hkl[..., 0, :]), jnp.ceil(hkl[..., 1, :])
        return jnp.stack((hkl_min, hkl_max), axis=-2, dtype=int)

    def patterns_to_kout(self, patterns: Patterns, state: InternalState) -> RealArray:
        return self.detector_to_kout(patterns.x, patterns.y, patterns.frames[..., None], state)

    def patterns_to_q(self, patterns: Patterns, state: InternalState) -> RealArray:
        kmin, kmax = self.lens.kin_min(state.lens), self.lens.kin_max(state.lens)
        kout = self.patterns_to_kout(patterns, state)
        kout_min = jnp.min(kout[..., :2], axis=-2)
        dkout = jnp.max(kout[..., :2], axis=-2) - kout_min
        return jnp.stack((kxy_to_k(kout_min) - kmin,
                          kxy_to_k(kout_min) - kxy_to_k(kmax[:2] - dkout)), axis=-2)

    def static_sampler(self, key: KeyArray, patterns: Patterns, offsets: IntArray,
                       num_points: int, state: InternalState) -> LaueSampler:
        miller = self.init_hkl(patterns, offsets, state)
        t = random.uniform(key, (num_points,) + miller.hkl.shape[:-1])
        points = patterns.points[:, 0] + t[..., None] * \
                (patterns.points[:, 1] - patterns.points[:, 0])

        def sampler(state: InternalState):
            kout = self.detector_to_kout(points[..., 0], points[..., 1],
                                         patterns.frames, state)
            q = self.xtal.hkl_to_q(miller, state.xtal)
            return LaueVectors(miller, kout, q, kout - q)

        return sampler

    def dynamic_sampler(self, key: KeyArray, patterns: Patterns, offsets: IntArray,
                        num_points: int):
        def sampler(state: InternalState):
            miller = self.init_hkl(patterns, offsets, state)
            t = uniform(key_combine(key, miller.hkl), num_points)
            points = patterns.points[:, 0] + t[..., None] * \
                    (patterns.points[:, 1] - patterns.points[:, 0])
            kout = self.detector_to_kout(points[..., 0], points[..., 1],
                                         patterns.frames, state)
            q = self.xtal.hkl_to_q(miller, state.xtal)
            return LaueVectors(miller, kout, q, kout - q)

        return sampler

    def line_projector(self, laue: LaueVectors, pupil: Pupil) -> RealArray:
        return self.lens.project_to_pupil(laue.kin_to_source_line(), pupil)

    def pupil_projector(self, laue: LaueVectors, pupil: Pupil) -> RealArray:
        return self.lens.project_to_pupil(laue.kin, pupil)

    def criterion(self, sampler: LaueSampler, projector: Projector, num_inliers: int) -> Criterion:
        def criterion(state: State) -> RealArray:
            int_state = self.to_internal(state)
            pupil = self.lens.pupil(int_state.lens)
            laue = sampler(int_state)
            proj = projector(laue, pupil)
            dist = jnp.mean(jnp.sum((laue.kin - proj)**2, axis=-1), axis=0)
            dist = jnp.min(dist, axis=0)
            return jnp.mean(jnp.sort(dist)[:num_inliers])

        return criterion
