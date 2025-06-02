from typing import Callable, Iterator, List, Literal, Protocol, Sequence, Tuple
from dataclasses import dataclass
import jax.numpy as jnp
from jax import random, vmap
from .cbc_data import (AnyPoints, CBData, CBDPoints, CircleState, LaueVectors, Miller,
                       MillerWithRLP, Patterns, PointsWithK, RLP, Rotograms, UCA)
from .cbc_setup import (EulerState, BaseLens, BaseSetup, BaseState, RotationState, TiltState,
                        TiltOverAxisState, XtalState)
from .geometry import (arange, det_to_k, k_to_det, k_to_smp, kxy_to_k, project_to_rect,
                       safe_divide, source_lines)
from .._src.annotations import ArrayNamespace, KeyArray, IntArray, JaxNumPy, NumPy, RealArray
from .._src.data_container import add_at, array_namespace
from .._src.src.bresenham import accumulate_lines
from .._src.src.label import PointSet3D, Structure3D, binary_dilation, center_of_mass, label
from .._src.state import State
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

    def to_patterns(self, streaks: Streaks) -> Patterns:
        xp = array_namespace(streaks)
        pts = xp.stack(self.to_coordinates(xp.asarray(streaks.x), xp.asarray(streaks.y)), axis=-1)
        return Patterns(xp.asarray(streaks.index), xp.reshape(pts, pts.shape[:-2] + (4,)))

    def to_streaks(self, patterns: Patterns) -> Streaks:
        xp = array_namespace(patterns)
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
    def hkl_in_aperture(self, theta: float | RealArray, hkl: IntArray, state: XtalState,
                        xp: ArrayNamespace) -> Miller:
        index = xp.broadcast_to(xp.expand_dims(xp.arange(len(state)),
                                               axis=tuple(range(1, hkl.ndim))),
                                (len(state),) + hkl.shape[:-1])
        hkl = xp.broadcast_to(hkl[None, ...], (len(state),) + hkl.shape)
        miller = Miller(hkl=hkl, index=index)

        miller = self.hkl_to_q(miller, state, xp)
        rec_abs = xp.sqrt((miller.q**2).sum(axis=-1))
        rec_th = xp.arccos(-miller.q[..., 2] / rec_abs)
        src_th = rec_th - xp.arccos(0.5 * rec_abs)
        return miller[xp.where((xp.abs(src_th) < theta))]

    def hkl_in_ball(self, q_abs: float | RealArray, state: XtalState, xp: ArrayNamespace
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

    def hkl_to_q(self, miller: Miller, state: XtalState, xp: ArrayNamespace) -> MillerWithRLP:
        basis = xp.reshape(state.basis, (-1, 3, 3))
        q = xp.sum(basis[miller.index] * miller.hkl_indices[..., None], axis=-2)
        return MillerWithRLP(index=miller.index, hkl=miller.hkl, q=q)

    def q_to_hkl(self, rlp: RLP, state: XtalState, xp: ArrayNamespace) -> MillerWithRLP:
        basis = xp.reshape(xp.linalg.inv(state.basis), (-1, 3, 3))
        hkl = xp.sum(basis[rlp.index] * rlp.q[..., None], axis=-2)
        return MillerWithRLP(index=rlp.index, hkl=hkl, q=rlp.q)

    def hkl_bounds(self, rlp1: RLP, rlp2: RLP, state: XtalState, xp: ArrayNamespace
                   ) -> Tuple[Miller, Miller]:
        miller1, miller2 = self.q_to_hkl(rlp1, state, xp), self.q_to_hkl(rlp2, state, xp)
        hkl_min, hkl_max = xp.sort(xp.stack((miller1.hkl, miller2.hkl)), axis=0)
        hkl_min, hkl_max = xp.floor(hkl_min), xp.ceil(hkl_max)
        return Miller(index=rlp1.index, hkl=hkl_min), Miller(index=rlp1.index, hkl=hkl_max)

    def hkl_offsets(self, hkl_min: Miller, hkl_max: Miller, xp: ArrayNamespace) -> IntArray:
        dhkl = xp.max(xp.reshape(hkl_max.hkl_indices - hkl_min.hkl_indices, (-1, 3)), axis=-2) + 1
        offsets = xp.meshgrid(xp.arange(-dhkl[0] // 2 + 1, dhkl[0] // 2 + 1),
                              xp.arange(-dhkl[1] // 2 + 1, dhkl[1] // 2 + 1),
                              xp.arange(-dhkl[2] // 2 + 1, dhkl[2] // 2 + 1))
        return xp.reshape(xp.stack(offsets, axis=-1), (-1, 3))

    def hkl_range(self, indices: Sequence[int] | IntArray, hkl: IntArray, state: XtalState,
                  xp: ArrayNamespace) -> Iterator[MillerWithRLP]:
        if len(state) == 1:
            q = xp.sum(xp.reshape(state.basis, (-1, 3, 3)) * hkl[..., None], axis=-2)
            for index in indices:
                yield MillerWithRLP(index=xp.full(hkl.shape[:-1], index), hkl=hkl, q=q)
        elif len(state) == len(indices):
            for index in indices:
                miller = Miller(index=xp.full(hkl.shape[:-1], index), hkl=hkl)
                yield self.hkl_to_q(miller, state, xp)
        else:
            raise ValueError(f'The length of state ({len(state):d}) is incompatible with ' \
                             f'the length of indices ({len(indices):d})')

class Lens():
    def kin_center(self, state: BaseLens | BaseSetup, xp: ArrayNamespace) -> RealArray:
        return det_to_k(xp.asarray(state.pupil_center), xp.asarray(state.foc_pos), xp.array(0), xp)

    def kin_max(self, state: BaseLens | BaseSetup, xp: ArrayNamespace) -> RealArray:
        return det_to_k(xp.asarray(state.pupil_max), xp.asarray(state.foc_pos), xp.array(0), xp)

    def kin_min(self, state: BaseLens | BaseSetup, xp: ArrayNamespace) -> RealArray:
        return det_to_k(xp.asarray(state.pupil_min), xp.asarray(state.foc_pos), xp.array(0), xp)

    def kin_edges(self, state: BaseLens | BaseSetup, xp: ArrayNamespace) -> RealArray:
        kmin, kmax = self.kin_min(state, xp), self.kin_max(state, xp)
        return xp.array([[[kmin[0], kmax[1]], [kmax[0], kmax[1]]],
                         [[kmax[0], kmax[1]], [kmax[0], kmin[1]]],
                         [[kmax[0], kmin[1]], [kmin[0], kmin[1]]],
                         [[kmin[0], kmin[1]], [kmin[0], kmax[1]]]])

    def kin_to_sample(self, kin: RealArray, z: RealArray, idxs: IntArray,
                      state: BaseLens | BaseSetup, xp: ArrayNamespace) -> RealArray:
        """Project incident wave-vectors to the sample planes.

        Args:
            setup : Experimental setup.
            kin : An array of incident wave-vectors.
            idxs : Sample indices.

        Returns:
            Array of sample coordinates.
        """
        if z.size > 1:
            z = xp.reshape(xp.ravel(z)[xp.ravel(idxs)], idxs.shape)
        return k_to_smp(kin, z, xp.asarray(state.foc_pos), xp)

    def source_lines(self, miller: MillerWithRLP, state: BaseLens | BaseSetup, xp: ArrayNamespace
                     ) -> LaueVectors:
        kin, is_good = source_lines(miller.q, self.kin_edges(state, xp), xp=xp)
        index = xp.broadcast_to(miller.index, miller.q.shape[:-1])
        laue = LaueVectors(index=index[..., None], hkl=miller.hkl, q=miller.q[..., None, :],
                           kin=kin, kout=kin + miller.q[..., None, :])
        return laue[is_good]

    def project_to_pupil(self, kin: RealArray, state: BaseLens | BaseSetup, xp: ArrayNamespace
                         ) -> RealArray:
        kin = safe_divide(kin, xp.sqrt(xp.sum(kin**2, axis=-1))[..., None], xp)
        kxy = project_to_rect(kin[..., :2], self.kin_min(state, xp)[:2],
                              self.kin_max(state, xp)[:2], xp)
        return kxy_to_k(kxy, xp)

    def zero_order(self, state: BaseLens | BaseSetup, xp: ArrayNamespace):
        return k_to_det(self.kin_center(state, xp), xp.asarray(state.foc_pos), xp.array(0), xp)

    def line_projector(self, laue: LaueVectors, state: BaseLens | BaseSetup, xp: ArrayNamespace
                       ) -> RealArray:
        return self.project_to_pupil(laue.source_line, state, xp)

    def pupil_projector(self, laue: LaueVectors, state: BaseLens | BaseSetup, xp: ArrayNamespace
                        ) -> RealArray:
        return self.project_to_pupil(laue.kin, state, xp)

class LaueSampler(Protocol):
    def __call__(self, state: BaseState) -> CBDPoints:
        ...

class Projector(Protocol):
    def __call__(self, laue: LaueVectors, state: BaseLens | BaseSetup, xp: ArrayNamespace) -> RealArray:
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

    def kin_to_sample(self, kin: RealArray, idxs: IntArray, state: BaseState | BaseSetup
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
        return self.lens.kin_to_sample(kin, xp.asarray(state.z), idxs, state, xp)

    def points_to_kout(self, points: AnyPoints, state: BaseState | BaseSetup) -> PointsWithK:
        xp = state.__array_namespace__()
        if isinstance(points, CBDPoints):
            kin = points.kin
        else:
            kin = self.lens.kin_center(state, xp)
        smp_pos = self.kin_to_sample(kin, points.index, state)
        kout = det_to_k(points.points, smp_pos, arange(smp_pos.shape[:-1]), xp)
        return PointsWithK(index=points.index, points=points.points, kout=kout)

    def kout_to_points(self, laue: LaueVectors, state: BaseState | BaseSetup) -> CBDPoints:
        xp = state.__array_namespace__()
        smp_pos = self.kin_to_sample(laue.kin, laue.index, state)
        points = k_to_det(laue.kout, smp_pos, arange(smp_pos.shape[:-1], xp), xp)
        return CBDPoints(**laue.to_dict(), points=points)

    def patterns_to_kout(self, patterns: Patterns, state: BaseState | BaseSetup
                         ) -> Tuple[RealArray, RealArray]:
        xp = state.__array_namespace__()
        pts = self.points_to_kout(patterns.points, state)
        return xp.min(pts.kout[..., :2], axis=-2), xp.max(pts.kout[..., :2], axis=-2)

    def patterns_to_q(self, patterns: Patterns, state: BaseState | BaseSetup) -> Tuple[RLP, RLP]:
        xp = state.__array_namespace__()
        kmin, kmax = self.lens.kin_min(state, xp), self.lens.kin_max(state, xp)
        kout_min, kout_max = self.patterns_to_kout(patterns, state)
        q1 = kxy_to_k(kout_min, xp) - kmin
        q2 = kxy_to_k(kout_max, xp) - kmax
        return RLP(index=patterns.index, q=q1), RLP(index=patterns.index, q=q2)

@dataclass
class CBDIndexer(CBDSetup):
    num_points  : int = 100
    circle      : Circle = Circle()

    @classmethod
    def rho_map(cls) -> float:
        return 1.0

    @classmethod
    def step(cls, shape: Tuple[int, ...], xp: ArrayNamespace) -> RealArray:
        return 2.0 * cls.rho_map() / (xp.asarray([shape[-3], shape[-2], shape[-1]]) - 1)

    def phi(self, xp: ArrayNamespace) -> RealArray:
        return xp.linspace(-2 * xp.pi, 0.0, self.num_points)

    def patterns_to_uca(self, patterns: Patterns, points: PointsWithK, state: BaseState | BaseSetup
                        ) -> UCA:
        xp = state.__array_namespace__()
        kmin, kmax = self.lens.kin_min(state, xp), self.lens.kin_max(state, xp)
        kout_min, kout_max = self.patterns_to_kout(patterns, state)
        xy = xp.stack((kout_min - kmin[:2], kout_max - kmax[:2]))
        xy_min, xy_max = xp.min(xy, axis=0), xp.max(xy, axis=0)
        return UCA(patterns.index, xp.arange(patterns.shape[0]), points.kout,
                   points.kout[:, :2] - xy_min, points.kout[:, :2] - xy_max)

    def candidates(self, candidates: MillerWithRLP, uca: UCA, xp: ArrayNamespace=NumPy
                   ) -> Tuple[MillerWithRLP, UCA]:
        min_res, max_res = uca.min_resolution, uca.max_resolution
        rlp_res = xp.sum(candidates.q**2, axis=-1)
        idxs, rlp_idxs = xp.where(uca.index[..., None] == candidates.index)
        mask = (rlp_res[rlp_idxs] > min_res[idxs]) & (rlp_res[rlp_idxs] < max_res[idxs])
        return candidates[rlp_idxs[mask]], uca[idxs[mask]]

    def intersection(self, rlp: MillerWithRLP, uca: UCA, xp: ArrayNamespace=NumPy) -> CircleState:
        rlp_res = xp.sum(rlp.q**2, axis=-1)
        radius = xp.sqrt(rlp_res - 0.25 * rlp_res**2)
        center = 0.5 * rlp_res[..., None] * uca.kout
        axis1 = xp.stack((uca.kout[..., 1], -uca.kout[..., 0], xp.zeros(uca.kout.shape[:-1])),
                          axis=-1)
        axis1 = axis1 / xp.sqrt(xp.sum(axis1**2, axis=-1, keepdims=True))
        axis2 = xp.cross(uca.kout, axis1)
        return CircleState(rlp.index, center, axis1, axis2, radius)

    def uca_endpoints(self, circle: CircleState, uca: UCA, xp: ArrayNamespace=NumPy) -> RealArray:
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

    def rotations(self, rlp: MillerWithRLP, midpoints: RealArray, xp: ArrayNamespace = NumPy
                  ) -> TiltOverAxisState:
        source = rlp.q / xp.sqrt(xp.sum(rlp.q**2, axis=-1, keepdims=True))
        target = midpoints / xp.sqrt(xp.sum(midpoints**2, axis=-1, keepdims=True))

        bisector = source + target
        bisector = bisector / xp.sqrt(xp.sum(bisector**2, axis=-1, keepdims=True))
        S, C = xp.sin(0.5 * self.phi(xp)), xp.cos(0.5 * self.phi(xp))
        prod = xp.sum(target * bisector, axis=-1)
        cross = xp.cross(bisector, target)

        theta = 2.0 * xp.arccos(-S * prod[..., None])
        S2 = xp.sin(0.5 * theta)
        axis = (C[..., None] * bisector[..., None, :] + S[..., None] * cross[..., None, :])
        return TiltOverAxisState(theta, axis / S2[..., None])

    def rotograms(self, rlp: MillerWithRLP, circle: CircleState, uca: UCA,
                  xp: ArrayNamespace) -> Rotograms:
        endpoints = self.uca_endpoints(circle, uca, xp)
        midpoints = self.circle(xp.mean(endpoints, axis=0), circle)
        tilts = self.rotations(rlp, midpoints, xp)
        return Rotograms.from_tilts(tilts, uca.index, uca.streak_id, xp)

    def index(self, candidates: MillerWithRLP, patterns: Patterns, points: PointsWithK,
              state: BaseState | BaseSetup):
        xp = state.__array_namespace__()
        patterns_uca = self.patterns_to_uca(patterns, points, state)
        rlp, uca = self.candidates(candidates, patterns_uca, xp)
        circles = self.intersection(rlp, uca, xp)
        return self.rotograms(rlp, circles, uca, xp)

    def rotogrid(self, shape: Tuple[int, int, int], return_step: bool=False,
                 xp: ArrayNamespace=NumPy) -> RealArray | Tuple[RealArray, RealArray]:
        indices = xp.stack(xp.meshgrid(xp.arange(shape[0]), xp.arange(shape[1]),
                                       xp.arange(shape[2])))
        coords = indices * self.step(shape, xp) - self.rho_map()
        if return_step:
            return coords, self.step(shape, xp)
        return coords

    def rotomap(self, shape: Tuple[int, int, int], rotograms: Rotograms,
                width: float, num_threads: int=1) -> RealArray:
        xp = rotograms.__array_namespace__()
        lines = (rotograms.lines + self.rho_map()) / xp.tile(self.step(shape, xp), 2)
        lines = xp.concatenate((lines, xp.full(rotograms.lines.shape[:-1] + (1,), width)),
                               axis=-1)
        _, indices, counts = xp.unique(rotograms.streak_id, return_index=True, return_counts=True)
        frames = rotograms.inverse()

        rmap = accumulate_lines(lines, (len(rotograms),) + shape, lines.shape[-2] * counts,
                                frames[indices], kernel='gaussian', in_overlap='max',
                                out_overlap='sum', num_threads=num_threads)
        return xp.asarray(rmap)

    def to_peaks(self, rotomap: RealArray, threshold: float, n_max: int=30) -> List[PointSet3D]:
        xp = array_namespace(rotomap)

        rotomap = rotomap / xp.max(rotomap, axis=(-3, -2, -1), keepdims=True)
        f, z, y, x = xp.where(rotomap > threshold)
        indices = xp.lexsort((rotomap[f, z, y, x], f))
        frames, counts = xp.unique(f, return_counts=True)
        mask = xp.concatenate([xp.arange(size - 1, -1, -1) < n_max for size in counts])

        f, z, y, x = f[indices[mask]], z[indices[mask]], y[indices[mask]], x[indices[mask]]
        return [PointSet3D(x[f == frame], y[f == frame], z[f == frame]) for frame in frames]

    def refine_peaks(self, peaks: List[PointSet3D], rotomap: RealArray, vicinity: Structure3D,
                     connectivity: Structure3D=Structure3D(1, 1), num_threads: int=1
                     ) -> Tuple[IntArray, TiltOverAxisState]:
        xp = array_namespace(rotomap)
        mask = binary_dilation(xp.zeros(rotomap.shape, dtype=bool), vicinity, peaks,
                               num_threads=num_threads)
        regions = label(mask, connectivity, peaks, num_threads=num_threads)
        frames = xp.concatenate([xp.full(len(region), index)
                                 for index, region in enumerate(regions)])
        centers = xp.concatenate(center_of_mass(regions, rotomap))
        centers = centers * self.step(rotomap.shape, xp) - self.rho_map()
        return frames, TiltOverAxisState.from_point(centers)

class CBDModel(CBDSetup):
    def hkl_in_aperture(self, q_abs: float, state: BaseState) -> Miller:
        xp = state.__array_namespace__()
        hkl = self.xtal.hkl_in_ball(q_abs, state.xtal, xp)
        kz = xp.asarray([self.lens.kin_min(state, xp)[..., 2],
                         self.lens.kin_max(state, xp)[..., 2]])
        return self.xtal.hkl_in_aperture(xp.arccos(xp.min(kz)), hkl, state.xtal, xp)

    def init_hkl(self, hkl_min: Miller, hkl_max: Miller, offsets: IntArray) -> Miller:
        return Miller(index=hkl_min.index,
                      hkl=(hkl_min.hkl_indices + hkl_max.hkl_indices) // 2).offset(offsets)

    def init_patterns(self, miller: MillerWithRLP, state: BaseState | BaseSetup) -> Patterns:
        xp = state.__array_namespace__()
        laue = self.lens.source_lines(miller, state, xp)
        laue = self.kout_to_points(laue, state)
        return Patterns.from_points(laue)

    def init_data(self, key: KeyArray, patterns: Patterns, num_points: int,
                  state: BaseState, quantile: float=1.0, combine_key: bool=False) -> 'CBData':
        xp = state.__array_namespace__()
        if xp.clip(quantile, 0.0, 1.0) == 0.0:
            raise ValueError(f"Invalid quantile value: {quantile}")

        q1, q2 = self.patterns_to_q(patterns, state)
        hkl_min, hkl_max = self.xtal.hkl_bounds(q1, q2, state.xtal, xp)
        offsets = self.xtal.hkl_offsets(hkl_min, hkl_max, xp)
        miller = self.init_hkl(hkl_min, hkl_max, offsets)

        if combine_key:
            x = uniform(key_combine(key, miller.hkl_indices), num_points, xp)
        else:
            x = xp.asarray(random.uniform(key, (num_points,) + miller.hkl.shape[:-1]))

        if xp.clip(quantile, 0.0, 1.0) < 1.0:
            _, counts = xp.unique(patterns.index, return_counts=True)
            counts = xp.clip(counts, 1, jnp.inf)
            mask = xp.concatenate([xp.arange(size) < quantile * size for size in counts])
        else:
            mask = xp.ones(patterns.shape[0], dtype=bool)
        return CBData(miller, patterns.sample(x), mask)

    def line_loss(self, loss: Loss='l1', xp: ArrayNamespace = JaxNumPy
                  ) -> 'CBDLoss':
        return CBDLoss(self, self.lens.line_projector, loss_function(loss, xp))

    def pupil_loss(self, loss: Loss='l1', xp: ArrayNamespace = JaxNumPy
                   ) -> 'CBDLoss':
        return CBDLoss(self, self.lens.pupil_projector, loss_function(loss, xp))

@dataclass(frozen=True, unsafe_hash=True)
class CBDLoss():
    model           : CBDModel
    projector       : Projector
    loss_fn         : LossFn

    def project_data(self, data: CBData, state: BaseState) -> CBDPoints:
        xp = state.__array_namespace__()
        pts = self.model.points_to_kout(data.points, state)
        rlp = self.model.xtal.hkl_to_q(data.miller, state.xtal, xp)
        return CBDPoints(index=pts.index, points=pts.points, hkl=rlp.hkl,
                         q=rlp.q, kin=pts.kout - rlp.q, kout=pts.kout)

    def __call__(self, data: CBData, state: BaseState) -> RealArray:
        xp = state.__array_namespace__()
        points = self.project_data(data, state)
        projected = self.projector(points, state, xp)
        dist = xp.mean(xp.sum(self.loss_fn(projected, points.kin), axis=-1), axis=0)
        dist = xp.min(dist, axis=0)
        indices = xp.lexsort((dist, data.points.index), axis=0)
        return xp.mean(dist[indices] * data.mask)

    def index(self, data: CBData, state: BaseState) -> Miller:
        xp = state.__array_namespace__()
        points = self.project_data(data, state)
        projected = self.projector(points, state, xp)
        dist = xp.mean(xp.sum(self.loss_fn(projected, points.kin), axis=-1), axis=0)
        idxs = xp.argmin(dist, axis=0)
        hkl = xp.take_along_axis(points.hkl, idxs[None, ..., None], axis=0)[0]
        return Miller(index=points.index, hkl=hkl)

    def per_pattern(self, data: CBData, state: BaseState) -> RealArray:
        xp = state.__array_namespace__()
        points = self.project_data(data, state)
        projected = self.projector(points, state, xp)
        dist = xp.mean(xp.sum(self.loss_fn(projected, points.kin), axis=-1), axis=0)
        dist = xp.min(dist, axis=0)
        indices = xp.lexsort((dist, points.index), axis=0)
        return add_at(xp.zeros(len(state.xtal)), points.index, dist[indices] * data.mask, xp)

    def per_streak(self, data: CBData, state: BaseState) -> RealArray:
        xp = state.__array_namespace__()
        crit = self.per_pattern(data, state)
        n_streaks = add_at(xp.zeros(len(state.xtal)), data.points.index, data.mask, xp)
        return crit / n_streaks
