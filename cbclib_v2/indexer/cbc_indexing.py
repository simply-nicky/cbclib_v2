from typing import Callable, Iterator, Literal, Protocol, Sequence, Tuple
from dataclasses import dataclass
from .cbc_data import (AnyPoints, CBData, CBDataBest, CBDataMasked, CBDPoints, CircleState,
                       LaueVectors, MaskedLaueVectors, Miller, MillerWithRLP, Patterns, PointsWithK,
                       RLP, Rotograms, UCA)
from .cbc_setup import (BaseState, IndexingResult, ResolvedLens, ResolvedSetup, ResolvedState,
                        TiltOverAxisState, XtalState)
from .geometry import (broadcast_to, det_to_k, k_to_det, k_to_smp, kxy_to_k, project_to_rect,
                       safe_divide, source_lines)
from .._src.annotations import (AnyNamespace, BoolArray, AnyGenerator, IntArray, JaxNumPy, NumPy,
                                RealArray)
from .._src.array_api import add_at, array_namespace
from .._src.functions import Structure, accumulate_lines, binary_dilation, center_of_mass, label
from .._src.state import State

class Xtal():
    def hkl_meshgrid(self, h_vals: IntArray, k_vals: IntArray, l_vals: IntArray,
                     xp: AnyNamespace) -> IntArray:
        h_grid, k_grid, l_grid = xp.meshgrid(h_vals, k_vals, l_vals)
        return xp.stack((xp.reshape(h_grid, -1), xp.reshape(k_grid, -1),
                         xp.reshape(l_grid, -1)), axis=1)

    def hkl_in_aperture(self, theta: float | RealArray, hkl: IntArray, state: XtalState,
                        xp: AnyNamespace) -> MillerWithRLP:
        index = xp.broadcast_to(xp.arange(len(state))[:, None],
                                (len(state), hkl.size // hkl.shape[-1]))
        index = xp.reshape(index, (len(state),) + hkl.shape[:-1])
        hkl = xp.broadcast_to(hkl, (len(state),) + hkl.shape)
        miller = Miller(hkl=hkl, index=index)

        miller = self.hkl_to_q(miller, state, xp)
        rec_abs = xp.sqrt((miller.q**2).sum(axis=-1))
        rec_th = xp.acos(-miller.q[..., 2] / rec_abs)
        src_th = rec_th - xp.acos(0.5 * rec_abs)
        return miller[xp.where((xp.abs(src_th) < theta))]

    def hkl_in_ball(self, q_abs: float | RealArray, state: XtalState, xp: AnyNamespace
                    ) -> IntArray:
        lat_size = xp.asarray(xp.rint(q_abs / state.unit_cell.lengths), dtype=int)
        lat_size = xp.max(xp.reshape(lat_size, (-1, 3)), axis=0)
        hkl = self.hkl_meshgrid(xp.arange(-int(lat_size[0]), int(lat_size[0]) + 1),
                                xp.arange(-int(lat_size[1]), int(lat_size[1]) + 1),
                                xp.arange(-int(lat_size[2]), int(lat_size[2]) + 1), xp)
        hkl = hkl[xp.any(hkl != 0, axis=-1)]

        rec_vec = xp.tensordot(hkl, state.basis, axes=(-1, -2))
        rec_abs = xp.sqrt(xp.sum(rec_vec**2, axis=-1))
        rec_abs = xp.reshape(rec_abs, (hkl.shape[0], -1))
        return hkl[xp.any(rec_abs < q_abs, axis=-1)]

    def hkl_to_q(self, miller: Miller, state: XtalState, xp: AnyNamespace) -> MillerWithRLP:
        basis = xp.reshape(state.basis, (-1, 3, 3))
        q = xp.sum(basis[miller.index] * miller.hkl_indices[..., None], axis=-2)
        return MillerWithRLP(index=miller.index, hkl=miller.hkl, q=xp.reshape(q, miller.hkl.shape))

    def q_to_hkl(self, rlp: RLP, state: XtalState, xp: AnyNamespace) -> MillerWithRLP:
        basis = xp.reshape(xp.linalg.inv(state.basis), (-1, 3, 3))
        hkl = xp.sum(basis[rlp.index] * rlp.q[..., None], axis=-2)
        return MillerWithRLP(index=rlp.index, hkl=hkl, q=rlp.q)

    def hkl_bounds(self, rlp1: RLP, rlp2: RLP, state: XtalState, xp: AnyNamespace
                   ) -> Tuple[Miller, Miller]:
        miller1, miller2 = self.q_to_hkl(rlp1, state, xp), self.q_to_hkl(rlp2, state, xp)
        hkl_min, hkl_max = xp.sort(xp.stack((miller1.hkl, miller2.hkl)), axis=0)
        hkl_min, hkl_max = xp.floor(hkl_min), xp.ceil(hkl_max)
        return Miller(index=rlp1.index, hkl=hkl_min), Miller(index=rlp1.index, hkl=hkl_max)

    def hkl_offsets(self, hkl_min: Miller, hkl_max: Miller, xp: AnyNamespace) -> IntArray:
        dhkl = xp.max(xp.reshape(hkl_max.hkl_indices - hkl_min.hkl_indices, (-1, 3)), axis=-2) + 1
        offsets = xp.meshgrid(xp.arange(-int(dhkl[0] // 2 + 1), int(dhkl[0] // 2 + 1)),
                              xp.arange(-int(dhkl[1] // 2 + 1), int(dhkl[1] // 2 + 1)),
                              xp.arange(-int(dhkl[2] // 2 + 1), int(dhkl[2] // 2 + 1)))
        return xp.reshape(xp.stack(offsets, axis=-1), (-1, 3))

    def hkl_range(self, indices: Sequence[int] | IntArray, hkl: IntArray, state: XtalState,
                  xp: AnyNamespace) -> Iterator[MillerWithRLP]:
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
    def kin_center(self, state: ResolvedLens, xp: AnyNamespace) -> RealArray:
        return det_to_k(state.pupil_center, state.foc_pos, xp)

    def kin_max(self, state: ResolvedLens, xp: AnyNamespace) -> RealArray:
        return det_to_k(state.pupil_max, state.foc_pos, xp)

    def kin_min(self, state: ResolvedLens, xp: AnyNamespace) -> RealArray:
        return det_to_k(state.pupil_min, state.foc_pos, xp)

    def kin_edges(self, state: ResolvedLens, xp: AnyNamespace) -> RealArray:
        kmin, kmax = self.kin_min(state, xp), self.kin_max(state, xp)
        p00 = xp.stack((kmin[..., 0], kmax[..., 1]), axis=-1)
        p01 = xp.stack((kmax[..., 0], kmax[..., 1]), axis=-1)
        p10 = xp.stack((kmax[..., 0], kmin[..., 1]), axis=-1)
        p11 = xp.stack((kmin[..., 0], kmin[..., 1]), axis=-1)
        return xp.stack((xp.stack((p00, p01), axis=-2),
                         xp.stack((p01, p10), axis=-2),
                         xp.stack((p10, p11), axis=-2),
                         xp.stack((p11, p00), axis=-2)), axis=-3)

    def source_lines(self, miller: MillerWithRLP, state: ResolvedLens, xp: AnyNamespace
                     ) -> MaskedLaueVectors:
        edges = self.kin_edges(state, xp)
        edges = broadcast_to(edges, miller.index, (4, 2, 2), xp)
        kin, is_good = source_lines(miller.q, edges, xp=xp)
        index = xp.broadcast_to(miller.index, miller.q.shape[:-1])
        return MaskedLaueVectors(index=index[..., None], kout=kin + miller.q[..., None, :],
                                 hkl=miller.hkl, kin=kin, q=miller.q[..., None, :],
                                 mask=is_good[..., None])

    def project_to_pupil(self, kin: RealArray, idxs: IntArray, state: ResolvedLens,
                         xp: AnyNamespace) -> RealArray:
        kin = safe_divide(kin, xp.sqrt(xp.sum(kin**2, axis=-1))[..., None], xp)
        kmin_xy = broadcast_to(self.kin_min(state, xp)[..., :2], idxs, (2,), xp)
        kmax_xy = broadcast_to(self.kin_max(state, xp)[..., :2], idxs, (2,), xp)
        kxy = project_to_rect(kin[..., :2], kmin_xy, kmax_xy, xp)
        return kxy_to_k(kxy, xp)

    def zero_order(self, state: ResolvedLens, xp: AnyNamespace) -> RealArray:
        return k_to_det(self.kin_center(state, xp), xp.asarray(state.foc_pos), xp)

    def line_projector(self, laue: LaueVectors, state: ResolvedLens, xp: AnyNamespace
                       ) -> RealArray:
        return self.project_to_pupil(laue.source_line, laue.index, state, xp)

    def pupil_projector(self, laue: LaueVectors, state: ResolvedLens, xp: AnyNamespace
                        ) -> RealArray:
        return self.project_to_pupil(laue.kin, laue.index, state, xp)

class Projector(Protocol):
    def __call__(self, laue: LaueVectors, state: ResolvedLens, xp: AnyNamespace
                 ) -> RealArray:
        ...

Criterion = Callable[[State,], RealArray]
LossFn = Callable[[RealArray, RealArray], RealArray]
Loss = Literal['l1', 'l2', 'log_cosh']

def loss_function(loss: Loss, xp: AnyNamespace = JaxNumPy) -> LossFn:
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

    def points_to_kout(self, points: AnyPoints, state: ResolvedSetup, xp: AnyNamespace
                       ) -> PointsWithK:
        if isinstance(points, CBDPoints):
            kin = points.kin
            z = broadcast_to(state.z, points.index, (), xp)
            foc_pos = broadcast_to(state.lens.foc_pos, points.index, (3,), xp)
            smp_pos = k_to_smp(kin, z, foc_pos, xp)
        else:
            kin = self.lens.kin_center(state.lens, xp)
            smp_pos = k_to_smp(kin, state.z, state.lens.foc_pos, xp)
            smp_pos = broadcast_to(smp_pos, points.index, (3,), xp)

        kout = det_to_k(points.points, smp_pos, xp)
        return PointsWithK(index=points.index, points=points.points, kout=kout)

    def kout_to_points(self, laue: MaskedLaueVectors, state: ResolvedSetup, xp: AnyNamespace
                       ) -> CBDPoints:
        def wrapper(laue: MaskedLaueVectors, state: ResolvedSetup) -> RealArray:
            z = broadcast_to(state.z, laue.index, (), xp)
            foc_pos = broadcast_to(state.lens.foc_pos, laue.index, (3,), xp)
            smp_pos = k_to_smp(laue.kin, z, foc_pos, xp)
            return k_to_det(laue.kout, smp_pos, xp)

        points = xp.where(laue.mask[..., None], wrapper(laue, state), xp.nan)
        return CBDPoints(index=laue.index, points=points, q=laue.q, hkl=laue.hkl,
                         kin=laue.kin, kout=laue.kout)

    def patterns_to_kout(self, patterns: Patterns, state: ResolvedSetup, xp: AnyNamespace
                         ) -> Tuple[RealArray, RealArray]:
        pts = self.points_to_kout(patterns.points, state, xp)
        return xp.min(pts.kout[..., :2], axis=-2), xp.max(pts.kout[..., :2], axis=-2)

    def patterns_to_q(self, patterns: Patterns, state: ResolvedSetup, xp: AnyNamespace
                      ) -> Tuple[RLP, RLP]:
        kmin, kmax = self.lens.kin_min(state.lens, xp), self.lens.kin_max(state.lens, xp)
        kout_min, kout_max = self.patterns_to_kout(patterns, state, xp)

        kmin = broadcast_to(kmin, patterns.index, (3,), xp)
        kmax = broadcast_to(kmax, patterns.index, (3,), xp)
        q1 = kxy_to_k(kout_min, xp) - kmin
        q2 = kxy_to_k(kout_max, xp) - kmax
        return RLP(index=patterns.index, q=q1), RLP(index=patterns.index, q=q2)

    def init_patterns(self, miller: MillerWithRLP, state: ResolvedSetup, xp: AnyNamespace
                      ) -> Patterns:
        laue = self.lens.source_lines(miller, state.lens, xp)
        laue = self.kout_to_points(laue, state, xp)
        return Patterns.from_points(laue)

@dataclass
class CBDIndexer(CBDSetup):
    num_points  : int = 100

    @classmethod
    def rho_map(cls) -> float:
        return 1.0

    @classmethod
    def step(cls, shape: Tuple[int, ...], xp: AnyNamespace) -> RealArray:
        return 2.0 * cls.rho_map() / (xp.asarray([shape[-3], shape[-2], shape[-1]]) - 1)

    def phi(self, xp: AnyNamespace) -> RealArray:
        return xp.linspace(-2 * xp.pi, 0.0, self.num_points)

    def patterns_to_uca(self, patterns: Patterns, points: PointsWithK, state: ResolvedSetup,
                        xp: AnyNamespace) -> UCA:
        kmin, kmax = self.lens.kin_min(state.lens, xp), self.lens.kin_max(state.lens, xp)
        kout_min, kout_max = self.patterns_to_kout(patterns, state, xp)
        xy = xp.stack((kout_min - kmin[..., :2], kout_max - kmax[..., :2]))
        xy_min, xy_max = xp.min(xy, axis=0), xp.max(xy, axis=0)
        return UCA(xp.asarray(patterns.index), xp.arange(patterns.shape[0]), points.kout,
                   points.kout[..., :2] - xy_min, points.kout[..., :2] - xy_max)

    def candidates(self, candidates: MillerWithRLP, uca: UCA, xp: AnyNamespace
                   ) -> Tuple[MillerWithRLP, UCA]:
        min_res, max_res = uca.min_resolution, uca.max_resolution
        rlp_res = xp.sum(candidates.q**2, axis=-1)
        idxs, rlp_idxs = xp.where(uca.index[..., None] == candidates.index)
        mask = (rlp_res[rlp_idxs] > min_res[idxs]) & (rlp_res[rlp_idxs] < max_res[idxs])
        return candidates[rlp_idxs[mask]], uca[idxs[mask]]

    def intersection(self, rlp: MillerWithRLP, uca: UCA, xp: AnyNamespace) -> CircleState:
        rlp_res = xp.sum(rlp.q**2, axis=-1)
        radius = xp.sqrt(rlp_res - 0.25 * rlp_res**2)
        center = 0.5 * rlp_res[..., None] * uca.kout
        axis1 = xp.stack((uca.kout[..., 1], -uca.kout[..., 0], xp.zeros(uca.kout.shape[:-1])),
                          axis=-1)
        axis1 = axis1 / xp.sqrt(xp.sum(axis1**2, axis=-1, keepdims=True))
        axis2 = xp.linalg.cross(uca.kout, axis1)
        return CircleState(rlp.index, center, axis1, axis2, radius)

    def uca_endpoints(self, circle: CircleState, uca: UCA, xp: AnyNamespace) -> RealArray:
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

        delta_sq = a**2 + b**2 - c**2
        delta_sq = xp.where(delta_sq < 0, xp.inf, delta_sq)
        theta = xp.concat((2.0 * xp.atan((b - xp.sqrt(delta_sq)) / (a + c)),
                           2.0 * xp.atan((b + xp.sqrt(delta_sq)) / (a + c))))

        points = circle.points(theta)
        proj = project_to_rect(points[..., :2], xy_min[None], xy_max[None], xp)
        dist = xp.sqrt(xp.sum((points[..., :2] - proj)**2, axis=-1))
        return xp.take_along_axis(theta, xp.argsort(dist, axis=0)[:2], axis=0)

    def rotations(self, rlp: MillerWithRLP, midpoints: RealArray, xp: AnyNamespace
                  ) -> TiltOverAxisState:
        source = rlp.q / xp.sqrt(xp.sum(rlp.q**2, axis=-1, keepdims=True))
        target = midpoints / xp.sqrt(xp.sum(midpoints**2, axis=-1, keepdims=True))

        bisector = source + target
        bisector = bisector / xp.sqrt(xp.sum(bisector**2, axis=-1, keepdims=True))
        S, C = xp.sin(0.5 * self.phi(xp)), xp.cos(0.5 * self.phi(xp))
        prod = xp.sum(target * bisector, axis=-1)
        cross = xp.linalg.cross(bisector, target)

        theta = 2.0 * xp.acos(-S * prod[..., None])
        S2 = xp.sin(0.5 * theta)
        axis = (C[..., None] * bisector[..., None, :] + S[..., None] * cross[..., None, :])
        return TiltOverAxisState(theta, axis / S2[..., None])

    def rotograms(self, rlp: MillerWithRLP, circle: CircleState, uca: UCA,
                  xp: AnyNamespace) -> Rotograms:
        endpoints = self.uca_endpoints(circle, uca, xp)
        midpoints = circle.points(xp.mean(endpoints, axis=0))
        tilts = self.rotations(rlp, midpoints, xp)
        return Rotograms.from_tilts(tilts, uca.index, uca.streak_id, xp)

    def index(self, candidates: MillerWithRLP, patterns: Patterns, points: PointsWithK,
              state: ResolvedSetup) -> Rotograms:
        xp = array_namespace(candidates, patterns, points)
        patterns_uca = self.patterns_to_uca(patterns, points, state, xp)
        rlp, uca = self.candidates(candidates, patterns_uca, xp)
        circles = self.intersection(rlp, uca, xp)
        return self.rotograms(rlp, circles, uca, xp)

    def rotogrid(self, shape: Tuple[int, int, int], return_step: bool=False,
                 xp: AnyNamespace=NumPy) -> RealArray | Tuple[RealArray, RealArray]:
        indices = xp.stack(xp.meshgrid(xp.arange(shape[0]), xp.arange(shape[1]),
                                       xp.arange(shape[2])))
        coords = indices * self.step(shape, xp) - self.rho_map()
        if return_step:
            return coords, self.step(shape, xp)
        return coords

    def rotomap(self, shape: Tuple[int, int, int], rotograms: Rotograms, frames: IntArray,
                width: float) -> RealArray:
        xp = rotograms.__array_namespace__()

        step = xp.asarray(self.step(shape, xp), dtype=rotograms.points.dtype)
        points = (rotograms.points + self.rho_map()) / step

        rmap = xp.zeros((len(rotograms),) + shape, dtype=points.dtype)
        return accumulate_lines(rmap, points, rotograms.streak_id, frames, width=width,
                                kernel='gaussian',
                                in_overlap='max', out_overlap='sum')

    def to_peaks(self, rotomap: RealArray, threshold: float, n_max: int=30) -> BoolArray:
        xp = array_namespace(rotomap)

        rotomap = rotomap / xp.max(rotomap, axis=(-3, -2, -1), keepdims=True)
        f, z, y, x = xp.where(rotomap > threshold)
        indices = xp.lexsort(xp.stack((rotomap[f, z, y, x], f)))
        _, counts = xp.unique_counts(f)
        mask = xp.concat([xp.arange(int(size) - 1, -1, -1) < n_max for size in counts])

        f, z, y, x = f[indices[mask]], z[indices[mask]], y[indices[mask]], x[indices[mask]]
        mask = xp.zeros(rotomap.shape, dtype=bool)
        return add_at(mask, (f, z, y, x), True)

    def refine_peaks(self, mask: BoolArray, rotomap: RealArray, vicinity: Structure,
                     connectivity: Structure=Structure([1, 1, 1], 1)
                     ) -> Tuple[IntArray, TiltOverAxisState]:
        if vicinity.rank != 3:
            raise ValueError(f'Vicinity structure must have rank 3, but got {vicinity.rank:d}')
        if connectivity.rank != 3:
            raise ValueError('Connectivity structure must have rank 3, but got '
                             f'{connectivity.rank:d}')

        vicinity = vicinity.expand_dims(list(range(mask.ndim - 3)))
        connectivity = connectivity.expand_dims(list(range(mask.ndim - 3)))

        xp = array_namespace(rotomap)
        mask = binary_dilation(mask, vicinity)
        regions = label(mask, connectivity)
        centers = center_of_mass(regions, rotomap)

        points = centers[..., -1:-4:-1] * self.step(rotomap.shape, xp) - self.rho_map()
        indices = xp.round(centers[..., -4]).astype(int)
        return indices, TiltOverAxisState.from_point(points)

    def solutions(self, initial: XtalState, indices: IntArray, tilts: TiltOverAxisState,
                  patterns: Patterns) -> IndexingResult:
        if len(initial) == 1:
            return IndexingResult(index=patterns.unique_index()[indices],
                                  xtal=tilts.to_tilt().to_rotation() @ initial)
        if len(initial) == len(patterns):
            return IndexingResult(index=patterns.unique_index()[indices],
                                  xtal=tilts.to_tilt().to_rotation() @ initial[indices])
        raise ValueError(f'Number of crystals ({len(initial):d}) and patterns ({len(patterns):d}) '\
                         'are inconsistent')

class CBDModel(CBDSetup):
    def hkl_in_aperture(self, q_abs: float, state: ResolvedState, xp: AnyNamespace=JaxNumPy
                        ) -> MillerWithRLP:
        hkl = self.xtal.hkl_in_ball(q_abs, state.xtal, xp)
        kz = xp.asarray([self.lens.kin_min(state.setup.lens, xp)[..., 2],
                         self.lens.kin_max(state.setup.lens, xp)[..., 2]])
        return self.xtal.hkl_in_aperture(xp.acos(xp.min(kz)), hkl, state.xtal, xp)

    def init_miller(self, patterns: Patterns, state: ResolvedState, xp: AnyNamespace) -> Miller:
        q1, q2 = self.patterns_to_q(patterns, state.setup, xp)
        hkl_min, hkl_max = self.xtal.hkl_bounds(q1, q2, state.xtal, xp)
        offsets = self.xtal.hkl_offsets(hkl_min, hkl_max, xp)
        hkl = (hkl_min.hkl_indices + hkl_max.hkl_indices) // 2
        return Miller(index=hkl_min.index, hkl=hkl).offset(offsets)

    def _init_mask(self, index: IntArray, quantile: float, xp: AnyNamespace):
        if max(min(quantile, 1.0), 0.0) < 1.0:
            _, counts = xp.unique(index, return_counts=True)
            counts = xp.clip(counts, 1, None)
            return xp.concat([xp.arange(size) < quantile * size for size in counts])

        return xp.ones(index.shape[0], dtype=bool)

    def init_data(self, patterns: Patterns, state: ResolvedState,
                  values: Sequence[float]=(0.0, 1.0)
                  ) -> 'CBData':
        xp = state.__array_namespace__()

        miller = self.init_miller(patterns, state, xp)
        x = xp.broadcast_to(xp.asarray(values), miller.hkl.shape[:-1] + (len(values),))

        return CBData(miller, patterns.sample(x))

    def init_data_random(self, rng: AnyGenerator, patterns: Patterns, num_points: int,
                         state: ResolvedState) -> 'CBData':
        xp = state.__array_namespace__()

        miller = self.init_miller(patterns, state, xp)
        x = rng.uniform(size=miller.hkl.shape[:-1] + (num_points,))

        return CBData(miller, patterns.sample(x))

    def keep_best(self, data: CBData, quantile: float) -> CBDataBest:
        if max(min(quantile, 1.0), 0.0) == 0.0:
            raise ValueError(f"Invalid quantile value: {quantile}")

        xp = data.__array_namespace__()
        mask = self._init_mask(data.miller.index, quantile, xp)
        return CBDataBest(miller=data.miller, points=data.points, mask=mask)

    def keep_in_shell(self, data: CBData, q_abs: float, state: ResolvedState) -> CBDataMasked:
        xp = state.__array_namespace__()
        miller = self.xtal.hkl_to_q(data.miller, state.xtal, xp)
        mask = xp.all(xp.sqrt(xp.sum(miller.q**2, axis=-1)) < q_abs, axis=-1)
        return CBDataMasked(miller=data.miller, points=data.points, mask=mask)

    def keep_refined(self, threshold: float, loss: 'CBDLoss', data: CBData, state: ResolvedState
                     ) -> CBDataMasked:
        xp = state.__array_namespace__()
        ratios = loss.distance_ratios(data, state, xp)
        mask = ratios < threshold
        return CBDataMasked(miller=data.miller, points=data.points, mask=mask)

    def line_loss(self, loss: Loss='l1', xp: AnyNamespace=JaxNumPy) -> 'CBDLoss':
        return CBDLoss(self, self.lens.line_projector, loss_function(loss, xp))

    def pupil_loss(self, loss: Loss='l1', xp: AnyNamespace=JaxNumPy) -> 'CBDLoss':
        return CBDLoss(self, self.lens.pupil_projector, loss_function(loss, xp))

@dataclass(frozen=True, unsafe_hash=True)
class CBDLoss():
    model           : CBDModel
    projector       : Projector
    loss_fn         : LossFn

    def project_data(self, data: CBData, state: ResolvedState, xp: AnyNamespace) -> CBDPoints:
        pts = self.model.points_to_kout(data.points, state.setup, xp)
        rlp = self.model.xtal.hkl_to_q(data.miller, state.xtal, xp)
        return CBDPoints(index=pts.index, points=pts.points, hkl=rlp.hkl, q=rlp.q[..., None, :],
                         kin=pts.kout - rlp.q[..., None, :], kout=pts.kout)

    def distance_matrix(self, points: CBDPoints, state: ResolvedState,
                        xp: AnyNamespace) -> RealArray:
        projected = self.projector(points, state.setup.lens, xp)
        return xp.mean(xp.sum(self.loss_fn(projected, points.kin), axis=-1), axis=-1)

    def distances(self, data: CBData, points: CBDPoints, state: ResolvedState, xp: AnyNamespace
                  ) -> RealArray:
        dist = self.distance_matrix(points, state, xp)
        dist = xp.min(dist, axis=-1)

        if isinstance(data, CBDataBest):
            # Sorting distances according to frame_id
            # lexsort is not implemented in CuPy
            indices = xp.lexsort((dist, data.points.index), axis=0)
            return xp.asarray(dist[indices] * data.mask)

        if isinstance(data, CBDataMasked):
            return xp.asarray(dist * data.mask)

        return dist

    def __call__(self, data: CBData, state: BaseState) -> RealArray:
        xp = state.__array_namespace__()
        resolved = state.resolve(xp)
        points = self.project_data(data, resolved, xp)
        return self.distances(data, points, resolved, xp).mean()

    def index(self, data: CBData, state: BaseState) -> Miller:
        xp = state.__array_namespace__()
        resolved = state.resolve(xp)
        points = self.project_data(data, resolved, xp)
        dist = self.distance_matrix(points, resolved, xp)
        idxs = xp.argmin(dist, axis=-1)
        hkl = xp.take_along_axis(points.hkl, idxs[..., None, None], axis=-2)[..., 0, :]
        return Miller(index=points.index, hkl=hkl)

    def per_pattern(self, data: CBData, state: BaseState) -> RealArray:
        xp = state.__array_namespace__()
        resolved = state.resolve(xp)
        points = self.project_data(data, resolved, xp)
        dist = self.distances(data, points, resolved, xp)
        crit = add_at(xp.zeros(len(state.xtal)), data.points.index, dist)
        n_streaks = add_at(xp.zeros(len(state.xtal)), data.points.index,
                           xp.ones_like(data.points.index))
        return crit / n_streaks

    def distance_ratios(self, data: CBData, state: ResolvedState,
                        xp: AnyNamespace) -> RealArray:
        points = self.project_data(data, state, xp)
        kin_dist = self.distance_matrix(points, state, xp)
        kin_dist = xp.min(kin_dist, axis=-1)

        kin_min = self.model.lens.kin_min(state.setup.lens, xp)
        kin_max = self.model.lens.kin_max(state.setup.lens, xp)
        kin_na = xp.max(self.loss_fn(kin_min, kin_max), axis=-1)
        kin_na = broadcast_to(kin_na, data.points.index, (), xp)
        return safe_divide(kin_dist, kin_na, xp)

    def pattern_fitness(self, threshold: float, data: CBData, state: BaseState) -> RealArray:
        xp = state.__array_namespace__()
        ratios = self.distance_ratios(data, state.resolve(xp), xp)
        n_good = add_at(xp.zeros(len(state.xtal)), data.points.index, ratios < threshold)
        n_streaks = add_at(xp.zeros(len(state.xtal)), data.points.index,
                           xp.ones_like(data.points.index))
        return safe_divide(n_good, n_streaks, xp)
