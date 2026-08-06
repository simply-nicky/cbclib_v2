from typing import Callable, Iterator, Literal, Protocol, Sequence, Tuple
from dataclasses import dataclass
from .cbc_data import (AnyPoints, CBDPoints, CircleState, LaueVectors, Miller, MillerWithRLP,
                       Patterns, RefinerData, RefinerDataBest, RefinerDataMasked, RLP, Rotograms,
                       SimulatedVectors, UCA)
from .cbc_setup import (BaseSetup, IndexingResult, ResolvedLens, ResolvedGeometry, ResolvedSetup,
                        TiltOverAxisState, XtalState)
from .cbc_pupil import BasePupil, Rectangle, intersect
from .._src.annotations import (AnyNamespace, BoolArray, AnyGenerator, IntArray, JaxNumPy, NumPy,
                                RealArray)
from .._src.array_api import (add_at, array_namespace, broadcast_to, det_to_k, k_to_det,
                              k_to_smp, kxy_to_k, project_to_rect, safe_divide, safe_sqrt)
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
        hkl_min, hkl_max = xp.floor(hkl_min).astype(int), xp.ceil(hkl_max).astype(int)
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
    def kin_center(self, state: ResolvedLens | ResolvedGeometry, xp: AnyNamespace) -> RealArray:
        return det_to_k(state.pupil_center, state.foc_pos, xp)

    def kin_max(self, state: ResolvedLens | ResolvedGeometry, xp: AnyNamespace) -> RealArray:
        return det_to_k(state.pupil_max, state.foc_pos, xp)

    def kin_min(self, state: ResolvedLens | ResolvedGeometry, xp: AnyNamespace) -> RealArray:
        return det_to_k(state.pupil_min, state.foc_pos, xp)

    def pupil(self, state: ResolvedLens | ResolvedGeometry, xp: AnyNamespace) -> Rectangle:
        kmin, kmax = self.kin_min(state, xp), self.kin_max(state, xp)
        roi = xp.stack((kmin[..., 1], kmax[..., 1], kmin[..., 0], kmax[..., 0]), axis=-1)
        return Rectangle(roi=roi)

    def source_lines(self, miller: MillerWithRLP, pupil: BasePupil, xp: AnyNamespace,
                     ) -> SimulatedVectors:
        pupil_edges = pupil.edges
        edges = broadcast_to(pupil_edges, miller.index, pupil_edges.tau.shape[-2:], xp)
        kin, distance = intersect(miller.q, edges)
        index = xp.expand_dims(miller.index, axis=tuple(range(miller.index.ndim, miller.q.ndim)))
        return SimulatedVectors(index=index, kout=kin + miller.q[..., None, :],
                                hkl=miller.hkl, kin=kin, q=miller.q[..., None, :],
                                distance=distance)

    def project_to_pupil(self, kin: RealArray, idxs: IntArray, state: ResolvedLens | ResolvedGeometry,
                         xp: AnyNamespace) -> RealArray:
        kin = safe_divide(kin, safe_sqrt(xp.sum(kin**2, axis=-1), xp)[..., None], xp)
        kmin_xy = broadcast_to(self.kin_min(state, xp)[..., :2], idxs, (2,), xp)
        kmax_xy = broadcast_to(self.kin_max(state, xp)[..., :2], idxs, (2,), xp)
        kxy = project_to_rect(kin[..., :2], kmin_xy, kmax_xy, xp)
        return kxy_to_k(kxy, xp)

    def zero_order(self, state: ResolvedLens | ResolvedGeometry, xp: AnyNamespace) -> RealArray:
        return k_to_det(self.kin_center(state, xp), xp.asarray(state.foc_pos), xp)

    def line_projector(self, laue: LaueVectors, state: ResolvedLens | ResolvedGeometry, xp: AnyNamespace
                       ) -> RealArray:
        return self.project_to_pupil(laue.source_points().points, laue.index, state, xp)

    def pupil_projector(self, laue: LaueVectors, state: ResolvedLens | ResolvedGeometry, xp: AnyNamespace
                        ) -> RealArray:
        return self.project_to_pupil(laue.kin, laue.index, state, xp)

class Projector(Protocol):
    def __call__(self, laue: LaueVectors, state: ResolvedLens | ResolvedGeometry, xp: AnyNamespace
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

    def kin_to_sample(self, kin: RealArray, index: IntArray, geometry: ResolvedLens | ResolvedGeometry,
                      xp: AnyNamespace) -> RealArray:
        """Project incident wavevectors from the focus onto their sample planes.

        Args:
            kin: Incident unit wavevectors, shape ``(..., 3)``.
            index: Setup-state index for each wavevector.
            geometry: Resolved experimental geometry.
            xp: Array namespace used for the calculation.

        Returns:
            Sample positions in detector coordinates, shape ``(..., 3)``.
        """
        if isinstance(geometry, ResolvedGeometry):
            defocus = broadcast_to(geometry.defocus, index, (), xp)
            foc_pos = broadcast_to(geometry.foc_pos, index, (3,), xp)
            return k_to_smp(kin, defocus, foc_pos, xp)

        return broadcast_to(geometry.foc_pos, index, (3,), xp)

    def smp_center(self, index: IntArray, geometry: ResolvedLens | ResolvedGeometry, xp: AnyNamespace) -> RealArray:
        if isinstance(geometry, ResolvedGeometry):
            kin = self.lens.kin_center(geometry, xp)
            smp_pos = k_to_smp(kin, geometry.defocus, geometry.foc_pos, xp)
            return broadcast_to(smp_pos, index, (3,), xp)

        return broadcast_to(geometry.foc_pos, index, (3,), xp)

    def kout_to_points(self, simulated: SimulatedVectors, geometry: ResolvedLens | ResolvedGeometry,
                       xp: AnyNamespace, *, atol: float=2e-6) -> CBDPoints:
        def wrapper(simulated: SimulatedVectors, geometry: ResolvedLens | ResolvedGeometry) -> RealArray:
            smp_pos = self.kin_to_sample(simulated.kin, simulated.index, geometry, xp)
            return k_to_det(simulated.kout, smp_pos, xp)

        points = xp.where(xp.isclose(simulated.distance, 0.0, atol=atol)[..., None],
                          wrapper(simulated, geometry), xp.nan)
        return CBDPoints(index=simulated.index, points=points, q=simulated.q, hkl=simulated.hkl,
                         kin=simulated.kin, kout=simulated.kout)

    def points_to_kout(self, points: AnyPoints, smp_pos: RealArray, xp: AnyNamespace) -> RealArray:
        return det_to_k(points.points, smp_pos, xp)

    def patterns_to_kout(self, patterns: Patterns, geometry: ResolvedLens | ResolvedGeometry, xp: AnyNamespace
                         ) -> Tuple[RealArray, RealArray]:
        smp_pos = self.smp_center(patterns.index, geometry, xp)
        kout = self.points_to_kout(patterns.points, smp_pos, xp)
        return xp.min(kout[..., :2], axis=-2), xp.max(kout[..., :2], axis=-2)

    def patterns_to_q(self, patterns: Patterns, geometry: ResolvedLens | ResolvedGeometry, xp: AnyNamespace
                      ) -> Tuple[RLP, RLP]:
        kmin = self.lens.kin_min(geometry, xp)
        kmax = self.lens.kin_max(geometry, xp)
        kout_min, kout_max = self.patterns_to_kout(patterns, geometry, xp)

        kmin = broadcast_to(kmin, patterns.index, (3,), xp)
        kmax = broadcast_to(kmax, patterns.index, (3,), xp)
        q1 = kxy_to_k(kout_min, xp) - kmin
        q2 = kxy_to_k(kout_max, xp) - kmax
        return RLP(index=patterns.index, q=q1), RLP(index=patterns.index, q=q2)

    def init_patterns(self, miller: MillerWithRLP, geometry: ResolvedLens | ResolvedGeometry, xp: AnyNamespace,
                      *, atol: float=2e-6) -> Patterns:
        simulated = self.lens.source_lines(miller, self.lens.pupil(geometry, xp), xp)
        points = self.kout_to_points(simulated, geometry, xp, atol=atol)
        return Patterns.from_points(points)

    def points_to_kin(self, points: AnyPoints, miller: Miller, setup: ResolvedSetup,
                      xp: AnyNamespace) -> CBDPoints:
        smp_pos = self.smp_center(points.index, setup.geometry, xp)
        kout = self.points_to_kout(points, smp_pos, xp)
        rlp = self.xtal.hkl_to_q(miller, setup.xtal, xp)
        kin = kout - rlp.q[..., None, :]

        if isinstance(setup.geometry, ResolvedGeometry):
            smp_pos = self.kin_to_sample(kin, points.index, setup.geometry, xp)
            kout = self.points_to_kout(points, smp_pos, xp)
            kin = kout - rlp.q[..., None, :]

        return CBDPoints(index=points.index, points=points.points, hkl=rlp.hkl,
                         q=rlp.q[..., None, :], kin=kin, kout=kout)

    def line_loss(self, loss: Loss='l1', xp: AnyNamespace=JaxNumPy) -> 'RefinerLoss':
        return RefinerLoss(self, self.lens.line_projector, loss_function(loss, xp))

    def pupil_loss(self, loss: Loss='l1', xp: AnyNamespace=JaxNumPy) -> 'RefinerLoss':
        return RefinerLoss(self, self.lens.pupil_projector, loss_function(loss, xp))

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

    def patterns_to_uca(self, patterns: Patterns, kout: RealArray, geometry: ResolvedLens | ResolvedGeometry,
                        xp: AnyNamespace) -> UCA:
        kmin = self.lens.kin_min(geometry, xp)
        kmax = self.lens.kin_max(geometry, xp)
        kout_min, kout_max = self.patterns_to_kout(patterns, geometry, xp)
        xy = xp.stack((kout_min - kmin[..., :2], kout_max - kmax[..., :2]))
        xy_min, xy_max = xp.min(xy, axis=0), xp.max(xy, axis=0)
        return UCA(xp.asarray(patterns.index), xp.arange(patterns.shape[0]), kout,
                   kout[..., :2] - xy_min, kout[..., :2] - xy_max)

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

    def index(self, candidates: MillerWithRLP, patterns: Patterns, kout: RealArray,
              geometry: ResolvedLens | ResolvedGeometry) -> Rotograms:
        xp = array_namespace(candidates, patterns, kout)
        patterns_uca = self.patterns_to_uca(patterns, kout, geometry, xp)
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

class RefinerModel(CBDSetup):
    def hkl_in_aperture(self, q_abs: float, setup: ResolvedSetup, xp: AnyNamespace=JaxNumPy
                        ) -> MillerWithRLP:
        hkl = self.xtal.hkl_in_ball(q_abs, setup.xtal, xp)
        kz = xp.asarray([self.lens.kin_min(setup.geometry, xp)[..., 2],
                         self.lens.kin_max(setup.geometry, xp)[..., 2]])
        return self.xtal.hkl_in_aperture(xp.acos(xp.min(kz)), hkl, setup.xtal, xp)

    def init_miller(self, patterns: Patterns, setup: ResolvedSetup, xp: AnyNamespace) -> Miller:
        q1, q2 = self.patterns_to_q(patterns, setup.geometry, xp)
        hkl_min, hkl_max = self.xtal.hkl_bounds(q1, q2, setup.xtal, xp)
        offsets = self.xtal.hkl_offsets(hkl_min, hkl_max, xp)
        hkl = (hkl_min.hkl_indices + hkl_max.hkl_indices) // 2
        return Miller(index=hkl_min.index, hkl=hkl).offset(offsets)

    def _init_mask(self, index: IntArray, quantile: float, xp: AnyNamespace):
        if max(min(quantile, 1.0), 0.0) < 1.0:
            _, counts = xp.unique(index, return_counts=True)
            counts = xp.clip(counts, 1, None)
            return xp.concat([xp.arange(size) < quantile * size for size in counts])

        return xp.ones(index.shape[0], dtype=bool)

    def init_data(self, patterns: Patterns, setup: ResolvedSetup,
                  values: Sequence[float]=(0.0, 1.0)
                  ) -> 'RefinerData':
        xp = setup.__array_namespace__()

        miller = self.init_miller(patterns, setup, xp)
        x = xp.broadcast_to(xp.asarray(values), miller.hkl.shape[:-1] + (len(values),))

        return RefinerData(miller, patterns.sample(x))

    def init_data_random(self, rng: AnyGenerator, patterns: Patterns, num_points: int,
                         setup: ResolvedSetup) -> 'RefinerData':
        xp = setup.__array_namespace__()

        miller = self.init_miller(patterns, setup, xp)
        x = rng.uniform(size=miller.hkl.shape[:-1] + (num_points,))

        return RefinerData(miller, patterns.sample(x))

    def keep_best(self, data: RefinerData, quantile: float) -> RefinerDataBest:
        if max(min(quantile, 1.0), 0.0) == 0.0:
            raise ValueError(f"Invalid quantile value: {quantile}")

        xp = data.__array_namespace__()
        mask = self._init_mask(data.miller.index, quantile, xp)
        return RefinerDataBest(miller=data.miller, points=data.points, mask=mask)

    def keep_in_shell(self, data: RefinerData, q_abs: float,
                      setup: ResolvedSetup) -> RefinerDataMasked:
        xp = setup.__array_namespace__()
        miller = self.xtal.hkl_to_q(data.miller, setup.xtal, xp)
        mask = xp.all(xp.sqrt(xp.sum(miller.q**2, axis=-1)) < q_abs, axis=-1)
        return RefinerDataMasked(miller=data.miller, points=data.points, mask=mask)

    def keep_refined(self, threshold: float, loss: 'RefinerLoss', data: RefinerData,
                     setup: ResolvedSetup
                     ) -> RefinerDataMasked:
        xp = setup.__array_namespace__()
        ratios = loss.distance_ratios(data, setup, xp)
        mask = ratios < threshold
        return RefinerDataMasked(miller=data.miller, points=data.points, mask=mask)

@dataclass(frozen=True, unsafe_hash=True)
class RefinerLoss():
    model           : CBDSetup
    projector       : Projector
    loss_fn         : LossFn

    def project_data(self, data: RefinerData, setup: ResolvedSetup,
                     xp: AnyNamespace) -> CBDPoints:
        return self.model.points_to_kin(data.points, data.miller, setup, xp)

    def distance_matrix(self, points: CBDPoints, setup: ResolvedSetup,
                        xp: AnyNamespace) -> RealArray:
        projected = self.projector(points, setup.geometry, xp)
        return xp.mean(xp.sum(self.loss_fn(projected, points.kin), axis=-1), axis=-1)

    def distances(self, data: RefinerData, points: CBDPoints, setup: ResolvedSetup,
                  xp: AnyNamespace
                  ) -> RealArray:
        dist = self.distance_matrix(points, setup, xp)
        dist = xp.min(dist, axis=-1)

        if isinstance(data, RefinerDataBest):
            # Sorting distances according to frame_id
            # lexsort is not implemented in CuPy
            indices = xp.lexsort((dist, data.points.index), axis=0)
            return xp.asarray(dist[indices] * data.mask)

        if isinstance(data, RefinerDataMasked):
            return xp.asarray(dist * data.mask)

        return dist

    def __call__(self, data: RefinerData, setup: BaseSetup) -> RealArray:
        xp = setup.__array_namespace__()
        resolved_setup = setup.resolve(xp)
        points = self.project_data(data, resolved_setup, xp)
        return self.distances(data, points, resolved_setup, xp).mean()

    def index(self, data: RefinerData, setup: ResolvedSetup) -> Miller:
        xp = setup.__array_namespace__()
        points = self.project_data(data, setup, xp)
        dist = self.distance_matrix(points, setup, xp)
        idxs = xp.argmin(dist, axis=-1)
        hkl = xp.take_along_axis(points.hkl, idxs[..., None, None], axis=-2)[..., 0, :]

        if isinstance(data, RefinerDataMasked):
            hkl = xp.where(data.mask[..., None], hkl, xp.nan)

        return Miller(index=points.index, hkl=hkl)

    def per_pattern(self, data: RefinerData, setup: BaseSetup) -> RealArray:
        xp = setup.__array_namespace__()
        resolved_setup = setup.resolve(xp)
        points = self.project_data(data, resolved_setup, xp)
        dist = self.distances(data, points, resolved_setup, xp)
        crit = add_at(xp.zeros(len(setup.xtal)), data.points.index, dist)
        n_streaks = add_at(xp.zeros(len(setup.xtal)), data.points.index,
                           xp.ones_like(data.points.index))
        return safe_divide(crit, n_streaks, xp)

    def distance_ratios(self, data: RefinerData, setup: ResolvedSetup,
                        xp: AnyNamespace) -> RealArray:
        points = self.project_data(data, setup, xp)
        kin_dist = self.distance_matrix(points, setup, xp)
        kin_dist = xp.min(kin_dist, axis=-1)

        kin_min = self.model.lens.kin_min(setup.geometry, xp)
        kin_max = self.model.lens.kin_max(setup.geometry, xp)
        kin_na = xp.max(self.loss_fn(kin_min, kin_max), axis=-1)
        kin_na = broadcast_to(kin_na, data.points.index, (), xp)
        return safe_divide(kin_dist, kin_na, xp)

    def pattern_fitness(self, threshold: float, data: RefinerData,
                        setup: BaseSetup) -> RealArray:
        xp = setup.__array_namespace__()
        resolved_setup = setup.resolve(xp)
        ratios = self.distance_ratios(data, resolved_setup, xp)
        n_good = add_at(xp.zeros(len(setup.xtal)), data.points.index, ratios < threshold)
        n_streaks = add_at(xp.zeros(len(setup.xtal)), data.points.index,
                           xp.ones_like(data.points.index))
        return safe_divide(n_good, n_streaks, xp)
