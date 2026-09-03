from dataclasses import dataclass
from typing import Protocol
from typing_extensions import Self
import pandas as pd
from .cbc_pupil import DataContainer, EdgePoints
from .cbc_setup import TiltOverAxisState
from .._src.annotations import AnyNamespace, BoolArray, IntArray, IntSequence, RealArray, Shape
from .._src.array_api import (array_namespace, asnumpy, broadcast_to, kxy_to_k, safe_divide,
                              safe_sqrt)
from .._src.data_container import ILocIndexer, IndexLookup, IndexedContainer, LocIndexer
from .._src.state import State
from .._src.streaks import BaseLines, project_to_streak

class AnyPoints(Protocol):
    index   : IntArray
    points  : RealArray

    @property
    def shape(self) -> Shape: ...

    def __array_namespace__(self) -> AnyNamespace: ...

@dataclass
class Patterns(IndexedContainer, BaseLines):
    """Detector streak lines container. Provides an interface to draw a pattern for a set of
    lines.

    Args:
        x0 : x coordinates of the first point of a line.
        y0 : y coordinates of the first point of a line.
        x1 : x coordinates of the second point of a line.
        y1 : y coordinates of the second point of a line.
        length: Line's length in pixels.
        h : First Miller index.
        k : Second Miller index.
        l : Third Miller index.
        hkl_id : Bragg reflection index.
    """
    index       : IntArray
    lines       : RealArray

    @classmethod
    def from_points(cls, points: 'LinePoints') -> 'Patterns':
        xp = points.__array_namespace__()
        lines = xp.reshape(points.points, points.points.shape[:-2] + (4,))
        index = xp.reshape(points.index, points.index.shape[:lines.ndim - 1])
        return cls(index=index, lines=lines)

    @classmethod
    def import_xy(cls, index: IntArray, x: RealArray, y: RealArray) -> 'Patterns':
        xp = array_namespace(x, y, index)
        lines = xp.stack((x[..., 0], y[..., 0], x[..., 1], y[..., 1]), axis=-1)
        return cls(index=index, lines=lines)

    def sample(self, x: RealArray) -> RealArray:
        xp = self.__array_namespace__()
        x = xp.reshape(x, (x.shape[0], -1, 1))
        return self.pt0[..., None, :] + x * (self.pt1 - self.pt0)[..., None, :]

    def to_points(self) -> 'LinePoints':
        return LinePoints(self.reset_index(), self.points)

class Points(State, IndexedContainer):
    index   : IntArray
    points  : RealArray

    @property
    def shape(self) -> Shape:
        return self.points.shape[:-1]

    @property
    def x(self) -> RealArray:
        return self.points[..., 0]

    @property
    def y(self) -> RealArray:
        return self.points[..., 1]

class LinePoints(State, IndexedContainer):
    index   : IntArray
    points  : RealArray

    def __post_init__(self):
        super().__post_init__()
        if self.points.shape[-2:] != (2, 2):
            raise ValueError(f"Expected points of shape (..., 2, 2), got {self.points.shape}")

    @property
    def shape(self) -> Shape:
        return self.points.shape[:-2]

    @property
    def iloc(self: Self) -> ILocIndexer[Self]:
        return ILocIndexer(self, reset_index=True)

    @property
    def loc(self: Self) -> LocIndexer[Self]:
        return LocIndexer(self, reset_index=True)

    def take(self: Self, indices: IntSequence, reset_index: bool=True) -> Self:
        return super().take(indices, reset_index=reset_index)

    @property
    def x(self) -> RealArray:
        return self.points[..., 0]

    @property
    def y(self) -> RealArray:
        return self.points[..., 1]

    @property
    def pt0(self) -> RealArray:
        return self.points[..., 0, :]

    @property
    def pt1(self) -> RealArray:
        return self.points[..., 1, :]

    def sample(self, x: RealArray) -> Points:
        xp = self.__array_namespace__()
        shape = x.shape + (2,)
        x = xp.reshape(x, (x.shape[0], -1, 1))
        pts = self.pt0[..., None, :] + x * (self.pt1 - self.pt0)[..., None, :]
        return Points(self.index, xp.reshape(pts, shape))

@dataclass
class UCA(IndexedContainer):
    """Uncertainty Cap Area (UCA) is a region of all possible q vectors that might have
    given rise to the given point on the detector defined by the point's kout. The UCA
    extent is limited by the length of the streak line associated with the point.
    """
    index       : IntArray      # (N,) index of the streak point
    streak_id   : IntArray      # (N,) index of the streak line associated with the point
    kout        : RealArray     # (N, 3) kout vector of the streak point centers
    kxy_min     : RealArray     # (N, 2) minimum kxy vector of the streak point UCA
    kxy_max     : RealArray     # (N, 2) maximum kxy vector of the streak point UCA

    @property
    def shape(self) -> Shape:
        return self.kout.shape[:-1]

    @property
    def min_resolution(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.min(xp.sum(self.q_corners**2, axis=-1), axis=0)

    @property
    def max_resolution(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.max(xp.sum(self.q_corners**2, axis=-1), axis=0)

    @property
    def q_min(self) -> RealArray:
        xp = self.__array_namespace__()
        return self.kout - kxy_to_k(self.kxy_min, xp)

    @property
    def q_max(self) -> RealArray:
        xp = self.__array_namespace__()
        return self.kout - kxy_to_k(self.kxy_max, xp)

    @property
    def q_corners(self) -> RealArray:
        xp = self.__array_namespace__()
        kxy = xp.stack((self.kxy_min, self.kxy_max))
        kxy = xp.concat((kxy, xp.stack((kxy[..., 0], kxy[::-1, ..., 1]), axis=-1)))
        return self.kout - kxy_to_k(kxy, xp)

@dataclass
class CircleState(IndexedContainer):
    index   : IntArray  # (N,) index of the circle
    center  : RealArray # (N, 3) center of the circle
    axis1   : RealArray # (N, 3) first axis of the circle
    axis2   : RealArray # (N, 3) second axis of the circle
    radius  : RealArray # (N,) radius of the circle

    @property
    def shape(self) -> Shape:
        return self.center.shape[:-1]

    def points(self, theta: RealArray) -> RealArray:
        xp = array_namespace(theta)
        return (self.radius * xp.cos(theta))[..., None] * self.axis1 \
             + (self.radius * xp.sin(theta))[..., None] * self.axis2 + self.center

@dataclass
class Rotograms(IndexedContainer):
    index       : IntArray      # (N_curves,) index of the pattern
    streak_id   : IntArray      # (N_curves,) index of the streak line
    points      : RealArray     # (N_curves, N_points, 3) points of the rotogram curves

    @property
    def shape(self) -> Shape:
        return self.points.shape[:-2]

    @classmethod
    def from_tilts(cls, tilts: TiltOverAxisState, index: IntArray, streak_id: IntArray,
                   xp: AnyNamespace) -> 'Rotograms':
        points = tilts.axis * xp.atan(0.25 * tilts.angles[..., None])
        return cls(index, xp.asarray(streak_id), points)

    @property
    def angles(self) -> RealArray:
        xp = self.__array_namespace__()
        return safe_sqrt(xp.sum(self.points**2, axis=-1), xp)

    @property
    def axis(self) -> RealArray:
        xp = self.__array_namespace__()
        return safe_divide(self.points, self.angles[..., None], xp)

    @property
    def lines(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.concat((self.points[..., 1:, :], self.points[..., :-1, :]), axis=-1)

class BaseMiller(IndexedContainer):
    index   : IntArray
    hkl     : IntArray | RealArray

    @property
    def hkl_indices(self) -> IntArray:
        xp = self.__array_namespace__()
        return xp.array(xp.round(self.hkl), dtype=int)

    @property
    def h(self) -> IntArray:
        return self.hkl_indices[..., 0]

    @property
    def k(self) -> IntArray:
        return self.hkl_indices[..., 1]

    @property
    def l(self) -> IntArray:
        return self.hkl_indices[..., 2]

class Miller(State, BaseMiller):
    index   : IntArray
    hkl     : IntArray | RealArray

    @property
    def shape(self) -> Shape:
        return self.hkl.shape[:-1]

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, frames: IntArray, xp: AnyNamespace
                         ) -> 'Miller':
        index = xp.asarray(df['index'])

        lookup = IndexLookup.build(xp.reshape(frames, -1))
        if lookup.unique.size != frames.size:
            raise ValueError("State frames must be unique")

        try:
            index, _ = lookup.get_index(index)
        except KeyError as error:
            raise ValueError(
                "Miller indices reference frames absent from the state"
            ) from error

        hkl = xp.asarray(df[['h', 'k', 'l']])
        return cls(index=index, hkl=hkl)

    @classmethod
    def tile(cls, hkl: IntArray | RealArray, index: IntArray, xp: AnyNamespace) -> 'Miller':
        hkl = xp.reshape(hkl, (-1, 3))
        index = xp.reshape(index, (-1,))
        return cls(index=xp.repeat(index, hkl.shape[0]), hkl=xp.tile(hkl, (index.size, 1)))

    def arange(self, stop: IntArray) -> 'Candidates':
        xp = self.__array_namespace__()

        lower_hkl = xp.reshape(self.hkl_indices, (-1, 3))
        upper_hkl = xp.reshape(stop, (-1, 3))
        shape = upper_hkl - lower_hkl + 1

        # Number of Cartesian-product candidates belonging to each streak.
        counts = xp.prod(shape, axis=-1)
        stops = xp.cumsum(counts)
        total = int(xp.sum(counts))

        # Map every packed candidate to its source streak.
        packed_id = xp.arange(total)
        streak_id = xp.searchsorted(stops, packed_id, side='right')

        starts = xp.concat((xp.zeros((1,), dtype=int), stops[:-1]), axis=0)
        local_id = packed_id - starts[streak_id]
        local_shape = shape[streak_id]

        # Decode the local candidate index. The l index varies fastest, then k, then h.
        l_offset = local_id % local_shape[..., 2]
        local_id = local_id // local_shape[..., 2]
        k_offset = local_id % local_shape[..., 1]
        h_offset = local_id // local_shape[..., 1]

        offsets = xp.stack((h_offset, k_offset, l_offset), axis=-1)
        hkl = xp.asarray(lower_hkl[streak_id] + offsets)

        index = xp.reshape(self.index, (-1,))
        return Candidates(index=index[streak_id], hkl=hkl, streak_id=streak_id)

    def finite_only(self) -> 'Miller':
        xp = self.__array_namespace__()
        index = xp.reshape(xp.broadcast_to(self.index, self.hkl.shape[:-1]), (-1,))
        hkl = xp.reshape(self.hkl, (-1, 3))
        mask = xp.isfinite(hkl).all(axis=-1)
        return self.replace(hkl=hkl[mask], index=index[mask])

    def unique(self) -> 'Miller':
        xp = self.__array_namespace__()
        index = xp.reshape(xp.broadcast_to(self.index, self.hkl.shape[:-1]), (-1,))
        indices = xp.concat((index[..., None], xp.reshape(self.hkl, (-1, 3))), axis=-1)
        unique = xp.unique(indices, axis=0)
        return self.replace(hkl=unique[..., 1:], index=unique[..., 0].astype(int))

    def to_dataframe(self, frames: IntArray) -> pd.DataFrame:
        return pd.DataFrame({'index': asnumpy(frames[self.index]), 'h': asnumpy(self.h),
                             'k': asnumpy(self.k), 'l': asnumpy(self.l)})

class Candidates(Miller):
    streak_id   : IntArray

    @property
    def n_streaks(self) -> int:
        if self.size:
            return int(self.streak_id[-1]) + 1
        return 0

class RLP(State, IndexedContainer):
    index   : IntArray
    q       : RealArray

    @property
    def shape(self) -> Shape:
        return self.q.shape[:-1]

    @property
    def origin_points(self) -> RealArray:
        xp = self.__array_namespace__()
        rec_abs = safe_sqrt(xp.sum(self.q**2, axis=-1), xp)
        theta = xp.acos(0.5 * rec_abs) - xp.acos(safe_divide(-self.q[..., 2], rec_abs, xp))
        phi = xp.atan2(self.q[..., 1], self.q[..., 0])
        return xp.stack((xp.sin(theta) * xp.cos(phi), xp.sin(theta) * xp.sin(phi),
                         xp.cos(theta)), axis=-1)

class LaueVectors(RLP):
    kin     : RealArray # (..., 2, 3) endpoint kin vectors of the streak line
    kout    : RealArray # (..., 2, 3) endpoint kout vectors of the streak line

    @property
    def shape(self) -> Shape:
        return self.kin.shape[:-2]

    @property
    def source_points(self) -> RealArray:
        xp = self.__array_namespace__()
        q_mag = xp.sum(self.q**2, axis=-1)
        t = safe_divide(xp.sum(self.kin * self.q, axis=-1), q_mag, xp) + 0.5
        kin = self.kin - t[..., None] * self.q
        tau = kin + 0.5 * self.q
        tau_mag = xp.sum(tau**2, axis=-1)
        s = safe_divide(xp.sum(kin**2, axis=-1) - 1.0,
                        xp.sum(kin * tau, axis=-1) + safe_sqrt(tau_mag, xp), xp)
        pts = kin - s[..., None] * tau
        return pts

class MillerWithRLP(Miller, RLP):
    pass

class SimulatedVectors(State, IndexedContainer):
    index       : IntArray   # (N,) index of the streak point
    hkl         : IntArray   # (N, 3) Miller indices of the streak point
    q           : RealArray  # (N, 3) q vectors of the streak point
    kin         : RealArray  # (N, 2, 3) endpoint kin vectors of the streak line
    kout        : RealArray  # (N, 2, 3) endpoint kout vectors of the streak line
    distance    : RealArray  # (N, 2) endpoint distances to the pupil support

    @property
    def shape(self) -> Shape:
        return self.kin.shape[:-2]

    def project(self, index: IntArray, kout: RealArray, xp: AnyNamespace) -> EdgePoints:
        bounds = broadcast_to(self.kout[..., :2], index, (2, 2), xp)
        projection = project_to_streak(kout[..., :2], bounds[..., 0, :], bounds[..., 1, :], xp)
        return EdgePoints(tau=projection.tau, origin=projection.center, t=projection.t)

    def distance_to(self, index: IntArray, kout: RealArray, xp: AnyNamespace) -> RealArray:
        projection = self.project(index, kout, xp)
        return xp.sqrt(xp.sum((projection.points - kout)**2, axis=-1))

    def distance_at(self, index: IntArray) -> RealArray:
        """Return the mean endpoint support distance for indexed streak points."""
        xp = self.__array_namespace__()
        distance = xp.mean(self.distance, axis=-1)
        return broadcast_to(distance, index, (), xp)

class CBDPoints(LinePoints, LaueVectors):
    hkl     : IntArray      # (N, 3) Miller indices of the streak point

class RefinerData(State, DataContainer):
    miller  : Candidates    # (n_candidates,)
    points  : LinePoints    # (n_points,)

class RefinerDataBest(RefinerData):
    mask    : BoolArray     # (n_points,) boolean mask of the best points

    @classmethod
    def from_data(cls, data: RefinerData, mask: BoolArray) -> 'RefinerDataBest':
        if mask.shape != data.points.shape:
            raise ValueError(
                    f"Mask shape {mask.shape} does not match data.points shape "
                    f"{data.points.shape}"
            )
        return cls(miller=data.miller, points=data.points, mask=mask)

class RefinerDataMasked(RefinerData):
    mask    : BoolArray     # (n_points,) boolean mask of the masked points

    @classmethod
    def from_data(cls, data: RefinerData, mask: BoolArray) -> 'RefinerDataMasked':
        if mask.shape != data.points.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match data.points shape "
                f"{data.points.shape}"
            )
        return cls(miller=data.miller, points=data.points, mask=mask)
