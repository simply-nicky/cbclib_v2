from dataclasses import dataclass
from typing import Protocol
from .cbc_pupil import EdgePoints
from .cbc_setup import TiltOverAxisState
from .._src.annotations import AnyNamespace, BoolArray, IntArray, RealArray, Shape
from .._src.array_api import array_namespace, broadcast_to, kxy_to_k, safe_divide, safe_sqrt
from .._src.data_container import ArrayContainer, IndexedContainer
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
    def from_points(cls, points: AnyPoints) -> 'Patterns':
        if points.points.shape[-2:] != (2, 2):
            raise ValueError(f"Expected points of shape (..., 2, 2), got {points.points.shape}")

        xp = points.__array_namespace__()
        lines = xp.reshape(points.points, points.points.shape[:-2] + (4,))
        return cls(index=points.index[..., 0], lines=lines)

    @classmethod
    def import_xy(cls, index: IntArray, x: RealArray, y: RealArray) -> 'Patterns':
        xp = array_namespace(x, y, index)
        lines = xp.stack((x[..., 0], y[..., 0], x[..., 1], y[..., 1]), axis=-1)
        return cls(index=index, lines=lines)

    @property
    def points(self) -> 'Points':
        xp = self.__array_namespace__()
        return Points(index=xp.asarray(self.index)[..., None], points=super().points)

    @property
    def pt0(self) -> 'Points':
        xp = self.__array_namespace__()
        return Points(index=xp.asarray(self.index), points=super().pt0)

    @property
    def pt1(self) -> 'Points':
        xp = self.__array_namespace__()
        return Points(index=xp.asarray(self.index), points=super().pt1)

    def sample(self, x: RealArray) -> 'Points':
        xp = self.__array_namespace__()
        shape = x.shape + (2,)
        x = xp.reshape(x, (x.shape[0], -1, 1))
        pts = super().pt0[..., None, :] + x * (super().pt1 - super().pt0)[..., None, :]
        return Points(points=xp.reshape(pts, shape), index=xp.asarray(self.index))

class Points(State, ArrayContainer):
    index   : IntArray
    points  : RealArray

    @property
    def x(self) -> RealArray:
        return self.points[..., 0]

    @property
    def y(self) -> RealArray:
        return self.points[..., 1]

@dataclass
class UCA(ArrayContainer):
    """Uncertainty Cap Area (UCA) is a region of all possible q vectors that might have
    given rise to the given point on the detector defined by the point's kout. The UCA
    extent is limited by the length of the streak line associated with the point.
    """
    index       : IntArray
    streak_id   : IntArray
    kout        : RealArray
    kxy_min     : RealArray
    kxy_max     : RealArray

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
class CircleState(ArrayContainer):
    index   : IntArray
    center  : RealArray
    axis1   : RealArray
    axis2   : RealArray
    radius  : RealArray

    def points(self, theta: RealArray) -> RealArray:
        xp = array_namespace(theta)
        return (self.radius * xp.cos(theta))[..., None] * self.axis1 \
             + (self.radius * xp.sin(theta))[..., None] * self.axis2 + self.center

@dataclass
class Rotograms(IndexedContainer):
    index       : IntArray
    streak_id   : IntArray
    points      : RealArray

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

class Miller(State, ArrayContainer):
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

    def collapse(self) -> 'Miller':
        xp = self.__array_namespace__()
        index = xp.broadcast_to(self.index, self.hkl.shape[:-1])
        idxs = xp.concat((self.hkl, index[..., None]), axis=-1)
        idxs = xp.unique(xp.reshape(idxs, (-1,) + idxs.shape[-1:]), axis=0)
        return self.replace(hkl=idxs[..., :3], index=idxs[..., 3])

    def offset(self, offsets: IntArray) -> 'Miller':
        xp = self.__array_namespace__()
        hkl = self.hkl
        shape = hkl.shape[:-1] + offsets.shape[:-1] + hkl.shape[-1:]
        hkl = xp.reshape(xp.reshape(hkl, (-1, 3))[..., None, :] + offsets, shape)
        return self.replace(hkl=hkl, index=self.index[..., None])

    def finite_only(self) -> 'Miller':
        xp = self.__array_namespace__()
        index = xp.reshape(xp.broadcast_to(self.index, self.hkl.shape[:-1]), (-1,))
        hkl = xp.reshape(self.hkl, (-1, 3))
        mask = xp.isfinite(hkl).all(axis=-1)
        return self.replace(hkl=hkl[mask], index=index[mask])

    def unique(self) -> 'Miller':
        xp = self.__array_namespace__()
        index = xp.reshape(xp.broadcast_to(self.index, self.hkl.shape[:-1]), (-1,))
        hkl = xp.reshape(self.hkl, (-1, 3))
        hkl, is_unique = xp.unique(hkl, return_index=True, axis=0)
        indices = xp.argsort(is_unique)
        return self.replace(hkl=hkl[indices], index=index[is_unique[indices]])

class RLP(State, ArrayContainer):
    index   : IntArray
    q       : RealArray

    def origin_points(self) -> Points:
        xp = self.__array_namespace__()
        rec_abs = safe_sqrt(xp.sum(self.q**2, axis=-1), xp)
        theta = xp.acos(0.5 * rec_abs) - xp.acos(safe_divide(-self.q[..., 2], rec_abs, xp))
        phi = xp.atan2(self.q[..., 1], self.q[..., 0])
        pts = xp.stack((xp.sin(theta) * xp.cos(phi), xp.sin(theta) * xp.sin(phi),
                        xp.cos(theta)), axis=-1)
        return Points(points=pts, index=self.index)

class MillerWithRLP(Miller, RLP):
    pass

class LaueVectors(MillerWithRLP):
    kin     : RealArray
    kout    : RealArray

    def source_points(self) -> Points:
        xp = self.__array_namespace__()
        q_mag = xp.sum(self.q**2, axis=-1)
        t = safe_divide(xp.sum(self.kin * self.q, axis=-1), q_mag, xp) + 0.5
        kin = self.kin - t[..., None] * self.q
        tau = kin + 0.5 * self.q
        tau_mag = xp.sum(tau**2, axis=-1)
        s = safe_divide(xp.sum(kin**2, axis=-1) - 1.0,
                        xp.sum(kin * tau, axis=-1) + safe_sqrt(tau_mag, xp), xp)
        pts = kin - s[..., None] * tau
        return Points(points=pts, index=self.index)

class SimulatedVectors(MillerWithRLP):
    kin     : RealArray
    kout    : RealArray
    distance: RealArray  # (..., 2) endpoint distances to the pupil support

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

class CBDPoints(LaueVectors, Points):
    pass

class RefinerData(State, ArrayContainer):
    miller  : Miller
    points  : Points

class RefinerDataBest(RefinerData):
    mask    : BoolArray

class RefinerDataMasked(RefinerData):
    mask    : BoolArray
