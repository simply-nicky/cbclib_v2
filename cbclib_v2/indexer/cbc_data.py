from typing import Union
from .cbc_setup import TiltOverAxisState
from .geometry import safe_divide, kxy_to_k
from .._src.annotations import ArrayNamespace, BoolArray, IntArray, RealArray, Shape
from .._src.data_container import ArrayContainer, IndexedContainer, array_namespace
from .._src.state import State
from .._src.streaks import BaseLines

AnyPoints = Union['Points', 'PointsWithK', 'CBDPoints']

class Patterns(State, IndexedContainer, BaseLines):
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
        xp = array_namespace(points)
        lines = xp.reshape(points.points, points.shape[:-1] + (4,))
        index = xp.reshape(points.index, lines.shape[:-1])
        return cls(index=index, lines=lines)

    @property
    def points(self) -> 'Points':
        return Points(index=self.index[..., None], points=super().points)

    @property
    def pt0(self) -> 'Points':
        return Points(index=self.index, points=super().pt0)

    @property
    def pt1(self) -> 'Points':
        return Points(index=self.index, points=super().pt1)

    def sample(self, x: RealArray) -> 'Points':
        pts = super().pt0 + x[..., None] * (super().pt1 - super().pt0)
        return Points(points=pts, index=self.index)

class Points(State, ArrayContainer):
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

class PointsWithK(Points):
    kout    : RealArray

class UCA(State, ArrayContainer):
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
        return self.kout - kxy_to_k(self.kxy_min, self.__namespace__)

    @property
    def q_max(self) -> RealArray:
        return self.kout - kxy_to_k(self.kxy_max, self.__namespace__)

    @property
    def q_corners(self) -> RealArray:
        xp = self.__array_namespace__()
        kxy = xp.stack((self.kxy_min, self.kxy_max))
        kxy = xp.concatenate((kxy, xp.stack((kxy[..., 0], kxy[::-1, ..., 1]), axis=-1)))
        return self.kout - kxy_to_k(kxy, xp)

class CircleState(State, ArrayContainer):
    index   : IntArray
    center  : RealArray
    axis1   : RealArray
    axis2   : RealArray
    radius  : RealArray

class Rotograms(State, IndexedContainer, BaseLines):
    index       : IntArray
    streak_id   : IntArray
    lines       : RealArray

    @classmethod
    def from_tilts(cls, tilts: TiltOverAxisState, index: IntArray, streak_id: IntArray,
                   xp: ArrayNamespace) -> 'Rotograms':
        points = tilts.axis * xp.arctan(0.25 * tilts.angles[..., None])
        lines = xp.concatenate((points[..., 1:, :], points[..., :-1, :]), axis=-1)
        return cls(xp.asarray(index), xp.asarray(streak_id), lines)

    @property
    def angles(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.sqrt(xp.sum(self.points**2, axis=-1))

    @property
    def axis(self) -> RealArray:
        return self.points / self.angles[..., None]

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
        idxs = xp.concatenate((self.hkl, index[..., None]), axis=-1)
        idxs = xp.unique(xp.reshape(idxs, (-1,) + idxs.shape[-1:]), axis=0)
        return self.replace(hkl=idxs[..., :3], index=idxs[..., 3])

    def offset(self, offsets: IntArray) -> 'Miller':
        xp = self.__array_namespace__()
        hkl = self.hkl
        shape = offsets.shape[:-1] + hkl.shape
        hkl = xp.reshape(xp.reshape(hkl, (-1, 3)) + offsets[..., None, :], shape)
        return self.replace(hkl=hkl)

class RLP(State, ArrayContainer):
    index   : IntArray
    q       : RealArray

    @property
    def source_points(self) -> RealArray:
        xp = self.__array_namespace__()
        rec_abs = xp.sqrt(xp.sum(self.q**2, axis=-1))
        theta = xp.arccos(0.5 * rec_abs) - xp.arccos(safe_divide(-self.q[..., 2], rec_abs, xp))
        phi = xp.arctan2(self.q[..., 1], self.q[..., 0])
        return xp.stack((xp.sin(theta) * xp.cos(phi), xp.sin(theta) * xp.sin(phi),
                         xp.cos(theta)), axis=-1)

class MillerWithRLP(Miller, RLP):
    pass

class LaueVectors(MillerWithRLP):
    kin     : RealArray
    kout    : RealArray

    @property
    def source_line(self) -> RealArray:
        xp = self.__array_namespace__()
        q_mag = xp.sum(self.q**2, axis=-1)
        t = safe_divide(xp.sum(self.kin * self.q, axis=-1), q_mag, xp) + 0.5
        kin = self.kin - t[..., None] * self.q
        tau = kin + 0.5 * self.q
        tau_mag = xp.sum(tau**2, axis=-1)
        s = safe_divide(xp.sum(kin**2, axis=-1) - 1.0,
                        xp.sum(kin * tau, axis=-1) + xp.sqrt(tau_mag), xp)
        return kin - s[..., None] * tau

class CBDPoints(LaueVectors, Points):
    pass

class CBData(State, ArrayContainer):
    miller  : Miller
    points  : Points
    mask    : BoolArray
