from typing import List, Optional, TypeVar, Union
from dataclasses import fields
import jax.numpy as jnp
from .geometry import safe_divide
from .state import State
from .._src.annotations import IntArray, RealArray, Shape
from .._src.data_container import ArrayContainer
from .._src.streaks import BaseLines

D = TypeVar("D", bound="BaseData")
AnyPoints = Union['Points', 'PointsWithK', 'CBDPoints']

class BaseData(State, ArrayContainer):
    ...

class Patterns(BaseData, BaseLines):
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
    kout        : Optional[RealArray] = None
    hkl         : Optional[IntArray] = None
    q           : Optional[RealArray] = None
    kin         : Optional[RealArray] = None

    @classmethod
    def extra_attributes(cls) -> List[str]:
        return [field.name for field in fields(cls)
                if field.name not in ['index', 'lines']]

    @classmethod
    def from_points(cls, points: AnyPoints) -> 'Patterns':
        lines = jnp.reshape(points.points, points.shape[:-1] + (4,))
        index = jnp.reshape(points.index, lines.shape[:-1])
        extra = {attr: getattr(points, attr) for attr in cls.extra_attributes()
                 if hasattr(points, attr)}
        return cls(index=index, lines=lines, **extra)

    @property
    def length(self) -> RealArray:
        return jnp.sqrt((self.lines[..., 2] - self.lines[..., 0])**2 +
                        (self.lines[..., 3] - self.lines[..., 1])**2)

    @property
    def pt0(self) -> 'Points':
        return Points(points=self.lines[..., :2], index=self.index)

    @property
    def pt1(self) -> 'Points':
        return Points(points=self.lines[..., 2:], index=self.index)

    def sample(self, x: RealArray) -> 'Points':
        pts = self.pt0.points + x[..., None] * (self.pt1.points - self.pt0.points)
        return Points(points=pts, index=self.index)

    def to_points(self) -> 'Points':
        points = jnp.reshape(self.lines, self.shape + (2, 2))
        return Points(index=self.index[..., None], points=points)

class Points(BaseData):
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

class Miller(BaseData):
    hkl     : Union[IntArray, RealArray]
    index   : IntArray

    @property
    def hkl_indices(self) -> IntArray:
        return jnp.array(jnp.round(self.hkl), dtype=int)

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
        index = jnp.broadcast_to(self.index, self.hkl.shape[:-1])
        idxs = jnp.concatenate((self.hkl, index[..., None]), axis=-1)
        idxs = jnp.unique(jnp.reshape(idxs, (-1,) + idxs.shape[-1:]), axis=0)
        return self.replace(hkl=idxs[..., :3], index=idxs[..., 3])

    def offset(self, offsets: IntArray) -> 'Miller':
        hkl = self.hkl
        shape = offsets.shape[:-1] + hkl.shape
        hkl = jnp.reshape(jnp.reshape(hkl, (-1, 3)) + offsets[..., None, :], shape)
        return self.replace(hkl=hkl)

class RLP(BaseData):
    q       : RealArray
    index   : IntArray

    def source_points(self) -> RealArray:
        rec_abs = jnp.sqrt(jnp.sum(self.q**2, axis=-1))
        theta = jnp.arccos(0.5 * rec_abs) - jnp.arccos(safe_divide(-self.q[..., 2], rec_abs))
        phi = jnp.arctan2(self.q[..., 1], self.q[..., 0])
        return jnp.stack((jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi),
                          jnp.cos(theta)), axis=-1)

class MillerWithRLP(Miller, RLP):
    pass

class LaueVectors(MillerWithRLP):
    kin     : RealArray
    kout    : RealArray

    def kin_to_source_line(self) -> RealArray:
        q_mag = jnp.sum(self.q**2, axis=-1)
        t = safe_divide(jnp.sum(self.kin * self.q, axis=-1), q_mag) + 0.5
        kin = self.kin - t[..., None] * self.q
        tau = kin + 0.5 * self.q
        tau_mag = jnp.sum(tau**2, axis=-1)
        s = safe_divide(jnp.sum(kin**2, axis=-1) - 1.0,
                        jnp.sum(kin * tau, axis=-1) + jnp.sqrt(tau_mag))
        return kin - s[..., None] * tau

class CBDPoints(LaueVectors, Points):
    pass

class CBData(BaseData):
    miller  : Miller
    points  : Points
