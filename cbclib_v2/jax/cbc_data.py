from typing import List, Optional, Tuple, TypeVar, Union
from dataclasses import fields
import pandas as pd
import numpy as np
import jax.numpy as jnp
from .geometry import safe_divide
from .state import State
from ..ndimage import draw_line_image, draw_line_mask, draw_line_table
from .._src.annotations import (BoolArray, Indices, IntArray, IntSequence, RealArray,
                                RealSequence, Shape)

D = TypeVar("D", bound="BaseData")
AnyPoints = Union['Points', 'PointsWithK', 'CBDPoints']

class BaseData(State):
    def filter(self: D, idxs: Union[Indices, BoolArray]) -> D:
        data = {attr: None for attr in self.to_dict()}
        data = data | {attr: val[idxs] for attr, val in self.contents().items()}
        return self.replace(**data)

    def to_dataframe(self) -> pd.DataFrame:
        """Export a streaks container into :class:`pandas.DataFrame`.

        Returns:
            A dataframe with all the data specified in :class:`cbclib.Streaks`.
        """
        return pd.DataFrame(self.to_dict().items())

class Patterns(BaseData):
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
    def shape(self) -> Shape:
        return self.lines.shape[:-1]

    @property
    def pt0(self) -> 'Points':
        return Points(points=self.lines[..., :2], index=self.index)

    @property
    def pt1(self) -> 'Points':
        return Points(points=self.lines[..., 2:], index=self.index)

    @property
    def x(self) -> RealArray:
        return self.lines[..., ::2]

    @property
    def y(self) -> RealArray:
        return self.lines[..., 1::2]

    def pattern_dataframe(self, width: float, shape: Shape, kernel: str='rectangular',
                     num_threads: int=1) -> pd.DataFrame:
        """Draw a pattern in the :class:`dict` format.

        Args:
            width : Lines width in pixels.
            shape : Detector grid shape.
            kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
                are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

        Returns:
            A pattern in dictionary format.
        """
        table = draw_line_table(lines=self.to_lines(width=width), shape=shape, idxs=self.index,
                                kernel=kernel, num_threads=num_threads)
        ids, idxs = np.array(list(table)).T
        normalised_shape = (np.prod(shape[:-2], dtype=int),) + shape[-2:]
        frames, y, x = jnp.unravel_index(idxs, normalised_shape)
        vals = np.array(list(table.values()))

        data = {'index': ids, 'frames': frames, 'y': y, 'x': x, 'value': vals}
        data = data | {attr: getattr(self, attr)[ids] for attr in self.extra_attributes()}

        return pd.DataFrame(data)

    def pattern_image(self, width: float, shape: Tuple[int, int], kernel: str='gaussian',
                      num_threads: int=1) -> RealArray:
        """Draw a pattern in the :class:`numpy.ndarray` format.

        Args:
            width : Lines width in pixels.
            shape : Detector grid shape.
            kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
                are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

        Returns:
            A pattern in :class:`numpy.ndarray` format.
        """
        return draw_line_image(self.to_lines(width=width), shape=shape, idxs=self.index,
                               kernel=kernel, num_threads=num_threads)

    def pattern_mask(self, width: float, shape: Tuple[int, int], max_val: int=1,
                     kernel: str='rectangular', num_threads: int=1) -> IntArray:
        """Draw a pattern mask.

        Args:
            width : Lines width in pixels.
            shape : Detector grid shape.
            max_val : Mask maximal value.
            kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
                are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

        Returns:
            A pattern mask.
        """
        return draw_line_mask(self.to_lines(width=width), shape=shape, idxs=self.index,
                              max_val=max_val, kernel=kernel, num_threads=num_threads)

    def sample(self, x: RealArray) -> 'Points':
        pts = self.pt0.points + x[..., None] * (self.pt1.points - self.pt0.points)
        return Points(points=pts, index=self.index)

    def to_lines(self, frames: Optional[IntSequence]=None,
                 width: Optional[RealSequence]=None) -> RealArray:
        """Export a streaks container into line parameters ``x0, y0, x1, y1, width``:

        * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
        * `width` : Line's width.

        Returns:
            An array of line parameters.
        """
        if width is None:
            lines = self.lines
        else:
            widths = jnp.broadcast_to(jnp.asarray(width), self.shape)
            lines = jnp.concatenate((self.lines, widths[..., None]), axis=-1)

        if frames is not None:
            lines = jnp.concat([lines[self.index == frame] for frame in np.atleast_1d(frames)])
        return lines

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
