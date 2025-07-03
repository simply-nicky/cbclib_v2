from dataclasses import dataclass
from typing import Tuple, TypeVar
import numpy as np
import pandas as pd
from .annotations import BoolArray, RealArray, RealSequence, Shape
from .data_container import ArrayContainer, ArrayNamespace, IndexArray, IndexedContainer, NumPy
from .src.bresenham import draw_lines, write_lines

L = TypeVar("L", bound='BaseLines')

class BaseLines(ArrayContainer):
    lines       : RealArray

    @property
    def ndim(self) -> int:
        return self.lines.shape[-1] // 2

    @property
    def length(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.sqrt(xp.sum((self.pt1 - self.pt0)**2, axis=-1))

    @property
    def shape(self) -> Shape:
        return self.lines.shape[:-1]

    @property
    def points(self) -> RealArray:
        return self.lines.reshape(self.lines.shape[:-1] + (2, self.ndim))

    @property
    def pt0(self) -> RealArray:
        return self.lines[..., :self.ndim]

    @property
    def pt1(self) -> RealArray:
        return self.lines[..., self.ndim:]

    @property
    def x(self) -> RealArray:
        return self.lines[..., ::self.ndim]

    @property
    def y(self) -> RealArray:
        return self.lines[..., 1::self.ndim]

    def intersection(self: L, other: L) -> RealArray:
        def vector_dot(a: RealArray, b: RealArray) -> RealArray:
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

        tau = self.pt1 - self.pt0
        other_tau = other.pt1 - other.pt0

        t = vector_dot(other.pt0 - self.pt0, other_tau) / vector_dot(tau, other_tau)
        return self.pt0 + t[..., None] * tau

    def project(self, point: RealArray) -> RealArray:
        xp = self.__array_namespace__()
        tau = self.pt1 - self.pt0
        center = 0.5 * (self.pt0 + self.pt1)
        r = point - center
        r_tau = xp.sum(tau * r, axis=-1) / xp.sum(tau**2, axis=-1)
        r_tau = xp.clip(r_tau[..., None], -0.5, 0.5)
        return tau * r_tau + center

    def to_lines(self, width: RealSequence | None=None) -> RealArray:
        """Export a streaks container into line parameters ``x0, y0, x1, y1, width``:

        * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
        * `width` : Line's width.

        Returns:
            An array of line parameters.
        """
        xp = self.__array_namespace__()
        if width is None:
            lines = self.lines
        else:
            widths = xp.broadcast_to(xp.asarray(width), self.shape)
            lines = xp.concatenate((self.lines, widths[..., None]), axis=-1)

        return lines

@dataclass
class Lines(BaseLines):
    lines       : RealArray

@dataclass
class Streaks(IndexedContainer, BaseLines):
    index       : IndexArray
    lines       : RealArray

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame, xp: ArrayNamespace=NumPy) -> 'Streaks':
        lines = xp.stack((df['x_0'].to_numpy(), df['y_0'].to_numpy(),
                          df['x_1'].to_numpy(), df['y_1'].to_numpy()), axis=-1)
        return cls(index=IndexArray(xp.asarray(df['index'].to_numpy())),
                   lines=xp.asarray(lines))

    def concentric_only(self, x_ctr: float, y_ctr: float, threshold: float=0.33) -> BoolArray:
        xp = self.__array_namespace__()
        centers = xp.mean(self.lines.reshape(-1, 2, 2), axis=1)
        norm = xp.stack([self.lines[:, 3] - self.lines[:, 1],
                         self.lines[:, 0] - self.lines[:, 2]], axis=-1)
        r = centers - xp.asarray([x_ctr, y_ctr])
        prod = xp.sum(norm * r, axis=-1)[..., None]
        proj = r - prod * norm / xp.sum(norm**2, axis=-1)[..., None]
        mask = xp.sqrt(xp.sum(proj**2, axis=-1)) / xp.sqrt(xp.sum(r**2, axis=-1)) < threshold
        return mask

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
        xp = self.__array_namespace__()
        idxs, ids, values = write_lines(lines=self.to_lines(width=width), shape=shape,
                                        idxs=xp.asarray(self.index), kernel=kernel,
                                        num_threads=num_threads)
        normalised_shape = (np.prod(shape[:-2], dtype=int),) + shape[-2:]
        frames, y, x = xp.unravel_index(idxs, normalised_shape)

        data = {'index': ids, 'frames': frames, 'y': y, 'x': x, 'value': values}
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
        xp = self.__array_namespace__()
        return draw_lines(self.to_lines(width=width), shape=shape, idxs=xp.asarray(self.index),
                          kernel=kernel, num_threads=num_threads)

    def to_dataframe(self) -> pd.DataFrame:
        """Export a streaks container into :class:`pandas.DataFrame`.

        Returns:
            A dataframe with all the data specified in :class:`cbclib_v2.Streaks`.
        """
        return pd.DataFrame({'index': self.index,
                             'x_0': self.x[:, 0], 'y_0': self.y[:, 0],
                             'x_1': self.x[:, 1], 'y_1': self.y[:, 1]})
