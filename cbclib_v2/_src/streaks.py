from dataclasses import dataclass
from math import prod
from typing import Tuple, TypeVar
import pandas as pd
from .annotations import BoolArray, IntArray, RealArray, RealSequence, Shape
from .data_container import ArrayContainer, ArrayNamespace, IndexedContainer, NumPy, array_namespace
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
            widths = xp.broadcast_to(xp.asarray(width), self.shape[:-1])
            lines = xp.concatenate((self.lines, widths[..., None]), axis=-1)

        return lines

@dataclass
class Lines(BaseLines):
    lines       : RealArray

class BaseStreaks(IndexedContainer, BaseLines):
    index       : IntArray
    lines       : RealArray

    @property
    def flat_index(self) -> IntArray:
        return self.index

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame, filename: str | None=None,
                         xp: ArrayNamespace=NumPy) -> Tuple[IntArray, RealArray]:
        if filename is None:
            index = xp.asarray(df['index'].to_numpy())
            lines = xp.stack((df['x_0'].to_numpy(), df['y_0'].to_numpy(),
                              df['x_1'].to_numpy(), df['y_1'].to_numpy()), axis=-1)
        else:
            df = df[df['file'] == filename]
            index = xp.asarray(df['frames'].to_numpy())
            lines = xp.stack((df['x_0'].to_numpy(), df['y_0'].to_numpy(),
                              df['x_1'].to_numpy(), df['y_1'].to_numpy()), axis=-1)
        return index, lines

    @classmethod
    def import_xy(cls, index: IntArray, x: RealArray, y: RealArray
                  ) -> Tuple[IntArray, RealArray]:
        xp = array_namespace(x, y)
        lines = xp.stack((x[..., 0], y[..., 0], x[..., 1], y[..., 1]), axis=-1)
        return index, lines

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
        index, streak_id, value = write_lines(lines=self.to_lines(width=width), shape=shape,
                                              idxs=xp.asarray(self.flat_index), kernel=kernel,
                                              num_threads=num_threads)
        normalised_shape = (prod(shape[:-2]),) + shape[-2:]
        frames, y, x = xp.unravel_index(index, normalised_shape)

        data = {'index': streak_id, 'frames': frames, 'y': y, 'x': x, 'value': value}
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
        return draw_lines(self.to_lines(width=width), idxs=xp.asarray(self.flat_index),
                          shape=shape, kernel=kernel, num_threads=num_threads)

    def to_dataframe(self) -> pd.DataFrame:
        """Export a streaks container into :class:`pandas.DataFrame`.

        Returns:
            A dataframe with all the data specified in :class:`cbclib_v2.Streaks`.
        """
        return pd.DataFrame({'index': self.index,
                             'x_0': self.x[:, 0], 'y_0': self.y[:, 0],
                             'x_1': self.x[:, 1], 'y_1': self.y[:, 1]})

@dataclass
class Streaks(BaseStreaks):
    index       : IntArray
    lines       : RealArray

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame, filename: str | None=None,
                         xp: ArrayNamespace=NumPy) -> 'Streaks':
        index, lines = super(Streaks, cls).import_dataframe(df, filename, xp)
        return cls(index=index, lines=lines)

    @classmethod
    def import_xy(cls, index: IntArray, x: RealArray, y: RealArray) -> 'Streaks':
        index, lines = super(Streaks, cls).import_xy(index, x, y)
        return cls(index=index, lines=lines)

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

@dataclass
class StackedStreaks(BaseStreaks):
    index       : IntArray
    module_id   : IntArray
    lines       : RealArray
    n_modules   : int = 1

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame, n_modules: int=1, filename: str | None=None,
                         xp: ArrayNamespace=NumPy) -> 'StackedStreaks':
        index, lines = super(StackedStreaks, cls).import_dataframe(df, filename, xp)
        return cls(index=index, module_id=xp.asarray(df['module_id'].to_numpy()), lines=lines,
                   n_modules=n_modules)

    @classmethod
    def import_xy(cls, index: IntArray, module_id: IntArray, x: RealArray, y: RealArray,
                  n_modules: int=1) -> 'StackedStreaks':
        index, lines = super(StackedStreaks, cls).import_xy(index, x, y)
        return cls(index=index, module_id=module_id, lines=lines, n_modules=n_modules)

    @property
    def flat_index(self) -> IntArray:
        return self.n_modules * self.index + self.module_id

    def pattern_dataframe(self, width: float, shape: Shape, kernel: str='rectangular',
                          num_threads: int=1) -> pd.DataFrame:
        dataframe = super().pattern_dataframe(width, shape, kernel, num_threads)
        if self.n_modules > 1:
            dataframe['module_id'] = dataframe['frames'] % self.n_modules
            dataframe['frames'] = dataframe['frames'] // self.n_modules
            return dataframe.loc[:, ['index', 'frames', 'module_id', 'y', 'x', 'value']]
        return dataframe

    def to_dataframe(self) -> pd.DataFrame:
        """Export a streaks container into :class:`pandas.DataFrame`.

        Returns:
            A dataframe with all the data specified in :class:`cbclib_v2.StackedStreaks`.
        """
        dataframe = super().to_dataframe()
        if self.n_modules > 1:
            dataframe['module_id'] = self.module_id
        return dataframe
