from dataclasses import dataclass
from typing import Tuple
from typing_extensions import Self
import pandas as pd
from .annotations import AnyNamespace, BoolArray, IntArray, RealArray, RealSequence
from .array_api import array_namespace, asnumpy
from .data_container import ArrayContainer, IndexedContainer, NumPy
from .functions import draw_lines

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

    def intersection(self: Self, other: Self) -> RealArray:
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
            widths = xp.broadcast_to(xp.asarray(width), self.lines.shape[:-1] + (1,))
            lines = xp.concat((self.lines, widths), axis=-1)

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
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace=NumPy
                         ) -> Tuple[IntArray, RealArray]:
        index = xp.asarray(df['index'])
        lines = xp.stack((xp.asarray(df['x_0']), xp.asarray(df['y_0']),
                          xp.asarray(df['x_1']), xp.asarray(df['y_1'])), axis=-1)
        return index, lines

    @classmethod
    def import_xy(cls, index: IntArray, x: RealArray, y: RealArray
                  ) -> Tuple[IntArray, RealArray]:
        xp = array_namespace(x, y)
        lines = xp.stack((x[..., 0], y[..., 0], x[..., 1], y[..., 1]), axis=-1)
        return index, lines

    def pattern_image(self, out: RealArray, width: float, kernel: str='gaussian'
                      ) -> RealArray:
        """Draw a pattern in the :class:`numpy.ndarray` format.

        Args:
            out : Output array where the pattern will be drawn.
            width : Lines width in pixels.
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
        return draw_lines(out=out, lines=self.to_lines(width=width),
                          idxs=xp.asarray(self.flat_index), kernel=kernel)

    def to_dataframe(self) -> pd.DataFrame:
        """Export a streaks container into :class:`pandas.DataFrame`.

        Returns:
            A dataframe with all the data specified in :class:`cbclib_v2.Streaks`.
        """
        return pd.DataFrame({'index': asnumpy(self.index),
                             'x_0': asnumpy(self.x[:, 0]), 'y_0': asnumpy(self.y[:, 0]),
                             'x_1': asnumpy(self.x[:, 1]), 'y_1': asnumpy(self.y[:, 1])})

@dataclass
class Streaks(BaseStreaks):
    index       : IntArray
    lines       : RealArray

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace=NumPy) -> 'Streaks':
        index, lines = super(Streaks, cls).import_dataframe(df, xp)
        return cls(index=index, lines=lines)

    @classmethod
    def import_xy(cls, index: IntArray, x: RealArray, y: RealArray) -> 'Streaks':
        index, lines = super(Streaks, cls).import_xy(index, x, y)
        return cls(index=index, lines=lines)

    def concentric_only(self, x_ctr: float, y_ctr: float, threshold: float=0.33) -> BoolArray:
        xp = self.__array_namespace__()
        centers = xp.mean(self.lines.reshape((-1, 2, 2)), axis=1)
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
    num_modules : int = 1

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, num_modules: int=1,
                         xp: AnyNamespace=NumPy) -> 'StackedStreaks':
        index, lines = super(StackedStreaks, cls).import_dataframe(df, xp)
        if num_modules > 1:
            module_id = xp.asarray(df['module_id'])
        else:
            module_id = xp.zeros_like(index)
        return cls(index=index, module_id=module_id, lines=lines, num_modules=num_modules)

    @classmethod
    def import_xy(cls, index: IntArray, module_id: IntArray, x: RealArray, y: RealArray,
                  num_modules: int=1) -> 'StackedStreaks':
        index, lines = super(StackedStreaks, cls).import_xy(index, x, y)
        return cls(index=index, module_id=module_id, lines=lines, num_modules=num_modules)

    @property
    def flat_index(self) -> IntArray:
        return self.num_modules * self.index + self.module_id

    def to_dataframe(self) -> pd.DataFrame:
        """Export a streaks container into :class:`pandas.DataFrame`.

        Returns:
            A dataframe with all the data specified in :class:`cbclib_v2.StackedStreaks`.
        """
        dataframe = super().to_dataframe()
        if self.num_modules > 1:
            dataframe['module_id'] = asnumpy(self.module_id)
        return dataframe
