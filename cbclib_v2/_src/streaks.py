from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, TypeVar
import numpy as np
import pandas as pd
from .annotations import IntArray, IntSequence, RealArray, RealSequence, Shape
from .data_container import ArrayContainer, ArrayNamespace, NumPy, to_list
from .src.bresenham import draw_lines, write_lines

L = TypeVar("L", bound="BaseLines")

class BaseLines(ArrayContainer):
    index       : IntArray
    lines       : RealArray

    @property
    def shape(self) -> Shape:
        return self.lines.shape[:-1]

    @property
    def x(self) -> RealArray:
        return self.lines[..., ::2]

    @property
    def y(self) -> RealArray:
        return self.lines[..., 1::2]

    def __iter__(self: L) -> Iterator[L]:
        xp = self.__array_namespace__()
        indices = xp.unique(self.index)
        for index in indices:
            yield self[self.index == index]

    def __len__(self) -> int:
        xp = self.__array_namespace__()
        return xp.unique(self.index).size

    def select(self: L, indices: IntSequence) -> L:
        xp = self.__array_namespace__()
        patterns = list(iter(self))
        result = [patterns[index].replace(index=xp.full(patterns[index].index.size, new_index))
                  for new_index, index in enumerate(to_list(indices))]
        return type(self).concatenate(*result)

    def to_lines(self, width: Optional[RealSequence]=None) -> RealArray:
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
class Streaks(BaseLines):
    index       : IntArray
    lines       : RealArray

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame, xp: ArrayNamespace=NumPy) -> 'Streaks':
        lines = xp.stack((df['x_0'].to_numpy(), df['y_0'].to_numpy(),
                          df['x_1'].to_numpy(), df['y_1'].to_numpy()), axis=-1)
        return cls(index=xp.asarray(df['index'].to_numpy()),
                   lines=xp.asarray(lines))

    def concentric_only(self, x_ctr: float, y_ctr: float, threshold: float) -> 'Streaks':
        xp = self.__array_namespace__()
        centers = xp.mean(self.lines.reshape(-1, 2, 2), axis=1)
        norm = xp.stack([self.lines[:, 3] - self.lines[:, 1],
                         self.lines[:, 0] - self.lines[:, 2]], axis=-1)
        r = centers - xp.asarray([x_ctr, y_ctr])
        prod = xp.sum(norm * r, axis=-1)[..., None]
        proj = r - prod * norm / xp.sum(norm**2, axis=-1)[..., None]
        mask = xp.sqrt(xp.sum(proj**2, axis=-1)) / xp.sqrt(xp.sum(r**2, axis=-1)) < threshold
        return self[mask]

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
        table = write_lines(lines=self.to_lines(width=width), shape=shape, idxs=self.index,
                            kernel=kernel, num_threads=num_threads)
        ids, idxs = np.array(list(table)).T
        normalised_shape = (np.prod(shape[:-2], dtype=int),) + shape[-2:]
        frames, y, x = np.unravel_index(idxs, normalised_shape)
        vals = np.array(list(table.values()))

        data = {'index': ids, 'frames': frames, 'y': y, 'x': x, 'value': vals}
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
        return draw_lines(self.to_lines(width=width), shape=shape, idxs=self.index,
                          kernel=kernel, num_threads=num_threads)

    def to_dataframe(self) -> pd.DataFrame:
        """Export a streaks container into :class:`pandas.DataFrame`.

        Returns:
            A dataframe with all the data specified in :class:`cbclib_v2.Streaks`.
        """
        return pd.DataFrame({'index': self.index,
                             'x_0': self.x[:, 0], 'y_0': self.y[:, 0],
                             'x_1': self.x[:, 1], 'y_1': self.y[:, 1]})
