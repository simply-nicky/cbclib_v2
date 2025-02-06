from dataclasses import dataclass
from typing import Optional, Tuple, TypeVar
import numpy as np
import pandas as pd
from .annotations import IntArray, IntSequence, RealArray, RealSequence, Shape
from .data_container import ArrayContainer
from .src.bresenham import draw_line_image, draw_line_mask, draw_line_table

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

    def get_frames(self: L, frames: IntSequence) -> L:
        mask = np.any(self.index[..., None] == np.atleast_1d(frames), axis=-1)
        return self.filter(np.asarray(mask, dtype=bool))

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
            widths = np.broadcast_to(np.asarray(width), self.shape)
            lines = np.concatenate((self.lines, widths[..., None]), axis=-1)

        if frames is not None:
            lines = np.concat([lines[self.index == frame] for frame in np.atleast_1d(frames)])
        return lines

@dataclass
class Streaks(BaseLines):
    index       : IntArray
    lines       : RealArray

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame) -> 'Streaks':
        lines = np.stack((df['x_0'], df['y_0'], df['x_1'], df['y_1']), axis=-1)
        return cls(index=df['index'].to_numpy(), lines=lines)

    def concentric_only(self, x_ctr: float, y_ctr: float, threshold: float) -> 'Streaks':
        centers = np.mean(self.lines.reshape(-1, 2, 2), axis=1)
        norm = np.stack([self.lines[:, 3] - self.lines[:, 1],
                         self.lines[:, 0] - self.lines[:, 2]], axis=-1)
        r = centers - np.asarray([x_ctr, y_ctr])
        prod = np.sum(norm * r, axis=-1)[..., None]
        proj = r - prod * norm / np.sum(norm**2, axis=-1)[..., None]
        mask = np.sqrt(np.sum(proj**2, axis=-1)) / np.sqrt(np.sum(r**2, axis=-1)) < threshold
        return self.filter(mask)

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
        frames, y, x = np.unravel_index(idxs, normalised_shape)
        vals = np.array(list(table.values()))

        data = {'index': ids, 'frames': frames, 'y': y, 'x': x, 'value': vals}
        # data = data | {attr: getattr(self, attr)[ids] for attr in self.extra_attributes()}

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

    def to_dataframe(self) -> pd.DataFrame:
        """Export a streaks container into :class:`pandas.DataFrame`.

        Returns:
            A dataframe with all the data specified in :class:`cbclib_v2.Streaks`.
        """
        return pd.DataFrame({'index': self.index,
                             'x_0': self.x[:, 0], 'y_0': self.y[:, 0],
                             'x_1': self.x[:, 1], 'y_1': self.y[:, 1]})
