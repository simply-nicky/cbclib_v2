from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from typing_extensions import Self
import pandas as pd
from .annotations import AnyNamespace, BoolArray, IntArray, NumPy, RealArray, RealSequence
from .array_api import array_namespace, asnumpy
from .data_container import ArrayContainer, IndexedContainer
from .functions import draw_lines

class BaseLines(ArrayContainer):
    """Base class for line-segment containers.

    A line segment is parameterised by two endpoints stored in a flat array
    of shape ``(..., 2 * ndim)`` in the order ``(x0, y0, ..., x1, y1, ...)``.
    Subclasses inherit geometry methods for intersection, projection, and
    distance computation.

    Attributes:
        lines: Float array of shape ``(..., 2 * ndim)`` with the endpoint
            coordinates of each line segment.
    """

    lines       : RealArray

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions (half the size of the last axis)."""
        return self.lines.shape[-1] // 2

    @property
    def length(self) -> RealArray:
        """Euclidean length of each line segment."""
        xp = self.__array_namespace__()
        return xp.sqrt(xp.sum((self.pt1 - self.pt0)**2, axis=-1))

    @property
    def points(self) -> RealArray:
        """Endpoints reshaped to ``(..., 2, ndim)``."""
        return self.lines.reshape(self.lines.shape[:-1] + (2, self.ndim))

    @property
    def pt0(self) -> RealArray:
        """First endpoint, shape ``(..., ndim)``."""
        return self.lines[..., :self.ndim]

    @property
    def pt1(self) -> RealArray:
        """Second endpoint, shape ``(..., ndim)``."""
        return self.lines[..., self.ndim:]

    @property
    def x(self) -> RealArray:
        """x-coordinates of both endpoints, shape ``(..., 2)``."""
        return self.lines[..., ::self.ndim]

    @property
    def y(self) -> RealArray:
        """y-coordinates of both endpoints, shape ``(..., 2)``."""
        return self.lines[..., 1::self.ndim]

    def intersection(self: Self, other: Self) -> RealArray:
        """Compute the intersection point of each line pair ``(self[i], other[i])``.

        Uses the cross-product formula for line–line intersection in 2-D.
        The result lies on the infinite extension of *self*; no clamping to
        the segment endpoints is performed.

        Args:
            other: Another :class:`BaseLines` container broadcastable with *self*.

        Returns:
            Array of shape ``(..., ndim)`` with the intersection coordinates.
        """
        def vector_dot(a: RealArray, b: RealArray) -> RealArray:
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

        tau = self.pt1 - self.pt0
        other_tau = other.pt1 - other.pt0

        t = vector_dot(other.pt0 - self.pt0, other_tau) / vector_dot(tau, other_tau)
        return self.pt0 + t[..., None] * tau

    def project(self, point: RealArray) -> RealArray:
        """Project *point* onto the nearest location on each segment.

        The projection is clamped to the segment: parameterised as
        ``center + t * tau`` with ``t ∈ [-0.5, 0.5]``, where *center* is
        the segment midpoint and *tau* is its direction vector.

        Args:
            point: Array broadcastable to ``(..., ndim)``.

        Returns:
            Array of shape ``(..., ndim)`` with the clamped projection
            coordinates.
        """
        xp = self.__array_namespace__()
        tau = self.pt1 - self.pt0
        center = 0.5 * (self.pt0 + self.pt1)
        r = point - center
        tau_mag = xp.sum(tau**2, axis=-1)
        tau_mag_safe = xp.where(tau_mag != 0, tau_mag, 1)
        r_tau = xp.where(tau_mag != 0, xp.sum(tau * r, axis=-1) / tau_mag_safe, 0)
        r_tau = xp.clip(r_tau[..., None], -0.5, 0.5)
        return tau * r_tau + center

    def distance(self, point: RealArray) -> RealArray:
        """Euclidean distance from *point* to the nearest location on each segment.

        Delegates to :meth:`project` and returns the distance between *point*
        and its projection.

        Args:
            point: Array broadcastable to ``(..., ndim)``.

        Returns:
            Array of distances, shape ``(...,)``.
        """
        xp = self.__array_namespace__()
        return xp.sqrt(xp.sum((self.project(point) - point)**2, axis=-1))

    def to_lines(self, width: RealSequence | None=None) -> RealArray:
        """Return line parameters ``(x0, y0, x1, y1[, width])`` as a plain array.

        Args:
            width: Line width in pixels, broadcast to the leading shape of
                *self*.  If ``None``, width is omitted.

        Returns:
            Array of shape ``(..., 2 * ndim)`` when *width* is ``None``, or
            ``(..., 2 * ndim + 1)`` otherwise.
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
    """Minimal line-segment container without a frame index.

    Stores a plain array of line endpoints and exposes the full
    :class:`BaseLines` geometry API.  Use this class when line segments do
    not need to be grouped by frame.

    Attributes:
        lines: Float array of shape ``(N, 2 * ndim)`` with the endpoint
            coordinates ``(x0, y0, ..., x1, y1, ...)``.
    """

    lines       : RealArray

class BaseStreaks(IndexedContainer, BaseLines):
    """Frame-indexed line-segment container.

    Extends :class:`BaseLines` by adding an integer ``index`` field that
    maps each streak to its parent frame.  The :attr:`flat_index` property
    returns a frame address suitable for passing to
    :func:`~cbclib_v2.ndimage.draw_lines`.

    Attributes:
        index: Integer frame index for each streak, shape ``(N,)``.
        lines: Float array of shape ``(N, 4)`` with the endpoint
            coordinates ``(x0, y0, x1, y1)``.
    """

    index       : IntArray
    lines       : RealArray

    @property
    def flat_index(self) -> IntArray:
        """Flat frame address for each streak; defaults to :attr:`index`."""
        return self.index

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace=NumPy
                         ) -> Tuple[IntArray, RealArray]:
        """Construct ``(index, lines)`` arrays from a :class:`~pandas.DataFrame`.

        The dataframe must contain columns ``'index'``, ``'x_0'``, ``'y_0'``,
        ``'x_1'``, ``'y_1'``.

        Args:
            df: Source dataframe or series.
            xp: Array namespace used for the output arrays.

        Returns:
            Tuple ``(index, lines)`` ready for the subclass constructor.
        """
        index = xp.asarray(df['index'])
        lines = xp.stack((xp.asarray(df['x_0']), xp.asarray(df['y_0']),
                          xp.asarray(df['x_1']), xp.asarray(df['y_1'])), axis=-1)
        return index, lines

    @classmethod
    def import_xy(cls, index: IntArray, x: RealArray, y: RealArray
                  ) -> Tuple[IntArray, RealArray]:
        """Construct ``(index, lines)`` arrays from separate coordinate arrays.

        Args:
            index: Integer frame index for each streak, shape ``(N,)``.
            x: x-coordinates of both endpoints, shape ``(N, 2)``.
            y: y-coordinates of both endpoints, shape ``(N, 2)``.

        Returns:
            Tuple ``(index, lines)`` with *lines* of shape ``(N, 4)`` in
            ``(x0, y0, x1, y1)`` order.
        """
        xp = array_namespace(x, y)
        lines = xp.stack((x[..., 0], y[..., 0], x[..., 1], y[..., 1]), axis=-1)
        return index, lines

    def pattern_image(self, out: RealArray, width: float, kernel: str='gaussian'
                      ) -> RealArray:
        """Rasterise all streaks onto *out* as thick line segments.

        Args:
            out: Output array of shape ``(n_frames, height, width)``.
            width: Line half-width in pixels.
            kernel: Radial profile for antialiasing.  Choose one of the
                supported `kernel functions
                <https://en.wikipedia.org/wiki/Kernel_(statistics)>`_:

                * ``'biweight'`` — quartic (biweight) kernel.
                * ``'gaussian'`` — Gaussian kernel.
                * ``'parabolic'`` — Epanechnikov (parabolic) kernel.
                * ``'rectangular'`` — uniform (box) kernel.
                * ``'triangular'`` — triangular kernel.

        Returns:
            *out* with streaks drawn in-place.
        """
        xp = self.__array_namespace__()
        return draw_lines(out=out, lines=self.to_lines(width=width),
                          idxs=xp.asarray(self.flat_index), kernel=kernel)

    def to_dataframe(self) -> pd.DataFrame:
        """Export the streak container to a :class:`~pandas.DataFrame`.

        Returns:
            DataFrame with columns ``'index'``, ``'x_0'``, ``'y_0'``,
            ``'x_1'``, ``'y_1'``.
        """
        return pd.DataFrame({'index': asnumpy(self.index),
                             'x_0': asnumpy(self.x[:, 0]), 'y_0': asnumpy(self.y[:, 0]),
                             'x_1': asnumpy(self.x[:, 1]), 'y_1': asnumpy(self.y[:, 1])})

@dataclass
class Streaks(BaseStreaks):
    """Streak container for a flat, single-module detector.

    Each streak is a 2-D line segment ``(x0, y0, x1, y1)`` associated with
    a frame index.

    Attributes:
        index: Integer frame index for each streak, shape ``(N,)``.
        lines: Float array of shape ``(N, 4)`` with the endpoint
            coordinates ``(x0, y0, x1, y1)``.
    """

    index       : IntArray
    lines       : RealArray

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace=NumPy) -> 'Streaks':
        """Construct a :class:`Streaks` from a :class:`~pandas.DataFrame`.

        The dataframe must contain columns ``'index'``, ``'x_0'``, ``'y_0'``,
        ``'x_1'``, ``'y_1'``.

        Args:
            df: Source dataframe or series.
            xp: Array namespace for the output arrays.

        Returns:
            New :class:`Streaks` instance.
        """
        index, lines = super(Streaks, cls).import_dataframe(df, xp)
        return cls(index=index, lines=lines)

    @classmethod
    def import_xy(cls, index: IntArray, x: RealArray, y: RealArray) -> 'Streaks':
        """Construct a :class:`Streaks` from index and coordinate arrays.

        Args:
            index: Integer frame index for each streak, shape ``(N,)``.
            x: x-coordinates of both endpoints, shape ``(N, 2)``.
            y: y-coordinates of both endpoints, shape ``(N, 2)``.

        Returns:
            New :class:`Streaks` instance.
        """
        index, lines = super(Streaks, cls).import_xy(index, x, y)
        return cls(index=index, lines=lines)

    def concentric_only(self, x_ctr: float, y_ctr: float, threshold: float=0.33) -> BoolArray:
        """Return a boolean mask selecting streaks tangential to circles centred at *(x_ctr, y_ctr)*.

        A streak is considered concentric when its line direction aligns with
        the tangential direction at its midpoint — equivalently, the component
        of the midpoint–centre vector *along the streak axis* is less than
        *threshold* times its total length.

        Args:
            x_ctr: x-coordinate of the reference centre in pixels.
            y_ctr: y-coordinate of the reference centre in pixels.
            threshold: Maximum allowed ratio of the along-streak radial
                projection to the full midpoint–centre distance.
                Defaults to 0.33.

        Returns:
            Boolean array of shape ``(N,)``; ``True`` for concentric streaks.
        """
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
    """Streak container for a multi-module, stacked detector.

    Extends :class:`BaseStreaks` by adding a ``module_id`` field and a
    ``num_modules`` count.  The :attr:`flat_index` property combines frame
    and module into a single integer suitable for addressing a
    ``(n_frames * num_modules, height, width)`` image stack.

    Attributes:
        index: Integer frame index for each streak, shape ``(N,)``.
        module_id: Detector module index for each streak, shape ``(N,)``.
        lines: Float array of shape ``(N, 4)`` with the endpoint
            coordinates ``(x0, y0, x1, y1)``.
        num_modules: Total number of detector modules.
    """

    index       : IntArray
    module_id   : IntArray
    lines       : RealArray
    num_modules : int = 1

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, num_modules: int=1,
                         xp: AnyNamespace=NumPy) -> 'StackedStreaks':
        """Construct a :class:`StackedStreaks` from a :class:`~pandas.DataFrame`.

        The dataframe must contain columns ``'index'``, ``'x_0'``, ``'y_0'``,
        ``'x_1'``, ``'y_1'``, and ``'module_id'`` when *num_modules* > 1.

        Args:
            df: Source dataframe or series.
            num_modules: Number of detector modules.
            xp: Array namespace for the output arrays.

        Returns:
            New :class:`StackedStreaks` instance.
        """
        index, lines = super(StackedStreaks, cls).import_dataframe(df, xp)
        if num_modules > 1:
            module_id = xp.asarray(df['module_id'])
        else:
            module_id = xp.zeros_like(index)
        return cls(index=index, module_id=module_id, lines=lines, num_modules=num_modules)

    @classmethod
    def import_xy(cls, index: IntArray, module_id: IntArray, x: RealArray, y: RealArray,
                  num_modules: int=1) -> 'StackedStreaks':
        """Construct a :class:`StackedStreaks` from index and coordinate arrays.

        Args:
            index: Integer frame index for each streak, shape ``(N,)``.
            module_id: Detector module index for each streak, shape ``(N,)``.
            x: x-coordinates of both endpoints, shape ``(N, 2)``.
            y: y-coordinates of both endpoints, shape ``(N, 2)``.
            num_modules: Number of detector modules.

        Returns:
            New :class:`StackedStreaks` instance.
        """
        index, lines = super(StackedStreaks, cls).import_xy(index, x, y)
        return cls(index=index, module_id=module_id, lines=lines, num_modules=num_modules)

    @property
    def flat_index(self) -> IntArray:
        """Flat frame address computed as ``num_modules * index + module_id``."""
        return self.num_modules * self.index + self.module_id

    def to_dataframe(self) -> pd.DataFrame:
        """Export the streak container to a :class:`~pandas.DataFrame`.

        Returns:
            DataFrame with columns ``'index'``, ``'x_0'``, ``'y_0'``,
            ``'x_1'``, ``'y_1'``, and ``'module_id'`` when
            :attr:`num_modules` > 1.
        """
        dataframe = super().to_dataframe()
        if self.num_modules > 1:
            dataframe['module_id'] = asnumpy(self.module_id)
        return dataframe
