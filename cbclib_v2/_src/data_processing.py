""":class:`cbclib_v2.CrystData` stores all the data necessarry to process measured convergent
beam crystallography patterns and provides a suite of data processing tools to wor with the
detector data.

Examples:
    Load all the necessary data using a :func:`cbclib_v2.CrystData.load` function.

    >>> import cbclib as cbc
    >>> inp_file = cbc.CXIStore('data.cxi')
    >>> data = cbc.CrystData(inp_file)
    >>> data = data.load()
"""
from math import prod
import os
from typing import Literal, NamedTuple, Sequence, TypeVar, cast
from dataclasses import dataclass, field
from weakref import ref
import numpy as np
import pandas as pd
from .cxi_protocol import CXIProtocol, Kinds
from .data_container import DataContainer, list_indices
from .streak_finder import PeaksList, detect_peaks, detect_streaks, filter_peaks
from .streaks import StackedStreaks, Streaks
from .annotations import (Array, ArrayLike, BoolArray, Indices, IntArray, RealArray, ReferenceType,
                          ROI, Shape)
from .label import (center_of_mass, covariance_matrix, ellipse_fit, label, line_fit, mean,
                    moment_of_inertia, total_mass, RegionsList2D, Structure2D)
from .src.signal_proc import binterpolate
from .src.median import median, robust_mean, robust_lsq

MaskMethod = Literal['all-bad', 'no-bad', 'range', 'snr']
MDMethod = Literal['median-poisson', 'robust-mean-scale', 'robust-mean-poisson']
STDMethod = Literal['poisson', 'robust-scale']
WFMethod = Literal['median', 'robust-mean', 'robust-mean-scale']

DATA_PROTOCOL = os.path.join(os.path.dirname(__file__), 'config/cryst_data.ini')
METADATA_PROTOCOL = os.path.join(os.path.dirname(__file__), 'config/cryst_metadata.ini')

C = TypeVar("C", bound='CrystBase')

class CrystBase(DataContainer):
    protocol    : CXIProtocol

    @property
    def frame_shape(self) -> Shape:
        current, old = tuple(), tuple()
        for attr, data in self.contents().items():
            kind = self.protocol.get_kind(attr)
            if isinstance(data, Array):
                if kind == Kinds.frame:
                    current = data.shape
                elif kind == Kinds.stack:
                    current = data.shape[1:]
                else:
                    continue

                if old and current != old:
                    raise ValueError(f"Attribute {attr} has an incompatible shape: {current}")

            old = current

        return current

    @property
    def num_modules(self) -> int:
        return prod(self.frame_shape) // prod(self.frame_shape[-2:])

    def crop(self: C, roi: ROI) -> C:
        cropped = {}
        for attr, data in self.contents().items():
            if self.protocol.get_kind(attr) in (Kinds.frame, Kinds.stack):
                cropped[attr] = data[..., roi[0]:roi[1], roi[2]:roi[3]]
        return self.replace(**cropped)

class PCAProjection(NamedTuple):
    good_fields : Sequence[int] | IntArray
    projection  : RealArray

@dataclass
class CrystMetadata(CrystBase):
    eigen_field : RealArray = field(default_factory=lambda: np.array([]))
    eigen_value : RealArray = field(default_factory=lambda: np.array([]))
    flatfield   : RealArray = field(default_factory=lambda: np.array([]))
    frames      : IntArray  = field(default_factory=lambda: np.array([], dtype=int))
    data_frames : IntArray  = field(default_factory=lambda: np.array([], dtype=int))
    mask        : BoolArray = field(default_factory=lambda: np.array([], dtype=bool))
    std         : RealArray = field(default_factory=lambda: np.array([]))
    whitefield  : RealArray = field(default_factory=lambda: np.array([]))

    protocol    : CXIProtocol = field(default_factory=lambda: CXIProtocol.read(METADATA_PROTOCOL))

    def __post_init__(self):
        super().__post_init__()
        if not self.is_empty(self.mask):
            if not self.is_empty(self.std):
                self.std *= self.mask
            if not self.is_empty(self.whitefield):
                self.whitefield *= self.mask

        if self.is_empty(self.flatfield) and not self.is_empty(self.whitefield):
            self.flatfield = self.whitefield.mean(axis=0)

    @property
    def center_frames(self) -> IntArray:
        if not self.is_empty(self.data_frames):
            return np.asarray(median(self.data_frames, axis=-1), dtype=int)
        return np.array([], dtype=int)

    @classmethod
    def default_protocol(cls) -> CXIProtocol:
        return CXIProtocol.read(METADATA_PROTOCOL)

    def import_data(self, data: RealArray, frames: IntArray | int=np.array([], dtype=int),
                    whitefield: RealArray=np.array([])) -> 'CrystData':
        xp = self.__array_namespace__()
        frames = xp.atleast_1d(frames)

        if not whitefield.size:
            if self.is_empty(self.flatfield):
                raise ValueError('no flatfield in the container')
            return CrystData(data=data, frames=frames, mask=self.mask, std=self.std,
                             whitefield=self.flatfield)

        if whitefield.shape != data.shape:
            raise ValueError(f'whitefield shape {whitefield.shape} is incompatible with data '
                             f'{data.shape}')
        protocol = CrystData.default_protocol()
        protocol.kinds['whitefield'] = 'stack'
        return CrystData(data=data, frames=frames, mask=self.mask, std=self.std,
                         whitefield=whitefield, protocol=protocol)

    def interpolate(self, frames: IntArray, num_threads: int=1) -> RealArray:
        if self.is_empty(self.data_frames):
            raise ValueError('no data_frames in the container')
        if self.is_empty(self.whitefield):
            raise ValueError('no whitefield in the container')

        return binterpolate(self.whitefield, [self.center_frames,], frames, axis=0,
                            num_threads=num_threads)

    def pca(self) -> 'CrystMetadata':
        if self.is_empty(self.whitefield):
            raise ValueError('no whitefield in the container')
        if self.whitefield.shape[0] == 1:
            raise ValueError('A stack of several whitefields is needed to perform PCA')

        xp = self.__array_namespace__()
        fields = self.whitefield - self.flatfield
        axes = tuple(range(1, len(self.frame_shape) + 1))
        mat_svd = xp.tensordot(fields, fields, axes=(axes, axes))
        eig_vals, eig_vecs = xp.linalg.eig(mat_svd)
        effs = xp.tensordot(eig_vecs, fields, axes=((0,), (0,)))
        return self.replace(eigen_field=effs, eigen_value=eig_vals / eig_vals.sum())

    def projection(self, data: RealArray, good_fields: Indices=slice(None),
                   method: str="robust-lsq", r0: float=0.0, r1: float=0.5, n_iter: int=12,
                   lm: float=9.0, num_threads: int=1) -> PCAProjection:
        """Return a new :class:`CrystData` object with a new set of whitefields. A set of
        backgrounds is generated by robustly fitting a design matrix `W` to the measured
        patterns.

        Args:
            method : Choose one of the following methods to scale the white-field:

                * "median" : By taking a median of data and whitefield.
                * "robust-lsq" : By solving a least-squares problem with truncated
                  with the fast least k-th order statistics (FLkOS) estimator.

            r0 : A lower bound guess of ratio of inliers. We'd like to make a sample
                out of worst inliers from data points that are between `r0` and `r1`
                of sorted residuals.
            r1 : An upper bound guess of ratio of inliers. Choose the `r0` to be as
                high as you are sure the ratio of data is inlier.
            n_iter : Number of iterations of fitting a gaussian with the FLkOS
                algorithm.
            lm : How far (normalized by STD of the Gaussian) from the mean of the
                Gaussian, data is considered inlier.

        Raises:
            ValueError : If there is no ``data`` inside the container.
            ValueError : If there is no ``whitefield`` inside the container.

        Returns:
            An array of scale factors for each frame in the container.
        """
        if self.is_empty(self.flatfield):
            raise ValueError('No flatfield in the container')
        if self.is_empty(self.eigen_field):
            raise ValueError('No eigen_field in the container')

        good_fields = list_indices(good_fields, self.eigen_field.shape[0])
        fields = self.eigen_field[good_fields]

        xp = self.__array_namespace__()
        y: RealArray = xp.reshape(data - self.flatfield, (data.size // prod(self.frame_shape), -1))
        W: RealArray = xp.reshape(fields, (fields.size // prod(self.frame_shape), -1))

        if method == "robust-lsq":
            projection = robust_lsq(W=W, y=y, axis=1, r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                    num_threads=num_threads)
        elif method == "lsq":
            projection = xp.mean(y[..., None, :] * W, axis=-1) / xp.mean(W * W, axis=-1)
        else:
            raise ValueError(f"Invalid method argument: {method}")

        return PCAProjection(good_fields=good_fields, projection=projection)

    def project(self, projection: PCAProjection) -> RealArray:
        if self.is_empty(self.eigen_field):
            raise ValueError('No eigen_field in the container')

        xp = self.__array_namespace__()
        fields = xp.tensordot(projection.projection, self.eigen_field[projection.good_fields],
                              ((-1,), (0,)))
        return self.flatfield + fields

@dataclass
class CrystData(CrystBase):
    """Convergent beam crystallography data container class. Takes a :class:`cbclib_v2.CXIStore`
    file handler. Provides an interface to work with the detector images and detect the diffraction
    streaks. Also provides an interface to load from a file and save to a file any of the data
    attributes. The data frames can be tranformed using any of the :class:`cbclib_v2.Transform`
    classes.

    Args:
        data : Detector raw data.
        mask : Bad pixels mask.
        frames : List of frame indices inside the container.
        whitefield : Measured frames' white-field.
        snr : Signal-to-noise ratio.
    """
    data        : RealArray = field(default_factory=lambda: np.array([]))

    whitefield  : RealArray = field(default_factory=lambda: np.array([]))
    std         : RealArray = field(default_factory=lambda: np.array([]))
    snr         : RealArray = field(default_factory=lambda: np.array([]))

    frames      : IntArray = field(default_factory=lambda: np.array([], dtype=int))
    mask        : BoolArray = field(default_factory=lambda: np.array([], dtype=bool))

    protocol    : CXIProtocol = field(default_factory=lambda: CXIProtocol.read(DATA_PROTOCOL))

    def __post_init__(self):
        super().__post_init__()
        xp = self.__array_namespace__()
        if self.frames.size != self.num_frames:
            self.frames = xp.arange(self.num_frames)
        if self.mask.shape != self.frame_shape:
            self.mask = xp.ones(self.frame_shape, dtype=bool)

    @property
    def num_frames(self) -> int:
        current, old = 0, 0
        for attr, data in self.contents().items():
            kind = self.protocol.get_kind(attr)
            if kind == Kinds.stack and isinstance(data, Array):
                current = data.shape[0]

            if old and current != old:
                raise ValueError(f"Attribute {attr} has an incompatible shape: {data.shape}")

            old = current

        return current

    @property
    def num_whitefields(self) -> int:
        return self.whitefield.size // prod(self.frame_shape)

    @property
    def shape(self) -> Shape:
        return (self.num_frames,) + self.frame_shape

    @classmethod
    def default_protocol(cls) -> CXIProtocol:
        return CXIProtocol.read(DATA_PROTOCOL)

    def apply_mask(self) -> 'CrystData':
        attributes = {}
        if not self.is_empty(self.whitefield):
            attributes['whitefield'] = self.whitefield * self.mask
        if not self.is_empty(self.std):
            attributes['std'] = self.std * self.mask
        if not self.is_empty(self.snr):
            attributes['snr'] = self.snr * self.mask
        return self.replace(**attributes)

    def import_mask(self, mask: BoolArray, update: str='reset') -> 'CrystData':
        """Return a new :class:`CrystData` object with the new mask.

        Args:
            mask : New mask array.
            update : Multiply the new mask and the old one if 'multiply', use the
                new one if 'reset'.

        Raises:
            ValueError : If the mask shape is incompatible with the data.
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``mask``.
        """
        if self.is_empty(self.mask):
            raise ValueError('no mask in the container')
        if mask.shape != self.frame_shape:
            raise ValueError('mask and data have incompatible shapes: '\
                             f'{mask.shape} != {self.frame_shape}')

        if update == 'reset':
            return self.replace(mask=mask)
        if update == 'multiply':
            return self.replace(mask=mask * self.mask)
        raise ValueError(f'Invalid update keyword: {update:s}')

    def mask_region(self, roi: ROI) -> 'CrystData':
        """Return a new :class:`CrystData` object with the updated mask. The region
        defined by the `[y_min, y_max, x_min, x_max]` will be masked out.

        Args:
            roi : Bad region of interest in the detector plane. A set of four
                coordinates `[y_min, y_max, x_min, x_max]`.

        Raises:
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``mask``.
        """
        if self.is_empty(self.mask):
            raise ValueError('no mask in the container')

        xp = self.__array_namespace__()
        mask = xp.copy(self.mask)
        mask[roi[0]:roi[1], roi[2]:roi[3]] = False
        return self.replace(mask=mask).apply_mask()

    def region_detector(self, structure: Structure2D) -> 'RegionDetector':
        if self.is_empty(self.mask):
            raise ValueError('no mask in the container')
        if self.is_empty(self.snr):
            raise ValueError('no snr in the container')

        parent = cast(ReferenceType[CrystData], ref(self))
        return RegionDetector(data=self.snr, mask=self.mask, structure=structure, parent=parent)

    def reset_mask(self) -> 'CrystData':
        """Reset bad pixel mask. Every pixel is assumed to be good by default.

        Raises:
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the default ``mask``.
        """
        xp = self.__array_namespace__()
        return self.replace(mask=xp.array([], dtype=bool))

    def select(self, idxs: Indices | None=None):
        """Return a new :class:`CrystData` object with the new mask.

        Args:
            mask : New mask array.
            update : Multiply the new mask and the old one if 'multiply', use the
                new one if 'reset'.

        Raises:
            ValueError : If the mask shape is incompatible with the data.
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``mask``.
        """
        data_dict = {}
        for attr in self.contents():
            if self.protocol.get_kind(attr) in (Kinds.sequence, Kinds.stack):
                data_dict[attr] = getattr(self, attr)[idxs]
            else:
                data_dict[attr] = getattr(self, attr)
        return self.replace(**data_dict)

    def streak_detector(self, structure: Structure2D) -> 'StreakDetector':
        """Return a new :class:`cbclib_v2.StreakDetector` object that detects lines in SNR frames.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.
            ValueError : If there is no ``snr`` inside the container.

        Returns:
            A CBC pattern detector based on :class:`cbclib_v2.bin.LSD` Line Segment Detection [LSD]_
            algorithm.
        """
        if self.is_empty(self.mask):
            raise ValueError('no mask in the container')
        if self.is_empty(self.snr):
            raise ValueError('no snr in the container')

        parent = cast(ReferenceType[CrystData], ref(self))
        return StreakDetector(data=self.snr, mask=self.mask, structure=structure, parent=parent)

    def update_mask(self, method: MaskMethod='no-bad', vmin: int=0, vmax: int=65535,
                    snr_max: float=3.0, roi: ROI | None=None) -> 'CrystData':
        """Return a new :class:`CrystData` object with the updated bad pixels mask.

        Args:
            method : Bad pixels masking methods. The following keyword values are
                allowed:

                * 'all-bad' : Mask out all pixels.
                * 'no-bad' (default) : No bad pixels.
                * 'range' : Mask the pixels which values lie outside of (`vmin`,
                  `vmax`) range.
                * 'snr' : Mask the pixels which SNR values lie exceed the SNR
                  threshold `snr_max`. The snr is given by
                  :code:`abs(data - whitefield) / sqrt(whitefield)`.

            vmin : Lower intensity bound of 'range-bad' masking method.
            vmax : Upper intensity bound of 'range-bad' masking method.
            snr_max : SNR threshold.
            roi : Region of the frame undertaking the update. The whole frame is updated
                by default.

        Raises:
            ValueError : If there is no ``data`` inside the container.
            ValueError : If there is no ``snr`` inside the container.
            ValueError : If ``method`` keyword is invalid.
            ValueError : If ``vmin`` is larger than ``vmax``.

        Returns:
            New :class:`CrystData` object with the updated ``mask``.
        """
        if self.is_empty(self.data):
            raise ValueError('no data in the container')
        if self.is_empty(self.mask):
            raise ValueError('no mask in the container')

        xp = self.__array_namespace__()
        if vmin >= vmax:
            raise ValueError('vmin must be less than vmax')
        if roi is None:
            roi = (0, self.shape[-2], 0, self.shape[-1])

        data = (self.data * self.mask)[..., roi[0]:roi[1], roi[2]:roi[3]]

        if method == 'all-bad':
            mask = xp.zeros(self.frame_shape, dtype=bool)
        elif method == 'no-bad':
            mask = xp.ones(self.frame_shape, dtype=bool)
        elif method == 'range':
            mask = xp.all((data >= vmin) & (data < vmax), axis=0)
        elif method == 'snr':
            if self.snr is None:
                raise ValueError('No snr in the container')

            snr = self.snr[..., roi[0]:roi[1], roi[2]:roi[3]]
            mask = xp.mean(xp.abs(snr), axis=0) < snr_max
        else:
            raise ValueError(f'Invalid method argument: {method:s}')

        new_mask = xp.copy(self.mask)
        new_mask[..., roi[0]:roi[1], roi[2]:roi[3]] &= mask
        return self.replace(mask=new_mask)

    def update_snr(self, std_min: float=0.0) -> 'CrystData':
        """Return a new :class:`CrystData` object with new background corrected detector
        images.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``cor_data``.
        """
        if self.is_empty(self.mask):
            raise ValueError('no mask in the container')
        if self.is_empty(self.std):
            raise ValueError('no std in the container')
        if self.is_empty(self.whitefield):
            raise ValueError('no whitefield in the container')

        xp = self.__array_namespace__()
        std = xp.clip(self.std, std_min, xp.inf)
        snr = xp.where(std, (self.data * self.mask - self.whitefield) / std, 0.0)
        return self.replace(snr=snr)

    def update_metadata(self, method: MDMethod='robust-mean-scale', frames: Indices | None=None,
                        r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0,
                        num_threads: int=1) -> 'CrystData':
        if method == 'median-poisson':
            data = self.update_whitefield('median', frames, num_threads=num_threads)
            return data.update_std('poisson', frames)

        if method == 'robust-mean-scale':
            xp = self.__array_namespace__()
            if frames is None:
                frames = xp.arange(self.num_frames)

            whitefield, std = robust_mean(inp=self.data[frames] * self.mask, axis=0,
                                          r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                          return_std=True, num_threads=num_threads)
            return self.replace(whitefield=xp.asarray(whitefield), std=xp.asarray(std))

        if method == 'robust-mean-poisson':
            data = self.update_whitefield('robust-mean', frames, r0, r1, n_iter, lm, num_threads)
            return data.update_std('poisson')

        raise ValueError(f"Invalid method argument: {method}")

    def update_std(self, method: STDMethod='robust-scale', frames: Indices | None=None,
                   r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0,
                   num_threads: int=1) -> 'CrystData':
        xp = self.__array_namespace__()
        if frames is None:
            frames = xp.arange(self.num_frames)

        if method == 'robust-scale':
            if self.is_empty(self.data):
                raise ValueError('no data in the container')
            if self.is_empty(self.mask):
                raise ValueError('no mask in the container')

            _, std = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0, r1=r1,
                                 n_iter=n_iter, lm=lm, return_std=True,
                                 num_threads=num_threads)
            std = xp.asarray(std)
        elif method == 'poisson':
            if self.is_empty(self.whitefield):
                raise ValueError('no whitefield in the container')

            whitefields = xp.reshape(self.whitefield, (self.num_whitefields,) + self.frame_shape)
            std = xp.sqrt(xp.mean(whitefields, axis=0))
        else:
            raise ValueError(f"Invalid method argument: {method}")

        return self.replace(std=std)

    def update_whitefield(self, method: WFMethod='median', frames: Indices | None=None,
                          r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0,
                          num_threads: int=1) -> 'CrystData':
        """Return a new :class:`CrystData` object with new whitefield.

        Args:
            method : Choose method for white-field generation. The following keyword
                values are allowed:

                * 'median' : Taking a median through the stack of frames.
                * 'robust-mean' : Finding a robust mean through the stack of frames.

            frames : List of frames to use for the white-field estimation.
            r0 : A lower bound guess of ratio of inliers. We'd like to make a sample
                out of worst inliers from data points that are between `r0` and `r1`
                of sorted residuals.
            r1 : An upper bound guess of ratio of inliers. Choose the `r0` to be as
                high as you are sure the ratio of data is inlier.
            n_iter : Number of iterations of fitting a gaussian with the FLkOS
                algorithm.
            lm : How far (normalized by STD of the Gaussian) from the mean of the
                Gaussian, data is considered inlier.

        Raises:
            ValueError : If there is no ``data`` inside the container.
            ValueError : If ``method`` keyword is invalid.

        Returns:
            New :class:`CrystData` object with the updated ``whitefield``.
        """
        if self.is_empty(self.data):
            raise ValueError('no data in the container')
        if self.is_empty(self.mask):
            raise ValueError('no mask in the container')

        if frames is None:
            xp = self.__array_namespace__()
            frames = xp.arange(self.num_frames)

        if method == 'median':
            whitefield = median(inp=self.data[frames] * self.mask, axis=0,
                                num_threads=num_threads)
        elif method == 'robust-mean':
            whitefield = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0,
                                     r1=r1, n_iter=n_iter, lm=lm,
                                     num_threads=num_threads)
        else:
            raise ValueError('Invalid method argument')

        return self.replace(whitefield=whitefield, protocol=self.default_protocol())

D = TypeVar("D", bound="DetectorBase")

class DetectorBase(DataContainer):
    data            : RealArray
    mask            : BoolArray
    parent          : ReferenceType[CrystData]

    @property
    def shape(self) -> Shape:
        return self.data.shape

    def __getitem__(self: D, idxs: Indices) -> D:
        return self.replace(data=self.data[idxs], mask=self.mask[idxs])

    def clip(self: D, vmin: ArrayLike, vmax: ArrayLike) -> D:
        xp = self.__array_namespace__()
        return self.replace(data=xp.clip(self.data, vmin, vmax))

    def export_table(self, streaks: StackedStreaks | Streaks, width: float,
                     kernel: str='rectangular') -> pd.DataFrame:
        """Export normalised pattern into a :class:`pandas.DataFrame` table.

        Args:
            streaks : A set of diffraction streaks.
            width : Width of diffraction streaks in pixels.
            kernel : Choose one of the supported kernel functions [Krn]_. The following
                kernels are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

        Raises:
            ValueError : If there is no ``streaks`` inside the container.

        Returns:
            List of :class:`pandas.DataFrame` tables for each frame in ``frames`` if
            ``concatenate`` is False, a single :class:`pandas.DataFrame` otherwise. Table
            contains the following information:

            * `frames` : Frame index.
            * `x`, `y` : Pixel coordinates.
            * `snr` : Signal-to-noise values.
            * `rp` : Reflection profiles.
            * `I_raw` : Measured intensity.
            * `bgd` : Background values.
        """
        if self.parent() is None:
            raise ValueError('Invalid parent: the parent data container was deleted')

        table = streaks.pattern_dataframe(width, shape=self.parent().shape, kernel=kernel)
        if self.parent().num_modules > 1:
            index = table['frames'] * self.parent().num_modules + table['module_id']
        else:
            index = table['frames']

        indices, x, y = index.to_numpy(), table['x'].to_numpy(), table['y'].to_numpy()
        bgd = self.parent().whitefield.reshape((-1,)  + self.parent().frame_shape[-2:])
        data = self.parent().data.reshape((-1,)  + self.parent().frame_shape[-2:])
        snr = self.parent().snr.reshape((-1,)  + self.parent().frame_shape[-2:])
        table = {'bgd': bgd[indices % bgd.shape[0], y, x], 'I_raw': data[indices, y, x],
                 'frames': self.parent().frames[indices // self.parent().num_modules],
                 'snr': snr[indices, y, x], 'x': x, 'y': y}

        if self.parent().num_modules > 1:
            table['module_id'] = indices % self.parent().num_modules
            columns = ['frames', 'module_id', 'x', 'y', 'snr', 'I_raw', 'bgd']
        else:
            columns = ['frames', 'x', 'y', 'snr', 'I_raw', 'bgd']

        return pd.DataFrame(table).loc[:, columns]

@dataclass
class StreakDetector(DetectorBase):
    data            : RealArray
    mask            : BoolArray
    structure       : Structure2D
    parent          : ReferenceType[CrystData]

    def detect_peaks(self, vmin: float, npts: int, connectivity: Structure2D=Structure2D(1, 1),
                     rank: int | None=None, num_threads: int=1) -> PeaksList:
        """Find peaks in a pattern. Returns a sparse set of peaks which values are above a threshold
        ``vmin`` that have a supporing set of a size larger than ``npts``. The minimal distance
        between peaks is ``2 * structure.radius``.

        Args:
            vmin : Peak threshold. All peaks with values lower than ``vmin`` are discarded.
            npts : Support size threshold. The support structure is a connected set of pixels which
                value is above the threshold ``vmin``. A peak is discarded is the size of support
                set is lower than ``npts``.
            connectivity : Connectivity structure used in finding a supporting set.

        Returns:
            Set of detected peaks.
        """
        if rank is None:
            rank = self.structure.rank
        peaks = detect_peaks(self.data, self.mask, rank, vmin, num_threads=num_threads)
        filter_peaks(peaks, self.data, self.mask, connectivity, vmin, npts, num_threads=num_threads)
        return peaks

    def detect_streaks(self, peaks: PeaksList, xtol: float, vmin: float, min_size: int,
                       lookahead: int=0, nfa: int=0, num_threads: int=1
                       ) -> StackedStreaks | Streaks:
        """Streak finding algorithm. Starting from the set of seed peaks, the lines are iteratively
        extended with a connectivity structure.

        Args:
            peaks : A set of peaks used as seed locations for the streak growing algorithm.
            xtol : Distance threshold. A new linelet is added to a streak if it's distance to the
                streak is no more than ``xtol``.
            vmin : Value threshold. A new linelet is added to a streak if it's value at the center
                of mass is above ``vmin``.
            min_size : Minimum number of linelets required in a detected streak.
            lookahead : Number of linelets considered at the ends of a streak to be added to the
                streak.

        Returns:
            A list of detected streaks.
        """
        detected = detect_streaks(peaks, self.data, self.mask, self.structure, xtol, vmin, min_size,
                                  lookahead, nfa, num_threads=num_threads)
        if self.parent().num_modules > 1:
            return StackedStreaks(detected.index() // self.parent().num_modules,
                                  detected.index() % self.parent().num_modules,
                                  detected.to_lines(), self.parent().num_modules)
        return Streaks(detected.index(), detected.to_lines())

@dataclass
class RegionDetector(DetectorBase):
    data            : RealArray
    mask            : BoolArray
    structure       : Structure2D
    parent          : ReferenceType[CrystData]

    def detect_regions(self, vmin: float, npts: int, num_threads: int=1) -> RegionsList2D:
        return label((self.data > vmin) & self.mask, structure=self.structure, npts=npts,
                     num_threads=num_threads)

    def detect_streaks(self, regions: RegionsList2D) -> StackedStreaks | Streaks:
        lines = self.line_fit(regions)
        if self.parent().num_modules > 1:
            return StackedStreaks(regions.frames() // self.parent().num_modules,
                                  regions.frames() % self.parent().num_modules,
                                  lines, self.parent().num_modules)
        return Streaks(regions.frames(), lines)

    def ellipse_fit(self, regions: RegionsList2D) -> RealArray:
        return ellipse_fit(regions, self.data)

    def line_fit(self, regions: RegionsList2D) -> RealArray:
        return line_fit(regions, self.data)

    def total_mass(self, regions: RegionsList2D) -> RealArray:
        return total_mass(regions, self.data)

    def mean(self, regions: RegionsList2D) -> RealArray:
        return mean(regions, self.data)

    def center_of_mass(self, regions: RegionsList2D) -> RealArray:
        return center_of_mass(regions, self.data)

    def moment_of_inertia(self, regions: RegionsList2D) -> RealArray:
        return moment_of_inertia(regions, self.data)

    def covariance_matrix(self, regions: RegionsList2D) -> RealArray:
        return covariance_matrix(regions, self.data)
