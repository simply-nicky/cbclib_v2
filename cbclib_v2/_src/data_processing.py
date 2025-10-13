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
from typing import ClassVar, List, Literal, Tuple, TypeVar, cast
from dataclasses import dataclass, field
from weakref import ref
import numpy as np
import pandas as pd
from .cxi_protocol import CXIProtocol, Kinds
from .data_container import DataContainer, IndexArray
from .streak_finder import PeaksList, StreakList, detect_peaks, detect_streaks, filter_peaks
from .streaks import Streaks
from .annotations import (ArrayLike, BoolArray, Indices, IntArray, RealArray, RealSequence,
                          ReferenceType, ROI, Shape)
from .label import (label, ellipse_fit, total_mass, mean, center_of_mass, moment_of_inertia,
                    covariance_matrix, Regions2D, Structure2D)
from .src.signal_proc import binterpolate, kr_grid
from .src.median import median, robust_mean, robust_lsq

MaskMethod = Literal['all-bad', 'no-bad', 'range', 'snr']
MDMethod = Literal['median-poisson', 'robust-mean-scale', 'robust-mean-poisson']
STDMethod = Literal['poisson', 'robust-scale']
WFMethod = Literal['median', 'robust-mean', 'robust-mean-scale']

DATA_PROTOCOL = os.path.join(os.path.dirname(__file__), 'config/cryst_data.ini')
METADATA_PROTOCOL = os.path.join(os.path.dirname(__file__), 'config/cryst_metadata.ini')

@dataclass
class CrystMetadata(DataContainer):
    eigen_field : RealArray = field(default_factory=lambda: np.array([]))
    eigen_value : RealArray = field(default_factory=lambda: np.array([]))
    frames      : IntArray  = field(default_factory=lambda: np.array([], dtype=int))
    data_frames : IntArray  = field(default_factory=lambda: np.array([], dtype=int))
    mask        : BoolArray = field(default_factory=lambda: np.array([], dtype=bool))
    std         : RealArray = field(default_factory=lambda: np.array([]))
    whitefield  : RealArray = field(default_factory=lambda: np.array([]))

    protocol    : ClassVar[CXIProtocol] = CXIProtocol.read(METADATA_PROTOCOL)

    def __post_init__(self):
        super().__post_init__()
        if not self.is_empty(self.mask):
            if not self.is_empty(self.std):
                self.std *= self.mask
            if not self.is_empty(self.whitefield):
                self.whitefield *= self.mask

    @property
    def center_frames(self) -> IntArray:
        if not self.is_empty(self.data_frames):
            return np.asarray(median(self.data_frames, axis=-1), dtype=int)
        return np.array([], dtype=int)

    @property
    def eigen_fields(self) -> RealArray:
        xp = self.__array_namespace__()
        num_frames = self.eigen_field.size // prod(self.eigen_field.shape[-2:])
        return xp.reshape(self.eigen_field, (num_frames,) + self.eigen_field.shape[-2:])

    @property
    def whitefields(self) -> RealArray:
        xp = self.__array_namespace__()
        num_frames = self.whitefield.size // prod(self.whitefield.shape[-2:])
        return xp.reshape(self.whitefield, (num_frames,) + self.whitefield.shape[-2:])

    def crop(self, roi: ROI) -> 'CrystMetadata':
        cropped = {}
        for attr, data in self.contents().items():
            if self.protocol.get_kind(attr) in (Kinds.frame, Kinds.stack):
                cropped[attr] = data[..., roi[0]:roi[1], roi[2]:roi[3]]
        return self.replace(**cropped)

    def import_data(self, data: RealArray, frames: IntArray | int=np.array([], dtype=int),
                    whitefield: RealArray=np.array([])) -> 'CrystData':
        xp = self.__array_namespace__()
        if not whitefield.size:
            whitefield = xp.mean(self.whitefields, axis=0)
        return CrystData(data=data, frames=xp.atleast_1d(frames), mask=self.mask,
                         std=self.std, whitefield=whitefield)

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
        if self.whitefields.shape[0] == 1:
            raise ValueError('A stack of several whitefields is needed to perform PCA')

        xp = self.__array_namespace__()
        whitefield = self.whitefields
        mat_svd = xp.tensordot(whitefield, whitefield, axes=((1, 2), (1, 2)))
        eig_vals, eig_vecs = xp.linalg.eig(mat_svd)
        effs = xp.tensordot(eig_vecs, whitefield, axes=((0,), (0,)))
        return self.replace(eigen_field=effs, eigen_value=eig_vals / eig_vals.sum())

    def projection(self, data: RealArray, num_fields: int, method: str="robust-lsq",
                   r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0,
                   num_threads: int=1) -> RealArray:
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
        if self.is_empty(self.eigen_field):
            raise ValueError('No eigen_field in the container')

        xp = self.__array_namespace__()
        y: RealArray = xp.reshape(data, (data.size // prod(self.eigen_field.shape[-2:]), -1))
        W: RealArray = xp.reshape(self.eigen_fields[:num_fields], (num_fields, -1))

        if method == "robust-lsq":
            return robust_lsq(W=W, y=y, axis=1, r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                              num_threads=num_threads)

        if method == "lsq":
            return xp.mean(y[..., None, :] * W, axis=-1) / xp.mean(W * W, axis=-1)

        raise ValueError(f"Invalid method argument: {method}")

    def project(self, projection: RealArray) -> RealArray:
        if self.is_empty(self.eigen_field):
            raise ValueError('No eigen_field in the container')

        xp = self.__array_namespace__()
        return xp.tensordot(projection, self.eigen_fields[:projection.shape[-1]], ((-1,), (0,)))

@dataclass
class CrystData(DataContainer):
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

    protocol    : ClassVar[CXIProtocol] = CXIProtocol.read(DATA_PROTOCOL)

    def __post_init__(self):
        super().__post_init__()
        xp = self.__array_namespace__()
        if self.frames.size != self.shape[0]:
            self.frames = xp.arange(self.shape[0])
        if self.mask.shape != self.shape[1:]:
            self.mask = xp.ones(self.shape[1:], dtype=bool)

    @property
    def shape(self) -> Shape:
        shape = [0, 0, 0]
        for attr, data in self.contents().items():
            if self.protocol.get_kind(attr) == Kinds.sequence:
                shape[0] = data.shape[0]
                break

        for attr, data in self.contents().items():
            if self.protocol.get_kind(attr) == Kinds.frame:
                shape[1:] = data.shape[-2:]
                break

        for attr, data in self.contents().items():
            if self.protocol.get_kind(attr) == Kinds.stack:
                shape[0] = prod(data.shape[:-2])
                shape[1:] = data.shape[-2:]
                break

        return tuple(shape)

    def apply_mask(self) -> 'CrystData':
        attributes = {}
        if not self.is_empty(self.whitefield):
            attributes['whitefield'] = self.whitefield * self.mask
        if not self.is_empty(self.std):
            attributes['std'] = self.std * self.mask
        if not self.is_empty(self.snr):
            attributes['snr'] = self.snr * self.mask
        return self.replace(**attributes)

    def crop(self, roi: ROI) -> 'CrystData':
        cropped = {}
        for attr, data in self.contents().items():
            if self.protocol.get_kind(attr) in (Kinds.frame, Kinds.stack):
                cropped[attr] = data[..., roi[0]:roi[1], roi[2]:roi[3]]
        return self.replace(**cropped)

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
        if mask.shape != self.shape[1:]:
            raise ValueError('mask and data have incompatible shapes: '\
                             f'{mask.shape} != {self.shape[1:]}')

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

    def region_detector(self, structure: Structure2D):
        if self.is_empty(self.mask):
            raise ValueError('no mask in the container')
        if self.is_empty(self.snr):
            raise ValueError('no snr in the container')

        xp = self.__array_namespace__()
        parent = cast(ReferenceType[CrystData], ref(self))
        idxs = xp.arange(self.shape[0])
        return RegionDetector(indices=idxs, data=self.snr, mask=self.mask, structure=structure,
                              parent=parent)

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

        xp = self.__array_namespace__()
        parent = cast(ReferenceType[CrystData], ref(self))
        idxs = xp.arange(self.shape[0])
        return StreakDetector(indices=idxs, data=self.snr, mask=self.mask, structure=structure,
                              parent=parent)

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
            roi = (0, self.shape[1], 0, self.shape[2])

        data = (self.data * self.mask)[:, roi[0]:roi[1], roi[2]:roi[3]]

        if method == 'all-bad':
            mask = xp.zeros(self.shape[1:], dtype=bool)
        elif method == 'no-bad':
            mask = xp.ones(self.shape[1:], dtype=bool)
        elif method == 'range':
            mask = xp.all((data >= vmin) & (data < vmax), axis=0)
        elif method == 'snr':
            if self.snr is None:
                raise ValueError('No snr in the container')

            snr = self.snr[:, roi[0]:roi[1], roi[2]:roi[3]]
            mask = xp.mean(xp.abs(snr), axis=0) < snr_max
        else:
            raise ValueError(f'Invalid method argument: {method:s}')

        new_mask = xp.copy(self.mask)
        new_mask[roi[0]:roi[1], roi[2]:roi[3]] &= mask
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
                frames = xp.arange(self.shape[0])

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
            frames = xp.arange(self.shape[0])

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

            num_whitefields = self.whitefield.size // prod(self.shape[-2:])
            whitefields = xp.reshape(self.whitefield, (num_whitefields,) + self.shape[-2:])
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

        xp = self.__array_namespace__()
        if frames is None:
            frames = xp.arange(self.shape[0])

        if method == 'median':
            whitefield = median(inp=self.data[frames] * self.mask, axis=0,
                                num_threads=num_threads)
            return self.replace(whitefield=xp.asarray(whitefield))
        if method == 'robust-mean':
            whitefield = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0,
                                     r1=r1, n_iter=n_iter, lm=lm,
                                     num_threads=num_threads)
            return self.replace(whitefield=xp.asarray(whitefield))

        raise ValueError('Invalid method argument')

class ScaleTransform(DataContainer):
    scale   : float

    def interpolate(self, data: RealArray) -> RealArray:
        xp = self.__array_namespace__()
        x, y = xp.arange(0, data.shape[-1]), xp.arange(0, data.shape[-2])

        xx = self.scale * xp.arange(0, data.shape[-1] / self.scale)
        yy = self.scale * xp.arange(0, data.shape[-2] / self.scale)
        pts = xp.stack(xp.meshgrid(xx, yy), axis=-1)
        return xp.asarray(binterpolate(data, (x, y), pts, (-1, -2)))

    def kernel_regression(self, data: RealArray, sigma: float, num_threads: int=1) -> RealArray:
        xp = self.__array_namespace__()
        x, y = xp.arange(0, data.shape[-1]), xp.arange(0, data.shape[-2])
        pts = xp.stack(xp.meshgrid(x, y), axis=-1)

        xx = self.scale * xp.arange(0, data.shape[-1] / self.scale)
        yy = self.scale * xp.arange(0, data.shape[-2] / self.scale)
        return xp.asarray(kr_grid(data, pts, (xx, yy), sigma=sigma, num_threads=num_threads)[0])

    def to_detector(self, x: RealSequence, y: RealSequence) -> Tuple[RealArray, RealArray]:
        xp = self.__array_namespace__()
        return self.scale * xp.asarray(x), self.scale * xp.asarray(y)

    def to_scaled(self, x: RealSequence, y: RealSequence) -> Tuple[RealArray, RealArray]:
        xp = self.__array_namespace__()
        return xp.asarray(x) / self.scale, xp.asarray(y) / self.scale

DetBase = TypeVar("DetBase", bound="DetectorBase")

class DetectorBase(ScaleTransform):
    indices         : IntArray
    data            : RealArray
    mask            : BoolArray
    scale           : float
    parent          : ReferenceType[CrystData]

    @property
    def shape(self) -> Shape:
        return self.data.shape

    def __getitem__(self: DetBase, idxs: Indices) -> DetBase:
        return self.replace(data=self.data[idxs], mask=self.mask[idxs], indices=self.indices[idxs])

    def clip(self: DetBase, vmin: ArrayLike, vmax: ArrayLike) -> DetBase:
        xp = self.__array_namespace__()
        return self.replace(data=xp.clip(self.data, vmin, vmax))

    def export_coordinates(self, indices: IntArray, y: IntArray, x: IntArray) -> pd.DataFrame:
        table = {'bgd': self.parent().whitefield[y, x], 'frames': self.parent().frames[indices],
                 'snr': self.parent().snr[indices, y, x], 'I_raw': self.parent().data[indices, y, x],
                 'x': x, 'y': y}
        return pd.DataFrame(table)

    def downscale(self: DetBase, scale: float, sigma: float, num_threads: int=1) -> DetBase:
        xp = self.__array_namespace__()
        data = self.kernel_regression(self.data, sigma, num_threads)
        mask = self.interpolate(xp.asarray(self.mask, dtype=float))
        return self.replace(data=data, mask=xp.asarray(mask, dtype=bool),
                            scale=scale)

    def to_detector(self, streaks: Streaks) -> Streaks:
        xp = self.__array_namespace__()
        pts = xp.stack(super().to_detector(streaks.x, streaks.y), axis=-1)
        return Streaks(streaks.index, xp.reshape(pts, pts.shape[:-2] + (4,)))

@dataclass
class StreakDetector(DetectorBase):
    indices         : IntArray
    data            : RealArray
    mask            : BoolArray
    structure       : Structure2D
    parent          : ReferenceType[CrystData]
    scale           : float = 1.0

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
        peaks = detect_peaks(self.data, self.mask, rank, vmin,
                             num_threads=num_threads)
        filter_peaks(peaks, self.data, self.mask, connectivity, vmin, npts,
                     num_threads=num_threads)
        return peaks

    def detect_streaks(self, peaks: PeaksList, xtol: float, vmin: float, min_size: int,
                       lookahead: int=0, nfa: int=0, num_threads: int=1
                       ) -> List[StreakList]:
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
        return detect_streaks(peaks, self.data, self.mask, self.structure, xtol, vmin, min_size,
                              lookahead, nfa, num_threads=num_threads)

    def to_streaks(self, result: List[StreakList]) -> Streaks:
        xp = self.__array_namespace__()
        streaks = [xp.asarray(pattern.to_lines()) for pattern in result]
        idxs = xp.concatenate([xp.full((len(pattern),), idx)
                                for idx, pattern in zip(self.indices, streaks)])
        lines = xp.concatenate(streaks)
        return Streaks(index=IndexArray(idxs), lines=lines)

    def export_table(self, streaks: Streaks, width: float,
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
        table2 = self.export_coordinates(table['frames'].to_numpy(),
                                         table['y'].to_numpy(), table['x'].to_numpy())
        columns = set(table.columns) - {'frames', 'y', 'x'}
        return table2.assign(**{key: table[key] for key in columns})

@dataclass
class RegionDetector(DetectorBase):
    indices         : IntArray
    data            : RealArray
    mask            : BoolArray
    structure       : Structure2D
    parent          : ReferenceType[CrystData]
    scale           : float = 1.0

    def detect_regions(self, vmin: float, npts: int, num_threads: int=1) -> List[Regions2D]:
        regions = label((self.data > vmin) & self.mask, structure=self.structure, npts=npts,
                        num_threads=num_threads)
        if isinstance(regions, Regions2D):
            return [regions,]
        return regions

    def export_table(self, regions: List[Regions2D]) -> pd.DataFrame:
        xp = self.__array_namespace__()
        frames, y, x = [], [], []
        for frame, pattern in zip(self.indices, regions):
            size = sum(len(region.x) for region in pattern)
            frames.extend(size * [frame,])
            y.extend(pattern.y)
            x.extend(pattern.x)
        return self.export_coordinates(xp.array(frames), xp.array(y), xp.array(x))

    def ellipse_fit(self, regions: List[Regions2D]) -> List[RealArray]:
        return ellipse_fit(regions, self.data)

    def total_mass(self, regions: List[Regions2D]) -> List[RealArray]:
        xp = self.__array_namespace__()
        arrays = total_mass(regions, self.data)
        return [xp.asarray(array) for array in arrays]

    def mean(self, regions: List[Regions2D]) -> List[RealArray]:
        xp = self.__array_namespace__()
        arrays = mean(regions, self.data)
        return [xp.asarray(array) for array in arrays]

    def center_of_mass(self, regions: List[Regions2D]) -> List[RealArray]:
        xp = self.__array_namespace__()
        arrays = center_of_mass(regions, self.data)
        return [xp.asarray(array) for array in arrays]

    def moment_of_inertia(self, regions: List[Regions2D]) -> List[RealArray]:
        xp = self.__array_namespace__()
        arrays = moment_of_inertia(regions, self.data)
        return [xp.asarray(array) for array in arrays]

    def covariance_matrix(self, regions: List[Regions2D]) -> List[RealArray]:
        xp = self.__array_namespace__()
        arrays = covariance_matrix(regions, self.data)
        return [xp.asarray(array) for array in arrays]
