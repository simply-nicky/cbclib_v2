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
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast
from dataclasses import dataclass, field
from weakref import ref
import numpy as np
import pandas as pd
from .cxi_protocol import CrystProtocol, FileStore, Kinds
from .data_container import StringFormatting, DataContainer, Transform
from .streak_finder import PatternsStreakFinder, Peaks
from .streaks import Streaks
from .annotations import (Indices, IntSequence, NDArrayLike, NDBoolArray, NDIntArray, NDRealArray,
                          RealSequence, ReferenceType, ROI, Shape)
from .src.label import (label, center_of_mass, central_moments, gauss_fit, ellipse_fit, line_fit,
                        moments, Regions, Structure)
from .src.signal_proc import binterpolate, kr_grid
from .src.median import median, robust_mean, robust_lsq

AnyCryst = Union['CrystData', 'CrystDataPart', 'CrystDataFull']

def from_dict(**data: Any) -> AnyCryst:
    if 'data' in data:
        if 'whitefield' in data and 'snr' in data:
            return CrystDataFull(**data)
        return CrystDataPart(**data)
    return CrystData(**data)

def read_hdf(input_file: FileStore, attributes: Optional[Union[str, List[str]]]=None,
             idxs: Optional[IntSequence]=None, transform: Optional[Transform]=None,
             processes: int=1, verbose: bool=True) -> AnyCryst:
    """Load data attributes from the input files in `files` file handler object.

    Args:
        attributes : List of attributes to load. Loads all the data attributes contained in
            the file(s) by default.
        idxs : List of frame indices to load.
        processes : Number of parallel workers used during the loading.
        verbose : Set the verbosity of the loading process.

    Raises:
        ValueError : If attribute is not existing in the input file(s).
        ValueError : If attribute is invalid.

    Returns:
        New :class:`CrystData` object with the attributes loaded.
    """
    if input_file.is_empty():
        input_file.update()
    shape = input_file.read_frame_shape()

    if attributes is None:
        attributes = input_file.attributes()
    attributes = StringFormatting.str_to_list(attributes)

    if idxs is None:
        idxs = np.arange(input_file.size)
    idxs = np.atleast_1d(idxs)

    if CrystProtocol.has_kind(attributes, Kinds.STACK):
        data_dict: Dict[str, Any] = {'frames': idxs}
    else:
        data_dict: Dict[str, Any] = {}

    for attr in attributes:
        if attr not in input_file.attributes():
            raise ValueError(f"No '{attr}' attribute in the input files")

        if transform and shape[0] * shape[1]:
            ss_idxs, fs_idxs = np.indices(shape)
            ss_idxs, fs_idxs = transform.index_array(ss_idxs, fs_idxs)
            data = input_file.load(attr, idxs=idxs, ss_idxs=ss_idxs,
                                   fs_idxs=fs_idxs, processes=processes,
                                   verbose=verbose)
        else:
            data = input_file.load(attr, idxs=idxs, processes=processes,
                                   verbose=verbose)

        data_dict[attr] = data

    return from_dict(**data_dict)

def write_hdf(container: AnyCryst, output_file: FileStore,
              attributes: Union[str, List[str], None]=None, good_frames: Optional[Indices]=None,
              transform: Optional[Transform] = None, input_file: Optional[FileStore]=None,
              mode: str='overwrite', idxs: Optional[Indices]=None):
    """Save data arrays of the data attributes contained in the container to an output file.

    Args:
        attributes : List of attributes to save. Saves all the data attributes contained in
            the container by default.
        apply_transform : Apply `transform` to the data arrays if True.
        mode : Writing modes. The following keyword values are allowed:

            * `append` : Append the data array to already existing dataset.
            * `insert` : Insert the data under the given indices `idxs`.
            * `overwrite` : Overwrite the existing dataset.

        idxs : Indices where the data is saved. Used only if ``mode`` is set to 'insert'.

    Raises:
        ValueError : If the ``output_file`` is not defined inside the container.
    """
    if transform:
        if input_file is None:
            raise ValueError("'input_file' is not defined inside the container")

        shape = input_file.read_frame_shape()

    if attributes is None:
        attributes = list(container.contents())

    if good_frames is None:
        good_frames = np.arange(container.shape[0])

    for attr in StringFormatting.str_to_list(attributes):
        data = np.asarray(getattr(container, attr))
        if data is not None:
            kind = container.get_kind(attr)

            if kind in (Kinds.STACK, Kinds.SEQUENCE):
                data = data[good_frames]

            if transform:
                if kind in (Kinds.STACK, Kinds.FRAME):
                    out = np.zeros(data.shape[:-2] + shape, dtype=data.dtype)
                    data = transform.backward(data, out)

            output_file.save(attr, data, mode=mode, idxs=idxs)

@dataclass
class CrystDataBase(CrystProtocol, DataContainer):
    """Convergent beam crystallography data container class. Takes a :class:`cbclib_v2.CXIStore` file
    handler. Provides an interface to work with the detector images and detect the diffraction
    streaks. Also provides an interface to load from a file and save to a file any of the data
    attributes. The data frames can be tranformed using any of the :class:`cbclib_v2.Transform`
    classes.

    Args:
        input_file : Input file :class:`cbclib_v2.CXIStore` file handler.
        transform : An image transform object.
        num_threads : Number of threads used in the calculations.
        output_file : On output file :class:`cbclib_v2.CXIStore` file handler.
        data : Detector raw data.
        mask : Bad pixels mask.
        frames : List of frame indices inside the container.
        whitefield : Measured frames' white-field.
        snr : Signal-to-noise ratio.
        whitefields : A set of white-fields generated for each pattern separately.
    """
    data        : Optional[NDRealArray]

    whitefield  : Optional[NDRealArray]
    snr         : Optional[NDRealArray]

    frames      : Optional[NDIntArray] = field(default=None)
    mask        : Optional[NDBoolArray] = field(default=None)
    std         : Optional[NDRealArray] = field(default=None)
    scales      : Optional[NDRealArray] = field(default=None)
    num_threads : int = field(default=cpu_count())

    @property
    def shape(self) -> Shape:
        shape = [0, 0, 0]
        for attr, data in self.contents().items():
            if data is not None and self.get_kind(attr) == Kinds.SEQUENCE:
                shape[0] = data.shape[0]
                break

        for attr, data in self.contents().items():
            if data is not None and self.get_kind(attr) == Kinds.FRAME:
                shape[1:] = data.shape
                break

        for attr, data in self.contents().items():
            if data is not None and self.get_kind(attr) == Kinds.STACK:
                shape[0] = np.prod(data.shape[:-2])
                shape[1:] = data.shape[-2:]
                break

        return tuple(shape)

    def select_frames(self, idxs: Optional[Indices]=None):
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
            if self.get_kind(attr) in (Kinds.SEQUENCE, Kinds.STACK):
                data_dict[attr] = getattr(self, attr)[idxs]
            else:
                data_dict[attr] = getattr(self, attr)
        return self.replace(**data_dict)

Cryst = TypeVar("Cryst", bound="CrystDataPartBase")

@dataclass
class CrystDataPartBase(CrystDataBase):
    data        : NDRealArray
    frames      : NDIntArray = field(default_factory=lambda: np.array([], dtype=int))
    mask        : NDBoolArray = field(default_factory=lambda: np.array([], dtype=bool))
    scales      : NDRealArray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if self.frames.size != self.shape[0]:
            self.frames = np.arange(self.shape[0])
        if self.mask.shape != self.shape[1:]:
            self.mask = np.ones(self.shape[1:], dtype=bool)
        if self.scales.shape != (self.shape[0],):
            self.scales = np.ones(self.shape[0])

    def mask_data(self: Cryst) -> Cryst:
        attributes = {}
        if self.whitefield is not None:
            attributes['whitefield'] = self.whitefield * self.mask
        if self.std is not None:
            attributes['std'] = self.std * self.mask
        if self.snr is not None:
            attributes['snr'] = self.snr * self.mask
        return self.replace(**attributes)

    def mask_region(self: Cryst, roi: ROI) -> Cryst:
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
        mask = np.copy(self.mask)
        mask[roi[0]:roi[1], roi[2]:roi[3]] = False
        return self.replace(mask=mask).mask_data()

    def import_mask(self: Cryst, mask: NDBoolArray, update: str='reset') -> Cryst:
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
        if mask.shape != self.shape[1:]:
            raise ValueError('mask and data have incompatible shapes: '\
                             f'{mask.shape:s} != {self.shape[1:]:s}')

        if update == 'reset':
            return self.replace(mask=mask)
        if update == 'multiply':
            return self.replace(mask=mask * self.mask)

        raise ValueError(f'Invalid update keyword: {update:s}')

    def reset_mask(self: Cryst) -> Cryst:
        """Reset bad pixel mask. Every pixel is assumed to be good by default.

        Raises:
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the default ``mask``.
        """
        return self.replace(mask=np.array([], dtype=bool))

    def update_mask(self: Cryst, method: str='no-bad', vmin: int=0, vmax: int=65535,
                    snr_max: float=3.0, roi: Optional[ROI]=None) -> Cryst:
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
        if vmin >= vmax:
            raise ValueError('vmin must be less than vmax')
        if roi is None:
            roi = (0, self.shape[1], 0, self.shape[2])

        data = (self.data * self.mask)[:, roi[0]:roi[1], roi[2]:roi[3]]

        if method == 'all-bad':
            mask = np.zeros(self.shape[1:], dtype=bool)
        elif method == 'no-bad':
            mask = np.ones(self.shape[1:], dtype=bool)
        elif method == 'range':
            mask = np.all((data >= vmin) & (data < vmax), axis=0)
        elif method == 'snr':
            if self.snr is None:
                raise ValueError('No snr in the container')

            snr = self.snr[:, roi[0]:roi[1], roi[2]:roi[3]]
            mask = np.mean(np.abs(snr), axis=0) < snr_max
        else:
            raise ValueError(f'Invalid method argument: {method:s}')

        new_mask = np.copy(self.mask)
        new_mask[roi[0]:roi[1], roi[2]:roi[3]] &= mask
        return self.replace(mask=new_mask)

    def update_whitefield(self: Cryst, method: str='median', frames: Optional[Indices]=None,
                          r0: float=0.0, r1: float=0.5, n_iter: int=12,
                          lm: float=9.0) -> Cryst:
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
        if frames is None:
            frames = np.arange(self.shape[0])
        std = self.std

        if method == 'median':
            whitefield = median(inp=self.data[frames] * self.mask, axis=0,
                                num_threads=self.num_threads)
        elif method == 'robust-mean':
            whitefield = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0,
                                     r1=r1, n_iter=n_iter, lm=lm,
                                     num_threads=self.num_threads)
        elif method == 'robust-mean-scale':
            whitefield, std = robust_mean(inp=self.data[frames] * self.mask, axis=0,
                                          r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                          return_std=True, num_threads=self.num_threads)
        else:
            raise ValueError('Invalid method argument')

        return self.replace(whitefield=whitefield, std=std)

    def update_std(self: Cryst, method="robust-scale", frames: Optional[Indices]=None,
                   r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0) -> Cryst:
        if frames is None:
            frames = np.arange(self.shape[0])
        if self.whitefield is None:
            raise ValueError("No whitefield in the container")

        if method == "robust-scale":
            _, std = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0, r1=r1,
                                 n_iter=n_iter, lm=lm, return_std=True,
                                 num_threads=self.num_threads)
        elif method == "poisson":
            std = np.sqrt(self.whitefield)
        else:
            raise ValueError(f"Invalid method argument: {method}")

        return self.replace(std=std)

    def update_snr(self) -> 'CrystDataFull':
        """Return a new :class:`CrystData` object with new background corrected detector
        images.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``cor_data``.
        """
        if self.whitefield is None:
            raise ValueError("No whitefield in the container")
        if self.std is None:
            raise ValueError("No std in the container")

        whitefields = self.scales[:, None, None] * self.whitefield
        snr = np.where(self.std, (self.data * self.mask - whitefields) / self.std, 0.0)
        return CrystDataFull(**dict(self.to_dict(), snr=snr))

    def scale_whitefield(self: Cryst, method: str="robust-lsq", r0: float=0.0, r1: float=0.5,
                         n_iter: int=12, lm: float=9.0) -> Cryst:
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
        if self.std is None:
            raise ValueError("No std in the container")
        if self.whitefield is None:
            raise ValueError("No whitefield in the container")

        mask = self.mask & (self.std > 0.0)
        y: NDRealArray = np.where(mask, self.data / self.std, 0.0)[:, mask]
        W: NDRealArray = np.where(mask, self.whitefield / self.std, 0.0)[None, mask]

        if method == "robust-lsq":
            scales = robust_lsq(W=W, y=y, axis=1, r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                num_threads=self.num_threads)
            return self.replace(scales=scales.ravel())

        if method == "median":
            scales = median(y * W, axis=1, num_threads=self.num_threads)[:, None] / \
                     median(W * W, axis=1, num_threads=self.num_threads)[:, None]
            return self.replace(scales=scales.ravel())

        raise ValueError(f"Invalid method argument: {method}")

@dataclass
class CrystData(CrystDataBase):
    data        : Optional[NDRealArray] = field(default=None)
    frames      : Optional[NDIntArray] = field(default=None)
    snr         : Optional[NDRealArray] = field(default=None)
    whitefield  : Optional[NDRealArray] = field(default=None)

@dataclass
class CrystDataPart(CrystDataPartBase):
    snr         : Optional[NDRealArray] = field(default=None)
    whitefield  : Optional[NDRealArray] = field(default=None)

@dataclass
class CrystDataFull(CrystDataPartBase):
    snr         : NDRealArray
    whitefield  : NDRealArray

    def streak_detector(self, structure: Structure) -> 'StreakDetector':
        """Return a new :class:`cbclib_v2.StreakDetector` object that detects lines in SNR frames.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.
            ValueError : If there is no ``snr`` inside the container.

        Returns:
            A CBC pattern detector based on :class:`cbclib_v2.bin.LSD` Line Segment Detection [LSD]_
            algorithm.
        """
        parent = cast(ReferenceType[CrystDataFull], ref(self))
        idxs = np.arange(self.shape[0])
        return StreakDetector(data=self.snr, mask=self.mask, parent=parent, indices=idxs,
                              structure=structure, transform=ScaleTransform(),
                              num_threads=self.num_threads)

    def region_detector(self, structure: Structure):
        parent = cast(ReferenceType[CrystDataFull], ref(self))
        idxs = np.arange(self.shape[0])
        return RegionDetector(data=self.snr, mask=self.mask, parent=parent, indices=idxs,
                              structure=structure, transform=ScaleTransform(),
                              num_threads=self.num_threads)

DetBase = TypeVar("DetBase", bound="DetectorBase")

@dataclass
class DetectorBase(DataContainer):
    indices         : NDIntArray
    data            : NDRealArray
    parent          : ReferenceType[CrystDataFull]

    @property
    def shape(self) -> Shape:
        return self.data.shape

    def get_frames(self: DetBase, idxs: Indices) -> DetBase:
        raise NotImplementedError

    def clip(self: DetBase, vmin: NDArrayLike, vmax: NDArrayLike) -> DetBase:
        return self.replace(data=np.clip(self.data, vmin, vmax))

    def export_coordinates(self, indices: NDIntArray, y: NDIntArray, x: NDIntArray) -> pd.DataFrame:
        table = {'bgd': self.parent().scales[indices] * self.parent().whitefield[y, x],
                 'frames': self.parent().frames[indices], 'snr': self.parent().snr[indices, y, x],
                 'I_raw': self.parent().data[indices, y, x], 'x': x, 'y': y}
        return pd.DataFrame(table)

MDet = TypeVar("MDet", bound="MaskedDetector")

@dataclass
class ScaleTransform():
    scale           : float = 1.0

    def interpolate(self, data: NDRealArray) -> NDRealArray:
        x, y = np.arange(0, data.shape[-1]), np.arange(0, data.shape[-2])

        xx = self.scale * np.arange(0, data.shape[-1] / self.scale)
        yy = self.scale * np.arange(0, data.shape[-2] / self.scale)
        pts = np.stack(np.meshgrid(xx, yy), axis=-1)
        return binterpolate(data, (x, y), pts)

    def kernel_regression(self, data: NDRealArray, sigma: float, num_threads: int=1
                          ) -> NDRealArray:
        x, y = np.arange(0, data.shape[-1]), np.arange(0, data.shape[-2])
        pts = np.stack(np.meshgrid(x, y), axis=-1)

        xx = self.scale * np.arange(0, data.shape[-1] / self.scale)
        yy = self.scale * np.arange(0, data.shape[-2] / self.scale)
        return kr_grid(data, pts, (xx, yy), sigma=sigma, num_threads=num_threads)[0]

    def to_detector(self, x: RealSequence, y: RealSequence) -> Tuple[NDRealArray, NDRealArray]:
        return self.scale * np.asarray(x), self.scale * np.asarray(y)

    def to_scaled(self, x: RealSequence, y: RealSequence) -> Tuple[NDRealArray, NDRealArray]:
        return np.asarray(x) / self.scale, np.asarray(y) / self.scale

@dataclass
class MaskedDetector(DetectorBase):
    mask            : NDBoolArray
    num_threads     : int
    transform       : ScaleTransform

    def downscale(self: MDet, scale: float, sigma: float) -> MDet:
        transform = ScaleTransform(scale)
        data = transform.kernel_regression(self.data, sigma, self.num_threads)
        mask = transform.interpolate(np.asarray(self.mask, dtype=float))
        return self.replace(data=data, mask=np.asarray(mask, dtype=bool),
                            transform=transform)

    def get_frames(self: MDet, idxs: Indices) -> MDet:
        return self.replace(data=self.data[idxs], mask=self.mask[idxs], indices=self.indices[idxs])

    def to_detector(self, streaks: Streaks) -> Streaks:
        pts = np.stack(self.transform.to_detector(streaks.x, streaks.y), axis=-1)
        return Streaks(streaks.index, np.reshape(pts, pts.shape[:-2] + (4,)))

@dataclass
class StreakDetector(MaskedDetector, PatternsStreakFinder):
    def detect_streaks(self, peaks: List[Peaks], xtol: float, vmin: float, min_size: int,
                       lookahead: int=0, nfa: int=0) -> Streaks:
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
        streaks = super().detect_streaks(peaks, xtol, vmin, min_size, lookahead, nfa)
        idxs = np.concatenate([np.full((len(pattern),), idx)
                               for idx, pattern in zip(self.indices, streaks)])
        lines = np.concatenate(streaks)
        return Streaks(index=idxs, lines=lines)

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
class RegionDetector(MaskedDetector):
    structure   : Structure

    def detect_regions(self, vmin: float, npts: int) -> List[Regions]:
        return label((self.data > vmin) & self.mask, self.structure, npts,
                     num_threads=self.num_threads)

    def export_table(self, regions: List[Regions]) -> pd.DataFrame:
        frames, y, x = [], [], []
        for frame, pattern in zip(self.indices, regions):
            size = sum(region.size for region in pattern)
            frames.extend(size * [frame,])
            y.extend(pattern.y)
            x.extend(pattern.x)
        return self.export_coordinates(np.array(frames), np.array(y), np.array(x))

    def center_of_mass(self, regions: List[Regions]) -> List[NDRealArray]:
        return center_of_mass(regions, self.data)

    def central_moments(self, regions: List[Regions]) -> List[NDRealArray]:
        return central_moments(regions, self.data)

    def gauss_fit(self, regions: List[Regions]) -> List[NDRealArray]:
        return gauss_fit(regions, self.data)

    def ellipse_fit(self, regions: List[Regions]) -> List[NDRealArray]:
        return ellipse_fit(regions, self.data)

    def line_fit(self, regions: List[Regions]) -> List[NDRealArray]:
        return line_fit(regions, self.data)

    def moments(self, regions: List[Regions]) -> List[NDRealArray]:
        return moments(regions, self.data)
