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
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, TypeVar, cast
from dataclasses import dataclass
from weakref import ref
import numpy as np
import pandas as pd
from .cxi_protocol import CXIProtocol, FileStore, Kinds
from .data_container import DataContainer
from .streak_finder import PatternsStreakFinder, Peaks
from .streaks import Streaks
from .annotations import (Indices, IntSequence, NDArrayLike, NDBoolArray, NDIntArray,
                          NDRealArray, RealSequence, ReferenceType, ROI, Shape)
from .label import (label, ellipse_fit, total_mass, mean, center_of_mass, moment_of_inertia,
                    covariance_matrix, Regions2D, Structure2D)
from .src.signal_proc import binterpolate, kr_grid
from .src.median import median, robust_mean, robust_lsq

WFMethod = Literal['median', 'robust-mean', 'robust-mean-scale']

def read_hdf(input_file: FileStore, *attributes: str,
             indices: None | IntSequence | Tuple[IntSequence, Indices, Indices]=None,
             processes: int=1, verbose: bool=True) -> 'CrystData':
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
    if not attributes:
        attributes = tuple(input_file.attributes())

    if indices is None:
        frames = list(range(input_file.size))
        ss_idxs, fs_idxs = slice(None), slice(None)
    elif isinstance(indices, (tuple, list)):
        frames, ss_idxs, fs_idxs = indices
    else:
        frames = indices
        ss_idxs, fs_idxs = slice(None), slice(None)
    frames = np.atleast_1d(frames)

    if input_file.protocol.has_kind(*attributes, kind=Kinds.stack):
        data_dict: Dict[str, Any] = {'frames': frames}
    else:
        data_dict: Dict[str, Any] = {}

    for attr in attributes:
        if attr not in input_file.attributes():
            raise ValueError(f"No '{attr}' attribute in the input files")

        data = input_file.load(attr, idxs=frames, ss_idxs=ss_idxs, fs_idxs=fs_idxs,
                               processes=processes, verbose=verbose)

        data_dict[attr] = data

    return CrystData(**data_dict)

def write_hdf(container: 'CrystData', output_file: FileStore, *attributes: str,
              mode: str='overwrite', indices: Optional[Indices]=None):
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
    if not attributes:
        attributes = tuple(container.contents())

    for attr in attributes:
        data = np.asarray(getattr(container, attr))
        if data is not None:
            output_file.save(attr, data, mode=mode, idxs=indices)

@dataclass
class CrystData(DataContainer):
    """Convergent beam crystallography data container class. Takes a :class:`cbclib_v2.CXIStore`
    file handler. Provides an interface to work with the detector images and detect the diffraction
    streaks. Also provides an interface to load from a file and save to a file any of the data
    attributes. The data frames can be tranformed using any of the :class:`cbclib_v2.Transform`
    classes.

    Args:
        input_file : Input file :class:`cbclib_v2.CXIStore` file handler.
        transform : An image transform object.
        output_file : On output file :class:`cbclib_v2.CXIStore` file handler.
        data : Detector raw data.
        mask : Bad pixels mask.
        frames : List of frame indices inside the container.
        whitefield : Measured frames' white-field.
        snr : Signal-to-noise ratio.
        whitefields : A set of white-fields generated for each pattern separately.
    """
    data        : NDRealArray = np.array([])

    whitefield  : NDRealArray = np.array([])
    std         : NDRealArray = np.array([])
    snr         : NDRealArray = np.array([])

    frames      : NDIntArray = np.array([], dtype=int)
    mask        : NDBoolArray = np.array([], dtype=bool)
    scales      : NDRealArray = np.array([])

    protocol    : ClassVar[CXIProtocol] = CXIProtocol.read()

    def __post_init__(self):
        super().__post_init__()
        if self.frames.size != self.shape[0]:
            self.frames = np.arange(self.shape[0])
        if self.mask.shape != self.shape[1:]:
            self.mask = np.ones(self.shape[1:], dtype=bool)
        if self.scales.shape != (self.shape[0],):
            self.scales = np.ones(self.shape[0])

    @property
    def shape(self) -> Shape:
        shape = [0, 0, 0]
        for attr, data in self.contents().items():
            if data is not None and self.protocol.get_kind(attr) == Kinds.sequence:
                shape[0] = data.shape[0]
                break

        for attr, data in self.contents().items():
            if data is not None and self.protocol.get_kind(attr) == Kinds.frame:
                shape[1:] = data.shape
                break

        for attr, data in self.contents().items():
            if data is not None and self.protocol.get_kind(attr) == Kinds.stack:
                shape[0] = np.prod(data.shape[:-2])
                shape[1:] = data.shape[-2:]
                break

        return tuple(shape)

    def apply_mask(self) -> 'CrystData':
        attributes = {}
        if len(self.whitefield):
            attributes['whitefield'] = self.whitefield * self.mask
        if len(self.std):
            attributes['std'] = self.std * self.mask
        if len(self.snr):
            attributes['snr'] = self.snr * self.mask
        return self.replace(**attributes)

    def import_mask(self, mask: NDBoolArray, update: str='reset') -> 'CrystData':
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
        if len(self.mask) == 0:
            raise ValueError('no mask in the container')
        if mask.shape != self.shape[1:]:
            raise ValueError('mask and data have incompatible shapes: '\
                             f'{mask.shape:s} != {self.shape[1:]:s}')

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
        if len(self.mask) == 0:
            raise ValueError('no mask in the container')

        mask = np.copy(self.mask)
        mask[roi[0]:roi[1], roi[2]:roi[3]] = False
        return self.replace(mask=mask).apply_mask()

    def region_detector(self, structure: Structure2D):
        if len(self.mask) == 0:
            raise ValueError('no mask in the container')
        if len(self.snr) == 0:
            raise ValueError('no snr in the container')

        parent = cast(ReferenceType[CrystData], ref(self))
        idxs = np.arange(self.shape[0])
        return RegionDetector(data=self.snr, mask=self.mask, parent=parent, indices=idxs,
                              structure=structure, transform=ScaleTransform())

    def reset_mask(self) -> 'CrystData':
        """Reset bad pixel mask. Every pixel is assumed to be good by default.

        Raises:
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the default ``mask``.
        """
        return self.replace(mask=np.array([], dtype=bool))

    def scale_whitefield(self, method: str="robust-lsq", r0: float=0.0, r1: float=0.5,
                         n_iter: int=12, lm: float=9.0, num_threads: int=1) -> 'CrystData':
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
        if len(self.data) == 0:
            raise ValueError('no data in the container')
        if len(self.mask) == 0:
            raise ValueError('no mask in the container')
        if len(self.std) == 0:
            raise ValueError('no std in the container')
        if len(self.whitefield) == 0:
            raise ValueError('no whitefield in the container')

        mask = self.mask & (self.std > 0.0)
        y: NDRealArray = np.where(mask, self.data / self.std, 0.0)[:, mask]
        W: NDRealArray = np.where(mask, self.whitefield / self.std, 0.0)[None, mask]

        if method == "robust-lsq":
            scales = robust_lsq(W=W, y=y, axis=1, r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                num_threads=num_threads)
            return self.replace(scales=scales.ravel())

        if method == "median":
            scales = median(y * W, axis=1, num_threads=num_threads)[:, None] / \
                     median(W * W, axis=1, num_threads=num_threads)[:, None]
            return self.replace(scales=scales.ravel())

        raise ValueError(f"Invalid method argument: {method}")

    def select(self, idxs: Optional[Indices]=None):
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
        if len(self.mask) == 0:
            raise ValueError('no mask in the container')
        if len(self.snr) == 0:
            raise ValueError('no snr in the container')

        parent = cast(ReferenceType[CrystData], ref(self))
        idxs = np.arange(self.shape[0])
        return StreakDetector(data=self.snr, mask=self.mask, parent=parent, indices=idxs,
                              structure=structure, transform=ScaleTransform())

    def update_mask(self, method: str='no-bad', vmin: int=0, vmax: int=65535,
                    snr_max: float=3.0, roi: Optional[ROI]=None) -> 'CrystData':
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
        if len(self.data) == 0:
            raise ValueError('no data in the container')
        if len(self.mask) == 0:
            raise ValueError('no mask in the container')
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

    def update_snr(self) -> 'CrystData':
        """Return a new :class:`CrystData` object with new background corrected detector
        images.

        Raises:
            ValueError : If there is no ``whitefield`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``cor_data``.
        """
        if len(self.mask) == 0:
            raise ValueError('no mask in the container')
        if len(self.std) == 0:
            raise ValueError('no std in the container')
        if len(self.whitefield) == 0:
            raise ValueError('no whitefield in the container')

        whitefields = self.scales[:, None, None] * self.whitefield
        snr = np.where(self.std, (self.data * self.mask - whitefields) / self.std, 0.0)
        return self.replace(snr=snr)

    def update_std(self, method="robust-scale", frames: Optional[Indices]=None,
                   r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0,
                   num_threads: int=1) -> 'CrystData':
        if frames is None:
            frames = np.arange(self.shape[0])

        if method == "robust-scale":
            if len(self.data) == 0:
                raise ValueError('no data in the container')
            if len(self.mask) == 0:
                raise ValueError('no mask in the container')

            _, std = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0, r1=r1,
                                 n_iter=n_iter, lm=lm, return_std=True,
                                 num_threads=num_threads)
        elif method == "poisson":
            if len(self.whitefield) == 0:
                raise ValueError('no whitefield in the container')

            std = np.sqrt(self.whitefield)
        else:
            raise ValueError(f"Invalid method argument: {method}")

        return self.replace(std=std)

    def update_whitefield(self, method: WFMethod='median', frames: Optional[Indices]=None,
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
        if len(self.data) == 0:
            raise ValueError('no data in the container')
        if len(self.mask) == 0:
            raise ValueError('no mask in the container')

        if frames is None:
            frames = np.arange(self.shape[0])

        if method == 'median':
            whitefield = median(inp=self.data[frames] * self.mask, axis=0,
                                num_threads=num_threads)
            return self.replace(whitefield=whitefield)
        if method == 'robust-mean':
            whitefield = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0,
                                     r1=r1, n_iter=n_iter, lm=lm,
                                     num_threads=num_threads)
            return self.replace(whitefield=whitefield)
        if method == 'robust-mean-scale':
            whitefield, std = robust_mean(inp=self.data[frames] * self.mask, axis=0,
                                          r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                          return_std=True, num_threads=num_threads)
            return self.replace(whitefield=whitefield, std=std)

        raise ValueError('Invalid method argument')

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

DetBase = TypeVar("DetBase", bound="DetectorBase")

@dataclass
class DetectorBase(DataContainer):
    indices         : NDIntArray
    data            : NDRealArray
    mask            : NDBoolArray
    transform       : ScaleTransform
    parent          : ReferenceType[CrystData]

    @property
    def shape(self) -> Shape:
        return self.data.shape

    def __getitem__(self: DetBase, idxs: Indices) -> DetBase:
        return self.replace(data=self.data[idxs], mask=self.mask[idxs], indices=self.indices[idxs])

    def clip(self: DetBase, vmin: NDArrayLike, vmax: NDArrayLike) -> DetBase:
        return self.replace(data=np.clip(self.data, vmin, vmax))

    def export_coordinates(self, indices: NDIntArray, y: NDIntArray, x: NDIntArray) -> pd.DataFrame:
        table = {'bgd': self.parent().scales[indices] * self.parent().whitefield[y, x],
                 'frames': self.parent().frames[indices], 'snr': self.parent().snr[indices, y, x],
                 'I_raw': self.parent().data[indices, y, x], 'x': x, 'y': y}
        return pd.DataFrame(table)

    def downscale(self: DetBase, scale: float, sigma: float, num_threads: int=1) -> DetBase:
        transform = ScaleTransform(scale)
        data = transform.kernel_regression(self.data, sigma, num_threads)
        mask = transform.interpolate(np.asarray(self.mask, dtype=float))
        return self.replace(data=data, mask=np.asarray(mask, dtype=bool),
                            transform=transform)

    def to_detector(self, streaks: Streaks) -> Streaks:
        pts = np.stack(self.transform.to_detector(streaks.x, streaks.y), axis=-1)
        return Streaks(streaks.index, np.reshape(pts, pts.shape[:-2] + (4,)))

@dataclass
class StreakDetector(DetectorBase, PatternsStreakFinder):
    def detect_streaks(self, peaks: List[Peaks], xtol: float, vmin: float, min_size: int,
                       lookahead: int=0, nfa: int=0, num_threads: int=1) -> Streaks:
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
        streaks = super().detect_streaks(peaks, xtol, vmin, min_size, lookahead, nfa, num_threads)
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
class RegionDetector(DetectorBase):
    structure   : Structure2D

    def detect_regions(self, vmin: float, npts: int, num_threads: int=1) -> List[Regions2D]:
        regions = label((self.data > vmin) & self.mask, structure=self.structure, npts=npts,
                        num_threads=num_threads)
        if isinstance(regions, Regions2D):
            return [regions,]
        return regions

    def export_table(self, regions: List[Regions2D]) -> pd.DataFrame:
        frames, y, x = [], [], []
        for frame, pattern in zip(self.indices, regions):
            size = sum(len(region.x) for region in pattern)
            frames.extend(size * [frame,])
            y.extend(pattern.y)
            x.extend(pattern.x)
        return self.export_coordinates(np.array(frames), np.array(y), np.array(x))

    def ellipse_fit(self, regions: List[Regions2D]) -> List[NDRealArray]:
        return ellipse_fit(regions, self.data)

    def total_mass(self, regions: List[Regions2D]) -> List[NDRealArray]:
        return total_mass(regions, self.data)

    def mean(self, regions: List[Regions2D]) -> List[NDRealArray]:
        return mean(regions, self.data)

    def center_of_mass(self, regions: List[Regions2D]) -> List[NDRealArray]:
        return center_of_mass(regions, self.data)

    def moment_of_inertia(self, regions: List[Regions2D]) -> List[NDRealArray]:
        return moment_of_inertia(regions, self.data)

    def covariance_matrix(self, regions: List[Regions2D]) -> List[NDRealArray]:
        return covariance_matrix(regions, self.data)
