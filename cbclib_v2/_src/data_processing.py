""":class:`cbclib_v2.CrystData` stores all the data necessarry to process measured convergent
beam crystallography patterns and provides a suite of data processing tools to wor with the
detector data.

Examples:
    Load all the necessary data using a :func:`cbclib_v2.CrystData.load` function.

    >>> import cbclib as cbc
    >>> inp_file = cbc.H5Handler('data.cxi')
    >>> data = cbc.CrystData(inp_file)
    >>> data = data.load()
"""
from math import prod
import os
from typing import Literal, NamedTuple, Sequence, cast
from dataclasses import dataclass, field
from weakref import ref
from typing_extensions import Self
import numpy as np
from .cxi_protocol import H5Protocol, Kinds
from .data_container import DataContainer, list_indices
from .streak_finder import PeaksList, detect_peaks, detect_streaks, filter_peaks
from .streaks import StackedStreaks, Streaks
from .annotations import (Array, ArrayLike, BoolArray, CPIntArray, Indices, IntArray, NumPy, RealArray, ReferenceType,
                          ROI, Shape)
from .functions import (LabelResult, Structure, center_of_mass, covariance_matrix, ellipse_fit,
                        min_at, label, line_fit, median, robust_mean, robust_lsq)

MaskMethod = Literal['all-bad', 'no-bad', 'range', 'snr']
MDMethod = Literal['median-poisson', 'robust-mean-scale', 'robust-mean-poisson']
STDMethod = Literal['poisson', 'robust-scale']
WFMethod = Literal['median', 'robust-mean', 'robust-mean-scale']

DATA_PROTOCOL = os.path.join(os.path.dirname(__file__), 'config/cryst_data.ini')
METADATA_PROTOCOL = os.path.join(os.path.dirname(__file__), 'config/cryst_metadata.ini')

class CrystBase(DataContainer):
    protocol    : H5Protocol

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

        if current:
            return current
        return (0,)

    @property
    def num_modules(self) -> int:
        return prod(self.frame_shape) // prod(self.frame_shape[-2:])

    def crop(self: Self, roi: ROI) -> Self:
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
    mask        : BoolArray = field(default_factory=lambda: np.array([], dtype=bool))
    std         : RealArray = field(default_factory=lambda: np.array([]))
    whitefields : RealArray = field(default_factory=lambda: np.array([]))

    protocol    : H5Protocol = field(default_factory=lambda: H5Protocol.read(METADATA_PROTOCOL))

    def __post_init__(self):
        if not self.is_empty(self.mask):
            if not self.is_empty(self.std):
                self.std *= self.mask
            if not self.is_empty(self.whitefields):
                self.whitefields *= self.mask

        if self.is_empty(self.flatfield) and not self.is_empty(self.whitefields):
            self.flatfield = self.whitefields.mean(axis=0)

    @classmethod
    def default_protocol(cls) -> H5Protocol:
        return H5Protocol.read(METADATA_PROTOCOL)

    def import_data(self, data: RealArray, frames: IntArray | int=np.array([], dtype=int),
                    whitefield: RealArray=np.array([])) -> 'CrystData':
        xp = self.__array_namespace__()
        if not isinstance(frames, Array):
            frames = xp.array([frames,], dtype=int)
        data = xp.reshape(data, (frames.size,) + self.frame_shape)

        if not whitefield.size:
            if self.is_empty(self.flatfield):
                raise ValueError('no flatfield in the container')
            return CrystData(data=data, frames=frames, mask=self.mask, std=self.std,
                             whitefield=self.flatfield)

        if whitefield.size != data.size:
            raise ValueError(f'whitefield size {whitefield.size} must be equal to data size '
                             f'{data.size}')

        protocol = CrystData.default_protocol()
        protocol.kinds['whitefield'] = 'stack'
        return CrystData(data=data, frames=frames, mask=self.mask, std=self.std,
                         whitefield=xp.reshape(whitefield, data.shape), protocol=protocol)

    def pca(self) -> 'CrystMetadata':
        if self.is_empty(self.whitefields):
            raise ValueError('no whitefield in the container')
        if self.whitefields.shape[0] == 1:
            raise ValueError('A stack of several whitefields is needed to perform PCA')

        xp = self.__array_namespace__()
        fields = self.whitefields - self.flatfield
        axes = tuple(range(1, len(self.frame_shape) + 1))
        mat_svd = xp.tensordot(fields, fields, axes=(axes, axes))
        eig_vals, eig_vecs = xp.linalg.eigh(mat_svd)
        effs = xp.tensordot(eig_vecs, fields, axes=((0,), (0,)))
        return self.replace(eigen_field=effs, eigen_value=eig_vals / eig_vals.sum())

    def projection(self, data: RealArray, good_fields: Indices=slice(None),
                   method: str="robust-lsq", r0: float=0.0, r1: float=0.5, n_iter: int=12,
                   lm: float=9.0) -> PCAProjection:
        """Return a new :class:`CrystData` object with a new set of whitefields. A set of
        backgrounds is generated by robustly fitting a design matrix `W` to the measured
        patterns.

        Args:
            method : Choose one of the following methods to scale the white-field:

                * "lsq" : By taking a least squares fit of data and whitefield.
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
            projection = robust_lsq(W=W, y=y, axis=1, r0=r0, r1=r1, n_iter=n_iter, lm=lm)
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
                              axes=((-1,), (0,)))
        return self.flatfield + fields

@dataclass
class CrystData(CrystBase):
    """Convergent beam crystallography data container class. Takes a :class:`cbclib_v2.H5Handler`
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

    protocol    : H5Protocol = field(default_factory=lambda: H5Protocol.read(DATA_PROTOCOL))

    def __post_init__(self):
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
    def default_protocol(cls) -> H5Protocol:
        return H5Protocol.read(DATA_PROTOCOL)

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

    def region_detector(self, structure: Structure) -> 'RegionDetector':
        if self.is_empty(self.snr):
            raise ValueError('no snr in the container')

        parent = cast(ReferenceType[CrystData], ref(self))
        return RegionDetector(data=self.snr, structure=structure, parent=parent)

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

    def streak_detector(self, structure: Structure) -> 'StreakDetector':
        """Return a new :class:`cbclib_v2.StreakDetector` object that detects lines in SNR frames.

        Raises:
            ValueError : If there is no ``snr`` inside the container.

        Returns:
            A CBC pattern detector based on bespoke streak detection algorithm.
        """
        if self.is_empty(self.snr):
            raise ValueError('no snr in the container')

        parent = cast(ReferenceType[CrystData], ref(self))
        return StreakDetector(data=self.snr, structure=structure, parent=parent)

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
                        r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0) -> 'CrystData':
        if method == 'median-poisson':
            data = self.update_whitefield('median', frames)
            return data.update_std('poisson', frames)

        if method == 'robust-mean-scale':
            xp = self.__array_namespace__()
            if frames is None:
                frames = xp.arange(self.num_frames)

            whitefield, std = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0, r1=r1,
                                          n_iter=n_iter, lm=lm, return_std=True)
            return self.replace(whitefield=xp.asarray(whitefield), std=xp.asarray(std))

        if method == 'robust-mean-poisson':
            data = self.update_whitefield('robust-mean', frames, r0, r1, n_iter, lm)
            return data.update_std('poisson')

        raise ValueError(f"Invalid method argument: {method}")

    def update_std(self, method: STDMethod='robust-scale', frames: Indices | None=None,
                   r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0
                   ) -> 'CrystData':
        xp = self.__array_namespace__()
        if frames is None:
            frames = xp.arange(self.num_frames)

        if method == 'robust-scale':
            if self.is_empty(self.data):
                raise ValueError('no data in the container')
            if self.is_empty(self.mask):
                raise ValueError('no mask in the container')

            _, std = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0, r1=r1,
                                 n_iter=n_iter, lm=lm, return_std=True)
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
                          r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0
                          ) -> 'CrystData':
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
            whitefield = median(inp=self.data[frames] * self.mask, axis=0)
        elif method == 'robust-mean':
            whitefield = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0,
                                     r1=r1, n_iter=n_iter, lm=lm)
        else:
            raise ValueError('Invalid method argument')

        return self.replace(whitefield=whitefield, protocol=self.default_protocol())

class DetectorBase(DataContainer):
    data            : RealArray
    parent          : ReferenceType[CrystData]

    @property
    def shape(self) -> Shape:
        return self.data.shape

    def __getitem__(self: Self, idxs: Indices) -> Self:
        return self.replace(data=self.data[idxs])

    def clip(self: Self, vmin: ArrayLike, vmax: ArrayLike) -> Self:
        xp = self.__array_namespace__()
        return self.replace(data=xp.clip(self.data, vmin, vmax))

@dataclass
class StreakDetector(DetectorBase):
    data            : RealArray
    structure       : Structure
    parent          : ReferenceType[CrystData]

    def detect_peaks(self, vmin: float, npts: int, connectivity: Structure=Structure([1, 1], 1)
                     ) -> CPIntArray | PeaksList:
        """Find peaks in a pattern. Returns a sparse set of peaks which values are above a threshold
        ``vmin`` that have a supporing set of a size larger than ``npts``. The minimal distance
        between peaks is ``2 * structure.radius``.

        Args:
            vmin : Peak threshold. All peaks with values lower than ``vmin`` are discarded.
            npts : Support size threshold. The support structure is a connected set of pixels which
                value is above the threshold ``vmin``. A peak is discarded is the size of support
                set is lower than ``npts``.
            connectivity : Connectivity structure used in finding a supporting set.
            radius : Minimal distance between peaks. If ``None``, uses the connectivity radius.

        Returns:
            Set of detected peaks.
        """
        peaks = detect_peaks(self.data, self.structure.connectivity, vmin)
        return filter_peaks(peaks, self.data, connectivity, vmin, npts)

    def detect_streaks(self, peaks: PeaksList | CPIntArray, xtol: float, vmin: float,
                       min_size: float, lookahead: int=0, nfa: int=0) -> StackedStreaks | Streaks:
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
            nfa : Number of linelet end points that can fail the distance threshold in a streak.

        Returns:
            A list of detected streaks.
        """
        xp = self.__array_namespace__()
        detected = detect_streaks(peaks, self.data, self.structure, xtol, vmin, min_size, lookahead,
                                  nfa)
        lines = NumPy.zeros((detected.total(), 4), dtype=self.data.dtype)
        lines = xp.asarray(detected.to_lines(lines))
        if self.parent().num_modules > 1:
            return StackedStreaks(xp.asarray(detected.index() // self.parent().num_modules),
                                  xp.asarray(detected.index() % self.parent().num_modules),
                                  lines, self.parent().num_modules)
        return Streaks(xp.asarray(detected.index()), lines)

@dataclass
class RegionDetector(DetectorBase):
    data            : RealArray
    structure       : Structure
    parent          : ReferenceType[CrystData]

    def __post_init__(self):
        if self.structure.rank != 2:
            raise ValueError('Only 2D connectivity structure is supported for streak detection')
        self.structure = self.structure.expand_dims(list(range(self.data.ndim - 2)))

    def detect_regions(self, vmin: float, npts: int) -> LabelResult:
        return label(self.data > vmin, structure=self.structure, npts=npts)

    def detect_streaks(self, regions: LabelResult) -> StackedStreaks | Streaks:
        lines = self.line_fit(regions)
        if self.parent().num_modules > 1:
            return StackedStreaks(min_at(regions, axis=0) // self.parent().num_modules,
                                  min_at(regions, axis=0) % self.parent().num_modules,
                                  lines, self.parent().num_modules)
        return Streaks(min_at(regions, axis=0), lines)

    def ellipse_fit(self, regions: LabelResult) -> RealArray:
        return ellipse_fit(regions, self.data)

    def line_fit(self, regions: LabelResult) -> RealArray:
        return line_fit(regions, self.data)

    def center_of_mass(self, regions: LabelResult) -> RealArray:
        return center_of_mass(regions, self.data)

    def covariance_matrix(self, regions: LabelResult) -> RealArray:
        return covariance_matrix(regions, self.data)
