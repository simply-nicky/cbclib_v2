from functools import partial
from math import log
import os
import logging
import sys
from multiprocessing import Pool
from typing import Any, Callable, Iterator, List, Literal, Tuple, Type, cast, overload
from dataclasses import InitVar, dataclass, field
import h5py
from jax import jit, value_and_grad
from optax import (GradientTransformation, Params, Schedule, Updates, adadelta, adam,
                   apply_updates, constant_schedule, cosine_decay_schedule,
                   cosine_onecycle_schedule, exponential_decay, global_norm,
                   linear_schedule, sgd)
import pandas as pd
from typing_extensions import Self
from tqdm.auto import tqdm
from .annotations import Array, AnyNamespace, IntArray, NDArray, NumPy, RealArray, ROI
from .array_api import asnumpy, default_api, Platform
from .config import get_cpu_config, set_cpu_pool_worker
from .crystfel import Detector
from .cxi_protocol import H5Handler, LoadWorker, TrainIndices
from .data_container import Container, array_namespace, list_indices
from .data_processing import CrystData, CrystMetadata
from .functions import Structure
from .parser import from_container, from_file
from .streaks import StackedStreaks, Streaks
from .polar_streak_finder import PolarStreakFinder, calibrate_center
from ..indexer.cbc_data import CBData, MillerWithRLP, Patterns
from ..indexer.cbc_indexing import CBDIndexer, CBDLoss, CBDModel
from ..indexer.cbc_setup import (BaseSetup, BaseState, FixedApertureSetup, FixedApertureState,
                                 FixedPupilSetup, FixedPupilState, FixedSetup, FixedState,
                                 TiltOverAxisState, XtalList, XtalState)

class BaseParameters(Container):
    """Base class for JSON-serialisable parameter containers.

    Extends :class:`~cbclib_v2.Container` with file I/O.  Every subclass can
    be written to and read from a JSON file, where the parameters are stored
    under a named section key (default ``"parameters"``).

    Example:

        .. code-block:: python

           params = StreakFinderConfig.read('detect_streaks.json')
           params.write('detect_streaks_updated.json')
    """

    @classmethod
    def read(cls: Type[Self], file: str, section: str='parameters') -> Self:
        """Load parameters from a JSON / INI file.

        Args:
            file: Path to the JSON / INI configuration file.
            section: Top-level JSON key under which the parameters are stored.
                Defaults to ``"parameters"``.

        Returns:
            New instance populated from the file.
        """
        parser = from_file(file, cls, section)
        return cls.from_dict(**parser.read(file))

    def write(self, file: str) -> None:
        """Write parameters to a JSON / INI file.

        Args:
            file: Destination path.  The parameters are written under the
                ``"parameters"`` key.
        """
        parser = from_container(file, self, 'parameters')
        parser.write(file, self)

@dataclass
class ROIParameters(BaseParameters):
    """Rectangular region-of-interest bounding box.

    All coordinates are in pixels; the default ``0`` values mean no clipping
    (the full frame is used).

    Attributes:
        xmin: Left column of the ROI (inclusive).
        xmax: Right column of the ROI (exclusive); ``0`` = full width.
        ymin: Top row of the ROI (inclusive).
        ymax: Bottom row of the ROI (exclusive); ``0`` = full height.
    """

    xmin    : int = 0
    xmax    : int = 0
    ymin    : int = 0
    ymax    : int = 0

    def to_roi(self) -> ROI:
        """Return the ROI as a ``(ymin, ymax, xmin, xmax)`` tuple."""
        return (self.ymin, self.ymax, self.xmin, self.xmax)

    def size(self) -> int:
        """Return the number of pixels in the ROI (0 if no ROI is set)."""
        return max((self.ymax - self.ymin) * (self.xmax - self.xmin), 0)

@dataclass
class StructureParameters(BaseParameters):
    """Connectivity-kernel specification for morphological operations.

    Wraps the ``radius`` and ``connectivity`` arguments of
    :class:`~cbclib_v2.label.Structure`.

    Attributes:
        radius: Half-size of the square kernel; the full kernel is
            ``(2 * radius + 1) x (2 * radius + 1)``.
        connectivity: Maximum squared distance from the centre for a pixel
            to be considered a neighbour.
    """

    radius          : int
    connectivity    : int

    def to_structure(self, rank: int=2) -> Structure:
        """Build a :class:`~cbclib_v2.label.Structure` for a *rank*-D array.

        Args:
            rank: Number of spatial dimensions.  The radius is replicated
                along each axis.

        Returns:
            New :class:`~cbclib_v2.label.Structure` instance.
        """
        return Structure([self.radius,] * rank, self.connectivity)

@dataclass
class MaskParameters(BaseParameters):
    """Bad-pixel mask strategy.

    Selects how pixels are classified as bad before background estimation.

    Attributes:
        method: Masking strategy:

            * ``'all-bad'`` — every pixel outside *roi* is masked bad.
            * ``'no-bad'`` — no pixels are masked.
            * ``'range'`` — pixels outside ``[vmin, vmax]`` are masked.
            * ``'snr'`` — pixels with SNR above *snr_max* are masked.
            * ``'std'`` — pixels with standard deviation outside
              ``[std_min, std_max]`` are masked (requires a prior background
              estimate).

        vmin: Lower intensity bound for ``'range'`` masking.
        vmax: Upper intensity bound for ``'range'`` masking.
        snr_max: SNR ceiling for ``'snr'`` masking.
        std_max: Standard-deviation upper bound for ``'std'`` masking
            (``0.0`` = no upper bound).
        std_min: Standard-deviation lower bound for ``'std'`` masking
            (``0.0`` = no lower bound).
    """

    method  : Literal['all-bad', 'no-bad', 'range', 'snr', 'std']
    vmin    : int = 0
    vmax    : int = 65535
    snr_max : float = 3.0
    std_max : float = 0.0
    std_min : float = 0.0

@dataclass
class BackgroundParameters(BaseParameters):
    """Background whitefield estimation method and parameters.

    Attributes:
        method: Estimation algorithm:

            * ``'mean-poisson'`` — pixel-wise mean whitefield with
              Poisson noise model.
            * ``'median-poisson'`` — pixel-wise median whitefield with
              Poisson noise model.
            * ``'robust-mean-scale'`` — robust mean with multiplicative
              scaling per frame.
            * ``'robust-mean-poisson'`` — robust mean with Poisson noise
              model.

        r0: Lower quantile bound used by robust methods to exclude outliers.
        r1: Upper quantile bound used by robust methods to exclude outliers.
        n_iter: Number of reweighting iterations for robust methods.
        lm: Regularisation parameter (effective number of modes) for robust
            methods.
    """

    method  : Literal['mean-poisson', 'median-poisson', 'robust-mean-scale', 'robust-mean-poisson']
    r0      : float = 0.0
    r1      : float = 0.5
    n_iter  : int = 12
    lm      : float = 9.0

@dataclass
class MetadataParameters(BaseParameters):
    """Combined configuration for the metadata-computation step.

    Groups the mask, background, and ROI parameters consumed by
    :func:`create_metadata`.

    Attributes:
        mask: Bad-pixel masking strategy.
        background: Background whitefield estimation method.
        roi: Optional region of interest (all zeros = full frame).
    """

    mask        : MaskParameters
    background  : BackgroundParameters
    roi         : ROIParameters = field(default_factory=ROIParameters)

def create_background(data: CrystData, params: BackgroundParameters, xp: AnyNamespace=NumPy
                      ) -> CrystData:
    """Compute the background whitefield and update *data* in-place.

    Dispatches to the appropriate :class:`~cbclib_v2.CrystData` update method
    based on ``params.method``.

    Args:
        data: Input crystal data container (raw detector frames).
        params: Background estimation parameters.
        xp: Array namespace to use for intermediate computations.

    Returns:
        Updated :class:`~cbclib_v2.CrystData` with ``whitefield`` and
        ``std`` populated.
    """
    if params.method == 'mean-poisson':
        data.whitefield = xp.mean(data.data * data.mask, axis=0)
        data = data.update_std(method='poisson')

    if params.method == 'median-poisson':
        data = data.update_metadata(method='median-poisson')

    if params.method == 'robust-mean-scale':
        data = data.update_metadata(method='robust-mean-scale', r0=params.r0,
                                    r1=params.r1, n_iter=params.n_iter, lm=params.lm)

    if params.method == 'robust-mean-poisson':
        data = data.update_metadata(method='robust-mean-poisson', r0=params.r0,
                                    r1=params.r1, n_iter=params.n_iter, lm=params.lm)
    return data

def create_metadata(frames: RealArray, params: MetadataParameters) -> CrystMetadata:
    """Compute and return a :class:`~cbclib_v2.CrystMetadata` from raw frames.

    Applies the mask strategy from *params*, then runs background estimation,
    and bundles the result into a :class:`~cbclib_v2.CrystMetadata` object
    suitable for use with :func:`scale_background`.

    Args:
        frames: Raw detector frame stack, shape ``(n_frames, *frame_shape)``.
        params: Metadata computation configuration (mask, background, ROI).

    Returns:
        :class:`~cbclib_v2.CrystMetadata` containing the whitefield, pixel
        mask, and per-pixel standard deviation.

    Raises:
        ValueError: If ``params.mask.method == 'all-bad'`` but no ROI is set.
        ValueError: If ``params.mask.method`` is not a recognised value.
    """
    xp = array_namespace(frames)
    data = CrystData(data=frames)

    if params.mask.method == 'all-bad':
        if params.roi.size() == 0:
            raise ValueError("No ROI is provided")
        data = data.update_mask(method='all-bad', roi=params.roi.to_roi())
        data = create_background(data, params.background, xp)

    elif params.mask.method == 'no-bad':
        data = create_background(data, params.background, xp)

    elif params.mask.method == 'range':
        data = data.update_mask(method='range', vmin=params.mask.vmin, vmax=params.mask.vmax)
        data = create_background(data, params.background, xp)

    elif params.mask.method == 'snr':
        data = data.update_mask(method='snr', snr_max=params.mask.snr_max)
        data = create_background(data, params.background, xp)

    elif params.mask.method == 'std':
        data = create_background(data, params.background, xp)
        mask = xp.ones(data.std.shape, dtype=bool)
        if params.mask.std_min > 0.0:
            mask &= data.std > params.mask.std_min
        if params.mask.std_max > 0.0:
            mask &= data.std < params.mask.std_max
        data = data.import_mask(mask)

    else:
        raise ValueError(f"The mask method keyword is invalid: {params.mask.method}")

    return data.metadata()

@dataclass
class ScalingParameters(Container):
    """Background scaling and subtraction configuration.

    Controls how the background whitefield stored in a metalist HDF5 file is
    scaled to each frame before subtraction.

    Attributes:
        method: Scaling algorithm:

            * ``'no-scale'`` — subtract the whitefield without scaling.
            * ``'lsq'`` — least-squares projection onto the PCA background
              components.
            * ``'robust-lsq'`` — robust least-squares projection (outlier
              resistant).

        good_fields: Indices of PCA eigen-fields to include in the
            background model.
        r0: Lower quantile bound for robust projection.
        r1: Upper quantile bound for robust projection.
        n_iter: Number of reweighting iterations for robust projection.
        lm: Regularisation parameter (effective number of modes).
    """

    method      : Literal['lsq', 'no-scale', 'robust-lsq']
    good_fields : Tuple[int, ...] = (0,)
    r0          : float = 0.5
    r1          : float = 0.99
    n_iter      : int = 3
    lm          : float = 9.0

    def metadata(self, metapath: str, xp: AnyNamespace=NumPy) -> CrystMetadata:
        """Load a :class:`~cbclib_v2.CrystMetadata` from a metalist HDF5 file.

        Reads the whitefield, pixel mask, standard deviation, and the
        selected PCA eigen-fields from *metapath* and bundles them into a
        :class:`~cbclib_v2.CrystMetadata`.

        Args:
            metapath: Path to the metalist HDF5 file produced by
                ``cbclib_cli metalist``.
            xp: Array namespace for the loaded arrays.

        Returns:
            :class:`~cbclib_v2.CrystMetadata` ready for
            :func:`scale_background`.
        """
        handler = H5Handler(CrystMetadata.default_protocol())
        idxs = handler.indices(metapath, 'eigen_field')
        good_fields = list_indices(self.good_fields, len(idxs))
        if len(good_fields) == 0:
            raise ValueError("No valid eigen-field indices found in the metadata file")

        eigen_field = handler.load(idxs[good_fields], verbose=False, xp=xp)

        flatfield = handler.load(handler.indices(metapath, 'flatfield'), verbose=False, xp=xp)
        mask = handler.load(handler.indices(metapath, 'mask'), verbose=False, xp=xp)
        std = handler.load(handler.indices(metapath, 'std'), verbose=False, xp=xp)
        return CrystMetadata(flatfield=flatfield, mask=mask, std=std,
                             eigen_field=eigen_field)

def scale_background(frames: IntArray | int, images: Array, metadata: CrystMetadata,
                     params: ScalingParameters) -> CrystData:
    """Subtract the background from *images* and return a :class:`~cbclib_v2.CrystData`.

    Projects each frame onto the PCA background components stored in
    *metadata* (unless ``params.method == 'no-scale'``), subtracts the
    reconstructed background, and wraps the result in a
    :class:`~cbclib_v2.CrystData` object.  Call
    :meth:`~cbclib_v2.CrystData.update_snr` on the result to compute SNR
    frames.

    Args:
        frames: Frame index (scalar) or array of frame indices mapping each
            image to its position in the full scan.
        images: Raw detector images, shape ``(n_frames, *frame_shape)``.
        metadata: Background model loaded with
            :meth:`ScalingParameters.metadata`.
        params: Scaling configuration.

    Returns:
        :class:`~cbclib_v2.CrystData` with background subtracted.

    Raises:
        ValueError: If *images* is less than 2-dimensional.
        ValueError: If ``params.method`` is not a recognised value.
    """
    if images.ndim < 2:
        raise ValueError("Image array must be at least 2 dimensional")

    if params.method == 'no-scale':
        return metadata.to_data(images, frames)

    if params.method in ['robust-lsq', 'lsq']:
        projection = metadata.project(images, params.good_fields, params.method, params.r0,
                                         params.r1, params.n_iter, params.lm)
        return metadata.to_data(images, frames, projection)

    raise ValueError(f'Invalid method keyword: {params.method}')

@dataclass
class RegionParameters(Container):
    """Detection parameters for the connected-region finder.

    Attributes:
        structure: Connectivity kernel used for region labeling.
        vmin: SNR threshold; pixels below this value are treated as
            background.
        npts: Minimum region size in pixels; smaller blobs are discarded.
    """

    structure   : StructureParameters
    vmin        : float
    npts        : int

@dataclass
class RegionFinderConfig(BaseParameters):
    """Full configuration for the region-detection pipeline.

    Passed to :func:`detect_regions`.

    Attributes:
        regions: Detection parameters for the connected-region finder.
        scaling: Background scaling configuration.
        center: Optional ``(x, y)`` detector centre in pixels.  When set,
            only concentric streaks (tangential to circles about this
            centre) are retained.
        roi: Optional region-of-interest crop applied before detection.
        std_min: Minimum per-pixel standard deviation; pixels below this
            threshold are masked before SNR computation.
    """

    regions     : RegionParameters
    scaling     : ScalingParameters
    center      : Tuple[float, float] | None = None
    roi         : ROIParameters = field(default_factory=ROIParameters)
    std_min     : float = 0.0

AllStreaks = StackedStreaks | Streaks

def detect_regions(frames: IntArray | int, images: Array, metadata: CrystMetadata,
                   params: RegionFinderConfig) -> AllStreaks:
    """Run the region-detection pipeline on a batch of frames.

    Subtracts the background, computes SNR, runs the connected-region finder,
    and returns detected streaks with global frame indices.

    Args:
        frames: Scalar frame index or array of frame indices.
        images: Raw detector images, shape ``(n_frames, *frame_shape)``.
        metadata: Background model from :meth:`ScalingParameters.metadata`.
        params: Full region-finder configuration.

    Returns:
        Detected streaks as :class:`~cbclib_v2.Streaks` or
        :class:`~cbclib_v2.StackedStreaks` with global frame indices.
    """
    data = scale_background(frames, images, metadata, params.scaling)

    data = data.update_snr(params.std_min)
    det_obj = data.region_detector(params.regions.structure.to_structure())
    regions = det_obj.detect_regions(params.regions.vmin, params.regions.npts)
    streaks = det_obj.detect_streaks(regions)
    if isinstance(frames, int):
        return streaks.replace(index=streaks.index + frames)
    return streaks.replace(index=frames[streaks.index])

@overload
def concentric_only(streaks: Streaks, center: Tuple[float, float],
                    detector: Detector | None=None) -> Streaks: ...

@overload
def concentric_only(streaks: StackedStreaks, center: Tuple[float, float],
                    detector: Detector | None=None) -> StackedStreaks: ...

def concentric_only(streaks: StackedStreaks | Streaks, center: Tuple[float, float],
                    detector: Detector | None=None) -> AllStreaks:
    """Filter streaks to keep only those tangential to circles centred at *center*.

    A streak is kept when its line direction is approximately tangential to a
    circle whose centre is *center* — equivalently, the radial component of
    the midpoint–centre vector along the streak axis is small relative to the
    total midpoint distance (see
    :meth:`~cbclib_v2.Streaks.concentric_only`).

    If *detector* is provided, streak coordinates are first transformed from
    per-module pixel coordinates to assembled lab-frame coordinates before
    the test is applied.

    Args:
        streaks: Input streak container.
        center: ``(x, y)`` reference centre in assembled detector
            coordinates (pixels).
        detector: Detector geometry used to assemble per-module coordinates
            into lab-frame coordinates.  Required for
            :class:`~cbclib_v2.StackedStreaks` with more than one module.

    Returns:
        Subset of *streaks* that pass the concentricity test, preserving the
        input container type.

    Raises:
        ValueError: If *streaks* is a :class:`~cbclib_v2.StackedStreaks` and
            *detector* is ``None``.
    """
    if detector is None:
        if isinstance(streaks, StackedStreaks):
            raise ValueError("Detector must be provided to apply the detector geometry")
        return streaks[streaks.concentric_only(center[0], center[1])]

    if isinstance(streaks, StackedStreaks) and detector.num_modules > 1:
        x, y, _ = detector.to_detector(streaks.module_id, streaks.y, streaks.x)
    else:
        x, y, _ = detector.to_detector(streaks.y, streaks.x)
    mask = Streaks.import_xy(streaks.index, x, y).concentric_only(center[0], center[1])
    return streaks[mask]


@dataclass
class PeakParameters(BaseParameters):
    """Detection parameters for the region detection step in the streak detection
    pipeline. The class is similar to :class:`~cbclib.scripts.RegionParameters`
    but the SNR threshold is taken from :class:`~cbclib_v2.scripts.StreakParameters`.

    Attributes:
        npts: Minimum region size in pixels; smaller blobs are discarded.
        structure: Connectivity kernel used for region labeling.
    """

    npts        : int
    structure   : StructureParameters = field(
        default_factory=lambda: StructureParameters(radius=1, connectivity=1)
    )

@dataclass
class StreakParameters(Container):
    """Detection parameters for the streak-growing step.

    Attributes:
        structure: Connectivity kernel controlling bin size and linelet
            neighbourhood.
        xtol: Collinearity tolerance in pixels.  Linelet endpoints must lie
            within *xtol* of the growing streak line.
        vmin: SNR threshold for foreground pixels.
        min_size: Minimum-support score threshold; streaks below this value
            are discarded (see
            :meth:`~cbclib_v2.streak_finder.PatternStreakFinder.min_support`).
        nfa: Maximum number of false-alarm linelet endpoints tolerated per
            streak (see :meth:`~cbclib_v2.streak_finder.PatternStreakFinder.detect_streaks`).
    """

    structure   : StructureParameters
    xtol        : float
    vmin        : float
    min_size    : float
    nfa         : int = 0
    keep_best   : float = 1.0

@dataclass
class StreakFinderConfig(BaseParameters):
    """Full configuration for the streak-detection pipeline.

    Passed to :func:`detect_streaks`.

    Attributes:
        peaks: Peak-detection parameters (region labeling + local maxima).
        streaks: Streak-growing parameters.
        scaling: Background scaling configuration.
        center: Optional ``(x, y)`` detector centre in pixels.  When set,
            only concentric streaks are retained after detection.
        std_min: Minimum per-pixel standard deviation; pixels below this
            threshold are masked before SNR computation.
    """

    peaks       : PeakParameters
    streaks     : StreakParameters
    scaling     : ScalingParameters
    center      : Tuple[float, float] | None = None
    std_min     : float = 0.0

@dataclass
class PolarStreakParameters(Container):
    """Detection parameters for the polar / tangential streak route (WIP).

    Attributes:
        na_length: Local streak-probe length in pixels (~ streak length, set by
            the numerical aperture); also the default seed non-max-suppression
            distance for centre calibration.
        r_bounds: ``(r_min, r_max)`` resolution annulus in pixels about the seed
            centre used when calibrating the beam centre.
        vmin: SNR threshold for foreground pixels.
        n_seeds: Maximum centre-calibration seeds per frame.
        n_angles: Number of probe orientations tested per seed.
        min_anisotropy: Minimum orientation contrast to accept a seed.
        weight_exponent: Resolution weight exponent in the centre solve.
    """

    na_length       : float
    r_bounds        : Tuple[float, float]
    vmin            : float
    n_seeds         : int = 20
    n_angles        : int = 64
    min_anisotropy  : float = 0.1
    weight_exponent : float = 1.0

@dataclass
class PolarStreakConfig(BaseParameters):
    """Full configuration for the polar / tangential streak route (WIP).

    Passed to :func:`detect_polar_streaks`.  Mirrors :class:`StreakFinderConfig`
    so it flows through :func:`run_detection` / :class:`StreaksWorker` unchanged.

    Attributes:
        polar: Polar / tangential detection parameters.
        scaling: Background scaling configuration.
        center: Calibrated ``(x, y)`` beam centre in pixels (from
            :func:`calibrate_center`).  Used for the polar remap and, when set,
            for the post-detection concentric filter.
        std_min: Minimum per-pixel standard deviation used in SNR computation.
    """

    polar       : PolarStreakParameters
    scaling     : ScalingParameters
    center      : Tuple[float, float] | None = None
    std_min     : float = 0.0

def detect_streaks(frames: IntArray | int, images: Array, metadata: CrystMetadata,
                   params: StreakFinderConfig) -> StackedStreaks | Streaks:
    """Run the full streak-detection pipeline on a batch of frames.

    Subtracts the background, computes SNR, runs the six-stage streak finder
    (region detection → peak detection → linelet fitting → streak growing →
    ranking → line fitting), filters by minimal support, and returns detected
    streaks with global frame indices.

    Args:
        frames: Scalar frame index or array of frame indices.
        images: Raw detector images, shape ``(n_frames, *frame_shape)``.
        metadata: Background model from :meth:`ScalingParameters.metadata`.
        params: Full streak-finder configuration.

    Returns:
        Detected streaks as :class:`~cbclib_v2.Streaks` or
        :class:`~cbclib_v2.StackedStreaks` with global frame indices.
    """
    data = scale_background(frames, images, metadata, params.scaling)

    data = data.update_snr(params.std_min)
    ndim = data.snr.ndim
    det_obj = data.streak_detector(params.streaks.structure.to_structure(ndim),
                                   params.streaks.vmin)
    regions = det_obj.detect_regions(params.peaks.npts,
                                     params.peaks.structure.to_structure(ndim))
    labels, peaks = det_obj.detect_peaks(regions)
    linelets, labels = det_obj.fit_linelets(labels, peaks)
    result = det_obj.detect_streaks(labels, peaks, linelets, params.streaks.xtol,
                                    nfa=params.streaks.nfa)
    labeled = det_obj.streak_labels(result, labels, peaks)
    lines = det_obj.line_fit(labeled)
    scores = det_obj.min_support(labeled, lines, params.streaks.xtol)
    streaks = det_obj.to_streaks(lines[scores >= params.streaks.min_size])
    if isinstance(frames, int):
        return streaks.replace(index=streaks.index + frames)
    return streaks.replace(index=frames[streaks.index])

def detect_polar_streaks(frames: IntArray | int, images: Array, metadata: CrystMetadata,
                         params: PolarStreakConfig) -> AllStreaks:
    """Run the polar / tangential streak route on a batch of frames (WIP).

    Subtracts the background, computes SNR, and runs
    :class:`~cbclib_v2.polar_streak_finder.PolarStreakFinder` about the
    calibrated ``params.center``.  The per-frame detection algorithm is the next
    milestone; today the finder returns an empty result so the pipeline runs
    end-to-end.  Use :func:`~cbclib_v2.polar_streak_finder.calibrate_center` to
    obtain ``params.center`` first.

    Args:
        frames: Scalar frame index or array of frame indices.
        images: Raw detector images, shape ``(n_frames, *frame_shape)``.
        metadata: Background model from :meth:`ScalingParameters.metadata`.
        params: Polar-route configuration.

    Returns:
        Detected streaks as :class:`~cbclib_v2.Streaks` /
        :class:`~cbclib_v2.StackedStreaks` with global frame indices.
    """
    data = scale_background(frames, images, metadata, params.scaling)
    data = data.update_snr(params.std_min)
    center = params.center if params.center is not None else (0.0, 0.0)
    finder = PolarStreakFinder(asnumpy(data.snr), center, data.num_modules)
    streaks = finder.detect()
    if isinstance(frames, int):
        return streaks.replace(index=streaks.index + frames)
    return streaks.replace(index=frames[streaks.index])

DetectionFunc = Callable[[IntArray | int, Array, CrystMetadata], AllStreaks]
FinderConfig = RegionFinderConfig | StreakFinderConfig | PolarStreakConfig

def detect_patterns(params: FinderConfig) -> DetectionFunc:
    if isinstance(params, StreakFinderConfig):
        return partial(detect_streaks, params=params)
    if isinstance(params, RegionFinderConfig):
        return partial(detect_regions, params=params)
    if isinstance(params, PolarStreakConfig):
        return partial(detect_polar_streaks, params=params)
    raise ValueError(f'Invalid parameters type: {type(params)}')

def run_detection(loader: LoadWorker[NDArray], indices: TrainIndices, metapath: str,
                  params: FinderConfig, platform: Platform = 'cpu',
                  detector: Detector | None=None) -> AllStreaks:
    """Run streak or region detection sequentially over all frames.

    Iterates over every frame in *indices*, loads raw data via *loader*,
    subtracts the background, detects patterns, and concatenates the results.
    Concentric filtering is applied if ``params.center`` is set.

    Args:
        loader: Frame loader; called with a single :class:`~cbclib_v2.H5Handler`
            index and returns a raw NumPy frame array.
        indices: Frame index table covering the chunk to process.
        metapath: Path to the metalist HDF5 file for background loading.
        params: Detection configuration (:class:`StreakFinderConfig` or
            :class:`RegionFinderConfig`).
        platform: Compute backend — ``'cpu'`` or ``'gpu'``.
        detector: Detector geometry for concentric filtering in assembled
            coordinates.  Required when ``params.center`` is set and the
            input is stacked (multi-module).

    Returns:
        All detected streaks concatenated into a single
        :class:`~cbclib_v2.Streaks` or :class:`~cbclib_v2.StackedStreaks`.
    """
    xp = default_api(platform)
    metadata = params.scaling.metadata(metapath, xp)
    detect = detect_patterns(params)

    streaks = []
    for frame, index in tqdm(zip(indices.index(), indices), total=len(indices),
                             desc='Detecting patterns'):
        data = xp.asarray(loader(index))
        pattern = detect(frame, data, metadata)
        if params.center is not None:
            pattern = concentric_only(pattern, params.center, detector)
        streaks.append(pattern)

    if streaks and isinstance(streaks[0], StackedStreaks):
        return StackedStreaks.concatenate(streaks)
    return Streaks.concatenate(streaks)

streaks_worker : 'StreaksWorker'

@dataclass
class StreaksWorker(LoadWorker[AllStreaks]):
    loader          : LoadWorker[NDArray]
    metapath        : InitVar[str]
    params          : FinderConfig
    platform        : InitVar[Platform]
    detector        : Detector | None

    def __post_init__(self, metapath, platform):
        self.xp = default_api(platform)
        self.metafile = self.params.scaling.metadata(metapath, self.xp)
        self.detect = detect_patterns(self.params)

    def __call__(self, indices: Tuple[int, Any]) -> AllStreaks:
        frame, index = indices
        data = self.xp.asarray(self.loader(index))
        pattern = self.detect(frame, data, self.metafile)
        if self.params.center is not None:
            pattern = concentric_only(pattern, self.params.center, self.detector)
        return pattern

    @classmethod
    def initializer(cls, loader: LoadWorker[NDArray], metapath: str, params: StreakFinderConfig,
                    platform: Platform, detector: Detector | None, is_pool: bool=False) -> None:
        set_cpu_pool_worker(is_pool)
        global streaks_worker
        streaks_worker = cls(loader, metapath, params, platform, detector)

    @staticmethod
    def run(index: Tuple[int, Any]) -> AllStreaks:
        return streaks_worker(index)

def pool_detection(loader: LoadWorker[NDArray], indices: TrainIndices, metapath: str,
                   params: FinderConfig, platform: Platform = 'cpu', detector: Detector | None=None
                   ) -> AllStreaks:
    """Run streak or region detection in parallel using a multiprocessing pool.

    Distributes frames across worker processes; each worker maintains its own
    copy of the background metadata and detection function.  Falls back to
    single-process execution when only one thread is configured or when
    *platform* is ``'gpu'`` (CUDA context cannot be shared across processes).

    Args:
        loader: Frame loader callable (must be picklable for multiprocessing).
        indices: Frame index table covering the chunk to process.
        metapath: Path to the metalist HDF5 file for background loading.
        params: Detection configuration (:class:`StreakFinderConfig` or
            :class:`RegionFinderConfig`).
        platform: Compute backend — ``'cpu'`` or ``'gpu'``.
        detector: Detector geometry for concentric filtering.

    Returns:
        All detected streaks concatenated into a single
        :class:`~cbclib_v2.Streaks` or :class:`~cbclib_v2.StackedStreaks`.
    """
    num_threads = get_cpu_config().effective_num_threads()
    initargs = (loader, metapath, params, platform, detector)

    streaks = []
    if platform == 'cpu' and num_threads > 1:
        with Pool(processes=num_threads, initializer=StreaksWorker.initializer,
                  initargs=(*initargs, True)) as pool:
            for pattern in tqdm(pool.imap(StreaksWorker.run, zip(indices.index(), indices)),
                                total=len(indices)):
                streaks.append(pattern)
    else:
        worker = StreaksWorker(*initargs)
        for frame, index in tqdm(zip(indices.index(), indices), total=len(indices)):
            streaks.append(worker((frame, index)))

    if streaks and isinstance(streaks[0], StackedStreaks):
        return StackedStreaks.concatenate(streaks)
    return Streaks.concatenate(streaks)

@dataclass
class IndexingConfig(BaseParameters):
    """Configuration for the CBC pattern-indexing pipeline.

    Attributes:
        shape: Shape ``(nz, ny, nx)`` of the 3-D rotogram accumulation
            buffer used to find orientation peaks.
        width: Gaussian width of each rotogram peak in voxels.
        threshold: Minimum rotogram peak height to be considered a valid
            orientation candidate.
        n_max: Maximum number of orientation candidates to extract per
            diffraction pattern.
        vicinity: Structuring element for peak-neighbourhood extraction
            during orientation refinement (3-D).
        connectivity: Structuring element controlling peak connectivity
            during refinement (3-D).
    """

    shape           : Tuple[int, int, int]
    q_max           : float
    width           : float
    threshold       : float
    n_max           : int
    vicinity        : StructureParameters
    connectivity    : StructureParameters

    def __post_init__(self):
        if not isinstance(self.shape, tuple):
            self.shape = (self.shape[0], self.shape[1], self.shape[2])

def index_patterns(candidates: MillerWithRLP, patterns: Patterns, indexer: CBDIndexer,
                   params: IndexingConfig, state: BaseSetup
                   ) -> Tuple[IntArray, TiltOverAxisState]:
    """Index a single diffraction pattern and return orientation candidates.

    Projects the pattern centre to reciprocal-space coordinates, accumulates
    a rotogram for each Miller-index candidate, builds a 3-D rotomap,
    extracts peaks, and refines them using the neighbourhood structures in
    *params*.

    Args:
        candidates: Miller indices with reciprocal-lattice points for this
            pattern.
        patterns: Diffraction patterns container (single pattern or batch).
        indexer: Convergent-beam diffraction indexer.
        params: Indexing pipeline configuration.
        state: Current crystal and detector geometry state.

    Returns:
        Tuple ``(peak_indices, tilt_states)`` where *peak_indices* selects
        the orientation peaks and *tilt_states* encodes the candidate tilt
        angles over the rotation axis.
    """
    xp = array_namespace(candidates, patterns)
    centers = patterns.sample(xp.full(patterns.shape[0], 0.5))
    points = indexer.points_to_kout(centers, state, xp)
    rotograms = indexer.index(candidates, patterns, points, state)
    rotomap = indexer.rotomap(params.shape, rotograms, patterns.reset_index().index, params.width)
    peaks = indexer.to_peaks(rotomap, params.threshold, params.n_max)
    return indexer.refine_peaks(peaks, rotomap, params.vicinity.to_structure(3),
                                params.connectivity.to_structure(3))

def indexing_candidates(indexer: CBDIndexer, patterns: Patterns, xtal: XtalState, hkl: IntArray,
                        xp: AnyNamespace=NumPy) -> Iterator[MillerWithRLP]:
    """Generate Miller-index candidates for each pattern in *patterns*.

    Yields a generator of :class:`~cbclib_v2.indexer.MillerWithRLP` per pattern
    from the given crystal state and Miller indices.

    Args:
        indexer: CBC indexer instance.
        patterns: Diffraction patterns container.
        xtal: Crystal state (unit cell and orientation).
        hkl: Miller indices to consider for indexing.
        xp: Array namespace.

    Yields:
        :class:`~cbclib_v2.indexer.MillerWithRLP` for each pattern.
    """
    return indexer.xtal.hkl_range(patterns.unique_index(), hkl, xtal, xp)

def run_indexing(patterns: Patterns, xtals: XtalState, state: BaseSetup, params: IndexingConfig,
                 xp: AnyNamespace=NumPy) -> XtalList:
    """Index all patterns sequentially and return a list of crystal solutions.

    Iterates over patterns, generates Miller-index candidates via
    :func:`indexing_candidates`, calls :func:`index_patterns` for each, and
    collects orientation solutions into an :class:`~cbclib_v2.indexer.XtalList`.

    Args:
        patterns: Diffraction patterns container.
        xtals: Crystal state — either a single crystal (broadcast to all
            patterns) or one state per pattern.
        state: Detector geometry state.
        params: Indexing configuration.
        xp: Array namespace.

    Returns:
        :class:`~cbclib_v2.indexer.XtalList` with one solution entry per
        pattern.

    Raises:
        ValueError: If the number of crystals is neither 1 nor equal to the
            number of patterns.
    """
    indexer = CBDIndexer()
    hkl = indexer.xtal.hkl_in_ball(params.q_max, xtals, xp)
    rlp_iterator = indexing_candidates(indexer, patterns, xtals, hkl, xp)
    solutions: List[XtalList] = []

    if len(xtals) == 1:
        iterator = zip(patterns, rlp_iterator)

        for pattern, candidates in tqdm(iterator, total=len(patterns)):
            idxs, tilts = index_patterns(candidates, pattern, indexer, params, state)
            solution = indexer.solutions(xtals, idxs, tilts, pattern)
            solutions.append(solution)
    elif len(xtals) == len(patterns):
        iterator = zip(patterns, rlp_iterator, xtals)

        for pattern, candidates, xtal in tqdm(iterator, total=len(patterns)):
            idxs, tilts = index_patterns(candidates, pattern, indexer, params, state)
            solution = indexer.solutions(xtal, idxs, tilts, pattern)
            solutions.append(solution)
    else:
        raise ValueError(f'Number of crystals ({len(xtals):d}) and patterns ({len(patterns):d}) '\
                         'are inconsistent')

    return XtalList.concatenate(solutions)

indexing_worker : 'IndexingWorker'

@dataclass
class IndexingWorker():
    state   : BaseSetup
    params  : IndexingConfig
    xtal    : XtalState
    indexer : CBDIndexer

    def __call__(self, args: Tuple[MillerWithRLP, Patterns]) -> XtalList:
        candidates, patterns = args
        idxs, tilts = index_patterns(candidates, patterns, self.indexer, self.params, self.state)
        initial = self.xtal if len(self.xtal) == 1 else self.xtal[idxs]
        return self.indexer.solutions(initial, idxs, tilts, patterns)

    @classmethod
    def initializer(cls, state: BaseSetup, params: IndexingConfig, xtal: XtalState,
                    indexer: CBDIndexer, is_pool: bool=False) -> None:
        set_cpu_pool_worker(is_pool)
        global indexing_worker
        indexing_worker = cls(state, params, xtal, indexer)

    @staticmethod
    def run(args: Tuple[MillerWithRLP, Patterns]) -> XtalList:
        return indexing_worker(args)

def pool_indexing(patterns: Patterns, xtals: XtalState, state: BaseSetup, params: IndexingConfig,
                  platform: Platform = 'cpu', xp: AnyNamespace=NumPy) -> XtalList:
    """Index all patterns in parallel using a multiprocessing pool.

    Distributes patterns across worker processes.  Falls back to
    single-process execution when only one thread is configured or when
    *platform* is ``'gpu'``.

    Args:
        patterns: Diffraction patterns container.
        xtals: Crystal state — single crystal or one per pattern.
        state: Detector geometry state.
        params: Indexing configuration.
        platform: Compute backend — ``'cpu'`` or ``'gpu'``.
        xp: Array namespace.

    Returns:
        :class:`~cbclib_v2.indexer.XtalList` with one solution entry per
        pattern.
    """
    num_threads = get_cpu_config().effective_num_threads()
    indexer = CBDIndexer()
    hkl = indexer.xtal.hkl_in_ball(params.q_max, xtals, xp)
    rlp_iterator = indexing_candidates(indexer, patterns, xtals, hkl, xp)

    solutions : List[XtalList] = []
    if platform == 'cpu' and num_threads > 1:
        with Pool(processes=num_threads, initializer=IndexingWorker.initializer,
                  initargs=(state, params, xtals, indexer, True)) as pool:
            iterator = zip(rlp_iterator, patterns)
            for solution in tqdm(pool.imap(IndexingWorker.run, iterator), total=len(patterns)):
                solutions.append(solution)
    else:
        worker = IndexingWorker(state, params, xtals, indexer)
        for candidates, pattern in tqdm(zip(rlp_iterator, patterns), total=len(patterns)):
            solutions.append(worker((candidates, pattern)))

    return XtalList.concatenate(solutions)

@dataclass
class ModelDataParameters(BaseParameters):
    keep        : Literal['best', 'in-shell', 'all']
    quantile    : float
    q_abs       : float
    points      : List[float]

    def cbd_data(self, patterns: Patterns, model: CBDModel, state: BaseState) -> CBData:
        data = model.init_data(patterns, state, values=self.points)
        if self.keep == 'best':
            return model.keep_best(data, quantile=self.quantile)
        if self.keep == 'in-shell':
            return model.keep_in_shell(data, q_abs=self.q_abs, state=state)
        if self.keep == 'all':
            return data
        raise ValueError(f'Invalid keep keyword: {self.keep}')

@dataclass
class LossParameters(BaseParameters):
    kind        : Literal['l1', 'l2', 'log_cosh']
    projector   : Literal['line', 'pupil']

    def cbd_loss(self, model: CBDModel) -> CBDLoss:
        if self.projector == 'line':
            return model.line_loss(loss=self.kind)
        if self.projector == 'pupil':
            return model.pupil_loss(loss=self.kind)
        raise ValueError(f'Invalid projector keyword: {self.projector}')

@dataclass
class ScheduleParameters(BaseParameters):
    kind            : Literal['constant', 'cosine', 'cosine-onecycle', 'exponential', 'linear']
    learning_rate   : float
    min_lr          : float
    num_steps       : int

    def scheduler(self) -> Schedule:
        if self.kind == 'constant':
            return constant_schedule(self.learning_rate)
        if self.kind == 'cosine':
            alpha = self.min_lr / self.learning_rate
            return cosine_decay_schedule(self.learning_rate, self.num_steps, alpha)
        if self.kind == 'cosine-onecycle':
            div_factor = self.learning_rate / self.min_lr
            return cosine_onecycle_schedule(self.num_steps, peak_value=self.learning_rate,
                                            final_div_factor=div_factor)
        if self.kind == 'exponential':
            decay = -log(self.min_lr / self.learning_rate) / self.num_steps
            return exponential_decay(self.learning_rate, self.num_steps, decay,
                                     end_value=self.min_lr)
        if self.kind == 'linear':
            return linear_schedule(self.learning_rate, end_value=self.min_lr,
                                   transition_steps=self.num_steps)
        raise ValueError(f'Invalid scheduler kind: {self.kind}')

@dataclass
class OptimiseParameters(BaseParameters):
    schedule        : ScheduleParameters
    method          : Literal['adadelta', 'adam', 'sgd']
    log_every       : int = 0
    trace_every     : int = 1

    def optimiser(self) -> Tuple[GradientTransformation, Schedule]:
        schedule = self.schedule.scheduler()
        if self.method == 'adadelta':
            return adadelta(schedule), schedule
        if self.method == 'adam':
            return adam(schedule), schedule
        if self.method == 'sgd':
            return sgd(schedule), schedule
        raise ValueError(f'Invalid optimiser method: {self.method}')

LossFn = Callable[[CBData, BaseState], RealArray]
LossGradFn = Callable[[CBData, BaseState], Tuple[RealArray, BaseState]]
ApplyUpdatesFn = Callable[[BaseState, Updates], BaseState]

@dataclass
class RefinementConfig(BaseParameters):
    data : ModelDataParameters
    loss : LossParameters
    optimise : OptimiseParameters
    setup : Literal['fixed', 'fixed-aperture', 'fixed-pupil']
    threshold : float

    def initial(self, xtal: XtalState, setup_file: str) -> BaseState:
        if self.setup == 'fixed':
            setup = FixedSetup.read(setup_file)
            return FixedState(xtal=xtal, setup=setup)
        if self.setup == 'fixed-aperture':
            setup = FixedApertureSetup.read(setup_file)
            return FixedApertureState(xtal=xtal, setup=setup)
        if self.setup == 'fixed-pupil':
            setup = FixedPupilSetup.read(setup_file)
            return FixedPupilState(xtal=xtal, setup=setup)
        raise ValueError(f'Invalid setup keyword: {self.setup}')

@dataclass
class RefinementStats(Container):
    step            : List[int] = field(default_factory=list)
    loss            : List[float] = field(default_factory=list)
    learning_rate   : List[float] = field(default_factory=list)
    grad_norm       : List[float] = field(default_factory=list)
    update_norm     : List[float] = field(default_factory=list)

    @staticmethod
    def norm(updates: Any | None) -> float:
        if updates is None:
            return 0.0
        if isinstance(updates, (int, float)):
            return abs(float(updates))
        return float(global_norm(updates))

    def append(self, step: int, loss: float, learning_rate: float,
               grad: BaseState, updates: Updates | None):
        self.step.append(step)
        self.loss.append(loss)
        self.learning_rate.append(learning_rate)
        self.grad_norm.append(self.norm(grad))
        self.update_norm.append(self.norm(updates))

    def to_dataframe(self) -> pd.DataFrame:
        """Return the optimisation trace as a tabular record."""
        return pd.DataFrame(self.to_dict())

    def save(self, file: str, key: str='stats', mode: str='a') -> None:
        """Write the optimisation trace to an HDF5 table.

        Args:
            file: Output HDF5 file path.
            key: HDF5 table key used for the trace.
            mode: Pandas HDF5 open mode.
        """
        self.to_dataframe().to_hdf(file, key=key, mode=mode)

@dataclass
class RefineResult(Container):
    index   : IntArray
    state   : BaseState
    loss    : RealArray
    fitness : RealArray
    stats   : RefinementStats

    def champions_only(self) -> 'RefineResult':
        xp = self.state.__array_namespace__()
        idxs = xp.lexsort((self.loss, self.index))
        sorted_index = self.index[idxs]
        firsts = xp.searchsorted(sorted_index, xp.unique_values(sorted_index))
        champions = idxs[firsts]
        state = self.state.replace(xtal=self.state.xtal[champions])
        return RefineResult(self.index[champions], state, self.loss[champions],
                            self.fitness[champions], self.stats)

    def to_dataframe(self) -> pd.DataFrame:
        """Return refined orientations with per-solution refinement metrics."""
        df = self.state.xtal.to_dataframe(index=self.index)
        df['loss'] = asnumpy(self.loss)
        df['fitness'] = asnumpy(self.fitness)
        return df

    def save(self, file: str, files: dict[str, str] | None=None) -> None:
        """Write champion solutions, all candidates, and refinement trace to HDF5.

        The ``data`` table contains one champion solution per pattern and is
        intended as the default downstream input.  The ``candidates`` and
        ``stats`` tables retain diagnostic information from refinement.

        Args:
            file: Output HDF5 file path.
            files: Optional source-file paths written under the ``files`` HDF5 group.
        """
        dir_path = os.path.dirname(file)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.champions_only().to_dataframe().to_hdf(file, key='data', mode='w')
        self.to_dataframe().to_hdf(file, key='candidates', mode='a')
        self.stats.save(file, mode='a')

        if files is not None:
            with h5py.File(file, 'a') as output_file:
                for name, path in files.items():
                    key = f'files/{name}'
                    if key in output_file:
                        del output_file[key]
                    output_file[key] = path

def optimisation_loop(data: CBData, initial: BaseState, optimiser: GradientTransformation,
                      schedule: Schedule, loss_grad_fn: LossGradFn,
                      num_steps: int, trace_every: int=1, log_every: int=0,
                      logger: logging.Logger | None=None
                      ) -> Tuple[BaseState, RefinementStats]:
    stats = RefinementStats()

    state = initial
    apply_updates_fn : ApplyUpdatesFn = cast(ApplyUpdatesFn, apply_updates)
    trace_every = max(trace_every, 1)
    log_every = max(log_every, 0)

    opt_state = optimiser.init(cast(Params, state))

    loss, grad = loss_grad_fn(data, state)
    stats.append(0, float(loss), float(schedule(0)), grad, None)
    if logger is not None:
        logger.info("step=%d loss=%.6e lr=%.6e grad_norm=%.6e update_norm=%.6e",
                    stats.step[-1], stats.loss[-1], stats.learning_rate[-1],
                    stats.grad_norm[-1], stats.update_norm[-1])

    for step in range(1, num_steps + 1):
        updates, opt_state = optimiser.update(cast(Updates, grad), opt_state)
        update_norm = global_norm(updates)
        state = apply_updates_fn(state, updates)
        loss, grad = loss_grad_fn(data, state)

        if step % trace_every == 0 or step == num_steps:
            stats.append(step, float(loss), float(schedule(step - 1)), grad, update_norm)

        if logger is not None and log_every and (step % log_every == 0 or step == num_steps):
            logger.info("step=%d loss=%.6e lr=%.6e grad_norm=%.6e update_norm=%.6e",
                        step, float(loss), float(schedule(step - 1)), float(global_norm(grad)),
                        float(update_norm))

    return state, stats

def default_refinement_logger() -> logging.Logger:
    """Return the default refinement logger writing progress messages to stdout."""
    logger = logging.getLogger(f'{__name__}.refinement')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    return logger

def refine_patterns(patterns: Patterns, xtal_list: XtalList, setup_file: str,
                    params: RefinementConfig, logger: logging.Logger | None=None
                    ) -> RefineResult:
    model = CBDModel()
    target = patterns.loc[xtal_list.index]
    initial = params.initial(xtal_list.to_xtals(), setup_file)

    loss = params.loss.cbd_loss(model)
    data = params.data.cbd_data(target, model, initial)
    loss_grad_fn : LossGradFn = jit(value_and_grad(jit(loss), argnums=1))

    solver, schedule = params.optimise.optimiser()

    if logger is None:
        logger = default_refinement_logger()
    logger.info("refining %d patterns with %s/%s/%s for %d steps",
                len(initial.xtal), params.optimise.method, params.loss.kind,
                params.loss.projector, params.optimise.schedule.num_steps)

    state, stats = optimisation_loop(data, initial, solver, schedule, loss_grad_fn,
                                     params.optimise.schedule.num_steps,
                                     params.optimise.trace_every, params.optimise.log_every,
                                     logger)
    criterion = loss.per_pattern(data, state)
    fitness = loss.pattern_fitness(params.threshold, target, data, state)
    return RefineResult(xtal_list.index, state, criterion, fitness, stats)
