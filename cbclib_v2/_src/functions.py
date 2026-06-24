"""Multidimensional image processing functions with CPU/CUDA backend support.

This module provides image processing operations that automatically dispatch to CPU or CUDA
backends based on the device context set via :mod:`cbclib_v2.device`.

See Also:
    :mod:`cbclib_v2.device`: Device context management for backend selection.
"""
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from math import prod
from typing import (TYPE_CHECKING, Callable, NamedTuple, Optional, Protocol, Sequence, Tuple,
                    cast, overload)
import warnings
from .annotations import (Array, BoolArray, CPArray, CPBoolArray, CPIntArray,
                          CPRealArray, CuPy, IntArray, IntSequence, JaxArray, JaxBoolArray,
                          JaxIntArray, JaxNumPy, JaxRealArray, NDArray, NDBoolArray, NDIntArray,
                          NDRealArray, NumPy, RealArray)
from .array_api import array_namespace, ascupy, asjax, asnumpy, get_platform
from .config import get_cpu_config
from .data_container import DataContainer
from .src import bresenham, label as cpu_label, median as cpu_median, online_detector, streak_finder
from .src.label import Structure
from .src.streak_finder import Streaks as NPStreaks

if TYPE_CHECKING:
    from .crystfel import Detector

def array_dispatch(dispatch_arg: str, cpu_impl: Callable, gpu_impl: Callable):
    """Dispatch to CPU or GPU implementation based on array namespace/device.

    Args:
        dispatch_arg: Name of the array argument used to determine dispatch.
        cpu_impl: Function implementing the CPU path.
        gpu_impl: Function implementing the GPU path.
    """
    def decorator(func):
        sig = signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            if dispatch_arg not in bound.arguments:
                raise TypeError(f"Missing dispatch argument '{dispatch_arg}'.")

            arr = bound.arguments[dispatch_arg]
            xp = array_namespace(arr)

            if xp is NumPy:
                return cpu_impl(*bound.args, **bound.kwargs)
            if xp is JaxNumPy:
                pl = get_platform(arr)
                if pl == 'cpu':
                    warnings.warn(f"{func.__name__} is not implemented for JAX backend. Falling " \
                                  "back to CPU implementation.", RuntimeWarning)

                    bound.arguments[dispatch_arg] = asnumpy(arr)
                    result = cpu_impl(*bound.args, **bound.kwargs)

                    if isinstance(result, NDArray):
                        return asjax(result)
                    return result

                warnings.warn(f"{func.__name__} is not implemented for JAX backend. Falling back " \
                              "to GPU implementation.", RuntimeWarning)

                bound.arguments[dispatch_arg] = ascupy(arr)
                result = gpu_impl(*bound.args, **bound.kwargs)

                if isinstance(result, CPArray):
                    return asjax(result)
                return result

            if xp is CuPy:
                return gpu_impl(*bound.args, **bound.kwargs)

            raise RuntimeError(f"Unkown Array API: {xp.__name__}. Supported backends are NumPy, "
                               "JAX and CuPy.")

        return wrapper

    return decorator

class PeakLabels(NamedTuple):
    """Bin-state descriptor produced by :func:`peak_labels`.

    Encodes the result of the peak-detection step: for each grid bin it
    records whether a local maximum was found (peak), a foreground pixel
    exists but no strict maximum (good), or the bin is empty (bad).

    Attributes:
        labels: Integer array of shape ``(*frame_shape[:-2], NY_bins, NX_bins)``.
            Positive values are 1-based peak indices; 0 marks a *good* bin;
            -1 marks a *bad* bin.
        n_seeds: Number of seed peaks that will be used as starting points
            for streak growing.  Controlled via :meth:`keep_best`.
        n_labels: Total number of bins that have been assigned a linelet
            (updated by :func:`fit_linelets` as propagation fills good bins).
        n_good: Number of bins that overlap at least one foreground region
            (peaks + good bins).
        radius: Bin side length in pixels; equals ``structure.connectivity``
            passed to :func:`detect_peaks`.
    """

    labels : IntArray
    n_seeds : int
    n_labels : int
    n_good : int
    radius : int

    def keep_best(self, quantile: float = 0.5) -> 'PeakLabels':
        """Restrict streak growing to the top fraction of peaks by intensity.

        Returns a copy of this descriptor with ``n_seeds`` reduced to
        ``int(n_seeds * quantile)``.  Only the first ``n_seeds`` entries of
        the peaks array (sorted by descending intensity) are used as seeds
        during :func:`detect_streaks`.

        Args:
            quantile: Fraction of peaks to keep.  ``0.5`` keeps the
                brightest half; ``1.0`` keeps all peaks.

        Returns:
            Updated :class:`PeakLabels` with a smaller ``n_seeds``.
        """
        return PeakLabels(self.labels, int(self.n_seeds * quantile), self.n_labels,
                          self.n_good, self.radius)

    def to_tuple(self) -> Tuple[IntArray, int, int, int, int]:
        return self.labels, self.n_seeds, self.n_labels, self.n_good, self.radius

if CuPy is not None or TYPE_CHECKING:
    from .src import (cuda_draw_lines, cuda_label, cuda_median, cuda_online_detector,
                      cuda_streak_finder)
    from cupyx.scipy import ndimage as _ndimage
    from .src.cuda_streak_finder import Streaks as CPStreaks

    Streaks = NPStreaks | CPStreaks

    class NDImageProtocol(Protocol):
        def binary_dilation(self, input: BoolArray, structure: BoolArray, iterations: int=1,
                            mask: BoolArray | None=None, output: BoolArray | None=None,
                            brute_force: bool=False) -> CPBoolArray: ...

        def minimum(self, input: RealArray | IntArray, labels: IntArray | None=None,
                    index: IntSequence | None=None) -> CPRealArray | CPIntArray: ...

    cupy_ndimage = cast(NDImageProtocol, _ndimage)

    def binary_structure(structure: Structure, shape: Sequence[int] | None=None) -> CPBoolArray:
        """Generate a binary structure array for CUDA backend."""
        if shape is None:
            shape = structure.shape
        return CuPy.asarray(structure.to_array(out=NumPy.zeros(shape, dtype=bool)))

    def shift_axis(inp: Array, axis: Tuple[int, ...]) -> Array:
        """Shift specified axis to the end for CUDA median implementation."""
        reduce_axis = []
        out_axis = []
        out_shape = []
        for i in range(inp.ndim):
            if i in axis or i - inp.ndim in axis:
                reduce_axis.append(i)
            else:
                out_axis.append(i)
                out_shape.append(inp.shape[i])

        inp = CuPy.permute_dims(inp, out_axis + reduce_axis)
        return CuPy.reshape(inp, (*out_shape, -1))
else:
    cuda_draw_lines = None
    cuda_label = None
    cuda_median = None
    cuda_online_detector = None
    cupy_ndimage = None
    cuda_streak_finder = None

    CPStreaks = None # type: ignore
    Streaks = NPStreaks

    def binary_structure(structure: Structure, shape: Sequence[int] | None=None) -> CPBoolArray:
        raise RuntimeError("CUDA backend is not available. Please, check if you have installed " \
                           "the cbclib_v2 with GPU support.")

    def shift_axis(inp: Array, axis: Tuple[int, ...]) -> Array:
        raise RuntimeError("CUDA backend is not available. Please, check if you have installed " \
                           "the cbclib_v2 with GPU support.")

def _accumulate_lines_cpu(out: RealArray, lines: RealArray, terms: IntArray,
                          frames: IntArray, width: RealArray | float=1.0,
                          max_val: float=1.0, kernel: str='rectangular', in_overlap: str='sum',
                          out_overlap: str='sum') -> NDRealArray:
    num_threads = get_cpu_config().effective_num_threads()

    xp = NumPy
    if isinstance(width, (int, float)):
        width = xp.asarray([width,], dtype=lines.dtype)
    lines, terms, frames = asnumpy(lines), asnumpy(terms), asnumpy(frames)
    return bresenham.accumulate_lines(out=out, lines=lines, terms=terms, frames=frames,
                                      widths=width, max_val=max_val, kernel=kernel,
                                      in_overlap=in_overlap, out_overlap=out_overlap,
                                      num_threads=num_threads)

def _accumulate_lines_gpu(out: RealArray, lines: RealArray, terms: IntArray,
                          frames: IntArray, width: RealArray | float=1.0,
                          max_val: float=1.0,
                          kernel: str='rectangular', in_overlap: str='sum',
                          out_overlap: str='sum') -> CPRealArray:
    if cuda_draw_lines is None:
        raise RuntimeError("accumulate_lines is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    if isinstance(width, (int, float)):
        width = xp.asarray([width,], dtype=lines.dtype)
    lines, terms, frames = ascupy(lines), ascupy(terms), ascupy(frames)
    widths = ascupy(width)
    if widths.ndim == 0:
        widths = widths.reshape((1,))
    return cuda_draw_lines.accumulate_lines(out=out, lines=lines, terms=terms, frames=frames,
                                            widths=widths, max_val=max_val, kernel=kernel,
                                            in_overlap=in_overlap,
                                            out_overlap=out_overlap)

@overload
def accumulate_lines(out: NDRealArray, lines: NDRealArray, terms: NDIntArray,
                     frames: NDIntArray, width: RealArray | float=1.0,
                     max_val: float=1.0, kernel: str='rectangular',
                     in_overlap: str='sum', out_overlap: str='sum') -> NDRealArray: ...

@overload
def accumulate_lines(out: CPRealArray, lines: CPRealArray, terms: CPIntArray,
                     frames: CPIntArray, width: RealArray | float=1.0,
                     max_val: float=1.0, kernel: str='rectangular',
                     in_overlap: str='sum', out_overlap: str='sum') -> CPRealArray: ...

@overload
def accumulate_lines(out: JaxRealArray, lines: JaxRealArray, terms: JaxIntArray,
                     frames: JaxIntArray, width: RealArray | float=1.0,
                     max_val: float=1.0, kernel: str='rectangular',
                     in_overlap: str='sum', out_overlap: str='sum') -> JaxRealArray: ...

@overload
def accumulate_lines(out: Array, lines: Array, terms: IntArray, frames: IntArray,
                     width: RealArray | float=1.0, max_val: float=1.0,
                     kernel: str='rectangular', in_overlap: str='sum',
                     out_overlap: str='sum') -> RealArray: ...

@array_dispatch("out", cpu_impl=_accumulate_lines_cpu, gpu_impl=_accumulate_lines_gpu)
def accumulate_lines(out: RealArray, lines: RealArray, terms: IntArray, frames: IntArray,
                     width: RealArray | float=1.0, max_val: float=1.0,
                     kernel: str='rectangular', in_overlap: str='sum',
                     out_overlap: str='sum') -> RealArray:
    """Accumulate thick lines or curves across multiple frames.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        out: Output array where the geometries will be accumulated.
        lines: Array of shape ``(..., 2 * ndim)`` with segment endpoints, or
            ``(..., n_points, ndim)`` with curve points. Curve points are sewn internally
            with a local maximum before overlap between curves is applied.
        terms: Term indices specifying to which term each line or curve belongs.
        frames: Frame indices specifying to which frame each term belongs.
        width: Line or curve width in pixels. A scalar applies to all geometries; arrays
            must broadcast to the leading geometry shape.
        max_val: Maximum pixel value of a drawn line.
        kernel: Choose one of the supported kernel functions. The following kernels
            are available:

            * 'biweight' : Quartic (biweight) kernel.
            * 'gaussian' : Gaussian kernel.
            * 'parabolic' : Epanechnikov (parabolic) kernel.
            * 'rectangular' : Uniform (rectangular) kernel.
            * 'triangular' : Triangular kernel.

        in_overlap: How to combine input overlapping pixels ('sum', 'max', 'min').
        out_overlap: How to combine output overlapping pixels ('sum', 'max', 'min').

    Returns:
        Output array with the lines or curves accumulated.

    See Also:
        :func:`draw_lines`: Draw lines on a single frame.
    """
    ...

def _draw_lines_cpu(out: RealArray, lines: RealArray, idxs: IntArray | None=None,
                    width: RealArray | float=1.0, max_val: float=1.0, kernel: str='rectangular',
                    overlap: str='sum') -> NDRealArray:
    num_threads = get_cpu_config().effective_num_threads()

    xp = NumPy
    if isinstance(width, (int, float)):
        width = xp.asarray([width,], dtype=lines.dtype)
    lines = asnumpy(lines)
    idxs = asnumpy(idxs) if idxs is not None else None
    return bresenham.draw_lines(out=out, lines=lines, idxs=idxs, widths=width, max_val=max_val,
                                kernel=kernel, overlap=overlap, num_threads=num_threads)

def _draw_lines_gpu(out: RealArray, lines: RealArray, idxs: IntArray | None=None,
                    width: RealArray | float=1.0, max_val: float=1.0, kernel: str='rectangular',
                    overlap: str='sum') -> CPRealArray:
    if cuda_draw_lines is None:
        raise RuntimeError("draw_lines is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    if isinstance(width, (int, float)):
        width = xp.asarray([width,], dtype=lines.dtype)
    lines = ascupy(lines)
    widths = ascupy(width)
    if widths.ndim == 0:
        widths = widths.reshape((1,))
    idxs = ascupy(idxs) if idxs is not None else None
    return cuda_draw_lines.draw_lines(out=out, lines=lines, widths=widths, idxs=idxs,
                                      max_val=max_val, kernel=kernel, overlap=overlap)

@overload
def draw_lines(out: NDRealArray, lines: NDRealArray, idxs: NDIntArray | None=None,
               width: RealArray | float=1.0, max_val: float=1.0,
               kernel: str='rectangular', overlap: str='sum'
               ) -> NDRealArray: ...

@overload
def draw_lines(out: CPRealArray, lines: CPRealArray, idxs: CPIntArray | None=None,
               width: RealArray | float=1.0, max_val: float=1.0,
               kernel: str='rectangular', overlap: str='sum'
               ) -> CPRealArray: ...

@overload
def draw_lines(out: JaxRealArray, lines: JaxRealArray, idxs: JaxIntArray | None=None,
               width: RealArray | float=1.0, max_val: float=1.0,
               kernel: str='rectangular', overlap: str='sum'
               ) -> JaxRealArray: ...

@overload
def draw_lines(out: Array, lines: Array, idxs: IntArray | None=None,
               width: RealArray | float=1.0, max_val: float=1.0,
               kernel: str='rectangular', overlap: str='sum') -> RealArray: ...

@array_dispatch("out", cpu_impl=_draw_lines_cpu, gpu_impl=_draw_lines_gpu)
def draw_lines(out: RealArray, lines: RealArray, idxs: IntArray | None=None,
               width: RealArray | float=1.0, max_val: float=1.0, kernel: str='rectangular',
               overlap: str='sum') -> RealArray:
    """Draw thick lines with variable thickness and antialiasing.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        out: Output array to draw lines on.
        lines: Array of shape ``(..., 2 * ndim)`` with segment endpoints.
        idxs: Optional frame indices for each line. If None, all lines drawn to single frame.
        width: Line width in pixels. A scalar applies to all lines; arrays must broadcast
            to the leading line shape.
        max_val: Maximum pixel value for drawn lines.
        kernel: Kernel function for antialiasing. Options:
            - 'rectangular': Uniform (box) kernel
            - 'gaussian': Gaussian kernel
            - 'parabolic': Epanechnikov kernel
            - 'biweight': Quartic (biweight) kernel
            - 'triangular': Triangular kernel
        overlap: How to combine overlapping pixels ('sum', 'max', 'min').

    Returns:
        Output array with drawn lines.

    See Also:
        :func:`accumulate_lines`: Accumulate lines across multiple frames.
        :mod:`cbclib_v2.device`: Set device context for backend selection.
    """
    ...

def _binary_dilation_cpu(inp: BoolArray, structure: Structure, iterations: int=1,
                         mask: Optional[BoolArray]=None) -> NDBoolArray:
    num_threads = get_cpu_config().effective_num_threads()
    mask = asnumpy(mask) if mask is not None else None
    return cpu_label.binary_dilation(inp=inp, structure=structure, iterations=iterations,
                                     mask=mask, num_threads=num_threads)

def _binary_dilation_gpu(inp: BoolArray, structure: Structure, iterations: int=1,
                         mask: Optional[BoolArray]=None) -> CPBoolArray:
    if cupy_ndimage is None:
        raise RuntimeError("binary_dilation is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    mask = ascupy(mask) if mask is not None else None
    sarray = binary_structure(structure)
    return cupy_ndimage.binary_dilation(input=inp, structure=sarray,
                                        iterations=iterations, mask=mask)

@overload
def binary_dilation(inp: NDBoolArray, structure: Structure, iterations: int=1,
                    mask: Optional[NDBoolArray]=None) -> NDBoolArray: ...

@overload
def binary_dilation(inp: CPBoolArray, structure: Structure, iterations: int=1,
                    mask: Optional[CPBoolArray]=None) -> CPBoolArray: ...

@overload
def binary_dilation(inp: JaxBoolArray, structure: Structure, iterations: int=1,
                    mask: Optional[JaxBoolArray]=None) -> JaxBoolArray: ...

@overload
def binary_dilation(inp: BoolArray, structure: Structure, iterations: int=1,
                    mask: Optional[BoolArray]=None) -> BoolArray: ...

@array_dispatch("inp", cpu_impl=_binary_dilation_cpu, gpu_impl=_binary_dilation_gpu)
def binary_dilation(inp: BoolArray, structure: Structure, iterations: int=1,
                    mask: Optional[BoolArray]=None) -> BoolArray:
    """Binary dilation of 2D binary image.

    Dispatches to the CPU or GPU implementation based on the Array API namespace of
    ``inp``. The CPU backend supports concurrent execution through the configured
    OpenMP thread count controlled by :func:`~cbclib_v2.set_cpu_config`; the GPU
    backend uses CuPy's CUDA implementation, following the behavior of
    :func:`scipy.ndimage.binary_dilation`.

    Args:
        inp: Input binary array.
        structure: Structuring element used for dilation.
        iterations: Number of dilation iterations.
        mask: Optional mask to limit dilation area.

    Returns:
        Dilated binary array.

    See Also:
        :doc:`/array_api`: Array API backend selection and conversion utilities.
        :func:`~cbclib_v2.set_cpu_config`: Configure the CPU thread count.
    """
    ...

def pixel_map(out: RealArray, geometry: 'Detector', half_pixel_shift: bool=True) -> RealArray:
    """Compute a CrystFEL detector pixel-coordinate map.

    The map stores the lab-frame coordinate of every detector pixel described
    by ``geometry``. The first axis has length 3 and contains ``x``, ``y``,
    and ``z`` coordinates in CrystFEL pixel units. The ``x``/``y`` axes follow
    the CrystFEL detector convention; ``z`` is the detector distance in pixels
    including panel ``coffset`` values.

    The result is the geometric bridge between a CrystFEL ``.geom`` file and
    array operations such as radial background estimation. NumPy and CuPy
    outputs use the native online-detector kernels; other array namespaces use
    a portable Python implementation that calls each panel's
    :meth:`~cbclib_v2.crystfel.Panel.to_detector` method.

    Args:
        out: Output array with shape ``(3, *detector_shape)``. Its namespace and
            dtype select the backend and native overload.
        geometry: Parsed CrystFEL detector geometry.
        half_pixel_shift: Add a 0.5-pixel offset before transforming panel
            coordinates, so returned coordinates refer to pixel centres rather
            than pixel corners.

    Returns:
        Real array with shape ``(3, *detector_shape)``. ``out[0]`` is ``x``,
        ``out[1]`` is ``y``, and ``out[2]`` is ``z`` in lab-frame pixel units.
    """
    xp = array_namespace(out)
    if xp is NumPy:
        return online_detector.pixel_map(
            out, geometry, half_pixel_shift=half_pixel_shift,
            num_threads=get_cpu_config().effective_num_threads()
        )
    if xp is CuPy:
        if cuda_online_detector is None:
            raise RuntimeError("online detector is not compiled for the current platform. "
                               "Please, check if you have installed the cbclib_v2 with GPU "
                               "support.")

        return cuda_online_detector.pixel_map(out, geometry, half_pixel_shift=half_pixel_shift)

    if out.shape != (3,) + geometry.shape:
        raise ValueError("pixel_map output shape mismatch")
    out[...] = 0
    for panel in geometry.panels.values():
        roi = panel.roi()
        ss_grid, fs_grid = xp.meshgrid(xp.arange(panel.shape[-2]),
                                       xp.arange(panel.shape[-1]), indexing='ij')

        x, y, z = panel.to_detector(ss_grid, fs_grid, half_pixel_shift)
        out[(0,) + roi] = x
        out[(1,) + roi] = y
        out[(2,) + roi] = z
    return out

def radius(out: RealArray, geometry: 'Detector', center: Tuple[int, int],
           half_pixel_shift: bool=True) -> RealArray:
    """Compute each detector pixel's radius from the beam centre.

    Radii are measured in the assembled detector plane using the same
    lab-frame pixel coordinates as :func:`pixel_map`. The ``center`` argument
    is a CrystFEL lab-frame coordinate, usually the direct-beam position. The
    detector bounds are subtracted internally so the returned radii align with
    the image grid used by :meth:`~cbclib_v2.Detector.assembler` and
    :func:`radial_index`.

    Args:
        out: Output array with shape ``detector_shape``. Its namespace and dtype
            select the backend and native overload.
        geometry: Parsed CrystFEL detector geometry.
        center: Beam centre ``(x, y)`` in CrystFEL lab-frame pixel units.
        half_pixel_shift: Add a 0.5-pixel offset before transforming panel
            coordinates, so distances are measured from pixel centres.

    Returns:
        Real array with the detector image shape. Each value is the Euclidean
        distance from ``center`` in pixels.
    """
    xp = array_namespace(out)
    if xp is NumPy:
        return online_detector.radius(
            out, geometry, center, half_pixel_shift=half_pixel_shift,
            num_threads=get_cpu_config().effective_num_threads()
        )
    if xp is CuPy:
        if cuda_online_detector is None:
            raise RuntimeError("online detector is not compiled for the current platform. "
                               "Please, check if you have installed the cbclib_v2 with GPU "
                               "support.")

        return cuda_online_detector.radius(out, geometry, center,
                                           half_pixel_shift=half_pixel_shift)

    pixel_out = xp.empty((3,) + geometry.shape, dtype=out.dtype)
    x, y, _ = pixel_map(pixel_out, geometry, half_pixel_shift=half_pixel_shift)
    result = xp.sqrt((x - center[0] - geometry.bounds[0]) ** 2 +
                     (y - center[1] - geometry.bounds[1]) ** 2)
    if out.shape != geometry.shape:
        raise ValueError("radius output shape mismatch")
    out[...] = result
    return out

def radial_index(out: IntArray, geometry: 'Detector', center: Tuple[int, int],
                 n_bins: int, half_pixel_shift: bool=True) -> IntArray:
    """Compute integer radial-bin indices for radial background estimation.

    The detector plane is divided into ``n_bins`` concentric annuli around
    ``center``. Each detector pixel receives the nearest integer radial-bin
    index, computed as ``round(radius / radius_step)`` where
    ``radius_step = geometry.max_radius(center) / (n_bins - 1)``. The compact
    profile arrays used by :func:`radial_profiles` and
    :class:`~cbclib_v2.RadialProfiles` are indexed with this map.

    Native CPU/CUDA kernels also mark pixels that do not belong to any panel as
    ``-1`` so radial-profile code can ignore gaps in the assembled detector
    image. Valid panel pixels are in the inclusive range ``[0, n_bins - 1]``.

    Args:
        out: Output array with shape ``detector_shape``. Its namespace and
            integer dtype select the backend and native overload.
        geometry: Parsed CrystFEL detector geometry.
        center: Beam centre ``(x, y)`` in CrystFEL lab-frame pixel units.
        n_bins: Number of radial bins. Must be at least 2 so a finite radial
            step can be computed.
        half_pixel_shift: Add a 0.5-pixel offset before transforming panel
            coordinates, so bins are assigned from pixel centres.

    Returns:
        Integer array with the detector image shape. Values are radial-bin
        indices; ``-1`` denotes non-panel pixels on native backends.
    """
    xp = array_namespace(out)
    if xp is NumPy:
        return online_detector.radial_index(
            out, geometry, center, n_bins, half_pixel_shift=half_pixel_shift,
            num_threads=get_cpu_config().effective_num_threads()
        )
    if xp is CuPy:
        if cuda_online_detector is None:
            raise RuntimeError("online detector is not compiled for the current platform. "
                               "Please, check if you have installed the cbclib_v2 with GPU "
                               "support.")

        return cuda_online_detector.radial_index(out, geometry, center, n_bins,
                                                 half_pixel_shift=half_pixel_shift)

    radius_step = geometry.max_radius(center) / (n_bins - 1)
    radius_out = xp.empty(geometry.shape, dtype=xp.float64)
    radii = radius(radius_out, geometry, center, half_pixel_shift=half_pixel_shift)
    result = xp.asarray(xp.round(radii / radius_step), dtype=out.dtype)
    if out.shape != geometry.shape:
        raise ValueError("radial_index output shape mismatch")
    out[...] = result
    return out

def _is_signal_cpu(data: IntArray | RealArray, whitefield: RealArray, std: RealArray,
                   radial_index: IntArray, min_snr: float=3.0, std_min: float=0.0
                   ) -> NDBoolArray:
    num_threads = get_cpu_config().effective_num_threads()
    return online_detector.is_signal(data=data, whitefield=whitefield, std=std,
                                     radial_index=radial_index, min_snr=min_snr, std_min=std_min,
                                     num_threads=num_threads)

def _is_signal_gpu(data: IntArray | RealArray, whitefield: RealArray, std: RealArray,
                   radial_index: IntArray, min_snr: float=3.0, std_min: float=0.0
                   ) -> CPBoolArray:
    if cuda_online_detector is None:
        raise RuntimeError("online detector is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    out = CuPy.empty(data.shape, dtype=bool)
    return cuda_online_detector.is_signal(out=out, data=data, whitefield=whitefield, std=std,
                                          radial_index=radial_index, min_snr=min_snr,
                                          std_min=std_min)

@array_dispatch("data", cpu_impl=_is_signal_cpu, gpu_impl=_is_signal_gpu)
def is_signal(data: IntArray | RealArray, whitefield: RealArray, std: RealArray,
              radial_index: IntArray, min_snr: float=3.0, std_min: float=0.0) -> BoolArray:
    """Identify pixels above a compact radial residual-SNR threshold.

    ``whitefield`` and ``std`` are compact per-frame radial profiles, indexed
    by ``radial_index``. For each pixel, the function compares the raw detector
    value to the profile value of that pixel's radial bin and returns ``True``
    when the residual SNR is at least ``min_snr``.

    Args:
        data: Raw detector frame stack. The trailing dimensions must match
            ``radial_index``.
        whitefield: Per-frame radial mean profile, shape
            ``(n_frames, n_bins)``.
        std: Per-frame radial standard-deviation profile, shape
            ``(n_frames, n_bins)``.
        radial_index: Integer radial-bin map for one detector frame. Valid
            pixels index ``whitefield``/``std``; ``-1`` pixels are ignored by
            native backends.
        min_snr: Minimum residual SNR for a pixel to be considered signal.
        std_min: Lower bound for the standard deviation used in the SNR
            denominator.

    Returns:
        Boolean array with the same shape as ``data``. ``True`` marks online
        signal pixels.
    """
    ...

@dataclass
class RadialProfiles(DataContainer):
    """Compact per-frame radial background profiles.

    Stores the online detector's background estimate as one value per radial
    bin instead of one value per detector pixel. The profiles are expanded back
    to pixels through a ``radial_index`` lookup when calling :meth:`is_signal`.

    Attributes:
        whitefield: Per-frame radial mean intensity, shape
            ``(n_frames, n_bins)``.
        std: Per-frame radial standard deviation, shape
            ``(n_frames, n_bins)``.
        counts: Number of valid pixels contributing to each radial bin, shape
            ``(n_frames, n_bins)``.
    """

    whitefield  : RealArray
    std         : RealArray
    counts      : IntArray

    @property
    def n_bins(self) -> int:
        return self.whitefield.shape[-1]

    def scale(self, scale: RealArray) -> 'RadialProfiles':
        """Scale whitefield and noise profiles by per-frame intensity factors.

        The mean profile scales linearly with intensity. The standard deviation
        is scaled by ``sqrt(scale)``, matching Poisson-like count statistics.

        Args:
            scale: Non-negative scale factor broadcastable to
                ``whitefield.shape``.

        Returns:
            New :class:`RadialProfiles` with scaled ``whitefield`` and ``std``.
        """
        xp = self.__array_namespace__()
        scale = xp.clip(xp.asarray(scale), 0.0, xp.inf)
        return self.replace(whitefield=self.whitefield * scale, std=self.std * xp.sqrt(scale))

    def is_signal(self, data: IntArray | RealArray, radial_index: IntArray,
                  min_snr: float=3.0, std_min: float=0.0) -> BoolArray:
        """Identify pixels that exceed this profile's radial SNR threshold.

        Args:
            data: Raw detector frame stack.
            radial_index: Integer radial-bin map matching one frame.
            min_snr: Minimum residual SNR for a pixel to be considered signal.
            std_min: Lower bound for the radial standard deviation used in the
                SNR denominator.

        Returns:
            Boolean signal mask with the same shape as ``data``.
        """
        return is_signal(data=data, whitefield=self.whitefield, std=self.std,
                         radial_index=radial_index, min_snr=min_snr, std_min=std_min)

    def to_array(self, radial_index: IntArray) -> Tuple[RealArray, RealArray]:
        """Expand compact radial profiles to full-frame arrays.

        Args:
            radial_index: Integer radial-bin map matching one detector frame.

        Returns:
            Tuple ``(whitefield, std)`` expanded to detector-pixel layout.
        """
        xp = self.__array_namespace__()
        indices = radial_index.reshape(-1)
        whitefield = xp.take_along_axis(self.whitefield, indices[None, ...], axis=-1)
        std = xp.take_along_axis(self.std, indices[None, ...], axis=-1)
        shape = (self.whitefield.shape[0],) + radial_index.shape
        return whitefield.reshape(shape), std.reshape(shape)

def _radial_profiles_cpu(data: NDIntArray | NDRealArray, radial_index: IntArray,
                         n_bins: int, interval: int=1, clip_snr: float=3.0, n_iter: int=3,
                         std_min: float=0.0) -> RadialProfiles:
    num_threads = get_cpu_config().effective_num_threads()
    whitefield, std, counts = online_detector.radial_profiles(
        data, radial_index, n_bins, interval, clip_snr, n_iter, std_min,
        num_threads=num_threads
    )
    return RadialProfiles(whitefield, std, counts)

def _radial_profiles_gpu(data: CPIntArray | CPRealArray, radial_index: CPIntArray,
                         n_bins: int, interval: int=1, clip_snr: float=3.0, n_iter: int=3,
                         std_min: float=0.0) -> RadialProfiles:
    if cuda_online_detector is None:
        raise RuntimeError("online detector is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    frame_size = int(radial_index.size)
    n_frames = int(data.size) // frame_size
    ftype = xp.float64 if data.dtype.itemsize >= 8 else xp.float32

    whitefield = xp.empty((n_frames, n_bins), dtype=ftype)
    std = xp.empty((n_frames, n_bins), dtype=ftype)
    counts = xp.empty((n_frames, n_bins), dtype=radial_index.dtype)
    whitefield, std, counts = cuda_online_detector.radial_profiles(
        whitefield, std, counts, data, radial_index, n_bins, interval, clip_snr, n_iter, std_min
    )
    return RadialProfiles(whitefield, std, counts)

@array_dispatch("data", cpu_impl=_radial_profiles_cpu, gpu_impl=_radial_profiles_gpu)
def radial_profiles(data: IntArray | RealArray, radial_index: IntArray, n_bins: int,
                    interval: int=1, clip_snr: float=3.0, n_iter: int=3,
                    std_min: float=0.0) -> RadialProfiles:
    """Compute compact radial background profiles for online detection.

    Pixels are grouped by ``radial_index`` and reduced to one mean and standard
    deviation per radial bin for each frame. The estimate is robustified by
    iteratively rejecting pixels above ``mean + clip_snr * std`` before
    recomputing the profiles, which keeps sparse diffraction signal from
    biasing the radial background.

    Args:
        data: Raw detector frame stack. The trailing dimensions must match
            ``radial_index``.
        radial_index: Integer radial-bin map for one detector frame, usually
            created with :meth:`~cbclib_v2.Detector.radial_index`.
        n_bins: Number of radial bins represented in the compact profiles.
        interval: Process every ``interval``-th radial bin together in the
            native kernels. The default keeps the binning exact.
        clip_snr: SNR threshold used to reject bright outliers between profile
            iterations.
        n_iter: Number of outlier-rejection iterations.
        std_min: Lower bound for the returned per-bin standard deviation.

    Returns:
        :class:`RadialProfiles` with ``whitefield``, ``std``, and ``counts``
        arrays of shape ``(n_frames, n_bins)``.
    """
    ...

class LabelResult(NamedTuple):
    """Result of a connected-component labeling operation (GPU/CuPy backend).

    Stores the label map as a dense CuPy integer array together with a
    1-D array of the region indices that are present.  Mirrors the interface
    of :class:`~cbclib_v2.label.LabelResult` for the CPU backend.

    Attributes:
        labels: CuPy integer array of the same shape as the input, with
            each pixel set to its region index (0 for background).
        index: 1-D CuPy integer array of region indices ``[1, …, n_labels]``.
    """
    labels      : IntArray
    index       : IntArray

def _label_cpu(inp: NDBoolArray | NDIntArray, structure: Structure, npts: int=1) -> LabelResult:
    num_threads = get_cpu_config().effective_num_threads()
    labels, index = cpu_label.label(inp=inp, structure=structure, npts=npts,
                                    num_threads=num_threads)
    return LabelResult(labels=labels, index=index)

def _label_gpu(inp: CPBoolArray | CPIntArray, structure: Structure, npts: int=1) -> LabelResult:
    if cuda_label is None:
        raise RuntimeError("label is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    labels = xp.empty_like(inp, dtype=xp.int32)
    labels, n_labels = cuda_label.label(out=labels, inp=inp, structure=structure, npts=npts)
    return LabelResult(labels=labels, index=xp.arange(1, n_labels + 1, dtype=int))

@array_dispatch("inp", cpu_impl=_label_cpu, gpu_impl=_label_gpu)
def label(inp: BoolArray | IntArray, structure: Structure, npts: int=1) -> LabelResult:
    """Label connected regions in a boolean or integer array.

    This function is similar to :func:`scipy.ndimage.label`: all non-zero
    pixels in *inp* are treated as foreground.  Connected foreground pixels
    are assigned the same positive integer label; background pixels are
    labeled 0.  Connectivity is determined by the structuring element
    *structure*.  Regions with fewer than *npts* pixels are discarded (their
    pixels reset to 0).

    Args:
        inp: Input boolean or integer array.  Non-zero values are foreground.
        structure: Structuring element that defines which pixels are
            considered neighbours (i.e. what counts as "connected").
        npts: Minimum region size in pixels.  Regions with fewer pixels are
            removed from the result.

    Returns:
        Labeled regions as a :class:`LabelResult`.
    """
    ...

def _center_of_mass_cpu(labels: LabelResult, data: NDRealArray) -> NDRealArray:
    num_threads = get_cpu_config().effective_num_threads()
    return cpu_label.center_of_mass(labels=(labels.labels, labels.index), data=data,
                                    num_threads=num_threads)

def _center_of_mass_gpu(labels: LabelResult, data: CPRealArray) -> CPRealArray:
    if cuda_label is None:
        raise RuntimeError("center_of_mass is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    out = xp.empty((labels.index.shape[0], data.ndim), dtype=data.dtype)
    return cuda_label.center_of_mass(out=out, labels=labels.labels, index=labels.index, data=data)

@overload
def center_of_mass(labels: LabelResult, data: NDRealArray) -> NDRealArray: ...

@overload
def center_of_mass(labels: LabelResult, data: CPRealArray) -> CPRealArray: ...

@overload
def center_of_mass(labels: LabelResult, data: JaxRealArray) -> JaxRealArray: ...

@overload
def center_of_mass(labels: LabelResult, data: RealArray) -> RealArray: ...

@array_dispatch("data", cpu_impl=_center_of_mass_cpu, gpu_impl=_center_of_mass_gpu)
def center_of_mass(labels: LabelResult, data: RealArray) -> RealArray:
    """Compute the intensity-weighted center of mass for each labeled region.

    The center of mass is the first-order
    `image moment <https://en.wikipedia.org/wiki/Image_moment>`_ normalised
    by the total intensity (zeroth-order moment):
    ``c_k = M_k / M_00`` along each axis *k*, where
    ``M_ij = Σ x^i y^j · w(x, y)`` and *w* are the values from *data*.

    Args:
        labels: Labeled regions returned by :func:`label`.
        data: Intensity (weight) array with the same spatial shape as the
            label array.

    Returns:
        Array of shape ``(N, ndim)`` with the center-of-mass coordinates for
        each of the *N* labeled regions.
    """
    ...

def _covariance_matrix_cpu(labels: LabelResult, data: NDRealArray) -> NDRealArray:
    num_threads = get_cpu_config().effective_num_threads()
    matrices = cpu_label.covariance_matrix(labels=(labels.labels, labels.index), data=data,
                                           num_threads=num_threads)
    return matrices.reshape(-1, data.ndim, data.ndim)

def _covariance_matrix_gpu(labels: LabelResult, data: CPRealArray) -> CPRealArray:
    if cuda_label is None:
        raise RuntimeError("covariance_matrix is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    out = xp.empty((labels.index.shape[0], data.ndim, data.ndim), dtype=data.dtype)
    return cuda_label.covariance_matrix(out=out, labels=labels.labels, index=labels.index,
                                        data=data)

@overload
def covariance_matrix(labels: LabelResult, data: NDRealArray) -> NDRealArray: ...

@overload
def covariance_matrix(labels: LabelResult, data: CPRealArray) -> CPRealArray: ...

@overload
def covariance_matrix(labels: LabelResult, data: JaxRealArray) -> JaxRealArray: ...

@overload
def covariance_matrix(labels: LabelResult, data: RealArray) -> RealArray: ...

@array_dispatch("data", cpu_impl=_covariance_matrix_cpu, gpu_impl=_covariance_matrix_gpu)
def covariance_matrix(labels: LabelResult, data: RealArray) -> RealArray:
    """Compute the intensity-weighted covariance matrix for each labeled region.

    The covariance matrix is built from second-order central
    `image moments <https://en.wikipedia.org/wiki/Image_moment>`_ weighted
    by the values in *data*:
    ``Cov[i, j] = μ_ij / M_00``, where
    ``μ_ij = Σ (x_i - x̄_i)(x_j - x̄_j) · w(x)`` and *w* are the values
    from *data*.

    Args:
        labels: Labeled regions returned by :func:`label`.
        data: Intensity (weight) array with the same spatial shape as the
            label array.

    Returns:
        Array of shape ``(N, ndim, ndim)`` with the covariance matrix for
        each of the *N* labeled regions.
    """
    ...

def to_ellipse(matrix: RealArray) -> RealArray:
    xp = array_namespace(matrix)
    if matrix.size == 0:
        return xp.empty((0, 3), dtype=matrix.dtype)

    mu_xx, mu_xy, mu_yy = matrix[..., -1, -1], matrix[..., -1, -2], matrix[..., -2, -2]
    theta = 0.5 * xp.atan(2 * mu_xy / (mu_xx - mu_yy))
    delta = xp.sqrt(4 * mu_xy**2 + (mu_xx - mu_yy)**2)
    a = xp.sqrt(2 * xp.log(2) * (mu_xx + mu_yy + delta))
    b = xp.sqrt(2 * xp.log(2) * (mu_xx + mu_yy - delta))
    return xp.stack((a, b, theta), axis=-1)

@overload
def ellipse_fit(labels: LabelResult, data: NDRealArray) -> NDRealArray: ...

@overload
def ellipse_fit(labels: LabelResult, data: CPRealArray) -> CPRealArray: ...

@overload
def ellipse_fit(labels: LabelResult, data: JaxRealArray) -> JaxRealArray: ...

@overload
def ellipse_fit(labels: LabelResult, data: RealArray) -> RealArray: ...

def ellipse_fit(labels: LabelResult, data: RealArray) -> RealArray:
    """Fit an ellipse to each labeled region using image moments.

    The ellipse parameters are derived from the second-order central
    `image moments <https://en.wikipedia.org/wiki/Image_moment>`_ weighted
    by the values in *data*.  The covariance matrix of the spatial
    coordinates is decomposed into its eigenvectors; the semi-axes of the
    ellipse correspond to the FWHM of the intensity distribution along those
    eigenvectors.

    Args:
        labels: Labeled regions returned by :func:`label`.
        data: Intensity (weight) array with the same spatial shape as the
            label array.

    Returns:
        Array of shape ``(N, 3)`` where each row is ``(a, b, theta)``:
        *a* and *b* are the FWHM of the major and minor axes, and *theta*
        is the orientation angle in radians.
    """
    covmat = covariance_matrix(labels, data)
    return to_ellipse(covmat)

def to_line(centers: CPRealArray, matrix: CPRealArray) -> CPRealArray:
    xp = array_namespace(centers, matrix)
    if centers.size == 0 and matrix.size == 0:
        return xp.empty((0, 2 * centers.shape[-1]), dtype=centers.dtype)

    mu_xx, mu_xy, mu_yy = matrix[..., -1, -1], matrix[..., -1, -2], matrix[..., -2, -2]
    theta = 0.5 * xp.atan2(2 * mu_xy, (mu_xx - mu_yy))

    tau = xp.zeros(centers.shape, dtype=centers.dtype)
    tau[..., 0] = xp.cos(theta)
    tau[..., 1] = xp.sin(theta)

    delta = xp.sqrt(4 * mu_xy**2 + (mu_xx - mu_yy)**2)
    hw = xp.sqrt(2 * xp.log(2) * (mu_xx + mu_yy + delta))
    return xp.concat((centers[..., ::-1] + hw[..., None] * tau,
                      centers[..., ::-1] - hw[..., None] * tau), axis=-1)

def _line_fit_cpu(labels: LabelResult, data: NDRealArray) -> NDRealArray:
    num_threads = get_cpu_config().effective_num_threads()
    return cpu_label.line_fit(labels=(labels.labels, labels.index), data=data,
                              num_threads=num_threads)

def _line_fit_gpu(labels: LabelResult, data: CPRealArray) -> CPRealArray:
    if cuda_label is None:
        raise RuntimeError("line_fit is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    centers = _center_of_mass_gpu(labels, data)
    covmat = _covariance_matrix_gpu(labels, data)
    return to_line(centers, covmat)

@overload
def line_fit(labels: LabelResult, data: NDRealArray) -> NDRealArray: ...

@overload
def line_fit(labels: LabelResult, data: JaxRealArray) -> JaxRealArray: ...

@overload
def line_fit(labels: LabelResult, data: CPRealArray) -> CPRealArray: ...

@overload
def line_fit(labels: LabelResult, data: RealArray) -> RealArray: ...

@array_dispatch("data", cpu_impl=_line_fit_cpu, gpu_impl=_line_fit_gpu)
def line_fit(labels: LabelResult, data: RealArray) -> RealArray:
    """Fit a line to each labeled region using image moments.

    The line is the major axis of the intensity-weighted covariance ellipse
    computed from second-order central
    `image moments <https://en.wikipedia.org/wiki/Image_moment>`_.  The
    eigenvector corresponding to the largest eigenvalue of the covariance
    matrix gives the orientation; the half-length of the segment is the FWHM
    of the distribution along that eigenvector.

    Args:
        labels: Labeled regions returned by :func:`label`.
        data: Intensity (weight) array with the same spatial shape as the
            label array.

    Returns:
        Array of shape ``(N, 2 * ndim)`` where each row is the concatenation
        of the two endpoint coordinates ``(x1, ..., x2, ...)`` of the fitted
        line segment.
    """
    ...

def _maximum_position_cpu(labels: LabelResult, data: NDIntArray | NDRealArray) -> NDIntArray:
    num_threads = get_cpu_config().effective_num_threads()
    return cpu_label.maximum_position(labels=(labels.labels, labels.index), data=data,
                                      num_threads=num_threads)

def _maximum_position_gpu(labels: LabelResult, data: CPIntArray | CPRealArray) -> CPIntArray:
    if cuda_label is None:
        raise RuntimeError("maximum_position is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")
    if labels.labels.shape != data.shape:
        raise ValueError("labels and data must have the same shape")

    xp = CuPy
    out = xp.empty((labels.index.shape[0], data.ndim), dtype=xp.int32)
    return cuda_label.maximum_position(out=out, labels=labels.labels, index=labels.index,
                                       data=data)

@overload
def maximum_position(labels: LabelResult, data: NDIntArray | NDRealArray) -> NDIntArray: ...

@overload
def maximum_position(labels: LabelResult, data: CPIntArray | CPRealArray) -> CPIntArray: ...

@overload
def maximum_position(labels: LabelResult, data: JaxIntArray | JaxRealArray) -> JaxIntArray: ...

@overload
def maximum_position(labels: LabelResult, data: IntArray | RealArray) -> IntArray: ...

@array_dispatch("data", cpu_impl=_maximum_position_cpu, gpu_impl=_maximum_position_gpu)
def maximum_position(labels: LabelResult, data: IntArray | RealArray) -> IntArray:
    """Find the first maximum position in each labeled region.

    For each label listed in ``labels.index``, the function returns the
    coordinate of the first pixel, in flat row-major traversal order, whose
    value is maximal within that labeled region.  Labels with no matching
    pixels return the coordinate of the first array element.

    Args:
        labels: Labeled regions returned by :func:`label`.
        data: Intensity array with the same spatial shape as the label array.

    Returns:
        Integer array of shape ``(N, data.ndim)`` with one coordinate row for
        each of the *N* labels in ``labels.index``.
    """
    ...

def _p_values_cpu(labels: LabelResult, lines: NDRealArray, data: NDRealArray, p0: float, vmin: float,
                  xtol: float) -> NDRealArray:
    num_threads = get_cpu_config().effective_num_threads()
    return cpu_label.p_values(labels=(labels.labels, labels.index), lines=lines, data=data, p0=p0,
                              vmin=vmin, xtol=xtol, num_threads=num_threads)

def _p_values_gpu(labels: LabelResult, lines: CPRealArray, data: CPRealArray, p0: float, vmin: float,
                  xtol: float) -> CPRealArray:
    if cuda_label is None:
        raise RuntimeError("label is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    out = xp.empty(labels.index.shape, dtype=data.dtype)
    return cuda_label.p_values(out=out, labels=labels.labels, index=labels.index, lines=lines,
                               data=data, p0=p0, vmin=vmin, xtol=xtol)

@overload
def p_values(labels: LabelResult, lines: NDRealArray, data: NDRealArray, p0: float, vmin: float,
             xtol: float) -> NDRealArray: ...

@overload
def p_values(labels: LabelResult, lines: JaxRealArray, data: JaxRealArray, p0: float, vmin: float,
             xtol: float) -> JaxRealArray: ...

@overload
def p_values(labels: LabelResult, lines: CPRealArray, data: CPRealArray, p0: float, vmin: float,
             xtol: float) -> CPRealArray: ...

@overload
def p_values(labels: LabelResult, lines: RealArray, data: RealArray, p0: float, vmin: float,
             xtol: float) -> RealArray: ...

@array_dispatch("data", cpu_impl=_p_values_cpu, gpu_impl=_p_values_gpu)
def p_values(labels: LabelResult, lines: RealArray, data: RealArray, p0: float, vmin: float,
             xtol: float) -> RealArray:
    """Compute the log-binomial tail probability for each labeled streak region.

    For each labeled region the function:

    1. Builds a footprint: the union of *structure* neighbourhoods around
       the region's peak pixels.
    2. Restricts the footprint to pixels within *xtol* of the fitted line,
       yielding *n* candidate pixels and *k* pixels above *vmin*.
    3. Returns ``log P(X ≥ k)`` for ``X ~ Binomial(n, p0)``, evaluated with
       the log-survival function of the binomial distribution.

    The returned values are the raw log tail probabilities.  Dividing by
    ``log(p0)`` converts them to the minimal-support score used by
    :meth:`~cbclib_v2.streak_finder.PatternStreakFinder.min_support`.

    Args:
        labels: Labeled streak regions returned by
            :func:`streak_labels` (after re-labeling with :func:`label`).
        lines: Fitted line endpoints of shape ``(N, 2 * data.ndim)`` in
            ``(x0, y0, x1, y1, ...)`` order, one row per labeled region.
        data: SNR frame stack of shape ``(n_frames, *frame_shape)``.
        p0: Background pixel probability — fraction of pixels above *vmin*
            across the full stack; used as the success probability of the
            null-hypothesis binomial.
        vmin: SNR threshold.  Pixels at or above *vmin* count as foreground
            in the binomial statistic.
        xtol: Distance tolerance in pixels matching the value used during
            streak growing.  Only pixels within *xtol* of the fitted line
            are included in the footprint count.

    Returns:
        Float array of length *N* containing ``log P(X ≥ k)`` for each
        region.  Values are negative; a more negative value indicates
        stronger statistical evidence for a real streak.
    """
    ...

@overload
def median(inp: NDRealArray, axis: IntSequence=0) -> NDRealArray: ...

@overload
def median(inp: NDIntArray, axis: IntSequence=0) -> NDIntArray: ...

@overload
def median(inp: CPRealArray | CPIntArray, axis: IntSequence=0) -> CPRealArray | CPIntArray: ...

@overload
def median(inp: JaxRealArray | JaxIntArray, axis: IntSequence=0) -> JaxRealArray | JaxIntArray: ...

def median(inp: RealArray | IntArray, axis: IntSequence=0) -> RealArray | IntArray:
    """Calculate a median along the axis.

    Dispatches to NumPy, CuPy, or JAX based on the input array's Array API backend.
    The NumPy backend supports concurrent execution through the configured OpenMP
    thread count controlled by :func:`~cbclib_v2.set_cpu_config`, while CuPy and
    JAX use their native accelerator-aware median implementations.

    Args:
        inp: Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        axis: Array axes along which median values are calculated.

    Returns:
        Array of medians along the given axis.

    See Also:
        :doc:`/array_api`: Array API backend selection and conversion utilities.
        :func:`~cbclib_v2.set_cpu_config`: Configure the CPU thread count.
        :func:`median_filter`: Multidimensional median filter.
        :func:`maximum_filter`: Multidimensional maximum filter.
    """
    if isinstance(inp, NDArray):
        num_threads = get_cpu_config().effective_num_threads()
        return cpu_median.median(inp=inp, axis=axis, num_threads=num_threads)
    if isinstance(inp, CPArray):
        return CuPy.median(inp, axis=axis)
    if isinstance(inp, JaxArray):
        return JaxNumPy.median(inp, axis=axis)
    raise RuntimeError("Unkown Array type: " + str(type(inp)) +
                       ". Supported types are NumPy, CuPy and JAX arrays.")

def _robust_mean_cpu(inp: IntArray | RealArray, axis: IntSequence=0, r0: float=0.0,
                     r1: float=0.5, n_iter: int=12, lm: float=9.0,
                     return_std: bool=False) -> NDRealArray:
    num_threads = get_cpu_config().effective_num_threads()
    return cpu_median.robust_mean(inp=inp, axis=axis, r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                  return_std=return_std, num_threads=num_threads)

def _robust_mean_gpu(inp: IntArray | RealArray, axis: int | Tuple[int, ...]=0, r0: float=0.0,
                     r1: float=0.5, n_iter: int=12, lm: float=9.0,
                     return_std: bool=False) -> CPRealArray:
    if CuPy is None or cuda_median is None:
        raise RuntimeError("robust_mean is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    if isinstance(axis, int):
        axis = (axis,)

    inp = shift_axis(inp, axis)
    n_reduce = inp.shape[-1]

    if n_reduce <= 64:
        out_dtype = CuPy.result_type(inp.dtype, CuPy.float32)
        mean = CuPy.empty(inp.shape[:-1] + (1,), dtype=out_dtype)

        if return_std:
            std = CuPy.empty_like(mean)
            mean, std = cuda_median.robust_mean_std(mean, std, inp, r0, r1, n_iter, lm)
            return CuPy.stack([mean, std], axis=0)[..., 0]

        return cuda_median.robust_mean(mean, inp, r0, r1, n_iter, lm)[..., 0]

    mean = CuPy.median(inp, axis=-1, keepdims=True)
    j0, j1 = int(r0 * n_reduce), int(r1 * n_reduce)

    for _ in range(n_iter):
        errors = (inp - mean) * (inp - mean)
        idxs = CuPy.argpartition(errors, (j0, j1), axis=-1)
        mean = CuPy.mean(CuPy.take_along_axis(inp, idxs[..., j0:j1], axis=-1),
                         axis=-1, keepdims=True)

    errors = (inp - mean) * (inp - mean)
    idxs = CuPy.argsort(errors, axis=-1)

    if return_std:
        mean, std = cuda_median.inliers_mean_std(mean, CuPy.zeros_like(mean), inp,
                                                 errors, idxs, lm)
        return CuPy.stack([mean, std], axis=0)[..., 0]

    return cuda_median.inliers_mean(mean, inp, errors, idxs, lm)[..., 0]

@overload
def robust_mean(inp: NDIntArray | NDRealArray, axis: int | Tuple[int, ...]=0, r0: float=0.0,
                r1: float=0.5, n_iter: int = 12, lm: float=9.0, return_std: bool = False
                ) -> NDRealArray: ...

@overload
def robust_mean(inp: CPIntArray | CPRealArray, axis: int | Tuple[int, ...]=0, r0: float=0.0,
                r1: float=0.5, n_iter: int = 12, lm: float=9.0, return_std: bool = False
                ) -> CPRealArray: ...

@overload
def robust_mean(inp: JaxIntArray | JaxRealArray, axis: int | Tuple[int, ...]=0, r0: float=0.0,
                r1: float=0.5, n_iter: int = 12, lm: float=9.0, return_std: bool = False
                ) -> JaxRealArray: ...

@array_dispatch("inp", cpu_impl=_robust_mean_cpu, gpu_impl=_robust_mean_gpu)
def robust_mean(inp: IntArray | RealArray, axis: int | Tuple[int, ...]=0, r0: float=0.0,
                r1: float=0.5, n_iter: int = 12, lm: float=9.0, return_std: bool = False
                ) -> RealArray:
    """Calculate a mean along the axis by robustly fitting a Gaussian to input vector.

    The algorithm performs n_iter times the fast least kth order statistics (FLkOS) algorithm
    to fit a gaussian to data.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        inp: Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        axis: Array axes along which median values are calculated.
        r0: A lower bound guess of ratio of inliers. We'd like to make a sample out of worst
            inliers from data points that are between r0 and r1 of sorted residuals.
        r1: An upper bound guess of ratio of inliers. Choose the r0 to be as high as you are
            sure the ratio of data is inlier.
        n_iter: Number of iterations of fitting a gaussian with the FLkOS algorithm.
        lm: How far (normalized by STD of the Gaussian) from the mean of the Gaussian, data is
            considered inlier.
        return_std: Return robust estimate of standard deviation if True.

    Returns:
        Array of robust mean and robust standard deviation (if return_std is True).

    See Also:
        :func:`robust_lsq`: Robust least-squares solution.
    """
    ...

def _robust_lsq_cpu(W: IntArray | RealArray, y: IntArray | RealArray,
                    axis: IntSequence=-1, r0: float=0.0, r1: float=0.5,
                    n_iter: int=12, lm: float=9.0) -> NDRealArray:
    num_threads = get_cpu_config().effective_num_threads()
    return cpu_median.robust_lsq(W=W, y=y, axis=axis, r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                 num_threads=num_threads)

def _robust_lsq_gpu(W: IntArray | RealArray, y: IntArray | RealArray,
                    axis: int | Tuple[int, ...]=-1, r0: float=0.0, r1: float=0.5,
                    n_iter: int=12, lm: float=9.0) -> CPRealArray:
    if CuPy is None or cuda_median is None:
        raise RuntimeError("robust_lsq is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    if isinstance(axis, int):
        axis = (axis,)

    if tuple(y.shape[ax] for ax in axis) != W.shape[-len(axis):]:
        raise ValueError("Shape of y along specified axis must match shape of W")

    y = shift_axis(y, axis)
    n_reduce = y.shape[-1]

    W = xp.reshape(W, (prod(W.shape[:-len(axis)]), n_reduce))

    fits = xp.sum(y[..., None, :] * W, axis=-1) / xp.sum(W * W, axis=-1)
    j0, j1 = int(r0 * n_reduce), int(r1 * n_reduce)

    for _ in range(n_iter):
        errors = (y - xp.tensordot(fits, W, axes=(-1, 0)))**2
        idxs = xp.argpartition(errors, (j0, j1), axis=-1)
        fits = cuda_median.lsq(fits, W, y, idxs[..., j0:j1])

    errors = (y - xp.tensordot(fits, W, axes=(-1, 0)))**2
    idxs = xp.argsort(errors, axis=-1)
    return cuda_median.inliers_lsq(fits, W, y, errors, idxs, lm)

@array_dispatch("y", cpu_impl=_robust_lsq_cpu, gpu_impl=_robust_lsq_gpu)
def robust_lsq(W: RealArray | IntArray, y: RealArray | IntArray, axis: int | Tuple[int, ...] = -1,
               r0: float=0.0, r1: float=0.5, n_iter: int = 12, lm: float=9.0) -> RealArray:
    """Robustly solve a linear least-squares problem with the fast least kth order statistics
    (FLkOS) algorithm.

    Given a (N[0], .., N[ndim]) target vector y and a design matrix W of the shape
    (M, N[axis[0]], .., N[axis[-1]]), robust_lsq solves the following problems:

        for i in range(0, prod(N[~axis])):
            minimize ||W x - y[i]||**2

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        W: Design matrix of the shape (M, N[axis[0]], .., N[axis[-1]]).
        y: Target vector of the shape (N[0], .., N[ndim]).
        axis: Array axes along which the design matrix is fitted to the target.
        r0: A lower bound guess of ratio of inliers. We'd like to make a sample out of worst
            inliers from data points that are between r0 and r1 of sorted residuals.
        r1: An upper bound guess of ratio of inliers. Choose the r0 to be as high as you are
            sure the ratio of data is inlier.
        n_iter: Number of iterations of fitting a gaussian with the FLkOS algorithm.
        lm: How far (normalized by STD of the Gaussian) from the mean of the Gaussian, data is
            considered inlier.

    Returns:
        The least-squares solution x of the shape N[~axis].

    See Also:
        :func:`robust_mean`: Robust mean calculation.
    """
    ...

# New GPU-friendly streak detection algorithm

def _detect_peaks_cpu(data: RealArray, labeled: LabelResult, radius: int, vmin: float) -> NDIntArray:
    num_threads = get_cpu_config().effective_num_threads()
    radii = [0,] * (data.ndim - 2) + [1, 1]
    return streak_finder.detect_peaks(labeled.labels, data, Structure(radii, 1), radius, vmin,
                                      num_threads=num_threads)

def binned_shape(shape: Tuple[int, ...], radius: int) -> Tuple[int, ...]:
    return shape[:-2] + ((shape[-2] + radius - 1) // radius, (shape[-1] + radius - 1) // radius)

def _detect_peaks_gpu(data: RealArray, labeled: LabelResult, radius: int, vmin: float) -> CPIntArray:
    if cuda_streak_finder is None:
        raise RuntimeError("detect_peaks is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    radii = [0,] * (data.ndim - 2) + [1, 1]

    out = xp.empty(binned_shape(data.shape, radius), dtype=xp.int32)
    return cuda_streak_finder.detect_peaks(out, labeled.labels, data, Structure(radii, 1), radius,
                                           vmin)

@overload
def detect_peaks(data: NDRealArray, labeled: LabelResult, radius: int, vmin: float) -> NDIntArray: ...

@overload
def detect_peaks(data: CPRealArray, labeled: LabelResult, radius: int, vmin: float) -> CPIntArray: ...

@overload
def detect_peaks(data: JaxRealArray, labeled: LabelResult, radius: int, vmin: float
                     ) -> NDIntArray | CPIntArray: ...

@overload
def detect_peaks(data: RealArray, labeled: LabelResult, radius: int, vmin: float) -> NDIntArray | CPIntArray: ...

@array_dispatch("data", cpu_impl=_detect_peaks_cpu, gpu_impl=_detect_peaks_gpu)
def detect_peaks(data: RealArray, labeled: LabelResult, radius: int, vmin: float
                     ) -> NDIntArray | CPIntArray:
    """Find local maxima within each grid bin of the SNR frame stack.

    The frame is divided into a regular grid of *radius* x *radius* bins.
    Within each bin that overlaps at least one labeled foreground region, the
    algorithm searches for the brightest pixel that is a **strict local
    maximum** — its SNR value exceeds every neighbour in the 3x3 spatial
    neighbourhood. Each bin is assigned one of three states:

    * **Peak** (``index >= 0 and index < data.size``) — flat pixel index of
      the local maximum.
    * **Good** (``index == data.size``) — bin overlaps a foreground region
      but contains no strict local maximum; can receive a linelet via
      propagation in :func:`fit_linelets`.
    * **Bad** (``index == -1``) — no foreground pixels in the bin; ignored
      in all later stages.

    Args:
        data: SNR frame stack of shape ``(n_frames, *frame_shape)``.
        labeled: Foreground regions returned by :func:`label`.  Only bins
            that overlap at least one labeled region are searched for peaks.
        radius: Bin side length in pixels.  Sets the minimum spacing between
            peaks and should equal ``structure.connectivity``.
        vmin: SNR threshold.  Pixels below *vmin* are not considered as peak
            candidates.

    Returns:
        Integer array of shape ``(*data.shape[:-2], NY_bins, NX_bins)``
        containing the raw bin state for each grid bin (flat pixel index,
        ``data.size`` sentinel, or ``-1``).  Pass to :func:`peak_labels` to
        obtain the :class:`PeakLabels` descriptor.
    """
    ...

def _cpu_peak_labels(indices: NDIntArray, data: RealArray, radius: int) -> Tuple[PeakLabels, NDIntArray]:
    xp = NumPy
    labels = xp.full(indices.shape, -1, dtype=xp.int64)
    labels[indices == data.size] = 0
    mask = (indices >= 0) & (indices < data.size)

    n_good, n_labels = int(xp.sum(indices >= 0)), int(mask.sum())

    peaks = xp.full(n_good, -1, dtype=xp.int64)
    sort_indices = xp.argsort(data.ravel()[indices[mask]])[::-1]
    peaks[:n_labels] = indices[mask][sort_indices]

    inverse = xp.empty_like(sort_indices)
    inverse[sort_indices] = xp.arange(n_labels, dtype=xp.int64) + 1

    labels[mask] = inverse

    return PeakLabels(labels, n_labels, n_labels, n_good, radius), peaks

def _gpu_peak_labels(indices: CPIntArray, data: RealArray, radius: int) -> Tuple[PeakLabels, CPIntArray]:
    xp = CuPy
    labels = xp.full(indices.shape, -1, dtype=xp.int32)
    labels[indices == data.size] = 0
    mask = (indices >= 0) & (indices < data.size)

    n_good, n_labels = int(xp.sum(indices >= 0)), int(mask.sum())

    peaks = xp.full(n_good, -1, dtype=xp.int32)
    sort_indices = xp.argsort(data.ravel()[indices[mask]])[::-1]
    peaks[:n_labels] = indices[mask][sort_indices]

    inverse = xp.empty(sort_indices.shape, dtype=xp.int32)
    inverse[sort_indices] = xp.arange(n_labels, dtype=xp.int32) + 1

    labels[mask] = inverse

    return PeakLabels(labels, n_labels, n_labels, n_good, radius), peaks

@overload
def peak_labels(indices: NDIntArray, data: RealArray, radius: int
                ) -> Tuple[PeakLabels, NDIntArray]: ...

@overload
def peak_labels(indices: CPIntArray, data: RealArray, radius: int
                ) -> Tuple[PeakLabels, CPIntArray]: ...

@overload
def peak_labels(indices: JaxIntArray, data: RealArray, radius: int
                ) -> Tuple[PeakLabels, NDIntArray | CPIntArray]: ...

@overload
def peak_labels(indices: IntArray, data: RealArray, radius: int
                ) -> Tuple[PeakLabels, IntArray]: ...

@array_dispatch("indices", cpu_impl=_cpu_peak_labels, gpu_impl=_gpu_peak_labels)
def peak_labels(indices: IntArray, data: RealArray, radius: int
                ) -> Tuple[PeakLabels, IntArray]:
    """Convert raw bin-state indices from :func:`detect_peaks` into a
    :class:`PeakLabels` descriptor.

    Assigns a positive 1-based label to every peak bin (sorted by descending
    intensity so that the strongest peaks have the lowest label indices),
    0 to every *good* bin, and -1 to every *bad* bin.  Also collects the
    flat pixel indices of all detected peaks into a 1-D array sorted by
    descending SNR value.

    Args:
        indices: Raw bin-state array returned by :func:`detect_peaks`, shape
            ``(*data.shape[:-2], NY_bins, NX_bins)``.
        data: SNR frame stack used for sorting peaks by intensity.
        radius: Bin side length in pixels; stored in the returned
            :class:`PeakLabels` for use by downstream functions.

    Returns:
        A tuple ``(labels, peaks)`` where *labels* is a :class:`PeakLabels`
        describing the state of every grid bin, and *peaks* is a 1-D integer
        array of flat pixel indices of the detected peaks sorted by
        descending intensity.
    """
    ...

def _cpu_fit_linelets(labels: PeakLabels, peaks: NDIntArray, data: RealArray, structure: Structure,
                      vmin: float) -> Tuple[NDRealArray, PeakLabels]:
    num_threads = get_cpu_config().effective_num_threads()
    linelets, new_labels = streak_finder.line_fit(labels.to_tuple(), peaks, data, structure, vmin,
                                                  num_threads=num_threads)
    return linelets, PeakLabels(*new_labels)

def _gpu_fit_linelets(labels: PeakLabels, peaks: CPIntArray, data: RealArray, structure: Structure,
                      vmin: float) -> Tuple[CPRealArray, PeakLabels]:
    if cuda_streak_finder is None:
        raise RuntimeError("fit_linelets is not compiled for the current platform. "
                        "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    out = xp.zeros((labels.n_good, 4), dtype=data.dtype)
    linelets, new_labels = cuda_streak_finder.line_fit(out, labels, peaks, data, structure, vmin)
    return linelets, PeakLabels(*new_labels)

@overload
def fit_linelets(labels: PeakLabels, peaks: NDIntArray, data: RealArray, structure: Structure,
                 vmin: float) -> Tuple[NDRealArray, PeakLabels]: ...

@overload
def fit_linelets(labels: PeakLabels, peaks: CPIntArray, data: RealArray, structure: Structure,
                 vmin: float) -> Tuple[CPRealArray, PeakLabels]: ...

@overload
def fit_linelets(labels: PeakLabels, peaks: JaxIntArray, data: RealArray, structure: Structure,
                 vmin: float) -> Tuple[NDRealArray | CPRealArray, PeakLabels]: ...

@overload
def fit_linelets(labels: PeakLabels, peaks: IntArray, data: RealArray, structure: Structure,
                 vmin: float) -> Tuple[RealArray, PeakLabels]: ...

@array_dispatch("peaks", cpu_impl=_cpu_fit_linelets, gpu_impl=_gpu_fit_linelets)
def fit_linelets(labels: PeakLabels, peaks: IntArray, data: RealArray, structure: Structure,
                 vmin: float) -> Tuple[RealArray, PeakLabels]:
    """Fit a linelet to each reachable grid bin and propagate to neighbours.

    For every peak bin, a *linelet* — a short line segment
    ``(x0, y0, x1, y1)`` — is fitted to the intensity distribution in the
    local neighbourhood defined by *structure* using intensity-weighted image
    moments.  The linelet direction is the eigenvector of the covariance
    matrix of pixel coordinates (weighted by SNR) corresponding to the larger
    eigenvalue; the half-length equals the FWHM of the distribution along
    that axis.

    Starting from each peak, the algorithm then **propagates** the linelet
    table forward and backward: at each step it traces the current linelet
    direction to the nearest bin boundary (ray–boundary intersection), fits a
    linelet at the new location if the pixel is above *vmin*, and writes it
    into the table.  *Good* bins encountered during propagation are promoted
    to labeled peaks and become eligible seeds for :func:`detect_streaks`.

    Args:
        labels: Bin-state descriptor returned by :func:`peak_labels`.
        peaks: Flat pixel indices of detected peaks returned by
            :func:`peak_labels`, sorted by descending intensity.
        data: SNR frame stack used for moment computation and the *vmin*
            threshold.
        structure: Structuring element defining the local pixel neighbourhood
            for linelet fitting and the propagation step size (via
            ``structure.connectivity``).
        vmin: SNR threshold.  Only pixels at or above *vmin* are included in
            the moment calculation and propagation.

    Returns:
        A tuple ``(linelets, labels)`` where *linelets* is an array of shape
        ``(n_labels, 4)`` containing ``(x0, y0, x1, y1)`` endpoint
        coordinates for each linelet, and *labels* is an updated
        :class:`PeakLabels` whose ``n_labels`` has grown to include bins
        filled during propagation.
    """
    ...

def _cpu_detect_streaks(labels: PeakLabels, peaks: NDIntArray, linelets: NDRealArray,
                        data: RealArray, structure: Structure, vmin: float, xtol: float, nfa: int
                        ) -> NPStreaks:
    num_threads = get_cpu_config().effective_num_threads()
    return streak_finder.detect_streaks(labels.to_tuple(), peaks, linelets, data, structure, vmin,
                                        xtol, nfa, num_threads=num_threads)

def _gpu_detect_streaks(labels: PeakLabels, peaks: CPIntArray, linelets: CPRealArray,
                        data: RealArray, structure: Structure, vmin: float, xtol: float, nfa: int
                        ) -> CPStreaks:
    if cuda_streak_finder is None:
        raise RuntimeError("detect_streaks is not compiled for the current platform. "
                        "Please, check if you have installed the cbclib_v2 with GPU support.")

    return cuda_streak_finder.detect_streaks(labels.to_tuple(), peaks, linelets, data, structure,
                                             vmin, xtol, nfa)

@overload
def detect_streaks(labels: PeakLabels, peaks: NDIntArray, linelets: NDRealArray,
                   data: RealArray, structure: Structure, vmin: float, xtol: float, nfa: int
                   ) -> NPStreaks: ...

@overload
def detect_streaks(labels: PeakLabels, peaks: CPIntArray, linelets: CPRealArray,
                   data: RealArray, structure: Structure, vmin: float, xtol: float, nfa: int
                   ) -> CPStreaks: ...

@overload
def detect_streaks(labels: PeakLabels, peaks: JaxIntArray, linelets: JaxRealArray,
                   data: RealArray, structure: Structure, vmin: float, xtol: float, nfa: int
                   ) -> Streaks: ...

@overload
def detect_streaks(labels: PeakLabels, peaks: IntArray, linelets: RealArray,
                   data: RealArray, structure: Structure, vmin: float, xtol: float, nfa: int
                   ) -> Streaks: ...

@array_dispatch("peaks", cpu_impl=_cpu_detect_streaks, gpu_impl=_gpu_detect_streaks)
def detect_streaks(labels: PeakLabels, peaks: IntArray, linelets: RealArray, data: RealArray,
                   structure: Structure, vmin: float, xtol: float, nfa: int) -> Streaks:
    """Grow streaks from seed peaks by aggregating aligned linelet bins.

    For each seed peak (the first ``labels.n_seeds`` entries of *peaks*), a
    streak is initialised from that single bin and then extended one bin at a
    time in both directions.  The current extent of a streak is the line
    connecting the two outermost linelet endpoints across all bins in the
    streak.  A candidate neighbouring bin is accepted when its linelet
    satisfies a **self-consistency check**: both endpoints must lie within
    *xtol* pixels of this total streak line, with at most *nfa* endpoint
    violations tolerated across the whole streak.  Growth stops when no
    aligned neighbour can be found in either direction.

    Args:
        labels: Bin-state descriptor returned by :func:`fit_linelets`.
            Use :meth:`~PeakLabels.keep_best` to restrict seeds to the
            strongest peaks.
        peaks: Flat pixel indices of detected peaks returned by
            :func:`peak_labels`, sorted by descending intensity.
        linelets: Linelet endpoint array of shape ``(n_labels, 4)`` returned
            by :func:`fit_linelets`.
        data: SNR frame stack (used internally for neighbour lookup).
        structure: Structuring element defining the bin radius and pixel
            neighbourhood for streak extension.
        vmin: SNR threshold.  Pixels below *vmin* are not considered as
            valid streak bins.
        xtol: Collinearity tolerance in pixels.  A candidate bin is accepted
            only if all linelet endpoints of the updated streak lie within
            *xtol* of the total streak line.  A value of 0.5–0.75 x
            ``structure.connectivity`` works well in practice.
        nfa: Maximum number of false alarms — linelet endpoints allowed to
            exceed *xtol* while still being accepted.  ``nfa=0`` enforces
            strict collinearity; ``nfa=1`` allows one outlier endpoint.

    Returns:
        List of detected :class:`Streaks`, one per seed peak.
    """
    ...

def _cpu_to_lines(streaks: NPStreaks, labels: NDIntArray, linelets: NDRealArray) -> NDRealArray:
    return streaks.to_lines(labels, linelets)

def _gpu_to_lines(streaks: CPStreaks, labels: CPIntArray, linelets: CPRealArray) -> CPRealArray:
    if cuda_streak_finder is None:
        raise RuntimeError("to_lines is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    out = xp.empty((len(streaks), 4), dtype=linelets.dtype)
    return streaks.to_lines(out, labels, linelets)

@array_dispatch("labels", cpu_impl=_cpu_to_lines, gpu_impl=_gpu_to_lines)
def to_lines(streaks: Streaks, labels: IntArray, linelets: RealArray) -> RealArray:
    """Extract the endpoint coordinates of each detected streak.

    Each streak's extent is represented by the line connecting the two
    outermost linelet endpoints across all bins in that streak.  This
    function looks up those endpoints from the *linelets* table using the
    bin-label indices stored in each streak object.

    Args:
        streaks: Detected streaks returned by :func:`detect_streaks`.
        labels: Bin-label array from :class:`PeakLabels` (``labels.labels``),
            used to map bin indices to linelet table entries.
        linelets: Linelet endpoint array of shape ``(n_labels, 4)`` returned
            by :func:`fit_linelets`.

    Returns:
        Array of shape ``(n_streaks, 4)`` where each row is
        ``(x0, y0, x1, y1)``, the pixel coordinates of the two endpoints
        of the fitted line segment for that streak.
    """
    ...

def _cpu_n_signal(streaks: NPStreaks, labels: PeakLabels, peaks: NDIntArray, data: RealArray,
                  structure: Structure, vmin: float) -> NDIntArray:
    num_threads = get_cpu_config().effective_num_threads()
    return streak_finder.n_signal(streaks, labels.to_tuple(), peaks, data, structure, vmin,
                                  num_threads=num_threads)

def _gpu_n_signal(streaks: CPStreaks, labels: PeakLabels, peaks: CPIntArray, data: RealArray,
                  structure: Structure, vmin: float) -> CPIntArray:
    if cuda_streak_finder is None:
        raise RuntimeError("n_signal is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    out = xp.empty(len(streaks), dtype=xp.uint32)
    return cuda_streak_finder.n_signal(out, streaks, labels.to_tuple(), peaks, data, structure,
                                       vmin)

@overload
def n_signal(streaks: NPStreaks, labels: PeakLabels, peaks: NDIntArray, data: RealArray,
             structure: Structure, vmin: float) -> NDIntArray: ...

@overload
def n_signal(streaks: CPStreaks, labels: PeakLabels, peaks: CPIntArray, data: RealArray,
             structure: Structure, vmin: float) -> CPIntArray: ...

@overload
def n_signal(streaks: Streaks, labels: PeakLabels, peaks: JaxIntArray, data: RealArray,
             structure: Structure, vmin: float) -> NDIntArray | CPIntArray: ...

@overload
def n_signal(streaks: Streaks, labels: PeakLabels, peaks: IntArray, data: RealArray,
             structure: Structure, vmin: float) -> IntArray: ...

@array_dispatch("peaks", cpu_impl=_cpu_n_signal, gpu_impl=_gpu_n_signal)
def n_signal(streaks: Streaks, labels: PeakLabels, peaks: IntArray, data: RealArray,
             structure: Structure, vmin: float) -> IntArray:
    """Count above-threshold pixels in each streak's footprint.

    The footprint of a streak is the union of the *structure* neighbourhood
    around every peak pixel belonging to that streak.  Pixels appearing in
    more than one peak's neighbourhood are counted only once.  The resulting
    count is used by :func:`~cbclib_v2.streak_finder.PatternStreakFinder.ranking`
    to rank streaks by signal strength.

    Args:
        streaks: Detected streaks returned by :func:`detect_streaks`.
        labels: Bin-state descriptor returned by :func:`peak_labels`.
        peaks: Flat pixel indices of detected peaks returned by
            :func:`peak_labels`.
        data: SNR frame stack used for the *vmin* threshold check.
        structure: Structuring element defining the pixel neighbourhood
            around each peak that forms the streak's footprint.
        vmin: SNR threshold.  A footprint pixel is counted as signal if its
            value is ≥ *vmin*.

    Returns:
        Integer array of length ``len(streaks)`` with the number of
        above-threshold pixels in each streak's footprint.
    """
    ...

def _streak_labels_cpu(out: NDIntArray, streaks: NPStreaks, ranks: IntArray, labels: PeakLabels,
                       peaks: IntArray, structure: Structure) -> NDIntArray:
    num_threads = get_cpu_config().effective_num_threads()
    return streak_finder.streak_labels(out, streaks, ranks, labels, peaks, structure,
                                            num_threads=num_threads)

def _streak_labels_gpu(out: CPIntArray, streaks: CPStreaks, ranks: IntArray, labels: PeakLabels,
                       peaks: IntArray, structure: Structure) -> CPIntArray:
    if cuda_streak_finder is None:
        raise RuntimeError("streaks_labels is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    return cuda_streak_finder.streak_labels(out.astype(xp.int32), streaks, ranks.astype(xp.int32),
                                            labels, peaks, structure)

@overload
def streak_labels(out: NDIntArray, streaks: NPStreaks, ranks: IntArray, labels: PeakLabels,
                  peaks: IntArray, structure: Structure) -> NDIntArray: ...

@overload
def streak_labels(out: CPIntArray, streaks: CPStreaks, ranks: IntArray, labels: PeakLabels,
                  peaks: IntArray, structure: Structure) -> CPIntArray: ...

@overload
def streak_labels(out: IntArray, streaks: Streaks, ranks: IntArray, labels: PeakLabels,
                  peaks: IntArray, structure: Structure) -> IntArray: ...

@array_dispatch("out", cpu_impl=_streak_labels_cpu, gpu_impl=_streak_labels_gpu)
def streak_labels(out: IntArray, streaks: Streaks, ranks: IntArray, labels: PeakLabels,
                  peaks: IntArray, structure: Structure) -> IntArray:
    """Paint ranked streaks onto a label image.

    For each streak, every pixel in its footprint (the *structure*
    neighbourhood around its peak pixels) is written to *out* with the
    streak's rank + 1 as its label.  When two streaks share a footprint
    pixel the one with the **lower rank** (higher signal count) wins,
    so stronger streaks are never overwritten by weaker ones.

    The result is a raw integer image suitable for re-labeling with
    :func:`label` to obtain clean connected regions for line fitting.

    Args:
        out: Pre-allocated integer array of the same shape as the SNR frame
            stack; initialised to 0 (background) before painting.
        streaks: Detected streaks returned by :func:`detect_streaks`.
        ranks: Rank array of length ``len(streaks)``; rank 0 is the strongest
            streak.  Typically produced by
            :meth:`~cbclib_v2.streak_finder.PatternStreakFinder.ranking`.
        labels: Bin-state descriptor returned by :func:`peak_labels`.
        peaks: Flat pixel indices of detected peaks returned by
            :func:`peak_labels`.
        structure: Structuring element defining the footprint around each
            peak pixel.

    Returns:
        *out* with streak footprints painted in rank order; background
        pixels remain 0.
    """
    ...
