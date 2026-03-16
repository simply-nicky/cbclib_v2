"""Multidimensional image processing functions with CPU/CUDA backend support.

This module provides image processing operations that automatically dispatch to CPU or CUDA
backends based on the device context set via :mod:`cbclib_v2.device`.

See Also:
    :mod:`cbclib_v2.device`: Device context management for backend selection.
"""
from functools import wraps
from inspect import signature
from math import prod
from typing import Callable, NamedTuple, Optional, Protocol, Sequence, Tuple, cast, overload
import warnings
from .annotations import (Array, BoolArray, CPArray, CPBoolArray, CPIntArray, CPRealArray,
                          CuPy, IntArray, IntSequence, JaxArray, JaxBoolArray, JaxIntArray,
                          JaxNumPy, JaxRealArray, NDArray, NDBoolArray, NDIntArray,
                          NDRealArray, NumPy, RealArray)
from .array_api import array_namespace, ascupy, asjax, asnumpy, get_platform
from .config import get_cpu_config
from .src import bresenham, label as cpu_label, median as cpu_median, streak_finder
from .src.label import Structure, LabelResult as NPLabelResult
from .src.streak_finder import PatternList, Peaks, PeaksList

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

if CuPy is not None:
    from .src import cuda_draw_lines, cuda_label, cuda_median
    from cupyx.scipy import ndimage as _ndimage

    class NDImageProtocol(Protocol):
        def binary_dilation(self, input: BoolArray, structure: BoolArray, iterations: int=1,
                            mask: BoolArray | None=None, output: BoolArray | None=None,
                            brute_force: bool=False) -> CPBoolArray: ...

        def sum_labels(self, input: RealArray, labels: IntArray | None,
                       index: IntSequence | None=None) -> CPRealArray: ...

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
    cupy_ndimage = None

    def binary_structure(structure: Structure, shape: Sequence[int] | None=None) -> CPBoolArray:
        raise RuntimeError("CUDA backend is not available. Please, check if you have installed " \
                           "the cbclib_v2 with GPU support.")

def _accumulate_lines_cpu(out: RealArray, lines: RealArray, terms: IntArray,
                          frames: IntArray, max_val: float=1.0,
                          kernel: str='rectangular', in_overlap: str='sum',
                          out_overlap: str='sum') -> NDRealArray:
    num_threads = get_cpu_config().effective_num_threads()
    lines, terms, frames = asnumpy(lines), asnumpy(terms), asnumpy(frames)
    return bresenham.accumulate_lines(out=out, lines=lines, terms=terms, frames=frames,
                                      max_val=max_val, kernel=kernel,
                                      in_overlap=in_overlap, out_overlap=out_overlap,
                                      num_threads=num_threads)

def _accumulate_lines_gpu(out: RealArray, lines: RealArray, terms: IntArray,
                          frames: IntArray, max_val: float=1.0,
                          kernel: str='rectangular', in_overlap: str='sum',
                          out_overlap: str='sum') -> CPRealArray:
    if cuda_draw_lines is None:
        raise RuntimeError("accumulate_lines is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    lines, terms, frames = ascupy(lines), ascupy(terms), ascupy(frames)
    return cuda_draw_lines.accumulate_lines(out=out, lines=lines, terms=terms, frames=frames,
                                            max_val=max_val, kernel=kernel,
                                            in_overlap=in_overlap, out_overlap=out_overlap)

@overload
def accumulate_lines(out: NDRealArray, lines: NDRealArray, terms: NDIntArray,
                     frames: NDIntArray, max_val: float=1.0, kernel: str='rectangular',
                     in_overlap: str='sum', out_overlap: str='sum') -> NDRealArray: ...

@overload
def accumulate_lines(out: CPRealArray, lines: CPRealArray, terms: CPIntArray,
                     frames: CPIntArray, max_val: float=1.0, kernel: str='rectangular',
                     in_overlap: str='sum', out_overlap: str='sum') -> CPRealArray: ...

@overload
def accumulate_lines(out: JaxRealArray, lines: JaxRealArray, terms: JaxIntArray,
                     frames: JaxIntArray, max_val: float=1.0, kernel: str='rectangular',
                     in_overlap: str='sum', out_overlap: str='sum') -> JaxRealArray: ...

@overload
def accumulate_lines(out: Array, lines: Array, terms: IntArray, frames: IntArray,
                     max_val: float=1.0, kernel: str='rectangular', in_overlap: str='sum',
                     out_overlap: str='sum') -> RealArray: ...

@array_dispatch("out", cpu_impl=_accumulate_lines_cpu, gpu_impl=_accumulate_lines_gpu)
def accumulate_lines(out: RealArray, lines: RealArray, terms: IntArray, frames: IntArray,
                     max_val: float=1.0, kernel: str='rectangular', in_overlap: str='sum',
                     out_overlap: str='sum') -> RealArray:
    """Accumulate thick lines with variable thickness across multiple frames.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        out: Output array where the lines will be accumulated.
        lines: A dictionary of the detected lines. Each array of lines must have a shape of
            (N, 5), where N is the number of lines. Each line is comprised of 5 parameters
            as follows:

            * [x0, y0], [x1, y1] : The coordinates of the line's ends.
            * width : Line's width.

        terms: Term indices specifying to which term each line belongs.
        frames: Frame indices specifying to which frame each term belongs.
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
        Output array with the lines accumulated.

    See Also:
        :func:`draw_lines`: Draw lines on a single frame.
    """
    ...

def _draw_lines_cpu(out: RealArray, lines: RealArray, idxs: IntArray | None=None,
                    max_val: float=1.0, kernel: str='rectangular',
                    overlap: str='sum') -> NDRealArray:
    num_threads = get_cpu_config().effective_num_threads()
    lines = asnumpy(lines)
    idxs = asnumpy(idxs) if idxs is not None else None
    return bresenham.draw_lines(out=out, lines=lines, idxs=idxs, max_val=max_val,
                                kernel=kernel, overlap=overlap, num_threads=num_threads)

def _draw_lines_gpu(out: RealArray, lines: RealArray, idxs: IntArray | None=None,
                    max_val: float=1.0, kernel: str='rectangular',
                    overlap: str='sum') -> CPRealArray:
    if cuda_draw_lines is None:
        raise RuntimeError("draw_lines is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    lines = ascupy(lines)
    idxs = ascupy(idxs) if idxs is not None else None
    return cuda_draw_lines.draw_lines(out=out, lines=lines, idxs=idxs, max_val=max_val,
                                      kernel=kernel, overlap=overlap)

@overload
def draw_lines(out: NDRealArray, lines: NDRealArray, idxs: NDIntArray | None=None,
               max_val: float=1.0, kernel: str='rectangular', overlap: str='sum'
               ) -> NDRealArray: ...

@overload
def draw_lines(out: CPRealArray, lines: CPRealArray, idxs: CPIntArray | None=None,
               max_val: float=1.0, kernel: str='rectangular', overlap: str='sum'
               ) -> CPRealArray: ...

@overload
def draw_lines(out: JaxRealArray, lines: JaxRealArray, idxs: JaxIntArray | None=None,
               max_val: float=1.0, kernel: str='rectangular', overlap: str='sum'
               ) -> JaxRealArray: ...

@overload
def draw_lines(out: Array, lines: Array, idxs: IntArray | None=None, max_val: float=1.0,
               kernel: str='rectangular', overlap: str='sum') -> RealArray: ...

@array_dispatch("out", cpu_impl=_draw_lines_cpu, gpu_impl=_draw_lines_gpu)
def draw_lines(out: RealArray, lines: RealArray, idxs: IntArray | None=None, max_val: float=1.0,
               kernel: str='rectangular', overlap: str='sum') -> RealArray:
    """Draw thick lines with variable thickness and antialiasing.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        out: Output array to draw lines on.
        lines: Array of shape (N, 5) with [x0, y0, x1, y1, width] for each line.
        idxs: Optional frame indices for each line. If None, all lines drawn to single frame.
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

    Args:
        inp: Input binary array.
        structure: Structuring element used for dilation.
        iterations: Number of dilation iterations.
        mask: Optional mask to limit dilation area.

    Returns:
        Dilated binary array.
    """
    ...

class CPLabelResult(NamedTuple):
    labels      : CPIntArray
    index       : CPIntArray

LabelResult = NPLabelResult | CPLabelResult

def _label_cpu(inp: Array, structure: Structure, npts: int=1) -> NPLabelResult:
    num_threads = get_cpu_config().effective_num_threads()
    return cpu_label.label(inp=inp, structure=structure, npts=npts,
                           num_threads=num_threads)

def _label_gpu(inp: Array, structure: Structure, npts: int=1) -> LabelResult:
    if cuda_label is None:
        raise RuntimeError("label is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    labels, n_labels = cuda_label.label(inp=inp, structure=structure, npts=npts)
    return CPLabelResult(labels=labels, index=CuPy.arange(1, n_labels + 1, dtype=int))

@overload
def label(inp: NDRealArray, structure: Structure, npts: int=1) -> NPLabelResult: ...

@overload
def label(inp: CPRealArray, structure: Structure, npts: int=1) -> CPLabelResult: ...

@overload
def label(inp: JaxRealArray, structure: Structure, npts: int=1) -> LabelResult: ...

@overload
def label(inp: Array, structure: Structure, npts: int=1) -> LabelResult: ...

@array_dispatch("inp", cpu_impl=_label_cpu, gpu_impl=_label_gpu)
def label(inp: Array, structure: Structure, npts: int=1) -> LabelResult:
    """Label connected components in a binary array.

    Args:
        inp: Input binary array.
        structure: Structuring element defining connectivity.
        npts: Minimum number of points for a region to be labeled.

    Returns:
        List of labeled regions.
    """
    ...

def _center_of_mass_cpu(labels: NPLabelResult, data: RealArray) -> NDRealArray:
    return cpu_label.center_of_mass(labels=labels, data=data)

def _center_of_mass_gpu(labels: CPLabelResult, data: RealArray) -> CPRealArray:
    if cupy_ndimage is None:
        raise RuntimeError("center_of_mass is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    if labels.index.size == 0:
        return xp.empty((0, data.ndim), dtype=data.dtype)

    mu = cupy_ndimage.sum_labels(data, labels=labels.labels, index=labels.index)
    grids = xp.ogrid[[slice(0, s) for s in data.shape]]
    mu_x = []
    for dim in range(data.ndim):
        x = cupy_ndimage.sum_labels(grids[dim] * data, labels=labels.labels, index=labels.index)
        mu_x.append(x)
    return xp.stack(mu_x, axis=-1) / mu[:, None]

@overload
def center_of_mass(labels: NPLabelResult, data: NDRealArray) -> NDRealArray: ...

@overload
def center_of_mass(labels: CPLabelResult, data: CPRealArray) -> CPRealArray: ...

@overload
def center_of_mass(labels: LabelResult, data: JaxRealArray) -> JaxRealArray: ...

@overload
def center_of_mass(labels: LabelResult, data: RealArray) -> RealArray: ...

@array_dispatch("data", cpu_impl=_center_of_mass_cpu, gpu_impl=_center_of_mass_gpu)
def center_of_mass(labels: LabelResult, data: RealArray) -> RealArray:
    """Calculate center of mass for each labeled region.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        labels: Labeled regions.
        data: Input data array.

    Returns:
        Array of center of mass coordinates for each region.
    """
    ...

def _covariance_matrix_cpu(labels: NPLabelResult, data: RealArray) -> NDRealArray:
    matrices = cpu_label.covariance_matrix(labels=labels, data=data)
    return matrices.reshape(-1, data.ndim, data.ndim)

def _covariance_matrix_gpu(labels: CPLabelResult, data: RealArray) -> CPRealArray:
    if cupy_ndimage is None:
        raise RuntimeError("covariance_matrix is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    if labels.index.size == 0:
        return xp.empty((0, data.ndim, data.ndim), dtype=data.dtype)

    mu = cupy_ndimage.sum_labels(data, labels=labels.labels, index=labels.index)
    grids = xp.ogrid[[slice(0, s) for s in data.shape]]
    mu_x = xp.stack([cupy_ndimage.sum_labels(grids[dim] * data, labels=labels.labels,
                                             index=labels.index)
                     for dim in range(data.ndim)])
    mu_x /= mu
    mu_xx = xp.stack([cupy_ndimage.sum_labels(grids[dim // data.ndim] *
                                              grids[dim % data.ndim] * data,
                                              labels=labels.labels, index=labels.index)
                      for dim in range(data.ndim * data.ndim)])
    mu_xx = xp.reshape(mu_xx, (data.ndim, data.ndim, -1)) / mu
    return xp.permute_dims((mu_xx - mu_x[:, None, :] * mu_x[None, :, :]), (2, 0, 1))

@overload
def covariance_matrix(labels: NPLabelResult, data: NDRealArray) -> NDRealArray: ...

@overload
def covariance_matrix(labels: CPLabelResult, data: CPRealArray) -> CPRealArray: ...

@overload
def covariance_matrix(labels: LabelResult, data: JaxRealArray) -> JaxRealArray: ...

@overload
def covariance_matrix(labels: LabelResult, data: RealArray) -> RealArray: ...

@array_dispatch("data", cpu_impl=_covariance_matrix_cpu, gpu_impl=_covariance_matrix_gpu)
def covariance_matrix(labels: LabelResult, data: RealArray) -> RealArray:
    """Calculate covariance matrix for each labeled region.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        labels: Labeled regions.
        data: Input data array.

    Returns:
        Array of covariance matrices for each region.
    """
    ...

@overload
def index(labels: NPLabelResult) -> NDIntArray: ...

@overload
def index(labels: CPLabelResult) -> CPIntArray: ...

def index(labels: LabelResult) -> NDIntArray | CPIntArray:
    """Get array of region indices."""
    if isinstance(labels, NPLabelResult):
        return NumPy.arange(1, len(labels.regions) + 1, dtype=int)
    if isinstance(labels, CPLabelResult):
        return labels.index
    raise ValueError("Invalid labels type. Expected NPLabelResult or CPLabelResult.")

@overload
def labels(labels: NPLabelResult) -> NDIntArray: ...

@overload
def labels(labels: CPLabelResult) -> CPIntArray: ...

def labels(labels: LabelResult) -> NDIntArray | CPIntArray:
    """Get array of region labels."""
    if isinstance(labels, NPLabelResult):
        return labels.to_mask(index(labels))
    if isinstance(labels, CPLabelResult):
        return labels.labels
    raise ValueError("Invalid labels type. Expected NPLabelResult or CPLabelResult.")

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
def ellipse_fit(labels: NPLabelResult, data: NDRealArray) -> NDRealArray: ...

@overload
def ellipse_fit(labels: CPLabelResult, data: CPRealArray) -> CPRealArray: ...

@overload
def ellipse_fit(labels: LabelResult, data: JaxRealArray) -> JaxRealArray: ...

@overload
def ellipse_fit(labels: LabelResult, data: RealArray) -> RealArray: ...

def ellipse_fit(labels: LabelResult, data: RealArray) -> RealArray:
    """ Fit ellipses to connected 2D regions in data. The fitted ellipse is defined by its
    major and minor axes (FWHM) and orientation.

    Parameters:
        labels: List of connected 2D regions.
        data: List of 2D arrays of data values.

    Returns:
        Array of shape (N, 3) where N is the number of regions. Each ellipse is represented by
        (a, b, theta) where a and b are the FWHM of the major and minor axes, and theta is the
        orientation angle in radians.
    """
    covmat = covariance_matrix(labels, data)
    return to_ellipse(covmat)

def to_line(centers: RealArray, matrix: RealArray) -> RealArray:
    xp = array_namespace(centers, matrix)
    if centers.size == 0 and matrix.size == 0:
        return xp.empty((0, 4), dtype=centers.dtype)

    mu_xx, mu_xy, mu_yy = matrix[..., -1, -1], matrix[..., -1, -2], matrix[..., -2, -2]
    theta = 0.5 * xp.atan2(2 * mu_xy, (mu_xx - mu_yy))

    tau = xp.zeros(centers.shape, dtype=centers.dtype)
    tau[..., 0] = xp.cos(theta)
    tau[..., 1] = xp.sin(theta)

    delta = xp.sqrt(4 * mu_xy**2 + (mu_xx - mu_yy)**2)
    hw = xp.sqrt(2 * xp.log(2) * (mu_xx + mu_yy + delta))
    return xp.concat((centers[..., ::-1] + hw[..., None] * tau,
                      centers[..., ::-1] - hw[..., None] * tau), axis=-1)

@overload
def line_fit(labels: NPLabelResult, data: NDRealArray) -> NDRealArray: ...

@overload
def line_fit(labels: LabelResult, data: JaxRealArray) -> JaxRealArray: ...

@overload
def line_fit(labels: CPLabelResult, data: CPRealArray) -> CPRealArray: ...

@overload
def line_fit(labels: LabelResult, data: RealArray) -> RealArray: ...

def line_fit(labels: LabelResult, data: RealArray) -> RealArray:
    """ Fit lines to connected 2D regions in data. The fitted line equals to the major axis of the
    covariance ellipse.

    Parameters:
        labels: List of connected 2D regions.
        data: 2D array of data values.

    Returns:
        Array of shape (N, 4) where N is the number of regions. Each line is represented by
        (x1, y1, x2, y2) coordinates of its endpoints.
    """
    if isinstance(labels, NPLabelResult):
        return cpu_label.line_fit(labels, data)
    if isinstance(labels, CPLabelResult):
        centers = center_of_mass(labels, data)
        covmat = covariance_matrix(labels, data)
        return to_line(centers, covmat)
    raise ValueError("Invalid labels type. Expected NPLabelResult or CPLabelResult.")

@overload
def median(inp: NDRealArray, axis: IntSequence=0) -> NDRealArray: ...

@overload
def median(inp: NDIntArray, axis: IntSequence=0) -> NDIntArray: ...

@overload
def median(inp: CPRealArray, axis: IntSequence=0) -> CPRealArray: ...

@overload
def median(inp: CPIntArray, axis: IntSequence=0) -> CPIntArray: ...

@overload
def median(inp: JaxRealArray, axis: IntSequence=0) -> JaxRealArray: ...

@overload
def median(inp: JaxIntArray, axis: IntSequence=0) -> JaxIntArray: ...

def median(inp: RealArray | IntArray, axis: IntSequence=0) -> RealArray | IntArray:
    """Calculate a median along the axis.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        inp: Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        axis: Array axes along which median values are calculated.

    Returns:
        Array of medians along the given axis.

    See Also:
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

    mean = CuPy.median(inp, axis=-1, keepdims=True)
    j0, j1 = int(r0 * n_reduce), int(r1 * n_reduce)

    for _ in range(n_iter):
        errors = (inp - mean) * (inp - mean)
        # idxs = CuPy.argsort(errors, axis=-1)
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
        # idxs = xp.argsort(errors, axis=-1)
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

def _detect_peaks_cpu(data: RealArray, radius: int, vmin: float, axes: Tuple[int, int] | None=None
                      ) -> PeaksList:
    num_threads = get_cpu_config().effective_num_threads()
    return streak_finder.detect_peaks(data=data, structure=Structure([1, 1], 1), radius=radius,
                                      vmin=vmin, axes=axes, num_threads=num_threads)

def _detect_peaks_gpu(data: RealArray, radius: int, vmin: float, axes: Tuple[int, int] | None=None
                      ) -> CPIntArray:
    if cuda_label is None:
        raise RuntimeError("detect_peaks is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    if axes is not None:
        data = shift_axis(data, axes)

    xp = CuPy
    peaks = xp.zeros(data.shape[:-2] + (data.shape[-2] // radius, data.shape[-1] // radius),
                     dtype=int)
    return cuda_label.detect_peaks(peaks=peaks, data=data, structure=Structure([1, 1], 1),
                                   radius=radius, vmin=vmin)

@overload
def detect_peaks(data: NDRealArray, radius: int, vmin: float, axes: Tuple[int, int] | None=None
                 ) -> PeaksList: ...

@overload
def detect_peaks(data: CPRealArray, radius: int, vmin: float, axes: Tuple[int, int] | None=None
                 ) -> CPIntArray: ...

@overload
def detect_peaks(data: JaxRealArray, radius: int, vmin: float, axes: Tuple[int, int] | None=None
                 ) -> CPIntArray | PeaksList: ...

@overload
def detect_peaks(data: RealArray, radius: int, vmin: float, axes: Tuple[int, int] | None=None
                 ) -> CPIntArray | PeaksList: ...

@array_dispatch("data", cpu_impl=_detect_peaks_cpu, gpu_impl=_detect_peaks_gpu)
def detect_peaks(data: RealArray, radius: int, vmin: float,axes: Tuple[int, int] | None=None
                 ) -> CPIntArray | PeaksList:
    """Detect sparse peaks in a set of images. The minimal distance between peaks is controlled
    by the radius parameter.

    Args:
        data: Input data array.
        radius: Minimum distance between peaks. The distance is measured as a number of pixels
            along the axes specified by the axes parameter.
        vmin: Minimum value to consider a pixel as part of a peak.
        axes: Axes along which to detect peaks. If None, last two axes are used.

    Returns:
        List of detected peaks with their properties.
    """
    ...

def _filter_peaks_cpu(peaks: PeaksList, data: RealArray, structure: Structure, vmin: float,
                      npts: int, axes: Tuple[int, int] | None=None) -> PeaksList:
    streak_finder.filter_peaks(peaks=peaks, data=data, structure=structure, vmin=vmin, npts=npts,
                               axes=axes)
    return peaks

def _filter_peaks_gpu(peaks: CPIntArray, data: RealArray, structure: Structure, vmin: float,
                      npts: int, axes: Tuple[int, int] | None=None) -> CPIntArray:
    if cuda_label is None:
        raise RuntimeError("filter_peaks is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    if axes is not None:
        data = shift_axis(data, axes)
    if structure.rank != 2:
        raise ValueError("Only 2D connectivity structure is supported for filter_peaks.")

    xp = CuPy
    structure = structure.expand_dims(list(range(data.ndim - 2)))
    labels, _ = cuda_label.label(data > vmin, structure, npts)
    mask = labels.reshape(-1)[peaks] > 0
    return xp.where(mask, peaks, -1)

@overload
def filter_peaks(peaks: PeaksList, data: NDRealArray, structure: Structure, vmin: float, npts: int,
                 axes: Tuple[int, int] | None=None) -> PeaksList: ...

@overload
def filter_peaks(peaks: CPIntArray, data: CPRealArray, structure: Structure, vmin: float, npts: int,
                 axes: Tuple[int, int] | None=None) -> CPIntArray: ...

@overload
def filter_peaks(peaks: CPIntArray | PeaksList, data: JaxRealArray, structure: Structure, vmin: float,
                 npts: int, axes: Tuple[int, int] | None=None) -> CPIntArray | PeaksList: ...

@overload
def filter_peaks(peaks: CPIntArray | PeaksList, data: RealArray, structure: Structure, vmin: float,
                 npts: int, axes: Tuple[int, int] | None=None) -> CPIntArray | PeaksList: ...

@array_dispatch("data", cpu_impl=_filter_peaks_cpu, gpu_impl=_filter_peaks_gpu)
def filter_peaks(peaks: CPIntArray | PeaksList, data: RealArray, structure: Structure, vmin: float,
                 npts: int, axes: Tuple[int, int] | None=None) -> CPIntArray | PeaksList:
    """Filter peaks by their local connectivity structure. A peak is kept if there are at least
    npts connected pixels in the neighborhood above the vmin threshold.

    Args:
        peaks : A set of peaks to be filtered.
        data : A 2D rasterised image.
        structure : A connectivity structure.
        vmin : Value threshold.
        npts : Size threshold. A peak is kept if there are at least npts connected pixels in its
            neighborhood above vmin.
        axes: Axes along which to filter peaks. If None, last two axes are used.

    Returns:
        A list of filtered peaks.
    """
    ...

@overload
def peaks_mask(peaks: PeaksList, data: RealArray) -> NDBoolArray: ...

@overload
def peaks_mask(peaks: CPIntArray, data: RealArray) -> CPBoolArray: ...

def peaks_mask(peaks: CPIntArray | PeaksList, data: RealArray) -> NDBoolArray | CPBoolArray:
    """Get indices of peaks in the original data array."""
    if isinstance(peaks, PeaksList):
        xp = NumPy
        frames = peaks.index()
        frame_indices = xp.concat([xp.asarray(list(pattern), dtype=int) for pattern in peaks])
        indices = xp.unravel_index(frames, data.shape[:-2])
        indices += xp.unravel_index(frame_indices, data.shape[-2:])
        mask = xp.zeros(data.shape, dtype=bool)
        mask[indices] = True
        return mask
    if isinstance(peaks, CPIntArray):
        xp = CuPy
        indices = peaks[peaks >= 0]
        mask = xp.zeros(data.shape, dtype=bool)
        mask[xp.unravel_index(indices, data.shape)] = True
        return mask
    raise ValueError("Invalid peaks type. Expected CPIntArray or PeaksList.")

def to_peaks_list(peaks: CPIntArray, n_frames: int, shape: Tuple[int, int], radius: int) -> PeaksList:
    xp = CuPy
    indices = peaks[peaks > 0]
    frames, frame_indices = divmod(indices, shape[0] * shape[1])
    offsets = xp.searchsorted(frames, xp.arange(0, n_frames + 1))
    starts, ends = offsets[:-1], offsets[1:]
    return PeaksList(Peaks(asnumpy(frame_indices[start:end]), shape, radius)
                     for start, end in zip(starts, ends))

def _detect_streaks_cpu(peaks: PeaksList, data: RealArray, structure: Structure, xtol: float,
                        vmin: float, min_size: int, lookahead: int=0, nfa: int=0,
                        axes: Tuple[int, int] | None=None) -> PatternList:
    num_threads = get_cpu_config().effective_num_threads()
    p0 = streak_finder.p0_values(data, vmin, axes, num_threads=num_threads)
    return streak_finder.detect_streaks(peaks, p0, data, structure, xtol, vmin, min_size,
                                        lookahead, nfa, num_threads=num_threads)

def _detect_streaks_gpu(peaks: CPIntArray, data: RealArray, structure: Structure, xtol: float,
                        vmin: float, min_size: int, lookahead: int=0, nfa: int=0,
                        axes: Tuple[int, int] | None=None) -> PatternList:
    if CuPy is None:
        raise RuntimeError("detect_streaks is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    if axes is None:
        axes = (-2, -1)

    xp = CuPy
    shape = (data.shape[axes[0]], data.shape[axes[1]])
    n_frames = data.size // prod(shape)

    n_signal = xp.sum(data >= vmin, axis=axes).reshape(-1)
    p0 = n_signal / prod(shape)

    num_threads = get_cpu_config().effective_num_threads()
    plist = to_peaks_list(peaks, n_frames, shape, structure.connectivity)
    return streak_finder.detect_streaks(plist, asnumpy(p0), asnumpy(data), structure, xtol, vmin,
                                        min_size, lookahead, nfa, num_threads=num_threads)

@overload
def detect_streaks(peaks: PeaksList, data: NDRealArray, structure: Structure, xtol: float, vmin: float,
                   min_size: float, lookahead: int=0, nfa: int=0, axes: Tuple[int, int] | None=None
                   ) -> PatternList: ...

@overload
def detect_streaks(peaks: CPIntArray, data: CPRealArray, structure: Structure, xtol: float, vmin: float,
                   min_size: float, lookahead: int=0, nfa: int=0, axes: Tuple[int, int] | None=None
                   ) -> PatternList: ...

@overload
def detect_streaks(peaks: CPIntArray | PeaksList, data: JaxRealArray, structure: Structure, xtol: float,
                   vmin: float, min_size: float, lookahead: int=0, nfa: int=0,
                   axes: Tuple[int, int] | None=None) -> PatternList: ...

@overload
def detect_streaks(peaks: CPIntArray | PeaksList, data: RealArray, structure: Structure, xtol: float,
                   vmin: float, min_size: float, lookahead: int=0, nfa: int=0,
                   axes: Tuple[int, int] | None=None) -> PatternList: ...

@array_dispatch("data", cpu_impl=_detect_streaks_cpu, gpu_impl=_detect_streaks_gpu)
def detect_streaks(peaks: CPIntArray | PeaksList, data: RealArray, structure: Structure, xtol: float,
                   vmin: float, min_size: float, lookahead: int=0, nfa: int=0,
                   axes: Tuple[int, int] | None=None) -> PatternList:
    """Streak finding algorithm. Starting from the set of seed peaks, the lines are iteratively
    extended with a connectivity structure.

    Args:
        peaks : A set of peaks used as seed locations for the streak growing algorithm.
        data : A 2D rasterised image.
        structure : A connectivity structure.
        xtol : Distance threshold. A new linelet is added to a streak if it's distance to the
            streak is no more than ``xtol``.
        vmin : Value threshold. A new linelet is added to a streak if it's value at the center of
            mass is above ``vmin``.
        log_eps : Detection threshold. A streak is added to the final list if it's p-value under
            null hypothesis is below ``np.exp(log_eps)``.
        lookahead : Number of linelets considered at the ends of a streak to be added to the streak.
        nfa : Number of false alarms, allowed number of unaligned points in a streak.

    Returns:
        A list of detected streaks.
    """
    ...
