"""Multidimensional image processing functions with CPU/CUDA backend support.

This module provides image processing operations that automatically dispatch to CPU or CUDA
backends based on the device context set via :mod:`cbclib_v2.device`.

See Also:
    :mod:`cbclib_v2.device`: Device context management for backend selection.
"""
from functools import wraps
from inspect import signature
from math import prod
from typing import (TYPE_CHECKING, Callable, NamedTuple, Optional, Protocol, Sequence, Tuple,
                    cast, overload)
import warnings
from .annotations import (Array, BoolArray, CPArray, CPBoolArray, CPIntArray, CPRealArray,
                          CuPy, IntArray, IntSequence, JaxArray, JaxBoolArray, JaxIntArray,
                          JaxNumPy, JaxRealArray, NDArray, NDBoolArray, NDIntArray,
                          NDRealArray, NumPy, RealArray)
from .array_api import array_namespace, ascupy, asjax, asnumpy, get_platform
from .config import get_cpu_config
from .src import bresenham, label as cpu_label, median as cpu_median, streak_finder
from .src.label import Structure, LabelResult as NPLabelResult
from .src.streak_finder import Streaks as NPStreaks

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
    labels : IntArray
    n_seeds : int
    n_labels : int
    n_good : int
    radius : int

    def keep_best(self, quantile: float = 0.5) -> 'PeakLabels':
        return PeakLabels(self.labels, int(self.n_seeds * quantile), self.n_labels, self.n_good,
                          self.radius)

    def to_tuple(self) -> Tuple[IntArray, int, int, int, int]:
        return self.labels, self.n_seeds, self.n_labels, self.n_good, self.radius

if CuPy is not None or TYPE_CHECKING:
    from .src import cuda_draw_lines, cuda_label, cuda_median, cuda_streak_finder
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
    cupy_ndimage = None
    cuda_streak_finder = None

    Streaks = NPStreaks

    def binary_structure(structure: Structure, shape: Sequence[int] | None=None) -> CPBoolArray:
        raise RuntimeError("CUDA backend is not available. Please, check if you have installed " \
                           "the cbclib_v2 with GPU support.")

    def shift_axis(inp: Array, axis: Tuple[int, ...]) -> Array:
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

def _label_cpu(inp: NDBoolArray | NDIntArray, structure: Structure, npts: int=1) -> NPLabelResult:
    num_threads = get_cpu_config().effective_num_threads()
    return cpu_label.label(inp=inp, structure=structure, npts=npts,
                           num_threads=num_threads)

def _label_gpu(inp: CPBoolArray | CPIntArray, structure: Structure, npts: int=1) -> LabelResult:
    if cuda_label is None:
        raise RuntimeError("label is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    labels = xp.empty_like(inp, dtype=xp.int32)
    labels, n_labels = cuda_label.label(out=labels, inp=inp, structure=structure, npts=npts)
    return CPLabelResult(labels=labels, index=xp.arange(1, n_labels + 1, dtype=int))

@overload
def label(inp: NDBoolArray | NDIntArray, structure: Structure, npts: int=1) -> NPLabelResult: ...

@overload
def label(inp: CPBoolArray | CPIntArray, structure: Structure, npts: int=1) -> CPLabelResult: ...

@overload
def label(inp: JaxBoolArray | JaxIntArray, structure: Structure, npts: int=1) -> LabelResult: ...

@overload
def label(inp: BoolArray | IntArray, structure: Structure, npts: int=1) -> LabelResult: ...

@array_dispatch("inp", cpu_impl=_label_cpu, gpu_impl=_label_gpu)
def label(inp: BoolArray | IntArray, structure: Structure, npts: int=1) -> LabelResult:
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
    if cuda_label is None:
        raise RuntimeError("center_of_mass is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    out = xp.empty((labels.index.shape[0], data.ndim), dtype=data.dtype)
    return cuda_label.center_of_mass(out=out, labels=labels.labels, index=labels.index, data=data)

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
    if cuda_label is None:
        raise RuntimeError("covariance_matrix is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    out = xp.empty((labels.index.shape[0], data.ndim, data.ndim), dtype=data.dtype)
    return cuda_label.covariance_matrix(out=out, labels=labels.labels, index=labels.index,
                                        data=data)

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
        return labels.to_array(index(labels))
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
        centers = _center_of_mass_gpu(labels, data)
        covmat = _covariance_matrix_gpu(labels, data)
        return to_line(centers, covmat)
    raise ValueError("Invalid labels type. Expected NPLabelResult or CPLabelResult.")

@overload
def p_values(labels: NPLabelResult, lines: NDRealArray, data: NDRealArray, p0: float, vmin: float,
             xtol: float) -> NDRealArray: ...

@overload
def p_values(labels: LabelResult, lines: JaxRealArray, data: JaxRealArray, p0: float, vmin: float,
             xtol: float) -> JaxRealArray: ...

@overload
def p_values(labels: CPLabelResult, lines: CPRealArray, data: CPRealArray, p0: float, vmin: float,
             xtol: float) -> CPRealArray: ...

@overload
def p_values(labels: LabelResult, lines: RealArray, data: RealArray, p0: float, vmin: float,
             xtol: float) -> RealArray: ...

def p_values(labels: LabelResult, lines: RealArray, data: RealArray, p0: float, vmin: float,
             xtol: float) -> RealArray:
    """Calculate p-values for each labeled region based on the data values.

    Args:
        labels: Labeled regions.
        lines : Line parameters for each region. Must have shape (N, 2 * data.ndim) and follow xyz
            format.
        data: Input data array.
        p0: Expected p-value for the null hypothesis.
        vmin: Minimum data value to consider for p-value calculation.
        xtol: Tolerance for convergence of p-value estimation.

    Returns:
        Array of p-values for each region.
    """
    if isinstance(labels, NPLabelResult):
        return cpu_label.p_values(labels=labels, lines=lines, data=data, p0=p0, vmin=vmin,
                                  xtol=xtol)
    if isinstance(labels, CPLabelResult):
        if cuda_label is None:
            raise RuntimeError("label is not compiled for the current platform. "
                               "Please, check if you have installed the cbclib_v2 with GPU support.")

        out = CuPy.empty(labels.index.shape, dtype=data.dtype)
        return cuda_label.p_values(out=out, labels=labels.labels, index=labels.index, lines=lines,
                                   data=data, p0=p0, vmin=vmin, xtol=xtol)
    raise ValueError("Invalid labels type. Expected NPLabelResult or CPLabelResult.")

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

# New GPU-friendly streak detection algorithm

def _detect_peaks_cpu(data: RealArray, labeled: NPLabelResult, radius: int, vmin: float) -> NDIntArray:
    num_threads = get_cpu_config().effective_num_threads()
    xp = NumPy
    labels_array = labeled.to_array(xp.arange(1, len(labeled.regions) + 1))
    radii = [0,] * (data.ndim - 2) + [1, 1]

    return streak_finder.detect_peaks(labels_array, data, Structure(radii, 1), radius, vmin,
                                          num_threads=num_threads)

def binned_shape(shape: Tuple[int, ...], radius: int) -> Tuple[int, ...]:
    return shape[:-2] + ((shape[-2] + radius - 1) // radius, (shape[-1] + radius - 1) // radius)

def _detect_peaks_gpu(data: RealArray, labeled: CPLabelResult, radius: int, vmin: float) -> CPIntArray:
    if cuda_streak_finder is None:
        raise RuntimeError("detect_peaks is not compiled for the current platform. "
                           "Please, check if you have installed the cbclib_v2 with GPU support.")

    xp = CuPy
    radii = [0,] * (data.ndim - 2) + [1, 1]

    out = xp.empty(binned_shape(data.shape, radius), dtype=xp.int32)
    return cuda_streak_finder.detect_peaks(out, labeled.labels, data, Structure(radii, 1), radius,
                                           vmin)

@overload
def detect_peaks(data: NDRealArray, labeled: NPLabelResult, radius: int, vmin: float) -> NDIntArray: ...

@overload
def detect_peaks(data: CPRealArray, labeled: CPLabelResult, radius: int, vmin: float) -> CPIntArray: ...

@overload
def detect_peaks(data: JaxRealArray, labeled: LabelResult, radius: int, vmin: float
                     ) -> NDIntArray | CPIntArray: ...

@overload
def detect_peaks(data: RealArray, labeled: LabelResult, radius: int, vmin: float) -> NDIntArray | CPIntArray: ...

@array_dispatch("data", cpu_impl=_detect_peaks_cpu, gpu_impl=_detect_peaks_gpu)
def detect_peaks(data: RealArray, labeled: LabelResult, radius: int, vmin: float
                     ) -> NDIntArray | CPIntArray:
    """Detect peaks in a set of images using the new GPU-friendly algorithm. The minimal distance
    between peaks is controlled by the radius parameter.

    Args:
        data: Input data array.
        labeled: Labeled regions used for peak detection.
        radius: Minimum distance between peaks. The distance is measured as a number of pixels
            along the axes specified by the axes parameter.
        vmin: Minimum value to consider a pixel as part of a peak.

    Returns:
        Array of bins with peak's index written in it. If the bin doesn't contain a peak, it is
        filled with -1. The shape of the output array is the same as the input array, except for
        the last two dimensions, which are divided by the radius.
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
    """Convert an array of peak indices to a labeled array and a list of peaks.

    Args:
        indices: Array of peak indices. The shape of the array is the same as the input data, except
            for the last two dimensions, which are divided by the radius. The values in the array
            are either -1 (if there is no peak in the corresponding bin) or an index of a peak in
            the original data array.
        data: Input data array.
        radius: Radius used for peak detection.

    Returns:
        A tuple of (labels, peaks) where labels is an array of bin labels and peaks is a flat array
        of peak indices in the original data array.
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
    """Fit linelets to the detected peaks.

    Args:
        labels: Array of bin labels. A bin with label -1 doesn't contain a peak, a bin with label
            0 doesn't have an assigned peak yet.
        peaks: Array of peak indices in the original data array.
        data: Input data array.
        structure: Connectivity structure used for linelet fitting.
        vmin: Minimum value to consider a pixel as part of a linelet.
        max_iter: Maximum number of iterations for linelet fitting.

    Returns:
        Array of linelet parameters. The shape of the array is (n_labels, 4) where n_labels is the
        number of labeled peaks. The last dimension contains linelet parameters in the following
        order: (x0, y0, x1, y1) where (x0, y0) and (x1, y1) are coordinates of the endpoints of the
        linelet.
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
    """Detect streaks using the new GPU-friendly algorithm.

    Args:
        labels: Array of bin labels. A bin with label <= 0 doesn't contain a peak.
        peaks: Array of peak indices in the original data array.
        linelets: Array of linelet parameters.
        data: Input data array.
        structure: Connectivity structure used for streak detection.
        vmin: Minimum value to consider a pixel as part of a streak.
        xtol: Distance threshold. A new linelet is added to a streak if it's distance to the
            streak is no more than ``xtol``.
        nfa: Number of false alarms, allowed number of unaligned points in a streak.

    Returns:
        A list of detected streaks.
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
    """Convert streaks to line parameters.

    Args:
        streaks: Detected streaks.
        labels: Array of bin labels. A bin with label <= 0 doesn't contain a peak.
        linelets: Array of linelet parameters.

    Returns:
        Array of line parameters. The shape of the array is (n_streaks, 4) where n_streaks is the
        number of detected streaks. The last dimension contains line parameters in the following
        order: (x0, y0, x1, y1) where (x0, y0) and (x1, y1) are coordinates of the endpoints of the
        line.
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
    """Calculate the number of pixels above the vmin threshold in each detected streak.

    Args:
        streaks: Detected streaks.
        labels: Array of bin labels. A bin with label <= 0 doesn't contain a peak.
        peaks: Array of peak indices in the original data array.
        data: Input data array.
        structure: Connectivity structure used for streak detection.
        vmin: Minimum value to consider a pixel as part of a streak.

    Returns:
        Array of the number of pixels above the vmin threshold in each detected streak.
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
    """Convert a list of detected streaks to a labeled array.

    Args:
        out: Output array. The shape of the array is the same as the input data.
        streaks: Detected streaks.
        ranks: Array of streak ranks. The shape of the array is (n_streaks,).
        labels: Array of bin labels. A bin with label <= 0 doesn't contain a peak.
        peaks: Array of peak indices in the original data array.
        structure: Connectivity structure used for streak detection.

    Returns:
        Labeled array of the same shape as the input data, where each pixel is labeled with the
        rank of the streak it belongs to, or -1 if it doesn't belong to any streak.
    """
    ...
