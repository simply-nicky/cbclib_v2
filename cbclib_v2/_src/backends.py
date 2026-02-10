"""Backend implementations initialized with device configurations.

Backends are initialized with a device instance and execute operations accordingly.
"""
from functools import wraps
from inspect import signature
from typing import (TYPE_CHECKING, Literal, NamedTuple, Protocol, Tuple, List, Optional, Sequence,
                    cast, overload)
from .annotations import (Array, ArrayDevice, BoolArray, ComplexArray, CPBoolArray, CPComplexArray,
                          CPDevice, CPIntArray, CPRealArray, CuPy, IntArray, IntSequence, JaxDevice,
                          Mode, NDComplexArray, NDIntArray, NDRealArray, Norm, NumPy, RealArray,
                          RealSequence, Shape)
from .device import CPUDevice, CUDADevice
from .src import (bresenham, fft_functions as fft, label, median, signal_proc as signal,
                  streak_finder)
from .src.label import LabelResult as NPLabelResult, Structure
from .src.streak_finder import PatternDoubleList, PatternFloatList, PeaksList

class CPLabelResult(NamedTuple):
    labels      : CPIntArray
    index       : CPIntArray

PatternList = PatternDoubleList | PatternFloatList
Platform = Literal['cpu', 'gpu']

LabelResult = CPLabelResult | NPLabelResult

def get_platform(device: str | ArrayDevice) -> Platform:
    """Get device type from array device.

    Args:
        device: Array device (JAX Device, CuPy Device, or 'cpu' string).

    Returns:
        Device type ('cpu' or 'gpu').

    Raises:
        ValueError: If device type cannot be determined.

    Examples:
        >>> import cupy as cp
        >>> from jax import devices
        >>> device.platform(cp.cuda.Device(0))
        'gpu'
        >>> device.platform('cpu')
        'cpu'
        >>> device.platform(devices()[0])
        'gpu'  # or 'cpu' depending on JAX device
    """
    if isinstance(device, str):
        if device == 'cpu':
            return 'cpu'
        if device in ('cuda', 'gpu'):
            return 'gpu'
        raise ValueError(f"Unknown device string: {device}")
    if isinstance(device, JaxDevice):
        return get_platform(device.platform)
    if CPDevice is not None and isinstance(device, CPDevice):
        return 'gpu'
    raise ValueError(f"Cannot determine platform for device type: {type(device)}")

def ensure_platform(*arguments: str):
    """Decorator to ensure specified array parameters are on correct platform."""
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            sig = signature(method)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()

            for argument in arguments:
                if argument in bound.arguments:
                    arr = bound.arguments[argument]
                    if arr is not None and hasattr(arr, 'device'):
                        platform = get_platform(arr.device)
                        expected = getattr(self.__class__, 'platform', None)
                        if platform != expected:
                            raise ValueError(f"Array argument '{argument}' is on platform "
                                                f"'{platform}', expected '{expected}'.")

            return method(self, *args, **kwargs)
        return wrapper
    return decorator

class BaseBackend():
    """Interface for image processing backends (CPU or CUDA).

    Implements Array API standard compatibility where possible.
    See: https://data-apis.org/array-api/2021.12/
    """
    platform: Platform

    def accumulate_lines(self, out: RealArray, lines: RealArray, terms: IntArray,
                         frames: IntArray, max_val: float=1.0, kernel: str='rectangular',
                         in_overlap: str='sum', out_overlap: str='sum') -> NDRealArray:
        raise NotImplementedError

    def draw_lines(self, out: RealArray, lines: RealArray, idxs: IntArray | None=None,
                   max_val: float=1.0, kernel: str='rectangular',
                   overlap: str='sum') -> NDRealArray:
        raise NotImplementedError

    def write_lines(self, lines: RealArray, shape: Shape, idxs: IntArray | None=None,
                    max_val: float=1.0, kernel: str='rectangular'
                    ) -> Tuple[NDIntArray, NDIntArray, NDRealArray]:
        raise NotImplementedError

    def fftn(self, inp: RealArray | ComplexArray, shape: IntSequence | None=None,
             axis: IntSequence | None=None, norm: Norm="backward") -> NDComplexArray:
        raise NotImplementedError

    @overload
    def fft_convolve(self, inp: RealArray, kernel: RealArray, axis: IntSequence | None=None
                     ) -> NDRealArray: ...

    @overload
    def fft_convolve(self, inp: RealArray, kernel: ComplexArray, axis: IntSequence | None=None
                     ) -> NDComplexArray: ...

    @overload
    def fft_convolve(self, inp: ComplexArray, kernel: RealArray, axis: IntSequence | None=None
                     ) -> NDComplexArray:
        raise NotImplementedError

    @overload
    def fft_convolve(self, inp: ComplexArray, kernel: ComplexArray,
                     axis: IntSequence | None=None) -> NDComplexArray: ...

    def fft_convolve(self, inp: RealArray | ComplexArray, kernel: RealArray | ComplexArray,
                     axis: IntSequence | None=None) -> NDRealArray | NDComplexArray:
        raise NotImplementedError

    @overload
    def gaussian_filter(self, inp: RealArray, sigma: RealSequence, order: IntSequence=0,
                        mode: Mode='reflect', cval: float=0.0, truncate: float=4.0
                        ) -> NDRealArray: ...

    @overload
    def gaussian_filter(self, inp: ComplexArray, sigma: RealSequence, order: IntSequence=0,
                        mode: Mode='reflect', cval: float=0.0, truncate: float=4.0
                        ) -> NDComplexArray: ...

    def gaussian_filter(self, inp: RealArray | ComplexArray, sigma: RealSequence,
                        order: IntSequence=0, mode: Mode='reflect', cval: float=0.0,
                        truncate: float=4.0) -> NDRealArray | NDComplexArray:
        raise NotImplementedError

    @overload
    def gaussian_gradient_magnitude(self, inp: RealArray, sigma: RealSequence,
                                    mode: Mode='reflect', cval: float=0.0, truncate: float=4.0
                                    ) -> NDRealArray: ...

    @overload
    def gaussian_gradient_magnitude(self, inp: ComplexArray, sigma: RealSequence,
                                    mode: Mode='reflect', cval: float=0.0, truncate: float=4.0
                                    ) -> NDComplexArray: ...

    def gaussian_gradient_magnitude(self, inp: RealArray | ComplexArray, sigma: RealSequence,
                                    mode: Mode='reflect', cval: float=0.0, truncate: float=4.0
                                    ) -> NDRealArray | NDComplexArray:
        raise NotImplementedError

    def ifftn(self, inp: RealArray | ComplexArray, shape: IntSequence | None=None,
              axis: IntSequence | None=None, norm: Norm="backward") -> NDComplexArray:
        raise NotImplementedError

    def binary_dilation(self, inp: BoolArray, structure: Structure, iterations: int=1,
                        mask: BoolArray | None=None) -> BoolArray:
        raise NotImplementedError

    def label(self, inp: Array, structure: Structure, npts: int=1) -> LabelResult:
        raise NotImplementedError

    def center_of_mass(self, labels: LabelResult, data: RealArray) -> RealArray:
        raise NotImplementedError

    def covariance_matrix(self, labels: LabelResult, data: RealArray) -> RealArray:
        raise NotImplementedError

    def index_at(self, labels: LabelResult, axis: int = 0) -> IntArray:
        raise NotImplementedError

    def median(self, inp: RealArray | IntArray, mask: BoolArray | None=None, axis: IntSequence=0
               ) -> NDRealArray | NDIntArray:
        raise NotImplementedError

    def median_filter(self, inp: RealArray | IntArray, size: IntSequence | None=None,
                      footprint: BoolArray | None=None, mode: Mode='reflect', cval: float=0.0
                      ) -> NDRealArray | NDIntArray:
        raise NotImplementedError

    def maximum_filter(self, inp: RealArray | IntArray, size: IntSequence | None=None,
                       footprint: BoolArray | None=None, mode: Mode='reflect', cval: float=0.0
                       ) -> NDRealArray | NDIntArray:
        raise NotImplementedError

    def robust_mean(self, inp: RealArray | IntArray, mask: BoolArray | None=None,
                    axis: IntSequence=0, r0: float=0.0, r1: float=0.5, n_iter: int=12,
                    lm: float=9.0, return_std: bool=False) -> NDRealArray:
        raise NotImplementedError

    def robust_lsq(self, W: RealArray | IntArray, y: RealArray | IntArray,
                   mask: BoolArray | None=None, axis: IntSequence=-1, r0: float=0.0,
                   r1: float=0.5, n_iter: int=12, lm: float=9.0) -> NDRealArray:
        raise NotImplementedError

    def binterpolate(self, inp: RealArray, grid: Sequence[RealArray | IntArray],
                     coords: RealArray | IntArray, axis: IntSequence) -> NDRealArray:
        raise NotImplementedError

    def kr_predict(self, y: RealArray, x: RealArray, x_hat: RealArray, sigma: float,
                   kernel: str='gaussian', w: Optional[RealArray]=None) -> NDRealArray:
        raise NotImplementedError

    def kr_grid(self, y: RealArray, x: RealArray, grid: Sequence[RealArray], sigma: float,
                kernel: str='gaussian', w: Optional[RealArray]=None
                ) -> Tuple[RealArray, List[float]]:
        raise NotImplementedError

    def local_maxima(self, inp: RealArray | IntArray, axis: IntSequence
                     ) -> NDRealArray | IntArray:
        raise NotImplementedError

    def detect_peaks(self, data: RealArray, mask: BoolArray, radius: int, vmin: float,
                     axes: Tuple[int, int] | None=None) -> PeaksList:
        raise NotImplementedError

    def detect_streaks(self, peaks: PeaksList, data: RealArray, mask: BoolArray,
                       structure: Structure, xtol: float, vmin: float, min_size: int,
                       lookahead: int=0, nfa: int=0, axes: Tuple[int, int] | None=None
                       ) -> PatternList:
        raise NotImplementedError

    def filter_peaks(self, peaks: PeaksList, data: RealArray, mask: BoolArray,
                     structure: Structure, vmin: float, npts: int,
                     axes: Tuple[int, int] | None=None) -> None:
        raise NotImplementedError

class CPUBackend(BaseBackend):
    """CPU backend using bresenham C++ extension."""
    platform = 'cpu'

    def __init__(self, device: CPUDevice):
        """Initialize CPU backend with device configuration.

        Args:
            device: CPUDevice instance with num_threads attribute.
        """
        self.device = device

    @ensure_platform('out', 'lines', 'terms', 'frames')
    def accumulate_lines(self, out: RealArray, lines: RealArray, terms: IntArray,
                         frames: IntArray, max_val: float=1.0, kernel: str='rectangular',
                         in_overlap: str='sum', out_overlap: str='sum') -> NDRealArray:
        """CPU implementation of accumulate_lines using OpenMP-parallelized C++."""
        return bresenham.accumulate_lines(out=out, lines=lines, terms=terms, frames=frames,
                                          max_val=max_val, kernel=kernel, in_overlap=in_overlap,
                                          out_overlap=out_overlap,
                                          num_threads=self.device.num_threads)

    @ensure_platform('out', 'lines')
    def draw_lines(self, out: RealArray, lines: RealArray, idxs: IntArray | None=None,
                   max_val: float=1.0, kernel: str='rectangular',
                   overlap: str='sum') -> NDRealArray:
        """CPU implementation of draw_lines using OpenMP-parallelized C++."""
        return bresenham.draw_lines(out=out, lines=lines, idxs=idxs, max_val=max_val,
                                    kernel=kernel, overlap=overlap,
                                    num_threads=self.device.num_threads)

    @ensure_platform('lines')
    def write_lines(self, lines: RealArray, shape: Shape, idxs: IntArray | None = None,
                    max_val: float = 1.0, kernel: str = 'rectangular'
                    ) -> Tuple[NDIntArray, NDIntArray, NDRealArray]:
        """CPU implementation of write_lines using OpenMP-parallelized C++."""
        return bresenham.write_lines(lines=lines, shape=shape, idxs=idxs,
                                     max_val=max_val, kernel=kernel,
                                     num_threads=self.device.num_threads)

    @ensure_platform('inp')
    def fftn(self, inp: RealArray | ComplexArray, shape: IntSequence | None=None,
             axis: IntSequence | None=None, norm: Norm="backward") -> NDComplexArray:
        """CPU implementation of fftn using OpenMP-parallelized C++."""
        return fft.fftn(inp, shape=shape, axis=axis, norm=norm,
                        num_threads=self.device.num_threads)

    @ensure_platform('inp', 'kernel')
    def fft_convolve(self, inp: RealArray | ComplexArray, kernel: RealArray | ComplexArray,
                     axis: IntSequence | None=None) -> NDRealArray | NDComplexArray:
        """CPU implementation of fft_convolve using OpenMP-parallelized C++."""
        return fft.fft_convolve(inp=inp, kernel=kernel, axis=axis,
                                num_threads=self.device.num_threads)

    @ensure_platform('inp')
    def gaussian_filter(self, inp: RealArray | ComplexArray, sigma: RealSequence,
                        order: IntSequence=0, mode: Mode='reflect', cval: float=0.0,
                        truncate: float=4.0) -> NDRealArray | NDComplexArray:
        """CPU implementation of gaussian_filter using OpenMP-parallelized C++."""
        return fft.gaussian_filter(inp=inp, sigma=sigma, order=order, mode=mode,
                                   cval=cval, truncate=truncate,
                                   num_threads=self.device.num_threads)

    @ensure_platform('inp')
    def gaussian_gradient_magnitude(self, inp: RealArray | ComplexArray, sigma: RealSequence,
                                    mode: Mode='reflect', cval: float=0.0, truncate: float=4.0
                                    ) -> NDRealArray | NDComplexArray:
        """CPU implementation of gaussian_gradient_magnitude using OpenMP-parallelized C++."""
        return fft.gaussian_gradient_magnitude(inp=inp, sigma=sigma, mode=mode,
                                               cval=cval, truncate=truncate,
                                               num_threads=self.device.num_threads)

    @ensure_platform('inp')
    def ifftn(self, inp: RealArray | ComplexArray, shape: IntSequence | None=None,
              axis: IntSequence | None=None, norm: Norm="backward") -> NDComplexArray:
        """CPU implementation of ifftn using OpenMP-parallelized C++."""
        return fft.ifftn(inp=inp, shape=shape, axis=axis, norm=norm,
                         num_threads=self.device.num_threads)

    @ensure_platform('inp')
    def binary_dilation(self, inp: BoolArray, structure: Structure, iterations: int=1,
                        mask: BoolArray | None=None) -> BoolArray:
        """CPU implementation of binary_dilation using OpenMP-parallelized C++."""
        return label.binary_dilation(inp=inp, structure=structure, iterations=iterations,
                                     mask=mask, num_threads=self.device.num_threads)

    @ensure_platform('inp')
    def label(self, inp: Array, structure: Structure, npts: int=1) -> NPLabelResult:
        """CPU implementation of label using OpenMP-parallelized C++."""
        return label.label(inp=inp, structure=structure, npts=npts,
                           num_threads=self.device.num_threads)

    @ensure_platform('data')
    def center_of_mass(self, labels: NPLabelResult, data: RealArray) -> RealArray:
        """CPU implementation of center_of_mass using OpenMP-parallelized C++.
        Swapping the order from xyz to zyx for consistency to scipy.ndimage.label."""
        return label.center_of_mass(labels=labels, data=data)[:, ::-1]

    @ensure_platform('data')
    def covariance_matrix(self, labels: NPLabelResult, data: RealArray) -> RealArray:
        """CPU implementation of covariance_matrix using OpenMP-parallelized C++.
        Swapping the order from xyz to zyx for consistency to scipy.ndimage.label."""
        return label.covariance_matrix(labels=labels, data=data)[:, ::-1, ::-1]

    def index_at(self, labels: NPLabelResult, axis: int = 0) -> NDIntArray:
        """CPU implementation of index_at using OpenMP-parallelized C++."""
        return labels.index_at(axis)

    @ensure_platform('inp', 'mask')
    def median(self, inp: RealArray | IntArray, mask: BoolArray | None=None, axis: IntSequence=0
               ) -> NDRealArray | NDIntArray:
        """CPU implementation of median using OpenMP-parallelized C++."""
        return median.median(inp=inp, mask=mask, axis=axis, num_threads=self.device.num_threads)

    @ensure_platform('inp')
    def median_filter(self, inp: RealArray | IntArray, size: IntSequence | None=None,
                      footprint: BoolArray | None=None, mode: Mode='reflect', cval: float=0.0
                      ) -> NDRealArray | NDIntArray:
        """CPU implementation of median_filter using OpenMP-parallelized C++."""
        return median.median_filter(inp=inp, size=size, footprint=footprint, mode=mode,
                                    cval=cval, num_threads=self.device.num_threads)

    @ensure_platform('inp')
    def maximum_filter(self, inp: RealArray | IntArray, size: IntSequence | None=None,
                       footprint: BoolArray | None=None, mode: Mode='reflect', cval: float=0.0
                       ) -> NDRealArray | NDIntArray:
        """CPU implementation of maximum_filter using OpenMP-parallelized C++."""
        return median.maximum_filter(inp=inp, size=size, footprint=footprint, mode=mode,
                                     cval=cval, num_threads=self.device.num_threads)

    @ensure_platform('inp', 'mask')
    def robust_mean(self, inp: RealArray | IntArray, mask: BoolArray | None=None,
                    axis: IntSequence=0, r0: float=0.0, r1: float=0.5, n_iter: int=12,
                    lm: float=9.0, return_std: bool=False) -> NDRealArray | NDIntArray:
        """CPU implementation of robust_mean using OpenMP-parallelized C++."""
        return median.robust_mean(inp=inp, mask=mask, axis=axis, r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                  return_std=return_std, num_threads=self.device.num_threads)

    @ensure_platform('W', 'y', 'mask')
    def robust_lsq(self, W: RealArray | IntArray, y: RealArray | IntArray,
                   mask: BoolArray | None=None, axis: IntSequence=-1, r0: float=0.0,
                   r1: float=0.5, n_iter: int=12, lm: float=9.0) -> NDRealArray | NDIntArray:
        """CPU implementation of robust_lsq using OpenMP-parallelized C++."""
        return median.robust_lsq(W=W, y=y, mask=mask, axis=axis, r0=r0, r1=r1, n_iter=n_iter, lm=lm,
                                 num_threads=self.device.num_threads)

    @ensure_platform('inp', 'coords')
    def binterpolate(self, inp: RealArray, grid: Sequence[RealArray | IntArray],
                     coords: RealArray | IntArray, axis: IntSequence) -> NDRealArray:
        """CPU implementation of binterpolate using OpenMP-parallelized C++."""
        return signal.binterpolate(inp=inp, grid=grid, coords=coords, axis=axis,
                                   num_threads=self.device.num_threads)

    @ensure_platform('y', 'x', 'x_hat')
    def kr_predict(self, y: RealArray, x: RealArray, x_hat: RealArray, sigma: float,
                   kernel: str='gaussian', w: Optional[RealArray]=None) -> NDRealArray:
        """CPU implementation of kr_predict using OpenMP-parallelized C++."""
        return signal.kr_predict(y=y, x=x, x_hat=x_hat, sigma=sigma, kernel=kernel, w=w,
                                 num_threads=self.device.num_threads)

    @ensure_platform('y', 'x')
    def kr_grid(self, y: RealArray, x: RealArray, grid: Sequence[RealArray], sigma: float,
                kernel: str='gaussian', w: Optional[RealArray]=None
                ) -> Tuple[RealArray, List[float]]:
        """CPU implementation of kr_grid using OpenMP-parallelized C++."""
        return signal.kr_grid(y=y, x=x, grid=grid, sigma=sigma, kernel=kernel, w=w,
                              num_threads=self.device.num_threads)

    @ensure_platform('inp')
    def local_maxima(self, inp: RealArray | IntArray, axis: IntSequence
                     ) -> NDRealArray | IntArray:
        """CPU implementation of local_maxima using OpenMP-parallelized C++."""
        return signal.local_maxima(inp=inp, axis=axis, num_threads=self.device.num_threads)

    @ensure_platform('data', 'mask')
    def detect_peaks(self, data: RealArray, mask: BoolArray, radius: int, vmin: float,
                     axes: Tuple[int, int] | None=None) -> PeaksList:
        """CPU implementation of detect_peaks using OpenMP-parallelized C++."""
        return streak_finder.detect_peaks(data=data, mask=mask, radius=radius, vmin=vmin,
                                          axes=axes, num_threads=self.device.num_threads)

    @ensure_platform('data', 'mask')
    def detect_streaks(self, peaks: PeaksList, data: RealArray, mask: BoolArray,
                       structure: Structure, xtol: float, vmin: float, min_size: int,
                       lookahead: int=0, nfa: int=0, axes: Tuple[int, int] | None=None
                       ) -> PatternList:
        """CPU implementation of detect_streaks using OpenMP-parallelized C++."""
        return streak_finder.detect_streaks(peaks=peaks, data=data, mask=mask, structure=structure,
                                            xtol=xtol, vmin=vmin, min_size=min_size,
                                            lookahead=lookahead, nfa=nfa, axes=axes,
                                            num_threads=self.device.num_threads)

    @ensure_platform('data', 'mask')
    def filter_peaks(self, peaks: PeaksList, data: RealArray, mask: BoolArray,
                     structure: Structure, vmin: float, npts: int,
                     axes: Tuple[int, int] | None=None) -> None:
        """CPU implementation of filter_peaks using OpenMP-parallelized C++."""
        streak_finder.filter_peaks(peaks=peaks, data=data, mask=mask, structure=structure,
                                   vmin=vmin, npts=npts, axes=axes,
                                   num_threads=self.device.num_threads)

if CuPy is not None or TYPE_CHECKING:
    from .src import cuda_draw_lines, cuda_label
    from cupyx.scipy import ndimage as cupy_ndimage
    from cupy import fft as cupy_fft

    xp = CuPy

    def binary_structure(structure: Structure, shape: Sequence[int] | None=None) -> CPBoolArray:
        """Generate a binary structure array for CUDA backend."""
        if shape is None:
            shape = structure.shape
        return CuPy.asarray(structure.to_array(out=NumPy.zeros(shape, dtype=bool)))

    class NDImageProtocol(Protocol):
        def binary_dilation(self, input: BoolArray, structure: BoolArray, iterations: int=1,
                            mask: BoolArray | None=None, output: BoolArray | None=None,
                            brute_force: bool=False) -> CPBoolArray: ...

        def gaussian_filter(self, input: RealArray | ComplexArray, sigma: RealSequence,
                            order: int | IntSequence=0, mode: Mode='reflect', cval: float=0.0,
                            truncate: float=4.0, output: RealArray | ComplexArray | None=None
                            ) -> CPRealArray | CPComplexArray: ...

        def gaussian_gradient_magnitude(self, input: RealArray | ComplexArray,
                                        sigma: RealSequence, mode: Mode='reflect',
                                        cval: float=0.0, truncate: float=4.0,
                                        output: RealArray | ComplexArray | None=None
                                        ) -> CPRealArray | CPComplexArray: ...

        def sum_labels(self, input: RealArray, labels: IntArray | None,
                       index: IntSequence | None=None) -> CPRealArray: ...

        def mean(self, input: RealArray | IntArray, labels: IntArray | None=None,
                 index: IntSequence | None=None) -> CPRealArray | CPIntArray: ...

        def median_filter(self, input: RealArray | IntArray, size: IntSequence | None=None,
                          footprint: BoolArray | None=None, mode: Mode='reflect',
                          cval: float=0.0, output: RealArray | IntArray | None=None
                          ) -> CPRealArray | CPIntArray: ...

        def maximum_filter(self, input: RealArray | IntArray, size: IntSequence | None=None,
                           footprint: BoolArray | None=None, mode: Mode='reflect',
                           cval: float=0.0, output: RealArray | IntArray | None=None
                           ) -> CPRealArray | CPIntArray: ...

    ndimage = cast(NDImageProtocol, cupy_ndimage)

    class CUDABackend(BaseBackend):
        """CUDA backend using cuda_functions extension."""
        platform = 'gpu'

        def __init__(self, device: CUDADevice):
            """Initialize CUDA backend with device configuration.

            Args:
                device: CUDADevice instance.
            """
            self.device = device

        @ensure_platform('out', 'lines', 'terms', 'frames')
        def accumulate_lines(self, out: RealArray, lines: RealArray, terms: IntArray,
                            frames: IntArray, max_val: float=1.0, kernel: str='rectangular',
                            in_overlap: str='sum', out_overlap: str='sum',
                            grid: Tuple[int, ...] | None = None) -> RealArray:
            """CUDA implementation of accumulate_lines using GPU acceleration."""
            return cuda_draw_lines.accumulate_lines(out=out, lines=lines, terms=terms,
                                                    frames=frames,  max_val=max_val,
                                                    kernel=kernel, in_overlap=in_overlap,
                                                    out_overlap=out_overlap, grid=grid)

        @ensure_platform('out', 'lines', 'idxs')
        def draw_lines(self, out: RealArray, lines: RealArray, idxs: IntArray | None = None,
                    max_val: float = 1.0, kernel: str = 'rectangular', overlap: str = 'sum',
                    grid: Tuple[int, ...] | None = None) -> RealArray:
            """CUDA implementation of draw_lines using GPU acceleration."""
            return cuda_draw_lines.draw_lines(out=out, lines=lines, idxs=idxs, max_val=max_val,
                                              kernel=kernel, overlap=overlap, grid=grid)

        @ensure_platform('inp')
        def fftn(self, inp: RealArray | ComplexArray, shape: IntSequence | None=None,
                 axis: IntSequence | None=None, norm: Norm="backward") -> ComplexArray:
            """CUDA implementation of fftn using GPU acceleration."""
            return cupy_fft.fftn(inp, s=shape, axes=axis, norm=norm)

        @ensure_platform('inp')
        def gaussian_filter(self, inp: RealArray | ComplexArray, sigma: RealSequence,
                            order: IntSequence=0, mode: Mode='reflect', cval: float=0.0,
                            truncate: float=4.0) -> RealArray | ComplexArray:
            """CUDA implementation of gaussian_filter using GPU acceleration."""
            return ndimage.gaussian_filter(input=inp, sigma=sigma, order=order, mode=mode,
                                           cval=cval, truncate=truncate)

        @ensure_platform('inp')
        def gaussian_gradient_magnitude(self, inp: RealArray | ComplexArray, sigma: RealSequence,
                                        mode: Mode='reflect', cval: float=0.0, truncate: float=4.0
                                        ) -> RealArray | ComplexArray:
            """CUDA implementation of gaussian_gradient_magnitude using GPU acceleration."""
            return ndimage.gaussian_gradient_magnitude(input=inp, sigma=sigma, mode=mode, cval=cval,
                                                       truncate=truncate)

        @ensure_platform('inp')
        def ifftn(self, inp: RealArray | ComplexArray, shape: IntSequence | None=None,
                  axis: IntSequence | None=None, norm: Norm="backward") -> ComplexArray:
            """CUDA implementation of ifftn using GPU acceleration."""
            return cupy_fft.ifftn(inp, s=shape, axes=axis, norm=norm)

        @ensure_platform('inp', 'mask')
        def median(self, inp: RealArray | IntArray, mask: BoolArray | None = None,
                   axis: int | Tuple[int, ...] = 0) -> RealArray | IntArray:
            """CUDA implementation of median using GPU acceleration."""
            out = xp.copy(inp)
            out[xp.asarray(mask)] = xp.nan
            return xp.nanmedian(out, axis=axis, overwrite_input=True)

        @ensure_platform('inp')
        def binary_dilation(self, inp: BoolArray, structure: Structure, iterations: int=1,
                            mask: BoolArray | None=None) -> BoolArray:
            """CUDA implementation of binary_dilation using GPU acceleration."""
            sarray = binary_structure(structure)
            return ndimage.binary_dilation(input=inp, structure=sarray, iterations=iterations,
                                           mask=mask)

        @ensure_platform('inp')
        def label(self, inp: Array, structure: Structure, npts: int=1) -> CPLabelResult:
            """CUDA implementation of label using GPU acceleration."""
            labels, n_labels = cuda_label.label(inp=inp, structure=structure, npts=npts)
            return CPLabelResult(labels=labels, index=xp.arange(1, n_labels + 1, dtype=int))

        @ensure_platform('data')
        def center_of_mass(self, labels: CPLabelResult, data: RealArray) -> RealArray:
            """CUDA implementation of center_of_mass using GPU acceleration."""
            mu = ndimage.sum_labels(data, labels=labels.labels, index=labels.index)
            grids = xp.ogrid[[slice(0, s) for s in data.shape]]
            mu_x = xp.stack([ndimage.sum_labels(grids[dim] * data, labels=labels.labels,
                                                index=labels.index) for dim in range(data.ndim)],
                            axis=-1)
            return mu_x / mu[:, None]

        @ensure_platform('data')
        def covariance_matrix(self, labels: CPLabelResult, data: RealArray) -> RealArray:
            """CUDA implementation of covariance_matrix for each region using GPU acceleration.
            """
            mu = ndimage.sum_labels(data, labels=labels.labels, index=labels.index)
            grids = xp.ogrid[[slice(0, s) for s in data.shape]]
            mu_x = xp.stack([ndimage.sum_labels(grids[dim] * data, labels=labels.labels,
                                                index=labels.index) for dim in range(data.ndim)])
            mu_x /= mu
            mu_xx = xp.stack([ndimage.sum_labels(grids[dim // data.ndim] * grids[dim % data.ndim] * data,
                                                 labels=labels.labels, index=labels.index)
                              for dim in range(data.ndim * data.ndim)])
            mu_xx = xp.reshape(mu_xx, (data.ndim, data.ndim, -1)) / mu
            return xp.permute_dims((mu_xx - mu_x[:, None, :] * mu_x[None, :, :]), (2, 0, 1))

        def index_at(self, labels: CPLabelResult, axis: int = 0) -> CPIntArray:
            """Get index of each region along a given axis"""
            grids = xp.ogrid[[slice(0, s) for s in labels.labels.shape]]
            indices = ndimage.mean(grids[axis], labels=labels.labels, index=labels.index)
            if not xp.allclose(indices - xp.floor(indices), 0.0):
                bad = xp.where(xp.invert(xp.isclose(indices - xp.floor(indices), 0.0)))[0]
                raise ValueError("Region at indices " + str(bad.tolist()) +
                                 " has no unique index along axis " + str(axis))
            return xp.asarray(indices, dtype=int)

        @ensure_platform('inp')
        def median_filter(self, inp: RealArray | IntArray, size: IntSequence | None=None,
                          footprint: BoolArray | None=None, mode: Mode='reflect', cval: float=0.0
                          ) -> CPRealArray | CPIntArray:
            """CUDA implementation of median_filter using GPU acceleration."""
            return ndimage.median_filter(input=inp, size=size, footprint=footprint, mode=mode,
                                         cval=cval)

        @ensure_platform('inp')
        def maximum_filter(self, inp: RealArray | IntArray, size: IntSequence | None=None,
                           footprint: BoolArray | None=None, mode: Mode='reflect', cval: float=0.0
                           ) -> CPRealArray | CPIntArray:
            """CUDA implementation of maximum_filter using GPU acceleration."""
            return ndimage.maximum_filter(input=inp, size=size, footprint=footprint, mode=mode,
                                          cval=cval)

def get_backend(device: CPUDevice | CUDADevice) -> BaseBackend:
    """Get appropriate backend for the given device.

    Args:
        device: Device instance (CPUDevice or CUDADevice).

    Returns:
        Backend instance implementing BackendProtocol.

    Examples:
        >>> from cbclib_v2 import device
        >>> dev = device.cpu(num_threads=8)
        >>> backend = get_backend(dev)
        >>> isinstance(backend, BackendProtocol)
        True
    """
    if isinstance(device, CUDADevice):
        return CUDABackend(device)
    if isinstance(device, CPUDevice):
        return CPUBackend(device)
    raise ValueError(f"Unsupported device type: {type(device)}")
