"""Multidimensional image processing functions with CPU/CUDA backend support.

This module provides image processing operations that automatically dispatch to CPU or CUDA
backends based on the device context set via :mod:`cbclib_v2.device`.

See Also:
    :mod:`cbclib_v2.device`: Device context management for backend selection.
"""
from typing import List, Optional, Sequence, Tuple, overload

from .annotations import (AnyNamespace, Array, BoolArray, ComplexArray, IntArray, IntSequence, Mode,
                          NDComplexArray, NDIntArray, NDRealArray, Norm, RealArray,
                          RealSequence, Shape)
from .array_api import array_namespace
from . import device
from .backends import LabelResult, PatternList, PeaksList, Structure, get_backend

def accumulate_lines(out: RealArray, lines: RealArray, terms: IntArray, frames: IntArray,
                     max_val: float=1.0, kernel: str='rectangular', in_overlap: str='sum',
                     out_overlap: str='sum') -> NDRealArray:
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
        :func:`write_lines`: Get pixel indices and values for lines.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.accumulate_lines(out=out, lines=lines, terms=terms, frames=frames,
                                            max_val=max_val, kernel=kernel, in_overlap=in_overlap,
                                            out_overlap=out_overlap)

def draw_lines(out: RealArray, lines: RealArray, idxs: IntArray | None=None, max_val: float=1.0,
               kernel: str='rectangular', overlap: str='sum') -> NDRealArray:
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
        :func:`write_lines`: Get pixel indices and values for lines.
        :mod:`cbclib_v2.device`: Set device context for backend selection.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.draw_lines(out=out, lines=lines, idxs=idxs, max_val=max_val,
                                      kernel=kernel, overlap=overlap)

def write_lines(lines: RealArray, shape: Shape, idxs: IntArray | None=None,
                max_val: float=1.0, kernel: str='rectangular'
                ) -> Tuple[NDIntArray, NDIntArray, NDRealArray]:
    """Return an array of rasterized thick lines indices and their corresponding pixel values.

    The lines are drawn with variable thickness and the antialiasing applied.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        lines: Array of shape (N, 5) with [x0, y0, x1, y1, width] for each line.
        shape: Shape of the image. All the lines outside the shape will be discarded.
        idxs: Optional frame indices for each line.
        max_val: Maximum pixel value of a drawn line.
        kernel: Kernel function for antialiasing. Options:
            - 'biweight' : Quartic (biweight) kernel.
            - 'gaussian' : Gaussian kernel.
            - 'parabolic' : Epanechnikov (parabolic) kernel.
            - 'rectangular' : Uniform (rectangular) kernel.
            - 'triangular' : Triangular kernel.

    Returns:
        Tuple of (line_indices, frame_indices, pixel_values).

    Raises:
        ValueError: If lines has an incompatible shape.

    See Also:
        :func:`draw_lines`: Draw lines directly on an output array.
        :func:`accumulate_lines`: Accumulate lines across multiple frames.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.write_lines(lines=lines, shape=shape, idxs=idxs, max_val=max_val,
                                       kernel=kernel)

def fftn(inp: RealArray | ComplexArray, shape: IntSequence | None=None,
         axis: IntSequence | None=None, norm: Norm = "backward") -> NDComplexArray:
    """Compute the N-dimensional discrete Fourier Transform.

    This function computes the N-dimensional discrete Fourier Transform over any number of
    axes in an M-dimensional array by means of the Fast Fourier Transform (FFT).

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        inp: Input array, can be complex.
        shape: Shape (length of each transformed axis) of the output (shape[0] refers to
            axis 0, shape[1] to axis 1, etc.). Along any axis, if the given shape is smaller
            than that of the input, the input is cropped. If it is larger, the input is padded
            with zeros. if shape is not given, the shape of the input along the axes specified
            by axis is used.
        axis: Axes over which to compute the FFT. If not given, the last len(shape) axes are
            used, or all axes if shape is also not specified. Repeated indices in axis
            means that the transform over that axis is performed multiple times.
        norm: Normalization mode. Default is "backward". Indicates which direction of the forward
            / backward pair of transforms is scaled and with what normalization factor.

    Returns:
        The truncated or zero-padded input, transformed along the axes indicated by axis, or by
        a combination of shape and inp, as explained in the parameters section above.

    See Also:
        :func:`ifftn`: Inverse FFT.
        :func:`fft_convolve`: Convolution via FFT.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.fftn(inp=inp, shape=shape, axis=axis, norm=norm)

@overload
def fft_convolve(inp: RealArray, kernel: RealArray, axis: IntSequence | None=None
                 ) -> NDRealArray: ...

@overload
def fft_convolve(inp: RealArray, kernel: ComplexArray, axis: IntSequence | None=None
                 ) -> NDComplexArray: ...

@overload
def fft_convolve(inp: ComplexArray, kernel: RealArray, axis: IntSequence | None=None
                 ) -> NDComplexArray: ...

@overload
def fft_convolve(inp: ComplexArray, kernel: ComplexArray, axis: IntSequence | None=None
                 ) -> NDComplexArray: ...

def fft_convolve(inp: RealArray | ComplexArray, kernel: RealArray | ComplexArray,
                 axis: IntSequence | None=None) -> NDRealArray | NDComplexArray:
    """Convolve a multi-dimensional array with one-dimensional kernel along the axis by means of
    FFT.

    Output has the same size as array.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        inp: Input array.
        kernel: Kernel array.
        axis: Array axis along which convolution is performed.

    Returns:
        A multi-dimensional array containing the discrete linear convolution of ``inp`` with
        ``kernel``.

    See Also:
        :func:`fftn`: Forward FFT.
        :func:`gaussian_filter`: Gaussian filtering via FFT convolution.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.fft_convolve(inp=inp, kernel=kernel, axis=axis)

@overload
def gaussian_filter(inp: RealArray, sigma: RealSequence, order: IntSequence=0,
                    mode: Mode='reflect', cval: float=0.0, truncate: float=4.0
                    ) -> NDRealArray: ...

@overload
def gaussian_filter(inp: ComplexArray, sigma: RealSequence, order: IntSequence=0,
                    mode: Mode='reflect', cval: float=0.0, truncate: float=4.0
                    ) -> NDComplexArray: ...

def gaussian_filter(inp: RealArray | ComplexArray, sigma: RealSequence, order: IntSequence=0,
                    mode: Mode='reflect', cval: float=0.0, truncate: float=4.0
                    ) -> NDRealArray | NDComplexArray:
    """Multidimensional Gaussian filter.

    The multidimensional filter is implemented as a sequence of 1-D FFT convolutions.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        inp: The input array.
        sigma: Standard deviation for Gaussian kernel. The standard deviations of the Gaussian
            filter are given for each axis as a sequence, or as a single number, in which case it
            is equal for all axes.
        order: The order of the filter along each axis is given as a sequence of integers, or as a
            single number. An order of 0 corresponds to convolution with a Gaussian kernel. A
            positive order corresponds to convolution with that derivative of a Gaussian.
        mode: The mode parameter determines how the input array is extended when the filter
            overlaps a border. Default value is 'reflect'. The valid values and their behavior is as
            follows:

            * 'constant', (k k k k | a b c d | k k k k) : The input is extended by filling all
              values beyond the edge with the same constant value, defined by the cval parameter.
            * 'nearest', (a a a a | a b c d | d d d d) : The input is extended by replicating the
              last pixel.
            * 'mirror', (c d c b | a b c d | c b a b) : The input is extended by reflecting about
              the center of the last pixel. This mode is also sometimes referred to as whole-sample
              symmetric.
            * 'reflect', (d c b a | a b c d | d c b a) : The input is extended by reflecting about
              the edge of the last pixel. This mode is also sometimes referred to as half-sample
              symmetric.
            * 'wrap', (a b c d | a b c d | a b c d) : The input is extended by wrapping around to
              the opposite edge.

        cval: Value to fill past edges of input if mode is 'constant'. Default is 0.0.
        truncate: Truncate the filter at this many standard deviations. Default is 4.0.

    Returns:
        Returned array of the same shape as inp.

    See Also:
        :func:`gaussian_gradient_magnitude`: Gradient magnitude using Gaussian derivatives.
        :func:`fft_convolve`: General FFT-based convolution.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.gaussian_filter(inp=inp, sigma=sigma, order=order, mode=mode, cval=cval,
                                           truncate=truncate)

@overload
def gaussian_gradient_magnitude(inp: RealArray, sigma: RealSequence, mode: Mode='reflect',
                                cval: float=0.0, truncate: float=4.0) -> NDRealArray: ...

@overload
def gaussian_gradient_magnitude(inp: ComplexArray, sigma: RealSequence, mode: Mode='reflect',
                                cval: float=0.0, truncate: float=4.0) -> NDComplexArray: ...

def gaussian_gradient_magnitude(inp: RealArray | ComplexArray, sigma: RealSequence,
                                mode: Mode='reflect', cval: float=0.0, truncate: float=4.0
                                ) -> NDRealArray | NDComplexArray:
    """Multidimensional gradient magnitude using Gaussian derivatives.

    The multidimensional filter is implemented as a sequence of 1-D FFT convolutions.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        inp: The input array.
        sigma: Standard deviation for Gaussian kernel. The standard deviations of the Gaussian
            filter are given for each axis as a sequence, or as a single number, in which case it
            is equal for all axes.
        mode: The mode parameter determines how the input array is extended when the filter
            overlaps a border. Default value is 'reflect'. The valid values and their behavior is
            as follows:

            * 'constant', (k k k k | a b c d | k k k k) : The input is extended by filling all
              values beyond the edge with the same constant value, defined by the cval parameter.
            * 'nearest', (a a a a | a b c d | d d d d) : The input is extended by replicating the
              last pixel.
            * 'mirror', (c d c b | a b c d | c b a b) : The input is extended by reflecting about
              the center of the last pixel. This mode is also sometimes referred to as whole-sample
              symmetric.
            * 'reflect', (d c b a | a b c d | d c b a) : The input is extended by reflecting about
              the edge of the last pixel. This mode is also sometimes referred to as half-sample
              symmetric.
            * 'wrap', (a b c d | a b c d | a b c d) : The input is extended by wrapping around to
              the opposite edge.

        cval: Value to fill past edges of input if mode is 'constant'. Default is 0.0.
        truncate: Truncate the filter at this many standard deviations. Default is 4.0.
        backend: Choose between numpy ('numpy') or FFTW ('fftw') backend library for the FFT
            implementation.

    Returns:
        Gaussian gradient magnitude array. The array is the same shape as inp.

    See Also:
        :func:`gaussian_filter`: Gaussian filtering.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.gaussian_gradient_magnitude(inp=inp, sigma=sigma, mode=mode, cval=cval,
                                                       truncate=truncate)

def ifftn(inp: RealArray | ComplexArray, shape: IntSequence | None=None,
          axis: IntSequence | None=None, norm: Norm = "backward") -> NDComplexArray:
    """Compute the N-dimensional discrete inverse Fourier Transform.

    This function computes the N-dimensional discrete Fourier Transform over any number of
    axes in an M-dimensional array by means of the Fast Fourier Transform (FFT).

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        inp: Input array, can be complex.
        shape: Shape (length of each transformed axis) of the output (shape[0] refers to
            axis 0, shape[1] to axis 1, etc.). Along any axis, if the given shape is smaller
            than that of the input, the input is cropped. If it is larger, the input is padded
            with zeros. if shape is not given, the shape of the input along the axes specified
            by axis is used.
        axis: Axes over which to compute the FFT. If not given, the last len(shape) axes are
            used, or all axes if shape is also not specified. Repeated indices in axis
            means that the transform over that axis is performed multiple times.
        norm: Normalization mode. Default is "backward". Indicates which direction of the forward
            / backward pair of transforms is scaled and with what normalization factor.

    Returns:
        The truncated or zero-padded input, transformed along the axes indicated by axis, or by
        a combination of shape and inp, as explained in the parameters section above.

    See Also:
        :func:`fftn`: Forward FFT.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.ifftn(inp=inp, shape=shape, axis=axis, norm=norm)

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
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.binary_dilation(inp=inp, structure=structure, iterations=iterations,
                                           mask=mask)

def label(inp: Array, structure: Structure, npts: int=1) -> LabelResult:
    """Label connected components in a binary array.

    Args:
        inp: Input binary array.
        structure: Structuring element defining connectivity.
        npts: Minimum number of points for a region to be labeled.

    Returns:
        List of labeled regions.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.label(inp=inp, structure=structure, npts=npts)

def center_of_mass(labels: LabelResult, data: RealArray) -> RealArray:
    """Calculate center of mass for each labeled region.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        labels: Labeled regions.
        data: Input data array.

    Returns:
        Array of center of mass coordinates for each region.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.center_of_mass(labels=labels, data=data)

def covariance_matrix(labels: LabelResult, data: RealArray) -> RealArray:
    """Calculate covariance matrix for each labeled region.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        labels: Labeled regions.
        data: Input data array.

    Returns:
        Array of covariance matrices for each region.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.covariance_matrix(labels=labels, data=data)

def index_at(labels: LabelResult, axis: int = 0) -> IntArray:
    """Get indices of labeled regions along specified axis.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        labels: Labeled regions.
        axis: Axis along which to get indices.

    Returns:
        Array of region indices along the specified axis.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.index_at(labels=labels, axis=axis)

def to_ellipse(matrix: RealArray, xp: AnyNamespace) -> RealArray:
    mu_xx, mu_xy, mu_yy = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 1, 1]
    theta = 0.5 * xp.atan(2 * mu_xy / (mu_xx - mu_yy))
    delta = xp.sqrt(4 * mu_xy**2 + (mu_xx - mu_yy)**2)
    a = xp.sqrt(2 * xp.log(2) * (mu_xx + mu_yy + delta))
    b = xp.sqrt(2 * xp.log(2) * (mu_xx + mu_yy - delta))
    return xp.stack((a, b, theta), axis=-1)

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
    xp = array_namespace(data)
    covmat = covariance_matrix(labels, data)
    return to_ellipse(covmat, xp)

def to_line(centers: RealArray, matrix: RealArray, xp: AnyNamespace) -> RealArray:
    mu_xx, mu_xy, mu_yy = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 1, 1]
    theta = 0.5 * xp.atan2(2 * mu_xy, (mu_xx - mu_yy))
    tau = xp.stack((xp.cos(theta), xp.sin(theta)), axis=-1)
    delta = xp.sqrt(4 * mu_xy**2 + (mu_xx - mu_yy)**2)
    hw = xp.sqrt(2 * xp.log(2) * (mu_xx + mu_yy - delta))
    return xp.concat((centers + hw[..., None] * tau,
                           centers - hw[..., None] * tau), axis=-1)

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
    xp = array_namespace(data)
    centers = center_of_mass(labels, data)
    covmat = covariance_matrix(labels, data)
    return to_line(centers, covmat, xp)

@overload
def median(inp: RealArray, mask: BoolArray | None=None, axis: IntSequence=0) -> NDRealArray:
    ...

@overload
def median(inp: IntArray, mask: BoolArray | None=None, axis: IntSequence=0) -> NDIntArray:
    ...

def median(inp: RealArray | IntArray, mask: BoolArray | None=None, axis: IntSequence=0
           ) -> NDRealArray | NDIntArray:
    """Calculate a median along the axis.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        inp: Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        mask: Output mask. Median is calculated only where mask is True, output array set to 0
            otherwise. Median is calculated over the whole input array by default.
        axis: Array axes along which median values are calculated.

    Returns:
        Array of medians along the given axis.

    See Also:
        :func:`median_filter`: Multidimensional median filter.
        :func:`maximum_filter`: Multidimensional maximum filter.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.median(inp=inp, mask=mask, axis=axis)

@overload
def median_filter(inp: RealArray, size: IntSequence | None=None,
                  footprint: BoolArray | None=None, mode: Mode='reflect', cval: float=0.0
                  ) -> NDRealArray: ...

@overload
def median_filter(inp: IntArray, size: IntSequence | None=None,
                  footprint: BoolArray | None=None, mode: Mode='reflect', cval: float=0.0
                  ) -> NDIntArray: ...

def median_filter(inp: RealArray | IntArray, size: IntSequence | None=None,
                  footprint: BoolArray | None=None, mode: Mode='reflect', cval: float=0.0
                  ) -> NDRealArray | NDIntArray:
    """Calculate a multidimensional median filter.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        inp: Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        size: See footprint, below. Ignored if footprint is given.
        footprint: Either size or footprint must be defined. size gives the shape that is taken
            from the input array, at every element position, to define the input to the filter
            function. footprint is a boolean array that specifies (implicitly) a shape, but also
            which of the elements within this shape will get passed to the filter function. Thus
            size=(n, m) is equivalent to footprint=np.ones((n, m)). We adjust size to the number of
            dimensions of the input array, so that, if the input array is shape (10, 10, 10), and
            size is 2, then the actual size used is (2, 2, 2). When footprint is given, size is
            ignored.
        mode: The mode parameter determines how the input array is extended when the filter
            overlaps a border. Default value is 'reflect'. The valid values and their behavior is as
            follows:

            * 'constant', (k k k k | a b c d | k k k k) : The input is extended by filling all
              values beyond the edge with the same constant value, defined by the cval parameter.
            * 'nearest', (a a a a | a b c d | d d d d) : The input is extended by replicating the
              last pixel.
            * 'mirror', (c d c b | a b c d | c b a b) : The input is extended by reflecting about
              the center of the last pixel. This mode is also sometimes referred to as whole-sample
              symmetric.
            * 'reflect', (d c b a | a b c d | d c b a) : The input is extended by reflecting about
              the edge of the last pixel. This mode is also sometimes referred to as half-sample
              symmetric.
            * 'wrap', (a b c d | a b c d | a b c d) : The input is extended by wrapping around to
              the opposite edge.

        cval: Value to fill past edges of input if mode is 'constant'. Default is 0.0.

    Returns:
        Filtered array. Has the same shape as inp.

    See Also:
        :func:`median`: Median along an axis.
        :func:`maximum_filter`: Multidimensional maximum filter.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.median_filter(inp=inp, size=size, footprint=footprint, mode=mode,
                                         cval=cval)

@overload
def maximum_filter(inp: RealArray, size: IntSequence | None=None,
                   footprint: BoolArray | None=None, mode: Mode='reflect', cval: float=0.0
                   ) -> NDRealArray: ...

@overload
def maximum_filter(inp: IntArray, size: IntSequence | None=None,
                   footprint: BoolArray | None=None, mode: Mode='reflect', cval: float=0.0
                   ) -> NDIntArray: ...

def maximum_filter(inp: RealArray | IntArray, size: IntSequence | None=None,
                   footprint: BoolArray | None=None, mode: Mode='reflect', cval: float=0.0
                   ) -> NDRealArray | NDIntArray:
    """Calculate a multidimensional maximum filter.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        inp: Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        size: See footprint, below. Ignored if footprint is given.
        footprint: Either size or footprint must be defined. size gives the shape that is taken
            from the input array, at every element position, to define the input to the filter
            function. footprint is a boolean array that specifies (implicitly) a shape, but also
            which of the elements within this shape will get passed to the filter function. Thus
            size=(n, m) is equivalent to footprint=np.ones((n, m)). We adjust size to the number of
            dimensions of the input array, so that, if the input array is shape (10, 10, 10), and
            size is 2, then the actual size used is (2, 2, 2). When footprint is given, size is
            ignored.
        mode: The mode parameter determines how the input array is extended when the filter
            overlaps a border. Default value is 'reflect'. The valid values and their behavior is as
            follows:

            * 'constant', (k k k k | a b c d | k k k k) : The input is extended by filling all
              values beyond the edge with the same constant value, defined by the cval parameter.
            * 'nearest', (a a a a | a b c d | d d d d) : The input is extended by replicating the
              last pixel.
            * 'mirror', (c d c b | a b c d | c b a b) : The input is extended by reflecting about
              the center of the last pixel. This mode is also sometimes referred to as whole-sample
              symmetric.
            * 'reflect', (d c b a | a b c d | d c b a) : The input is extended by reflecting about
              the edge of the last pixel. This mode is also sometimes referred to as half-sample
              symmetric.
            * 'wrap', (a b c d | a b c d | a b c d) : The input is extended by wrapping around to
              the opposite edge.

        cval: Value to fill past edges of input if mode is 'constant'. Default is 0.0.

    Returns:
        Filtered array. Has the same shape as inp.

    See Also:
        :func:`median_filter`: Multidimensional median filter.
        :func:`median`: Median along an axis.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.maximum_filter(inp=inp, size=size, footprint=footprint, mode=mode,
                                          cval=cval)

def robust_mean(inp: RealArray | IntArray, mask: BoolArray | None=None, axis: IntSequence=0,
                r0: float=0.0, r1: float=0.5, n_iter: int = 12, lm: float=9.0,
                return_std: bool = False) -> NDRealArray:
    """Calculate a mean along the axis by robustly fitting a Gaussian to input vector.

    The algorithm performs n_iter times the fast least kth order statistics (FLkOS) algorithm
    to fit a gaussian to data.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        inp: Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        mask: Input mask. Robust mean is calculated only where mask is True.
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
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.robust_mean(inp=inp, mask=mask, axis=axis, r0=r0, r1=r1, n_iter=n_iter,
                                       lm=lm, return_std=return_std)

def robust_lsq(W: RealArray | IntArray, y: RealArray | IntArray, mask: BoolArray | None=None,
               axis: IntSequence = -1, r0: float=0.0, r1: float=0.5, n_iter: int = 12,
               lm: float=9.0) -> NDRealArray:
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
        mask: Input mask. Robust solution is computed only where mask is True.
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
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.robust_lsq(W=W, y=y, mask=mask, axis=axis, r0=r0, r1=r1, n_iter=n_iter,
                                      lm=lm)

def binterpolate(inp: RealArray, grid: Sequence[RealArray | IntArray],
                 coords: RealArray | IntArray, axis: IntSequence) -> NDRealArray:
    """Perform bilinear multidimensional interpolation on regular grids.

    The integer grid starting from (0, 0, ...) to (inp.shape[0] - 1, inp.shape[1] - 1, ...)
    is implied.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        inp: The data on the regular grid in n dimensions.
        grid: A tuple of grid coordinates along each dimension (x, y, z, ...).
        coords: A list of N coordinates [(x_hat, y_hat, z_hat), ...] to sample the gridded
            data at.
        axis: Axes along which interpolation is performed.

    Returns:
        Interpolated values at input coordinates.

    See Also:
        :func:`kr_predict`: Kernel regression prediction.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.binterpolate(inp=inp, grid=grid, coords=coords, axis=axis)

def kr_predict(y: RealArray, x: RealArray, x_hat: RealArray, sigma: float,
               kernel: str='gaussian', w: Optional[RealArray] = None) -> NDRealArray:
    """Perform the multi-dimensional Nadaraya-Watson kernel regression.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        y: The data to fit.
        x: Coordinates array.
        x_hat: Set of coordinates where the fit is to be calculated.
        sigma: Kernel bandwidth.
        kernel: Choose one of the supported kernel functions. The following kernels
            are available:

            * 'biweight' : Quartic (biweight) kernel.
            * 'gaussian' : Gaussian kernel.
            * 'parabolic' : Epanechnikov (parabolic) kernel.
            * 'rectangular' : Uniform (rectangular) kernel.
            * 'triangular' : Triangular kernel.

        w: A set of weights, unitary weights are assumed if it's not provided.

    Returns:
        The regression result.

    See Also:
        :func:`kr_grid`: Kernel regression over a grid.
        :func:`binterpolate`: Bilinear interpolation.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.kr_predict(y=y, x=x, x_hat=x_hat, sigma=sigma, kernel=kernel, w=w)

def kr_grid(y: RealArray, x: RealArray, grid: Sequence[RealArray], sigma: float,
            kernel: str='gaussian', w: Optional[RealArray] = None) -> Tuple[RealArray, List[float]]:
    """Perform the multi-dimensional Nadaraya-Watson kernel regression over a grid of points.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        y: The data to fit.
        x: Coordinates array.
        grid: A tuple of grid coordinates along each dimension (x, y, z, ...).
        sigma: Kernel bandwidth.
        kernel: Choose one of the supported kernel functions. The following kernels
            are available:

            * 'biweight' : Quartic (biweight) kernel.
            * 'gaussian' : Gaussian kernel.
            * 'parabolic' : Epanechnikov (parabolic) kernel.
            * 'rectangular' : Uniform (rectangular) kernel.
            * 'triangular' : Triangular kernel.

        w: A set of weights, unitary weights are assumed if it's not provided.

    Returns:
        Tuple of (regression_result, region_of_interest).

    See Also:
        :func:`kr_predict`: Kernel regression at specific points.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.kr_grid(y=y, x=x, grid=grid, sigma=sigma, kernel=kernel, w=w)

@overload
def local_maxima(inp: RealArray, axis: IntSequence) -> NDRealArray: ...

@overload
def local_maxima(inp: IntArray, axis: IntSequence) -> IntArray: ...

def local_maxima(inp: RealArray | IntArray, axis: IntSequence) -> NDRealArray | IntArray:
    """Find local maxima in a multidimensional array along a set of axes.

    This function returns the indices of the maxima.

    Automatically dispatches to CPU or CUDA backend based on current device context.

    Args:
        inp: The array to search for local maxima.
        axis: Choose an axis along which the maxima are sought for.

    Returns:
        Indices of local maxima.

    See Also:
        :func:`maximum_filter`: Multidimensional maximum filter.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.local_maxima(inp=inp, axis=axis)

def detect_peaks(data: RealArray, mask: BoolArray, radius: int, vmin: float,
                 axes: Tuple[int, int] | None=None) -> PeaksList:
    """Detect sparse peaks in a set of images. The minimal distance between peaks is controlled
    by the radius parameter.

    Args:
        data: Input data array.
        mask: Boolean mask array where True indicates valid pixels.
        radius: Minimum distance between peaks.
        vmin: Minimum value to consider a pixel as part of a peak.
        axes: Axes along which to detect peaks. If None, last two axes are used.

    Returns:
        List of detected peaks with their properties.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.detect_peaks(data=data, mask=mask, radius=radius, vmin=vmin, axes=axes)

def detect_streaks(peaks: PeaksList, data: RealArray, mask: BoolArray, structure: Structure,
                   xtol: float, vmin: float, min_size: int, lookahead: int=0, nfa: int=0,
                   axes: Tuple[int, int] | None=None) -> PatternList:
    """Streak finding algorithm. Starting from the set of seed peaks, the lines are iteratively
    extended with a connectivity structure.

    Args:
        peaks : A set of peaks used as seed locations for the streak growing algorithm.
        data : A 2D rasterised image.
        mask : Mask of bad pixels. mask is False if the pixel is bad. Bad pixels are skipped in the
            streak detection algorithm.
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
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.detect_streaks(peaks=peaks, data=data, mask=mask, structure=structure,
                                          xtol=xtol, vmin=vmin, min_size=min_size,
                                          lookahead=lookahead, nfa=nfa, axes=axes)

def filter_peaks(peaks: PeaksList, data: RealArray, mask: BoolArray, structure: Structure,
                 vmin: float, npts: int, axes: Tuple[int, int] | None=None) -> None:
    """Filter out peaks that don't belong to a connected region of pixels above a threshold and of
    a minimal size.

    Args:
        peaks: List of detected peaks to be filtered in place.
        data: Input data array.
        mask: Boolean mask array where True indicates valid pixels.
        structure: Structuring element defining pixel connectivity.
        vmin: Minimum value to consider a pixel as part of a peak.
        npts: Minimum number of connected pixels above threshold for a peak to be kept.
        axes: Axes along which to filter peaks. If None, last two axes are used.

    Returns:
        None. The peaks list is modified in place.
    """
    current_device = device.get_device()
    current_backend = get_backend(current_device)
    return current_backend.filter_peaks(peaks=peaks, data=data, mask=mask, structure=structure,
                                        vmin=vmin, npts=npts, axes=axes)
