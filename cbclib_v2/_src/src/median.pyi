from typing import overload
from ..annotations import IntArray, IntSequence, NDIntArray, NDRealArray, RealArray

@overload
def median(inp: RealArray, axis: IntSequence=0, num_threads: int=1) -> NDRealArray:
    ...

@overload
def median(inp: IntArray, axis: IntSequence=0, num_threads: int=1) -> NDIntArray:
    ...

def median(inp: RealArray | IntArray, axis: IntSequence=0, num_threads: int=1
           ) -> NDRealArray | NDIntArray:
    """Calculate a median along the `axis`.

    Args:
        inp : Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        axis : Array axes along which median values are calculated.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If `mask` and `inp` have different shapes.
        TypeError : If `inp` has incompatible type.
        RuntimeError : If C backend exited with error.

    Returns:
        Array of medians along the given axis.
    """
    ...

def robust_mean(inp: RealArray | IntArray, axis: IntSequence=0, r0: float=0.0, r1: float=0.5,
                n_iter: int=12, lm: float=9.0, return_std: bool=False, num_threads: int=1
                ) -> NDRealArray:
    """Calculate a mean along the `axis` by robustly fitting a Gaussian to input vector [RFG]_.
    The algorithm performs `n_iter` times the fast least kth order statistics (FLkOS [FLKOS]_)
    algorithm to fit a gaussian to data.

    Args:
        inp : Input array. Must be one of the following types: np.float64, np.float32, np.int32,
            np.uint32, np.uint64.
        axis : Array axes along which median values are calculated.
        r0 : A lower bound guess of ratio of inliers. We'd like to make a sample out of worst
            inliers from data points that are between `r0` and `r1` of sorted residuals.
        r1 : An upper bound guess of ratio of inliers. Choose the `r0` to be as high as you are
            sure the ratio of data is inlier.
        n_iter : Number of iterations of fitting a gaussian with the FLkOS algorithm.
        lm : How far (normalized by STD of the Gaussian) from the mean of the Gaussian, data is
            considered inlier.
        return_std : Return robust estimate of standard deviation if True.
        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If `mask` and `inp` have different shapes.
        TypeError : If `inp` has incompatible type.
        RuntimeError : If C backend exited with error.

    References:
        .. [RFG] A. Sadri et al., "Automatic bad-pixel mask maker for X-ray pixel detectors with
                application to serial crystallography", J. Appl. Cryst. 55, 1549-1561 (2022).

        .. [FLKOS] A. Bab-Hadiashar and R. Hoseinnezhad, "Bridging Parameter and Data Spaces for
                  Fast Robust Estimation in Computer Vision," Digital Image Computing: Techniques
                  and Applications, pp. 1-8 (2008).

    Returns:
        Array of robust mean and robust standard deviation (if `robust_std` is True).
    """
    ...
