from typing import List
from .src.label import (RegionsList2D, RegionsList3D, Pixels2DDouble, Pixels2DFloat, PointSet2D, PointSet3D,
                        Regions2D, Regions3D, Structure2D, Structure3D)
from .src.label import (binary_dilation, label, total_mass, mean, center_of_mass, moment_of_inertia,
                        covariance_matrix)
from .annotations import ArrayNamespace, RealArray
from .data_container import array_namespace

def to_ellipse(matrix: RealArray, xp: ArrayNamespace) -> RealArray:
    mu_xx, mu_xy, mu_yy = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 1, 1]
    theta = 0.5 * xp.arctan(2 * mu_xy / (mu_xx - mu_yy))
    delta = xp.sqrt(4 * mu_xy**2 + (mu_xx - mu_yy)**2)
    a = xp.sqrt(2 * xp.log(2) * (mu_xx + mu_yy + delta))
    b = xp.sqrt(2 * xp.log(2) * (mu_xx + mu_yy - delta))
    return xp.stack((a, b, theta), axis=-1)

def ellipse_fit(regions: RegionsList2D, data: RealArray, axes: List[int] | None=None) -> RealArray:
    """ Fit ellipses to connected 2D regions in data. The fitted ellipse is defined by its
    major and minor axes (FWHM) and orientation.

    Parameters:
        regions: List of connected 2D regions.
        data: List of 2D arrays of data values.
        axes: Axes over which to compute the fit. If None, the last two axes are used.

    Returns:
        Array of shape (N, 3) where N is the number of regions. Each ellipse is represented by
        (a, b, theta) where a and b are the FWHM of the major and minor axes, and theta is the
        orientation angle in radians.
    """
    xp = array_namespace(data)
    covmat = covariance_matrix(regions, data, axes)
    return to_ellipse(covmat, xp)

def to_line(centers: RealArray, matrix: RealArray, xp: ArrayNamespace) -> RealArray:
    mu_xx, mu_xy, mu_yy = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 1, 1]
    theta = 0.5 * xp.arctan2(2 * mu_xy, (mu_xx - mu_yy))
    tau = xp.stack((xp.cos(theta), xp.sin(theta)), axis=-1)
    delta = xp.sqrt(4 * mu_xy**2 + (mu_xx - mu_yy)**2)
    hw = xp.sqrt(2 * xp.log(2) * (mu_xx + mu_yy - delta))
    return xp.concatenate((centers + hw[..., None] * tau,
                           centers - hw[..., None] * tau), axis=-1)

def line_fit(regions: RegionsList2D, data: RealArray, axes: List[int] | None=None) -> RealArray:
    """ Fit lines to connected 2D regions in data. The fitted line equals to the major axis of the
    covariance ellipse.

    Parameters:
        regions: List of connected 2D regions.
        data: 2D array of data values.
        axes: Axes over which to compute the fit. If None, the last two axes are used.

    Returns:
        Array of shape (N, 4) where N is the number of regions. Each line is represented by
        (x1, y1, x2, y2) coordinates of its endpoints.
    """
    xp = array_namespace(data)
    centers = center_of_mass(regions, data, axes)
    covmat = covariance_matrix(regions, data, axes)
    return to_line(centers, covmat, xp)
