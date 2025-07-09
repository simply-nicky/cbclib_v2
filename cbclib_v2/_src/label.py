from typing import List, overload
import numpy as np
from .src.label import (Pixels2DDouble, Pixels2DFloat, PointSet2D, PointSet3D, Regions2D, Regions3D,
                        Structure2D, Structure3D)
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

@overload
def ellipse_fit(regions: Regions2D, data: RealArray, axes: List[int] | None=None
                ) -> RealArray:
    ...

@overload
def ellipse_fit(regions: List[Regions2D], data: RealArray, axes: List[int] | None=None
                ) -> List[RealArray]:
    ...

def ellipse_fit(regions: Regions2D | List[Regions2D], data: RealArray,
                axes: List[int] | None=None) -> RealArray | List[RealArray]:
    xp = array_namespace(data)
    covmat = covariance_matrix(regions, data, axes)
    if isinstance(covmat, list):
        return [to_ellipse(mat, xp) for mat in covmat]
    return to_ellipse(covmat, xp)
