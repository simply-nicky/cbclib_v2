from typing import List, Optional, Union, overload
import numpy as np
from .src.label import PixelsDouble, PixelsFloat, PointsSet, Regions, Structure
from .src.label import label, total_mass, mean, center_of_mass, moment_of_inertia, covariance_matrix
from .annotations import NDRealArray

def to_ellipse(matrix: NDRealArray) -> NDRealArray:
    mu_xx, mu_xy, mu_yy = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 1, 1]
    theta = 0.5 * np.arctan(2 * mu_xy / (mu_xx - mu_yy))
    delta = np.sqrt(4 * mu_xy**2 + (mu_xx - mu_yy)**2)
    a = np.sqrt(2 * np.log(2) * (mu_xx + mu_yy + delta))
    b = np.sqrt(2 * np.log(2) * (mu_xx + mu_yy - delta))
    return np.stack((a, b, theta), axis=-1)

@overload
def ellipse_fit(regions: Regions, data: NDRealArray, axes: Optional[List[int]]=None
                ) -> NDRealArray:
    ...

@overload
def ellipse_fit(regions: List[Regions], data: NDRealArray, axes: Optional[List[int]]=None
                ) -> List[NDRealArray]:
    ...

def ellipse_fit(regions: Union[Regions, List[Regions]], data: NDRealArray,
                axes: Optional[List[int]]=None) -> Union[NDRealArray, List[NDRealArray]]:
    covmat = covariance_matrix(regions, data, axes)
    if isinstance(covmat, list):
        return [to_ellipse(mat) for mat in covmat]
    return to_ellipse(covmat)
