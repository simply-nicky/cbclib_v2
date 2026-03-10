from typing import Tuple
from ..annotations import CPRealArray, IntArray, RealArray

def inliers_mean(mean: RealArray, inp: IntArray | RealArray, errors: IntArray | RealArray,
                 indices: IntArray, lm: float = 9.0) -> CPRealArray: ...

def inliers_mean_std(mean: RealArray, std: RealArray, inp: IntArray | RealArray,
                     errors: IntArray | RealArray, indices: IntArray, lm: float = 9.0
                     ) -> Tuple[CPRealArray, CPRealArray]: ...

def lsq(fits: RealArray, W: RealArray, y: IntArray | RealArray, indices: IntArray) -> CPRealArray: ...

def inliers_lsq(fits: RealArray, W: RealArray, y: IntArray | RealArray, errors: IntArray | RealArray,
                indices: IntArray, lm: float = 9.0) -> CPRealArray: ...
