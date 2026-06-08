from typing import Tuple
from ..annotations import CPRealArray, IntArray, RealArray

def inliers_mean(mean: RealArray, inp: IntArray | RealArray, errors: RealArray,
                 indices: IntArray, lm: float = 9.0) -> CPRealArray: ...

def inliers_mean_std(mean: RealArray, std: RealArray, inp: IntArray | RealArray,
                     errors: RealArray, indices: IntArray, lm: float = 9.0
                     ) -> Tuple[CPRealArray, CPRealArray]: ...

def robust_mean(mean: RealArray, inp: IntArray | RealArray, r0: float = 0.0,
                r1: float = 0.5, n_iter: int = 12, lm: float = 9.0
                ) -> CPRealArray: ...

def robust_mean_std(mean: RealArray, std: RealArray, inp: IntArray | RealArray,
                    r0: float = 0.0, r1: float = 0.5, n_iter: int = 12,
                    lm: float = 9.0) -> Tuple[CPRealArray, CPRealArray]: ...

def lsq(fits: RealArray, W: RealArray, y: IntArray | RealArray, indices: IntArray
        ) -> CPRealArray: ...

def inliers_lsq(fits: RealArray, W: RealArray, y: IntArray | RealArray,
                errors: IntArray | RealArray, indices: IntArray, lm: float = 9.0
                ) -> CPRealArray: ...
