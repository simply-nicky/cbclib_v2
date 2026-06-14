from typing import Any, Tuple
from ..annotations import CPBoolArray, CPIntArray, CPRealArray, IntArray, RealArray

def pixel_map(out: RealArray, geometry: Any, half_pixel_shift: bool=True
              ) -> CPRealArray: ...

def radius(out: RealArray, geometry: Any, center: Tuple[float, float],
           half_pixel_shift: bool=True) -> CPRealArray: ...

def radial_index(out: IntArray, geometry: Any, center: Tuple[float, float], n_bins: int,
                 half_pixel_shift: bool=True) -> CPIntArray: ...

def radial_profiles(whitefield: CPRealArray, std: CPRealArray, counts: CPIntArray,
                    data: IntArray | RealArray, radial_index: IntArray, n_bins: int,
                    interval: int=1, clip_snr: float=3.0, n_iter: int=3,
                    std_min: float=0.0) -> Tuple[CPRealArray, CPRealArray, CPIntArray]: ...

def is_signal(out: CPBoolArray, data: IntArray | RealArray, whitefield: RealArray,
              std: RealArray, radial_index: IntArray, min_snr: float=3.0,
              std_min: float=0.0) -> CPBoolArray: ...
