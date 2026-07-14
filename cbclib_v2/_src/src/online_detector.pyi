from typing import Any, Tuple
from ..annotations import IntArray, NDBoolArray, NDIntArray, NDRealArray, RealArray

def pixel_map(out: RealArray, geometry: Any, half_pixel_shift: bool=True, num_threads: int=1
              ) -> NDRealArray: ...

def radius(out: RealArray, geometry: Any, center: Tuple[int, int],
           half_pixel_shift: bool=True, num_threads: int=1) -> NDRealArray: ...

def radial_index(out: IntArray, geometry: Any, center: Tuple[int, int], n_bins: int,
                 half_pixel_shift: bool=True, num_threads: int=1) -> NDIntArray: ...

def radial_profiles(data: NDIntArray | NDRealArray, radial_index: IntArray,
                    n_bins: int, interval: int=1, clip_snr: float=3.0, n_iter: int=3,
                    std_min: float=0.0, num_threads: int=1
                    ) -> Tuple[NDRealArray, NDRealArray, NDIntArray]: ...

def is_signal(data: IntArray | RealArray, whitefield: RealArray, std: RealArray,
              radial_index: IntArray, min_snr: float=3.0, std_min: float=0.0, num_threads: int=1
              ) -> NDBoolArray: ...
