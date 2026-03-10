from typing import Tuple
from ..annotations import BoolArray, CPIntArray, RealArray
from .label import Structure

def label(inp: BoolArray, structure: Structure, npts: int=1) -> Tuple[CPIntArray, int]: ...

def detect_peaks(peaks: CPIntArray, data: RealArray, structure: Structure, radius: int,
                 vmin: float) -> CPIntArray: ...
