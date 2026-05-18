from typing import Tuple
from ..annotations import BoolArray, CPIntArray, CPRealArray, IntArray, RealArray
from .label import Structure

def label(out: CPIntArray, inp: BoolArray | IntArray, structure: Structure, npts: int=1
          ) -> Tuple[CPIntArray, int]: ...

def center_of_mass(out: CPRealArray, labels: CPIntArray, index: CPIntArray, data: RealArray
                   ) -> CPRealArray: ...

def covariance_matrix(out: CPRealArray, labels: CPIntArray, index: CPIntArray, data: RealArray
                      ) -> CPRealArray: ...

def p_values(out: CPRealArray, labels: CPIntArray, index: CPIntArray, lines: RealArray,
             data: RealArray, p0: float, vmin: float, xtol: float) -> CPRealArray: ...
