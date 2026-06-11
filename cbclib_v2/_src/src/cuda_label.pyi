from typing import Tuple
from ..annotations import BoolArray, CPIntArray, CPRealArray, IntArray, RealArray
from .label import Structure

def label(out: CPIntArray, inp: BoolArray | IntArray, structure: Structure, npts: int=1
          ) -> Tuple[CPIntArray, int]: ...

def center_of_mass(out: CPRealArray, labels: IntArray, index: IntArray, data: CPRealArray
                   ) -> CPRealArray: ...

def covariance_matrix(out: CPRealArray, labels: IntArray, index: IntArray, data: CPRealArray
                      ) -> CPRealArray: ...

def p_values(out: CPRealArray, labels: IntArray, index: IntArray, lines: CPRealArray,
             data: RealArray, p0: float, vmin: float, xtol: float) -> CPRealArray: ...
