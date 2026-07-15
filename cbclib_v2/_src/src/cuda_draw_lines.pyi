from typing import Sequence
from ..annotations import CPIntArray, IntArray, CPRealArray, RealArray

def accumulate_lines(out: RealArray, lines: RealArray, terms: IntArray, frames: IntArray,
                     widths: RealArray, max_val: float=1.0, kernel: str="rectangular",
                     in_overlap: str="sum",
                     out_overlap: str="sum", grid: tuple[int, ...] | None=None
                     ) -> CPRealArray: ...

def draw_lines(out: RealArray, lines: RealArray, widths: RealArray,
               idxs: IntArray | None=None, max_val: float=1.0,
               kernel: str="rectangular", overlap: str="sum",
               grid: tuple[int, ...] | None=None) -> CPRealArray: ...

def write_lines(lines: RealArray, shape: Sequence[int], widths: RealArray,
                idxs: IntArray | None=None, max_val: float=1.0,
                kernel: str="rectangular"
                ) -> tuple[CPIntArray, CPIntArray, CPRealArray]: ...
