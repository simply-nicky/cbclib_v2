from typing import Tuple, overload
from ..annotations import IntArray, IntSequence

class Indexer:
    array           : IntArray
    is_increasing   : bool
    is_decreasing   : bool
    is_unique       : bool

    def __init__(self, array: IntArray): ...

    @overload
    def __getitem__(self, indices: int) -> slice: ...
    @overload
    def __getitem__(self, indices: IntSequence) -> Tuple[IntArray, IntArray]: ...

    def unique(self) -> IntArray: ...
