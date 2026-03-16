from typing import Tuple, Sequence, overload
from ..annotations import NDIntArray

class Indexer:
    array           : NDIntArray
    is_increasing   : bool
    is_decreasing   : bool
    is_unique       : bool

    def __init__(self, array: NDIntArray): ...

    @overload
    def __getitem__(self, indices: int) -> slice: ...
    @overload
    def __getitem__(self, indices: Sequence[int] | NDIntArray
                    ) -> Tuple[NDIntArray, NDIntArray]: ...

    def insert_index(self, indices: Sequence[int] | NDIntArray
                     ) -> Tuple[NDIntArray, NDIntArray]: ...

    def unique(self) -> NDIntArray: ...
