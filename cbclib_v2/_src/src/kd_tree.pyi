from typing import List, Tuple, Union, overload
from ..annotations import NDIntArray, NDRealArray

class KDTreeFloat:
    high    : List[float]
    low     : List[float]
    ndim    : int
    size    : int

    def __init__(self, database: NDRealArray):
        ...

    def find_nearest(self, query: NDRealArray, k: int=1, num_threads: int=1
                     ) -> Tuple[NDRealArray, NDIntArray]:
        ...

    def find_range(self, query: NDRealArray, range: float, num_threads: int=1
                   ) -> Tuple[List[List[float]], List[List[int]]]:
        ...

class KDTreeDouble(KDTreeFloat):
    ...

class KDTreeInt:
    high    : List[int]
    low     : List[int]
    ndim    : int
    size    : int

    def __init__(self, database: NDIntArray):
        ...

    def find_nearest(self, query: NDIntArray, k: int=1, num_threads: int=1
                     ) -> Tuple[NDRealArray, NDIntArray]:
        ...

    def find_range(self, query: NDIntArray, range: float, num_threads: int=1
                   ) -> Tuple[List[List[float]], List[List[int]]]:
        ...

@overload
def build_kd_tree(database: NDRealArray) -> Union[KDTreeDouble, KDTreeFloat]:
    ...

@overload
def build_kd_tree(database: NDIntArray) -> KDTreeInt:
    ...

def build_kd_tree(database: Union[NDRealArray, NDIntArray]) -> Union[KDTreeDouble, KDTreeFloat, KDTreeInt]:
    ...
