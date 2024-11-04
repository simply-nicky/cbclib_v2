from typing import Dict, List, Sequence, Tuple, Union, overload
from ..annotations import IntArray, RealArray

# 2D case : QuadTree

class QuadTreeFloat:
    box     : List[float]
    size    : int

    def __init__(self, database: RealArray, box: Sequence[float]):
        ...

    def find_k_nearest(self, query: RealArray, k: int=1, num_threads: int=1
                       ) -> Tuple[RealArray, IntArray]:
        ...

    def find_range(self, query: RealArray, range: float, num_threads: int=1
                   ) -> Tuple[List[List[float]], List[List[int]]]:
        ...

class QuadTreeDouble(QuadTreeFloat):
    ...

class QuadTreeInt:
    box     : List[int]
    size    : int

    def __init__(self, database: IntArray, box: Sequence[int]):
        ...

    def find_k_nearest(self, query: IntArray, k: int=1, num_threads: int=1
                       ) -> Tuple[RealArray, IntArray]:
        ...

    def find_range(self, query: IntArray, range: float, num_threads: int=1
                   ) -> Tuple[List[List[float]], List[List[int]]]:
        ...

@overload
def build_quad_tree(database: RealArray, box: Sequence[float]) -> Union[QuadTreeDouble, QuadTreeFloat]:
    ...

@overload
def build_quad_tree(database: IntArray, box: Sequence[int]) -> QuadTreeInt:
    ...

def build_quad_tree(database: Union[RealArray, IntArray], box: Union[Sequence[float], Sequence[int]]
                   ) -> Union[QuadTreeDouble, QuadTreeFloat, QuadTreeInt]:
    ...

class QuadStackFloat:
    trees : Dict[int, QuadTreeFloat]

    def __init__(self, database: RealArray, indices: IntArray, box: Sequence[float]):
        ...

    def find_k_nearest(self, query: RealArray, indices: IntArray, k: int=1, num_threads: int=1
                       ) -> Tuple[RealArray, IntArray]:
        ...

    def find_range(self, query: RealArray, indices: IntArray, range: float, num_threads: int=1
                   ) -> Tuple[List[List[float]], List[List[int]]]:
        ...

class QuadStackDouble(QuadStackFloat):
    trees : Dict[int, QuadTreeDouble]

class QuadStackInt:
    trees : Dict[int, QuadTreeInt]

    def __init__(self, database: IntArray, indices: IntArray, box: Sequence[int]):
        ...

    def find_k_nearest(self, query: IntArray, indices: IntArray, k: int=1, num_threads: int=1
                       ) -> Tuple[RealArray, IntArray]:
        ...

    def find_range(self, query: IntArray, indices: IntArray, range: float, num_threads: int=1
                   ) -> Tuple[List[List[float]], List[List[int]]]:
        ...

@overload
def build_quad_stack(database: RealArray, indices: IntArray, box: Sequence[float]
                     ) -> Union[QuadStackDouble, QuadStackFloat]:
    ...

@overload
def build_quad_stack(database: IntArray, indices: IntArray, box: Sequence[int]
                     ) -> QuadStackInt:
    ...

def build_quad_stack(database: Union[RealArray, IntArray], indices: IntArray,
                    box: Union[Sequence[float], Sequence[int]]
                    ) -> Union[QuadStackDouble, QuadStackFloat, QuadStackInt]:
    ...

# 3D case : Octree

class OctreeFloat:
    box     : List[float]
    size    : int

    def __init__(self, database: RealArray, box: Sequence[float]):
        ...

    def find_k_nearest(self, query: RealArray, k: int=1, num_threads: int=1
                       ) -> Tuple[RealArray, IntArray]:
        ...

    def find_range(self, query: RealArray, range: float, num_threads: int=1
                   ) -> Tuple[List[List[float]], List[List[int]]]:
        ...

class OctreeDouble(OctreeFloat):
    ...

class OctreeInt:
    box     : List[int]
    size    : int

    def __init__(self, database: IntArray, box: Sequence[int]):
        ...

    def find_k_nearest(self, query: IntArray, k: int=1, num_threads: int=1
                       ) -> Tuple[RealArray, IntArray]:
        ...

    def find_range(self, query: IntArray, range: float, num_threads: int=1
                   ) -> Tuple[List[List[float]], List[List[int]]]:
        ...

@overload
def build_octree(database: RealArray, box: Sequence[float]) -> Union[OctreeDouble, OctreeFloat]:
    ...

@overload
def build_octree(database: IntArray, box: Sequence[int]) -> OctreeInt:
    ...

def build_octree(database: Union[RealArray, IntArray], box: Union[Sequence[float], Sequence[int]]
                 ) -> Union[OctreeDouble, OctreeFloat, OctreeInt]:
    ...

class OctStackFloat:
    trees : Dict[int, OctreeFloat]

    def __init__(self, database: RealArray, indices: IntArray, box: Sequence[float]):
        ...

    def find_k_nearest(self, query: RealArray, indices: IntArray, k: int=1, num_threads: int=1
                       ) -> Tuple[RealArray, IntArray]:
        ...

    def find_range(self, query: RealArray, indices: IntArray, range: float, num_threads: int=1
                   ) -> Tuple[List[List[float]], List[List[int]]]:
        ...

class OctStackDouble(OctStackFloat):
    trees : Dict[int, OctreeDouble]

class OctStackInt:
    trees : Dict[int, OctreeInt]

    def __init__(self, database: IntArray, indices: IntArray, box: Sequence[int]):
        ...

    def find_k_nearest(self, query: IntArray, indices: IntArray, k: int=1, num_threads: int=1
                       ) -> Tuple[RealArray, IntArray]:
        ...

    def find_range(self, query: IntArray, indices: IntArray, range: float, num_threads: int=1
                   ) -> Tuple[List[List[float]], List[List[int]]]:
        ...

@overload
def build_oct_stack(database: RealArray, indices: IntArray, box: Sequence[float]
                   ) -> Union[OctStackDouble, OctStackFloat]:
    ...

@overload
def build_oct_stack(database: IntArray, indices: IntArray, box: Sequence[int]) -> OctStackInt:
    ...

def build_oct_stack(database: Union[RealArray, IntArray], indices: IntArray,
                   box: Union[Sequence[float], Sequence[int]]
                   ) -> Union[OctStackDouble, OctStackFloat, OctStackInt]:
    ...
