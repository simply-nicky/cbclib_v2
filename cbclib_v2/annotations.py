from typing import Any, Dict, List, Literal, NamedTuple, Sequence, Set, Tuple, Type, Union
import numpy as np
import numpy.typing as npt
from jax import Array as JaxArray

Scalar = Union[int, float, np.number]
Shape = Tuple[int, ...]

IntTuple = Tuple[int, ...]
FloatTuple = Tuple[float, ...]

JaxBoolArray = JaxArray
JaxIntArray = JaxArray
JaxRealArray = JaxArray
JaxComplexArray = JaxArray

NDArray = npt.NDArray[Any]
NDArrayLike = npt.ArrayLike
NDBoolArray = npt.NDArray[np.bool_]
NDIntArray = npt.NDArray[np.integer[Any]]
NDRealArray = npt.NDArray[np.floating[Any]]
NDComplexArray = npt.NDArray[Union[np.floating[Any], np.complexfloating[Any, Any]]]

Array = Union[JaxArray, NDArray]
BoolArray = Union[JaxBoolArray, NDBoolArray]
IntArray = Union[JaxIntArray, NDIntArray]
RealArray = Union[JaxRealArray, NDRealArray]

KeyArray = JaxArray

Indices = Union[int, slice, IntArray, Sequence[int]]

IntSequence = Union[int, np.integer[Any], Sequence[int], IntArray]
ROI = Union[List[int], Tuple[int, int, int, int], IntArray]
RealSequence = Union[float, np.floating[Any], Sequence[float], RealArray]
CPPIntSequence = Union[Sequence[int], IntArray]

PyTree = Any
Table = Dict[Tuple[int, int], float]
ExpandedType = Union[Type, Tuple[Type, List]]

Norm = Literal['backward', 'forward', 'ortho']
Mode = Literal['constant', 'nearest', 'mirror', 'reflect', 'wrap']

Line = List[float]
Streak = Tuple[Set[Tuple[int, int, float]], Dict[float, List[float]],
               Dict[float, List[int]], Line]

class Pattern(NamedTuple):
    index   : IntArray
    frames  : IntArray
    y       : IntArray
    x       : IntArray

class PatternWithHKL(NamedTuple):
    index   : IntArray
    frames  : IntArray
    y       : IntArray
    x       : IntArray
    rp      : RealArray
    h       : IntArray
    k       : IntArray
    l       : IntArray

class PatternWithHKLID(NamedTuple):
    index   : IntArray
    frames  : IntArray
    y       : IntArray
    x       : IntArray
    rp      : RealArray
    h       : IntArray
    k       : IntArray
    l       : IntArray
    hkl_id  : IntArray
