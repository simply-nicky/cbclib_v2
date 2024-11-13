from dataclasses import Field
from typing import (Any, Callable, ClassVar, Dict, Generic, List, Literal, Optional,
                    Protocol, Sequence, Set, Tuple, Type, TypeVar, Union, runtime_checkable)
import numpy as np
import numpy.typing as npt
from jax import Array as JaxArray

T = TypeVar('T')
Self = TypeVar('Self')

class ReferenceType(Generic[T]):
    __callback__: Callable[['ReferenceType[T]'], Any]
    def __new__(cls: type[Self], o: T,
                callback: Optional[Callable[['ReferenceType[T]'], Any]]=...) -> Self:
        ...
    def __call__(self) -> T:
        ...

@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Field[Any]]]

Scalar = Union[int, float, np.number]
Shape = Tuple[int, ...]

IntTuple = Tuple[int, ...]
FloatTuple = Tuple[float, ...]

JaxBoolArray = JaxArray
JaxComplexArray = JaxArray
JaxIntArray = JaxArray
JaxRealArray = JaxArray

NDArray = npt.NDArray[Any]
NDArrayLike = npt.ArrayLike
NDBoolArray = npt.NDArray[np.bool_]
NDComplexArray = npt.NDArray[Union[np.floating[Any], np.complexfloating[Any, Any]]]
NDIntArray = npt.NDArray[np.integer[Any]]
NDRealArray = npt.NDArray[np.floating[Any]]

Array = Union[JaxArray, NDArray]
BoolArray = Union[JaxBoolArray, NDBoolArray]
ComplexArray = Union[JaxComplexArray, NDComplexArray]
IntArray = Union[JaxIntArray, NDIntArray]
RealArray = Union[JaxRealArray, NDRealArray]

KeyArray = JaxArray

Indices = Union[int, slice, IntArray, Sequence[int], Tuple[IntArray, ...]]

IntSequence = Union[int, np.integer[Any], Sequence[int], IntArray]
CPPIntSequence = Union[Sequence[int], IntArray]
RealSequence = Union[float, np.floating[Any], Sequence[float], RealArray]
CPPRealSequence = Union[Sequence[float], RealArray]
ROI = Union[List[int], Tuple[int, int, int, int], IntArray]

PyTree = Any
Table = Dict[Tuple[int, int], float]
ExpandedType = Union[Type, Tuple[Type, List]]

Norm = Literal['backward', 'forward', 'ortho']
Mode = Literal['constant', 'nearest', 'mirror', 'reflect', 'wrap']

Line = List[float]
Streak = Tuple[Set[Tuple[int, int, float]], Dict[float, List[float]],
               Dict[float, List[int]], Line]
