"""Transforms are common image transformations. They can be chained together using
:class:`cbclib_v2.ComposeTransforms`. You pass a :class:`cbclib_v2.Transform` instance to a data
container :class:`cbclib_v2.CrystData`. All transform classes are inherited from the abstract
:class:`cbclib_v2.Transform` class.
"""
from collections import defaultdict
from dataclasses import InitVar, dataclass, fields
from typing import (Any, DefaultDict, Dict, Generic, Iterable, Iterator, List, Literal, Protocol,
                    Sequence, Set, Tuple, Type, TypeVar, Union, get_args, get_type_hints, overload)
import numpy as np
import jax.numpy as jnp
from .src.index import Indexer
from .annotations import (Array, ArrayNamespace, BoolArray, DataclassInstance, DType, Indices,
                          IntArray, IntSequence, JaxArray, JaxNumPy, NumPy, RealSequence, Scalar,
                          Shape, SupportsNamespace)

def add_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar,
           xp: ArrayNamespace=JaxNumPy) -> Array:
    if xp is jnp:
        return jnp.asarray(a).at[indices].add(b)
    np.add.at(np.asarray(a), indices, b)
    return np.asarray(a)

def argmin_at(a: Array, indices: IntArray, xp: ArrayNamespace=JaxNumPy) -> Array:
    sort_idxs = xp.argsort(a)
    idxs = set_at(xp.zeros(a.size, dtype=int), sort_idxs, xp.arange(a.size))
    result = xp.full(xp.unique(indices).size, a.size + 1, dtype=int)
    return sort_idxs[min_at(result, indices, idxs)]

def min_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar,
           xp: ArrayNamespace=JaxNumPy) -> Array:
    if xp is jnp:
        return jnp.asarray(a).at[indices].min(b)
    np.minimum.at(np.asarray(a), indices, b)
    return np.asarray(a)

def set_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar,
           xp: ArrayNamespace=JaxNumPy) -> Array:
    if xp is jnp:
        return jnp.asarray(a).at[indices].set(b)
    a[indices] = b
    return np.asarray(a)

@overload
def to_list(sequence: IntSequence) -> List[int]: ...

@overload
def to_list(sequence: RealSequence) -> List[float]: ...

@overload
def to_list(sequence: Sequence[str] | str) -> List[str]: ...

@overload
def to_list(sequence: Sequence[Any]) -> List[Any]: ...

def to_list(sequence: IntSequence |  RealSequence | str | Sequence[Any]
            ) -> List[int] | List[float] | List[str] | List[Any]:
    if isinstance(sequence, str):
        return [sequence,]
    if isinstance(sequence, (np.ndarray, JaxArray)):
        return to_list(sequence.tolist())
    if isinstance(sequence, (int, np.integer)):
        return [int(sequence),]
    if isinstance(sequence, (float, np.floating)):
        return [float(sequence),]
    return list(sequence)

def is_generic(t: Any) -> bool:
    return isinstance(t, (type(List[int]), type(Literal), type(list[int])))

def is_union(t: Any) -> bool:
    return isinstance(t, (type(list | int), type(Union[list, int])))

C = TypeVar("C", bound="Container")
D = TypeVar("D", bound="DataContainer")
A = TypeVar("A", bound="ArrayContainer")

class Container(DataclassInstance):
    @classmethod
    def from_dict(cls: Type[C], **values: Any) -> C:
        kwargs = {}
        types = get_type_hints(cls)
        for field in fields(cls):
            attr_type = types[field.name]
            value = values[field.name]
            if is_union(attr_type):
                if value is not None:
                    for t in get_args(attr_type):
                        if not is_generic(t) and issubclass(t, Container):
                            kwargs[field.name] = t.from_dict(**value)
                else:
                    kwargs[field.name] = value
            elif not is_generic(attr_type) and issubclass(attr_type, Container):
                kwargs[field.name] = attr_type.from_dict(**value)
            else:
                kwargs[field.name] = value
        return cls(**kwargs)

    @staticmethod
    def is_empty(data: Any) -> bool:
        return hasattr(data, 'size') and len(data) == 0

    def contents(self) -> Dict[str, Any]:
        """Return a list of the attributes stored in the container that are initialised.

        Returns:
            List of the attributes stored in the container.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)
                if not self.is_empty(getattr(self, f.name))}

    def replace(self: C, **kwargs: Any) -> C:
        """Return a new container object with a set of attributes replaced.

        Args:
            kwargs : A set of attributes and the values to to replace.

        Returns:
            A new container object with updated attributes.
        """
        return type(self)(**(self.to_dict() | kwargs))

    def to_dict(self) -> Dict[str, Any]:
        """Export the :class:`DataContainer` object to a :class:`dict`.

        Returns:
            A dictionary of :class:`DataContainer` object's attributes.
        """
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Container):
                result[field.name] = value.to_dict()
            else:
                result[field.name] = value
        return result

class DataContainer(Container):
    """Abstract data container class based on :class:`dataclass`. Has :class:`dict` interface,
    and :func:`DataContainer.replace` to create a new obj with a set of data attributes replaced.
    """
    def __post_init__(self):
        self.__namespace__ = array_namespace(*self.to_dict().values())

    def __array_namespace__(self) -> ArrayNamespace:
        return self.__namespace__

    def asjax(self: D) -> D:
        data = {attr: jnp.asarray(val) for attr, val in self.contents().items()
                if isinstance(val, np.ndarray)}
        return self.replace(**data)

    def asnumpy(self: D) -> D:
        data = {attr: np.asarray(val) for attr, val in self.contents().items()
                if isinstance(val, JaxArray)}
        return self.replace(**data)

class ArrayContainer(DataContainer):
    @classmethod
    def concatenate(cls: Type[A], containers: Iterable[A]) -> A:
        xp = array_namespace(*containers)
        concatenated : DefaultDict[str, List] = defaultdict(list)
        for container in containers:
            for key, val in container.contents().items():
                concatenated[key].append(val)
        result = {key: xp.concatenate(val) for key, val in concatenated.items()}
        result = {key: xp.array([]) for key in container.to_dict()} | result
        return cls(**result)

    def __getitem__(self: A, indices: Indices | BoolArray) -> A:
        data = self.to_dict() | {attr: val[indices] for attr, val in self.contents().items()}
        return self.replace(**data)

@overload
def split(containers: Iterable[A], size: int) -> Iterator[A]: ...

@overload
def split(containers: Iterable[Array], size: int) -> Iterator[Array]: ...

def split(containers: Iterable[A | Array], size: int) -> Iterator[A | Array]:
    chunk: List = []
    types: Set[Type[A | Array]] = set()

    for container in containers:
        chunk.append(container)
        types.add(type(container))

        if len(chunk) == size:
            if len(types) != 1:
                raise ValueError("Containers must have the same type")
            t = types.pop()
            if issubclass(t, ArrayContainer):
                yield t.concatenate(chunk)
            elif issubclass(t, (np.ndarray, np.number)):
                yield np.stack(chunk)
            elif issubclass(t, JaxArray):
                yield jnp.stack(chunk)
            else:
                raise ValueError(f"Containers have an invalid type: {t}")

            chunk.clear()

    if len(chunk) > 0:
        if len(types) != 1:
            raise ValueError("Containers must have the same type")
        t = types.pop()
        if issubclass(t, ArrayContainer):
            yield t.concatenate(chunk)
        elif issubclass(t, (np.ndarray, np.number)):
            yield np.stack(chunk)
        elif issubclass(t, JaxArray):
            yield jnp.stack(chunk)
        else:
            raise ValueError(f"Containers have an invalid type: {t}")

I = TypeVar("I", bound="Indexed")
IC = TypeVar("IC", bound="IndexedContainer")

@dataclass
class IndexArray():
    array       : InitVar[IntArray]

    def __post_init__(self, array: IntArray):
        xp = self.__namespace__ = array_namespace(array)
        self.index = Indexer(xp.asarray(xp.atleast_1d(array), dtype=int))

    def __array_namespace__(self) -> ArrayNamespace:
        return self.__namespace__

    def __reduce__(self) -> Tuple:
        return (self.__class__, (self.index.array,))

    # Comparisons

    def __eq__(self, other) -> BoolArray:
        return self.index.array.__eq__(other)

    def __ne__(self, other) -> BoolArray:
        return self.index.array.__ne__(other)

    def __lt__(self, other) -> BoolArray:
        return self.index.array.__lt__(other)

    def __le__(self, other) -> BoolArray:
        return self.index.array.__le__(other)

    def __gt__(self, other) -> BoolArray:
        return self.index.array.__gt__(other)

    def __ge__(self, other) -> BoolArray:
        return self.index.array.__ge__(other)

    # Logical Methods

    def __and__(self, other) -> IntArray:
        return self.index.array.__and__(other)

    def __rand__(self, other) -> IntArray:
        return self.index.array.__rand__(other)

    def __or__(self, other) -> IntArray:
        return self.index.array.__or__(other)

    def __ror__(self, other) -> IntArray:
        return self.index.array.__ror__(other)

    def __xor__(self, other) -> IntArray:
        return self.index.array.__xor__(other)

    def __rxor__(self, other) -> IntArray:
        return self.index.array.__rxor__(other)

    # Arithmetic Methods

    def __add__(self, other) -> IntArray:
        return self.index.array.__add__(other)

    def __radd__(self, other) -> IntArray:
        return self.index.array.__radd__(other)

    def __sub__(self, other) -> IntArray:
        return self.index.array.__sub__(other)

    def __rsub__(self, other) -> IntArray:
        return self.index.array.__rsub__(other)

    def __mul__(self, other) -> IntArray:
        return self.index.array.__mul__(other)

    def __rmul__(self, other) -> IntArray:
        return self.index.array.__rmul__(other)

    def __truediv__(self, other) -> IntArray:
        return self.index.array.__truediv__(other)

    def __rtruediv__(self, other) -> IntArray:
        return self.index.array.__rtruediv__(other)

    def __floordiv__(self, other) -> IntArray:
        return self.index.array.__floordiv__(other)

    def __rfloordiv__(self, other) -> IntArray:
        return self.index.array.__rfloordiv__(other)

    def __mod__(self, other) -> IntArray:
        return self.index.array.__mod__(other)

    def __rmod__(self, other) -> IntArray:
        return self.index.array.__rmod__(other)

    def __divmod__(self, other) -> IntArray:
        return self.index.array.__divmod__(other)

    def __rdivmod__(self, other) -> IntArray:
        return self.index.array.__rdivmod__(other)

    def __pow__(self, other) -> IntArray:
        return self.index.array.__pow__(other)

    def __rpow__(self, other) -> IntArray:
        return self.index.array.__rpow__(other)

    # Other Methods

    def __array__(self, dtype: DType | None=None) -> np.ndarray:
        return np.asarray(self.index.array, dtype=dtype)

    def __contains__(self, key: int) -> bool:
        return key in self.index.array

    def __getitem__(self, idxs: Indices) -> 'IndexArray':
        xp = self.__array_namespace__()
        return IndexArray(xp.asarray(self)[idxs])

    def __iter__(self) -> Iterator[np.integer[Any]]:
        return iter(self.index.array)

    def __len__(self) -> int:
        return len(self.index.array)

    def __repr__(self) -> str:
        return self.index.array.__repr__()

    def __setitem__(self, idxs: Indices, value: IntArray):
        xp = self.__array_namespace__()
        array = xp.asarray(self.index)
        array[idxs] = value
        self.index = Indexer(array)

    @property
    def is_decreasing(self) -> bool:
        return self.index.is_decreasing

    @property
    def is_increasing(self) -> bool:
        return self.index.is_increasing

    @property
    def size(self) -> int:
        return self.index.array.size

    @property
    def shape(self) -> Shape:
        return self.index.array.shape

    @overload
    def get_index(self, key: int) -> slice: ...

    @overload
    def get_index(self, key: IntSequence) -> Tuple[IntArray, IntArray]: ...

    def get_index(self, key: int | IntSequence) -> slice | Tuple[IntArray, IntArray]:
        return self.index[key]

    def unique(self) -> IntArray:
        return self.index.unique()

    def reset(self) -> 'IndexArray':
        return IndexArray(self.get_index(self.index.unique())[1])

class Indexed(Protocol):
    index       : IndexArray

    def __getitem__(self: I, indices: Indices | BoolArray) -> I: ...

    def replace(self: I, **kwargs: Any) -> I: ...

@dataclass
class GenericIndexer(Generic[I]):
    obj         : I

    def __getitem__(self, indices: IntSequence) -> I:
        indexer, new_index = self.obj.index.get_index(indices)
        return self.obj[indexer].replace(index=IndexArray(new_index))

@dataclass
class ILocIndexer(GenericIndexer[I]):
    def __getitem__(self, indices: IntSequence | IndexArray) -> I:
        if isinstance(indices, IndexArray):
            indices = self.obj.index.unique()[np.asarray(indices)]
        elif isinstance(indices, int):
            indices = self.obj.index.unique()[np.atleast_1d(indices)]
        else:
            indices = self.obj.index.unique()[indices]
        return super().__getitem__(indices)

@dataclass
class LocIndexer(GenericIndexer[I]):
    def __getitem__(self, indices: IntSequence | IndexArray) -> I:
        if isinstance(indices, IndexArray):
            indices = np.asarray(indices)
        elif isinstance(indices, int):
            indices = np.atleast_1d(indices)
        return super().__getitem__(indices)

class IndexedContainer(ArrayContainer):
    index       : IndexArray

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.index, IndexArray):
            self.index = IndexArray(self.index)

    def __iter__(self: IC) -> Iterator[IC]:
        for index in self.index.unique():
            yield self[self.index.get_index(index)]

    def __len__(self) -> int:
        return self.index.unique().size

    @property
    def iloc(self: IC) -> ILocIndexer[IC]:
        return ILocIndexer(self)

    @property
    def loc(self: IC) -> LocIndexer[IC]:
        return LocIndexer(self)

    def take(self: IC, indices: IntSequence) -> IC:
        if isinstance(indices, int):
            indexer = self.index.get_index(indices)
        else:
            indexer, _ = self.index.get_index(indices)
        return self[indexer]

def array_namespace(*arrays: SupportsNamespace | Any) -> ArrayNamespace:
    def namespaces(*arrays: SupportsNamespace | Any) -> Set:
        result = set()
        for array in arrays:
            if isinstance(array, dict):
                result |= namespaces(*array.values())
            elif isinstance(array, SupportsNamespace):
                result.add(array.__array_namespace__())
        return result

    nspaces = namespaces(*arrays)
    if len(nspaces) == 0:
        raise ValueError("namespace set should not be empty")
    return JaxNumPy if JaxNumPy in nspaces else NumPy
