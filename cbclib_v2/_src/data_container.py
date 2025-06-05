"""Transforms are common image transformations. They can be chained together using
:class:`cbclib_v2.ComposeTransforms`. You pass a :class:`cbclib_v2.Transform` instance to a data
container :class:`cbclib_v2.CrystData`. All transform classes are inherited from the abstract
:class:`cbclib_v2.Transform` class.
"""
from collections import defaultdict
from dataclasses import fields
from typing import (Any, DefaultDict, Dict, Iterable, Iterator, List, Sequence, Set, Sized, Tuple,
                    Type, TypeVar, get_origin, get_type_hints, overload)
import numpy as np
import jax.numpy as jnp
from .annotations import (Array, ArrayNamespace, BoolArray, DataclassInstance, Indices, IntArray,
                          IntSequence, JaxArray, JaxNumPy, NumPy, RealSequence, Scalar)

def add_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar,
           xp: ArrayNamespace = JaxNumPy) -> Array:
    if xp is jnp:
        return jnp.asarray(a).at[indices].add(b)
    np.add.at(np.asarray(a), indices, b)
    return np.asarray(a)

def argmin_at(a: Array, indices: IntArray, xp: ArrayNamespace = JaxNumPy) -> Array:
    sort_idxs = xp.argsort(a)
    idxs = set_at(xp.zeros(a.size, dtype=int), sort_idxs, xp.arange(a.size))
    result = xp.full(xp.unique(indices).size, a.size + 1, dtype=int)
    return sort_idxs[min_at(result, indices, idxs)]

def min_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar,
           xp: ArrayNamespace = JaxNumPy) -> Array:
    if xp is jnp:
        return jnp.asarray(a).at[indices].min(b)
    np.minimum.at(np.asarray(a), indices, b)
    return np.asarray(a)

def set_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar,
           xp: ArrayNamespace = JaxNumPy) -> Array:
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
            if get_origin(attr_type) is not None:
                attr_type = get_origin(attr_type)
            if issubclass(attr_type, Container):
                kwargs[field.name] = attr_type.from_dict(**values[field.name])
            else:
                kwargs[field.name] = attr_type(values[field.name])
        return cls(**kwargs)

    @staticmethod
    def is_empty(data: Any) -> bool:
        return isinstance(data, Sized) and len(data) == 0

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
        self.__namespace__ = array_namespace(*self.contents().values())

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
        result : DefaultDict[str, List] = defaultdict(list)
        for container in containers:
            for key, val in container.contents().items():
                result[key].append(val)
        return cls(**{key: xp.concatenate(val) for key, val in result.items()})

    def __getitem__(self: A, indices: Indices | BoolArray) -> A:
        data = {attr: None for attr in self.to_dict()}
        data = data | {attr: val[indices] for attr, val in self.contents().items()}
        return self.replace(**data)

def split(containers: Iterable[A], size: int) -> Iterator[A]:
    chunk: List[A] = []
    types: Set[Type[A]] = set()

    for container in containers:
        chunk.append(container)
        types.add(type(container))

        if len(chunk) == size:
            if len(types) != 1:
                raise ValueError("Containers must have the same type")
            t = types.pop()
            if not issubclass(t, ArrayContainer):
                raise ValueError(f"Containers have an invalid type: {t}")
            yield t.concatenate(chunk)

            chunk.clear()

    if len(chunk):
        if len(types) != 1:
            raise ValueError("Containers must have the same type")
        t = types.pop()
        if not issubclass(t, ArrayContainer):
            raise ValueError(f"Containers have an invalid type: {t}")
        yield t.concatenate(chunk)

I = TypeVar("I", bound="IndexedContainer")

class IndexedContainer(ArrayContainer):
    index       : IntArray

    def __post_init__(self):
        super().__post_init__()
        xp = self.__array_namespace__()
        self._indices = xp.asarray(xp.unique(self.index))

    def __iter__(self: I) -> Iterator[I]:
        for index in self._indices:
            yield self[self.index == index]

    def __len__(self) -> int:
        return self._indices.size

    def indices(self) -> IntArray:
        return self._indices

    def inverse(self) -> IntArray:
        xp = self.__array_namespace__()
        return xp.unique_inverse(self.index).inverse_indices

    def select(self: I, indices: IntSequence) -> I:
        xp = self.__array_namespace__()
        patterns = list(iter(self))
        result = [patterns[index].replace(index=xp.full(patterns[index].index.size, new_index))
                  for new_index, index in enumerate(to_list(indices))]
        return type(self).concatenate(result)

def array_namespace(*arrays: Array | DataContainer | Any) -> ArrayNamespace:
    namespaces = set(array.__array_namespace__() for array in arrays
                     if isinstance(array, (JaxArray, np.ndarray, DataContainer)))
    return JaxNumPy if JaxNumPy in namespaces else NumPy
