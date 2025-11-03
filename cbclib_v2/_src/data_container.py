"""Transforms are common image transformations. They can be chained together using
:class:`cbclib_v2.ComposeTransforms`. You pass a :class:`cbclib_v2.Transform` instance to a data
container :class:`cbclib_v2.CrystData`. All transform classes are inherited from the abstract
:class:`cbclib_v2.Transform` class.
"""
from collections import defaultdict
from dataclasses import InitVar, dataclass, fields
from math import prod
from typing import (Any, DefaultDict, Dict, Generic, Iterable, Iterator, List, Literal, Protocol,
                    Sequence, Set, Tuple, Type, TypeVar, Union, get_args, get_type_hints, overload)
import numpy as np
import jax.numpy as jnp
from .src.index import Indexer
from .annotations import (Array, ArrayNamespace, BoolArray, DataclassInstance, DType, Indices,
                          IntArray, IntSequence, JaxArray, JaxNumPy, MultiIndices, NDArray,
                          NumPy, RealSequence, Scalar, Shape, SupportsNamespace)

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

Item = TypeVar("Item")

@overload
def to_list(sequence: IntSequence) -> List[int]: ...

@overload
def to_list(sequence: RealSequence) -> List[float]: ...

@overload
def to_list(sequence: Sequence[str] | str) -> List[str]: ...

@overload
def to_list(sequence: Sequence[Item] | Sequence[List[Item] | Tuple[Item]]) -> List[Item]: ...

ToListSequence = Union[IntSequence, RealSequence, Sequence[str], str]

def to_list(sequence: ToListSequence | Sequence[Item] | Sequence[List[Item] | Tuple[Item]]
            ) -> List[int] | List[float] | List[str] | List[Item]:
    if isinstance(sequence, str):
        return [sequence,]
    if isinstance(sequence, Array):
        return to_list(sequence.ravel().tolist())
    if isinstance(sequence, (int, np.integer)):
        return [int(sequence),]
    if isinstance(sequence, (float, np.floating)):
        return [float(sequence),]
    result = []
    for item in sequence:
        if isinstance(item, (list, tuple)):
            result.extend(item)
        else:
            result.append(item)
    return result

def is_generic(t: Any) -> bool:
    return isinstance(t, (type(List[int]), type(Literal), type(list[int])))

def is_union(t: Any) -> bool:
    return isinstance(t, (type(list | int), type(Union[list, int])))

C = TypeVar("C", bound="Container")
D = TypeVar("D", bound="DataContainer")
A = TypeVar("A", bound="ArrayContainer")

class Container(DataclassInstance):
    """Lightweight dataclass-backed container base class.

    This class provides utilities for converting dataclass instances to and from
    dictionaries and for creating modified copies. It is expected that concrete
    data containers in this module inherit from `Container`.

    Attributes:
        (inherited from dataclass fields) Various typed attributes defined on
            subclasses.
    """
    def __reduce__(self) -> Tuple:
        return (self.__class__, tuple(getattr(self, field.name) for field in fields(self)))

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
                kwargs[field.name] = value
            elif not is_generic(attr_type) and issubclass(attr_type, Container):
                kwargs[field.name] = attr_type.from_dict(**value)
            else:
                kwargs[field.name] = value
        return cls(**kwargs)

    def replace(self: C, **kwargs: Any) -> C:
        """Create a new instance with selected fields replaced.

        This is a convenience that constructs a new object of the same type as
        ``self`` by merging the result of :meth:`to_dict` with ``kwargs``.

        Args:
            **kwargs: Field values to override on the new instance.

        Returns:
            C: A new instance of the same concrete container type with the
                provided fields replaced.
        """

        return type(self)(**({f.name: getattr(self, f.name) for f in fields(self)} | kwargs))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the container to a plain dictionary.

        Nested ``Container`` instances are converted recursively using their
        own :meth:`to_dict` implementation.

        Returns:
            Dict[str, Any]: Mapping of field names to their serialized values.
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
    """Base class for containers holding scalar and array-like data.

    Subclasses are dataclasses that represent structured collections of arrays
    or other values. This class wires up an array namespace (NumPy or JAX) and
    provides helpers to convert between NumPy and JAX arrays.

    Methods provided by this class operate on the dataclass fields and return
    appropriately-typed container instances (preserving the concrete subclass
    via :meth:`replace`).
    """
    def __post_init__(self):
        self.__namespace__ = array_namespace(*(getattr(self, f.name) for f in fields(self)))

    def __array_namespace__(self) -> ArrayNamespace:
        return self.__namespace__

    @classmethod
    def is_empty(cls, data: Any) -> bool:
        """A field is considered non-empty if it is an array with non-zero size."""
        if isinstance(data, Array):
            return data.size == 0
        return True

    def contents(self) -> Dict[str, Array]:
        """Return the non-empty array fields stored in the container.

        Only fields whose value is not considered empty by
        :meth:`Container.is_empty` are included.

        Returns:
            Dict[str, Any]: Mapping from field name to field value for all
                initialized (non-empty) array fields.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)
                if not self.is_empty(getattr(self, f.name))}

    def asjax(self: D) -> D:
        """Return a copy of this container with NumPy arrays converted to JAX.

        Only attributes that are :class:`numpy.ndarray` are converted. Other
        values are left unchanged.

        Returns:
            D: A new container instance with converted arrays.
        """
        data = {attr: jnp.asarray(val) for attr, val in self.contents().items()
                if isinstance(val, NDArray)}
        return self.replace(**data)

    def asnumpy(self: D) -> D:
        """Return a copy of this container with JAX arrays converted to NumPy.

        Only attributes that are recognised as JAX arrays are converted. Other
        values are left unchanged.

        Returns:
            D: A new container instance with converted arrays.
        """
        data = {attr: np.asarray(val) for attr, val in self.contents().items()
                if isinstance(val, JaxArray)}
        return self.replace(**data)

class ArrayContainer(DataContainer):
    """Container mixin for dataclasses that store only array-like fields with
    a uniform shape.

    Provides class methods to concatenate or stack multiple instances of the
    same concrete container type along a new leading dimension. The methods
    preserve the concrete subclass when constructing the result.
    """
    @classmethod
    def is_empty(cls, data: Any) -> bool:
        """A field is considered non-empty if it is an array."""
        return not isinstance(data, Array)

    @classmethod
    def concatenate(cls: Type[A], containers: Iterable[A]) -> A:
        """Concatenate a sequence of containers field-wise.

        For each field present in the container objects, the values are
        concatenated using the appropriate array namespace (NumPy or JAX)
        determined from the inputs.

        Args:
            containers: Iterable of container instances of the same concrete
                type to concatenate. Must be non-empty.

        Returns:
            A: A new container instance with concatenated array fields.

        Raises:
            ValueError: If ``containers`` is empty.
        """
        containers = list(containers)
        if len(containers) == 0:
            raise ValueError("containers must not be empty")

        defaults = {f.name: getattr(containers[0], f.name) for f in fields(containers[0])}

        xp = array_namespace(*containers)
        concatenated : DefaultDict[str, List] = defaultdict(list)
        for container in containers:
            for key, val in container.contents().items():
                concatenated[key].append(val)
        result = {key: xp.concatenate(val) for key, val in concatenated.items()}
        return cls(**(defaults | result))

    @classmethod
    def stack(cls: Type[A], containers: Iterable[A], axis: int=0) -> A:
        """Stack a sequence of containers along a new axis.

        Similar to :meth:`concatenate` but uses ``stack`` to create an extra
        axis. The axis parameter is forwarded to the underlying array
        ``stack`` implementation.

        Args:
            containers: Iterable of container instances of the same concrete
                type to stack. Must be non-empty.
            axis: Axis along which to stack the arrays.

        Returns:
            A: A new container instance with stacked array fields.

        Raises:
            ValueError: If ``containers`` is empty.
        """
        containers = list(containers)
        if len(containers) == 0:
            raise ValueError("containers must not be empty")

        defaults = {f.name: getattr(containers[0], f.name) for f in fields(containers[0])}

        xp = array_namespace(*containers)
        stacked : DefaultDict[str, List] = defaultdict(list)
        for container in containers:
            for key, val in container.contents().items():
                stacked[key].append(val)
        result = {key: xp.stack(val, axis=axis) for key, val in stacked.items()}
        return cls(**(defaults | result))

    @property
    def shape(self) -> Shape:
        shape: List[int] = []
        for lengths in zip(*(val.shape for val in self.contents().values())):
            if len(lengths) == len(self.contents()):
                if not all(l == lengths[0] for l in lengths):
                    raise ValueError("Inconsistent array shapes in the container")
                shape.append(lengths[0])
        return tuple(shape)

    def __getitem__(self: A, indices: MultiIndices | BoolArray) -> A:
        """Index into the container, returning a new container of the same type.

        Only the fields returned by :meth:`contents` are indexed; other fields
        present in :meth:`to_dict` are preserved as-is.

        Args:
            indices: Indices or boolean mask used to index array fields.

        Returns:
            A: A new container instance containing the indexed fields.
        """
        data = {attr: val[indices] for attr, val in self.contents().items()
                if isinstance(val, Array)}
        return self.replace(**data)

    def reshape(self: A, shape: int | Sequence[int] | None=None) -> A:
        """Reshape all array fields in the container to the given shape.

        Args:
            shape: New shape to apply to all array fields. If an integer is
                provided, it is treated as a single-element tuple. If ``None``, the arrays
                are flattened.

        Returns:
            A: A new container instance with reshaped array fields.
        """
        if shape is None:
            new_shape: Tuple[int, ...] = (-1,)
        elif isinstance(shape, int):
            new_shape = (shape,)
        else:
            new_shape = tuple(shape)

        data = {attr: val.reshape(new_shape + val.shape[len(self.shape):])
                for attr, val in self.contents().items()}
        return self.replace(**data)

@overload
def split(containers: Iterable[A], size: int) -> Iterator[A]: ...

@overload
def split(containers: Iterable[Array], size: int) -> Iterator[Array]: ...

@overload
def split(containers: Iterable[Any], size: int) -> Iterator[List]: ...

def split(containers: Iterable[A | Array | Any], size: int) -> Iterator[A | Array | List]:
    """Split an iterable of items into chunks of the given size.

    If the elements are container-like (subclasses of :class:`ArrayContainer`)
    the chunks are reassembled into instances of the same concrete type using
    :meth:`ArrayContainer.concatenate`. For plain arrays or JAX arrays, a
    stacked array is yielded. Otherwise, a plain Python list is yielded for
    each chunk.

    Args:
        containers: Iterable of items to chunk. Elements may be containers,
            NumPy/JAX arrays, or arbitrary Python objects.
        size: Chunk size (must be positive).

    Yields:
        Either container instances, stacked arrays, or lists depending on the
        element types in the input.
    """

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
                yield list(chunk)

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
            yield list(chunk)

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

    def __array__(self, dtype: DType | None=None) -> NDArray:
        return np.asarray(self.index.array, dtype=dtype)

    def __contains__(self, key: int) -> bool:
        return key in self.index.array

    def __getitem__(self, idxs: MultiIndices | BoolArray) -> 'IndexArray':
        xp = self.__array_namespace__()
        return IndexArray(xp.asarray(self)[idxs])

    def __iter__(self) -> Iterator[np.integer[Any]]:
        return iter(self.index.array)

    def __len__(self) -> int:
        return len(self.index.array)

    def __repr__(self) -> str:
        return self.index.array.__repr__()

    def __setitem__(self, idxs: MultiIndices, value: IntArray):
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

    def insert_index(self, key: IntSequence) -> Tuple[IntArray, IntArray]:
        return self.index.insert_index(key)

    def unique(self) -> IntArray:
        return self.index.unique()

    def reset(self) -> 'IndexArray':
        return IndexArray(self.get_index(self.index.unique())[1])

class Indexed(Protocol):
    index       : IntArray
    index_array : IndexArray

    def __getitem__(self: I, indices: Indices | BoolArray) -> I: ...

    def replace(self: I, **kwargs: Any) -> I: ...

@dataclass
class GenericIndexer(Generic[I]):
    obj         : I

    def __getitem__(self, indices: IntSequence) -> I:
        indexer, new_index = self.obj.index_array.get_index(indices)
        return self.obj[indexer].replace(index=new_index)

@dataclass
class ILocIndexer(GenericIndexer[I]):
    def __getitem__(self, indices: IntSequence | IndexArray) -> I:
        if isinstance(indices, IndexArray):
            indices = self.obj.index_array.unique()[np.asarray(indices)]
        elif isinstance(indices, int):
            indices = self.obj.index_array.unique()[np.atleast_1d(indices)]
        else:
            indices = self.obj.index_array.unique()[indices]
        return super().__getitem__(indices)

@dataclass
class LocIndexer(GenericIndexer[I]):
    def __getitem__(self, indices: IntSequence | IndexArray) -> I:
        if isinstance(indices, IndexArray):
            indices = np.asarray(indices)
        elif isinstance(indices, int):
            indices = np.atleast_1d(indices)
        return super().__getitem__(indices)

def concatenate_index(arrays: Iterable[IntArray], xp: ArrayNamespace=NumPy) -> IntArray:
    indices, last = [], 0
    for array in arrays:
        array = xp.atleast_1d(array)
        if len(array) != 0:
            if array[0] < last:
                index = array + last - array[0]
            else:
                index = array
            indices.append(index)
            last = int(index[-1]) + 1
    return xp.concatenate(indices)

class IndexedContainer(ArrayContainer):
    """Container with an integer index mapping items to groups.

    An :class:`IndexedContainer` stores array-like fields with a uniform shape
    (inherited from :class:`ArrayContainer`) together with an ``index`` array
    that groups rows or entries. The ``index`` field is not included in
    :meth:`contents` and is handled specially by methods such as :meth:`concatenate`
    and :meth:`take`.
    """
    index       : IntArray

    def __post_init__(self):
        super().__post_init__()
        try:
            self.index_array = IndexArray(self.index)
        except ValueError:
            xp = self.__array_namespace__()
            index = concatenate_index((chunk for chunk in self.index), xp)
            self.index_array = IndexArray(index).reset()
            self.index = xp.asarray(self.index_array)

    @classmethod
    def concatenate(cls: Type[IC], containers: Iterable[IC]) -> IC:
        """Concatenate several indexed containers preserving group indices.

        This concatenates all data fields using the array namespace determined
        from the inputs, and then builds a combined ``index`` by offsetting
        the indices of each input container so they remain unique.

        Args:
            containers: Iterable of :class:`IndexedContainer` instances of the
                same concrete type. Must be non-empty.

        Returns:
            IC: A new instance of the concrete subclass with concatenated
                data fields and adjusted ``index``.
        """
        obj = super(IndexedContainer, cls).concatenate(containers)
        xp = obj.__array_namespace__()
        index = concatenate_index((container.index for container in containers), xp)
        return cls(**(obj.to_dict() | {'index': index}))

    @classmethod
    def stack(cls: Type[IC], containers: Iterable[IC], axis: int=0) -> IC:
        obj = super(IndexedContainer, cls).stack(containers, axis)
        for container in containers:
            return cls(**(obj.to_dict() | {'index': container.index}))

        raise ValueError("containers must not be empty")

    def __getitem__(self: IC, indices: MultiIndices | BoolArray) -> IC:
        """Index the container and return a new container with sliced index.

        Args:
            indices: Position indices or boolean mask applied to data fields.

        Returns:
            IC: A new container instance containing the indexed data and the
                corresponding entries of the ``index`` field.
        """
        obj = super().__getitem__(indices)
        if isinstance(indices, tuple):
            index = self.index[indices[0]]
        elif isinstance(indices, Array) and indices.dtype == bool:
            xp = self.__array_namespace__()
            index = xp.expand_dims(self.index, list(range(1, indices.ndim)))
            index = xp.broadcast_to(index, indices.shape)[indices]
        else:
            index = self.index[indices]
        return type(self)(**(obj.to_dict() | {'index': index}))

    def __iter__(self: IC) -> Iterator[IC]:
        """Iterate over grouped items using unique indices.

        Yields:
            IC: Container slices corresponding to each unique value in
                ``self.index``.
        """
        for index in self.index_array.unique():
            yield self[self.index_array.get_index(index)]

    def __len__(self) -> int:
        """Return the number of unique index groups.

        Returns:
            int: Count of unique index values.
        """
        return self.index_array.unique().size

    @property
    def iloc(self: IC) -> ILocIndexer[IC]:
        """Indexer for integer-location based indexing.

        Returns:
            ILocIndexer[IC]: Helper that supports ``.iloc[...]`` style access.
        """
        return ILocIndexer(self)

    @property
    def loc(self: IC) -> LocIndexer[IC]:
        """Indexer for label/location based indexing.

        Returns:
            LocIndexer[IC]: Helper that supports ``.loc[...]`` style access.
        """
        return LocIndexer(self)

    def contents(self) -> Dict[str, Any]:
        """Return the non-empty data fields, excluding the index field.

        The ``index`` attribute is intentionally excluded from the returned
        mapping since it is treated specially by methods on this class.

        Returns:
            Dict[str, Any]: Mapping of data field names to values.
        """
        contents = super().contents()
        if 'index' in contents:
            del contents['index']
        return contents

    def reshape(self: IC, shape: int | Sequence[int] | None=None) -> IC:
        """Reshape all data fields in the container to the given shape.

        The ``index`` field is preserved as-is.

        Args:
            shape: New shape to apply to all data fields. If an integer is
                provided, it is treated as a single-element tuple. If ``None``, the arrays
                are flattened.

        Returns:
            IC: A new container instance with reshaped data fields.
        """
        obj = super().reshape(shape)
        xp = self.__array_namespace__()
        sizes = xp.cumprod(obj.shape)
        new_shape = xp.asarray(obj.shape)[sizes <= self.index.size]
        index = self.index.reshape((new_shape[0], -1))
        new_index = index[:, 0]
        if prod(index.shape[1:]) > 1 and not xp.all(index == xp.expand_dims(new_index, 1)):
            raise ValueError("Cannot reshape IndexedContainer: inconsistent index grouping")
        return type(self)(**(obj.to_dict() | {'index': new_index}))

    def take(self: IC, indices: IntSequence) -> IC:
        """Select elements by index groups and return the corresponding slice.

        Args:
            indices: Integer position or sequence of integer positions referring
                to groups (not raw row positions).

        Returns:
            IC: A new container corresponding to the requested groups.
        """
        if isinstance(indices, int):
            indexer = self.index_array.get_index(indices)
        else:
            indexer, _ = self.index_array.get_index(indices)
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
