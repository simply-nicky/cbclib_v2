"""Transforms are common image transformations. They can be chained together using
:class:`cbclib_v2.ComposeTransforms`. You pass a :class:`cbclib_v2.Transform` instance to a data
container :class:`cbclib_v2.CrystData`. All transform classes are inherited from the abstract
:class:`cbclib_v2.Transform` class.
"""
from collections import defaultdict
from dataclasses import InitVar, dataclass, fields
from math import prod
from typing import (Any, DefaultDict, Dict, Generic, Iterable, Iterator, List, Protocol, Sequence,
                    Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints, overload)
from typing_extensions import Self
import numpy as np
from .array_api import array_namespace, asjax, asnumpy
from .src.index import Indexer
from .annotations import (Array, AnyNamespace, BoolArray, DataclassInstance, DType, Indices,
                          IntArray, IntSequence, MultiIndices, NDArray, NDIntArray, NumPy,
                          RealSequence, Shape)

def compute_index(index: int, length: int) -> int:
    if index < 0:
        index = index + length
    if index < 0 or index >= length:
        raise ValueError(f'Index {index:d} is out of range [0, {length - 1:d}]')
    return index

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
        return to_list(sequence.reshape(-1).tolist())
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
    return isinstance(t, (type(List[int]), type(list[int])))

def is_union(t: Any) -> bool:
    return isinstance(t, (type(list | int), type(Union[list, int])))

def is_compound(t: Any) -> bool:
    return is_generic(t) or is_union(t)

def list_indices(indices: Indices, size: int) -> List[int]:
    if isinstance(indices, (int, np.integer)):
        return [indices,]
    if isinstance(indices, slice):
        start, stop, step = indices.indices(size)
        return list(range(start, stop, step))
    return list(indices)

def resolved_type(field: type['Container'], field_name: str,
                  data: Dict[str, Any]) -> type['Container']:
    if hasattr(field, 'type_resolver'):
        type_resolver = getattr(field, 'type_resolver')
        if callable(type_resolver):
            origin_type = type_resolver(data[field_name])
            if isinstance(origin_type, type) and issubclass(origin_type, Container):
                return origin_type
            raise ValueError(f"Type resolver for field '{field_name}' should return "
                             f"a dataclass instance, got {origin_type}")
    return field

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
    def from_dict(cls: Type[Self], **values: Any) -> Self:
        kwargs = {}
        types = get_type_hints(cls)
        for field in fields(cls):
            attr_type = types[field.name]
            value = values[field.name]

            # Handle Optional[Container] and Union[..., None]
            if is_union(attr_type) and type(None) in get_args(attr_type):
                if value is not None:
                    for t in get_args(attr_type):
                        if not is_compound(t) and issubclass(t, Container):
                            kwargs[field.name] = t.from_dict(**value)
                            break
                    else:
                        kwargs[field.name] = value
                else:
                    kwargs[field.name] = None

            # Handle Container types
            elif not is_compound(attr_type) and issubclass(attr_type, Container):
                attr_type = resolved_type(attr_type, field.name, values)
                kwargs[field.name] = attr_type.from_dict(**value)

            # Handle List[Container] and Tuple[Container, ...]
            elif is_generic(attr_type) and get_origin(attr_type) in (list, List, tuple, Tuple):
                elem_types = get_args(attr_type)
                if len(elem_types) == 2 and elem_types[1] is Ellipsis:
                    elem_type = elem_types[0]
                elif len(elem_types) == 1:
                    elem_type = elem_types[0]
                else:
                    elem_type = attr_type

                if not is_compound(elem_type) and issubclass(elem_type, Container):
                    elem_type = resolved_type(elem_type, field.name, values)
                    if get_origin(attr_type) in (tuple, Tuple):
                        kwargs[field.name] = tuple(elem_type.from_dict(**v) for v in value)
                    else:
                        kwargs[field.name] = [elem_type.from_dict(**v) for v in value]
                else:
                    kwargs[field.name] = value

            # Handle other types
            else:
                kwargs[field.name] = value
        return cls(**kwargs)

    def replace(self: Self, **kwargs: Any) -> Self:
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
            # Handle Container types
            if isinstance(value, Container):
                result[field.name] = value.to_dict()
            # Handle List[Container] and Tuple[Container, ...]
            elif isinstance(value, (list, tuple)):
                elements = []
                for elem in value:
                    if isinstance(elem, Container):
                        elements.append(elem.to_dict())
                    else:
                        elements.append(elem)
                result[field.name] = elements
            # Handle other types
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
    def __array_namespace__(self) -> AnyNamespace:
        contents = self.contents()

        if contents:
            return array_namespace(*contents.values())

        return NumPy

    @classmethod
    def is_empty(cls, data: Any) -> bool:
        """A field is considered non-empty if it is an array with non-zero size."""
        if isinstance(data, Array):
            return data.size == 0
        return False

    def contents(self) -> Dict[str, Array]:
        """Return the non-empty array fields stored in the container.

        Only fields whose value is not considered empty by
        :meth:`Container.is_empty` are included.

        Returns:
            Dict[str, Any]: Mapping from field name to field value for all
                initialized (non-empty) array fields.
        """
        data = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Array) and not self.is_empty(val):
                data[f.name] = val
            if isinstance(val, DataContainer) and len(val.contents()):
                data[f.name] = val
        return data

    def to_jax(self: Self) -> Self:
        """Return a copy of this container with NumPy arrays converted to JAX.

        Only attributes that are :class:`numpy.ndarray` are converted. Other
        values are left unchanged.

        Returns:
            A new container instance with converted arrays.
        """
        data = {}
        for attr, val in self.contents().items():
            if isinstance(val, Array):
                data[attr] = asjax(val)
            if isinstance(val, DataContainer):
                data[attr] = val.to_jax()
        return self.replace(**data)

    def to_numpy(self: Self) -> Self:
        """Return a copy of this container with JAX arrays converted to NumPy.

        Only attributes that are recognised as JAX arrays are converted. Other
        values are left unchanged.

        Returns:
            A new container instance with converted arrays.
        """
        data = {}
        for attr, val in self.contents().items():
            if isinstance(val, Array):
                data[attr] = asnumpy(val)
            if isinstance(val, DataContainer):
                data[attr] = val.to_numpy()
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
    def concatenate(cls: Type[Self], containers: Iterable[Self]) -> Self:
        """Concatenate a sequence of containers field-wise.

        For each field present in the container objects, the values are
        concatenated using the appropriate array namespace (NumPy or JAX)
        determined from the inputs.

        Args:
            containers: Iterable of container instances of the same concrete
                type to concatenate. Must be non-empty.

        Returns:
            A new container instance with concatenated array fields.

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
        result = {key: xp.concat(val) for key, val in concatenated.items()}
        return cls(**(defaults | result))

    @classmethod
    def stack(cls: Type[Self], containers: Iterable[Self], axis: int=0) -> Self:
        """Stack a sequence of containers along a new axis.

        Similar to :meth:`concatenate` but uses ``stack`` to create an extra
        axis. The axis parameter is forwarded to the underlying array
        ``stack`` implementation.

        Args:
            containers: Iterable of container instances of the same concrete
                type to stack. Must be non-empty.
            axis: Axis along which to stack the arrays.

        Returns:
            A new container instance with stacked array fields.

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
        ndim = 0
        for lengths in zip(*(val.shape for val in self.contents().values())):
            if len(lengths) == len(self.contents()):
                if all(l == lengths[0] for l in lengths):
                    shape.append(lengths[0])
                ndim += 1
        if len(shape) == 0 and ndim > 0:
            raise ValueError("No uniform shape found among array fields")
        return tuple(shape)

    def __getitem__(self: Self, indices: MultiIndices | BoolArray) -> Self:
        """Index into the container, returning a new container of the same type.

        Only the fields returned by :meth:`contents` are indexed; other fields
        present in :meth:`to_dict` are preserved as-is.

        Args:
            indices: Indices or boolean mask used to index array fields.

        Returns:
            A new container instance containing the indexed fields.
        """
        data = {attr: val[indices] for attr, val in self.contents().items()
                if isinstance(val, Array)}
        return self.replace(**data)

    def reshape(self: Self, shape: int | Sequence[int] | None=None) -> Self:
        """Reshape all array fields in the container to the given shape.

        Args:
            shape: New shape to apply to all array fields. If an integer is
                provided, it is treated as a single-element tuple. If ``None``, the arrays
                are flattened.

        Returns:
            A new container instance with reshaped array fields.
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

A = TypeVar("A", bound=ArrayContainer)
IC = TypeVar("IC", bound="IndexedContainer")

@overload
def split(containers: IC | Sequence[IC], n_chunks: int) -> Iterator[IC]: ...

@overload
def split(containers: Sequence[A], n_chunks: int) -> Iterator[A]: ...

@overload
def split(containers: Array | Sequence[Array], n_chunks: int) -> Iterator[Array]: ...

@overload
def split(containers: Sequence[Any], n_chunks: int) -> Iterator[List]: ...

def split(containers: IC | Array | Sequence[IC | A | Array | Any], n_chunks: int
          ) -> Iterator[IC | A | Array | List]:
    """Split an iterable of items into chunks of the given size.

    If the elements are container-like (subclasses of :class:`ArrayContainer`)
    the chunks are reassembled into instances of the same concrete type using
    :meth:`ArrayContainer.concatenate`. For plain arrays or JAX arrays, a
    stacked array is yielded. Otherwise, a plain Python list is yielded for
    each chunk.

    Args:
        containers: Iterable of items to chunk. Elements may be containers,
            NumPy/JAX arrays, or arbitrary Python objects.
        n_chunks: Number of chunks to split the input into (must be positive).

    Yields:
        Either container instances, stacked arrays, or lists depending on the
        element types in the input.
    """
    n_total = len(containers)
    q, r = divmod(n_total, n_chunks)

    chunks = []
    start = 0
    for chunk_size in [q + 1] * r + [q] * (n_chunks - r):
        chunks.append(containers[start:start + chunk_size])
        start += chunk_size

    yield from chunks

I = TypeVar("I", bound="Indexed")

@dataclass
class IndexArray():
    arr       : InitVar[IntArray]

    def __post_init__(self, arr: IntArray):
        xp = self.__namespace__ = NumPy
        arr = xp.asarray(asnumpy(arr), dtype=int)
        self.index = Indexer(xp.reshape(arr, (-1,) if arr.ndim == 0 else arr.shape))

    def __array_namespace__(self) -> AnyNamespace:
        return self.__namespace__

    def __reduce__(self) -> Tuple:
        return (self.__class__, (self.index.array,))

    def __array__(self, dtype: DType | None=None) -> NDArray:
        return NumPy.asarray(self.index.array, dtype=dtype)

    def __getitem__(self, idxs: MultiIndices | BoolArray) -> 'IndexArray':
        xp = self.__array_namespace__()
        return IndexArray(xp.asarray(self)[idxs])

    def __setitem__(self, idxs: MultiIndices, value: IntArray):
        xp = self.__array_namespace__()
        array = xp.asarray(self.index)
        array[idxs] = value
        self.index = Indexer(asnumpy(array))

    def __repr__(self) -> str:
        return f"IndexArray(index={self.index.array})"

    @property
    def array(self) -> NDIntArray:
        return self.index.array

    @property
    def is_decreasing(self) -> bool:
        return self.index.is_decreasing

    @property
    def is_increasing(self) -> bool:
        return self.index.is_increasing

    def get_index(self, key: IntSequence) -> Tuple[NDIntArray | slice, NDIntArray]:
        def to_indices(key: int | np.integer | Array) -> Tuple[slice, NDIntArray]:
            indices = self.index[int(key)]
            start, stop, step = indices.indices(self.index.array.size)
            return indices, NumPy.zeros((stop - start) // step, dtype=int)

        if isinstance(key, (int, np.integer)):
            return to_indices(key)

        if isinstance(key, Array):
            key = asnumpy(key)
            if key.ndim == 0:
                return to_indices(key)

        return self.index[key]

    def unique(self) -> NDIntArray:
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
    def __getitem__(self, indices: slice | IntSequence | IndexArray) -> I:
        xp = NumPy
        if isinstance(indices, IndexArray):
            idxs = self.obj.index_array.unique()[xp.asarray(indices)]
        elif isinstance(indices, int):
            idxs = self.obj.index_array.unique()[xp.atleast_1d(indices)]
        else:
            idxs = self.obj.index_array.unique()[indices]
        return super().__getitem__(idxs)

@dataclass
class LocIndexer(GenericIndexer[I]):
    def __getitem__(self, indices: slice | IntSequence | IndexArray) -> I:
        xp = NumPy
        if isinstance(indices, IndexArray):
            idxs = xp.asarray(indices)
        elif isinstance(indices, int):
            idxs = xp.atleast_1d(indices)
        elif isinstance(indices, slice):
            start, stop, step = indices.indices(self.obj.index.size)
            idxs = list(range(start, stop, step))
        else:
            idxs = indices
        return super().__getitem__(idxs)

def concatenate_index(arrays: Iterable[IntArray], xp: AnyNamespace=NumPy) -> IntArray:
    """Concatenate multiple index arrays while shifting following indices if they
    are lower than the last index of the previous array to preserve monotonicity.
    """
    indices, last = [], 0
    for array in arrays:
        array = xp.asarray(array)
        array = xp.reshape(array, (-1,) if array.ndim == 0 else array.shape)
        if len(array) != 0:
            if array[0] < last:
                index = array + last - array[0]
            else:
                index = array
            indices.append(index)
            last = int(index[-1]) + 1
    return xp.concat(indices)

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
        try:
            self.index_array = IndexArray(self.index)
        except ValueError:
            xp = self.__array_namespace__()
            indices = xp.argsort(self.index)
            for attr, val in self.contents().items():
                setattr(self, attr, val[indices])
            self.index = self.index[indices]
            self.index_array = IndexArray(self.index)

    @classmethod
    def concatenate(cls: Type[Self], containers: Iterable[Self]) -> Self:
        """Concatenate several indexed containers preserving group indices.

        This concatenates all data fields using the array namespace determined
        from the inputs, and then builds a combined ``index`` by offsetting
        the indices of each input container so they remain unique.

        Args:
            containers: Iterable of :class:`IndexedContainer` instances of the
                same concrete type. Must be non-empty.

        Returns:
            A new instance of the concrete subclass with concatenated
                data fields and adjusted ``index``.
        """
        obj = super(IndexedContainer, cls).concatenate(containers)
        xp = obj.__array_namespace__()
        index = concatenate_index((container.index for container in containers), xp)
        return cls(**(obj.to_dict() | {'index': index}))

    @classmethod
    def stack(cls: Type[Self], containers: Iterable[Self], axis: int=0) -> Self:
        obj = super(IndexedContainer, cls).stack(containers, axis)
        for container in containers:
            return cls(**(obj.to_dict() | {'index': container.index}))

        raise ValueError("containers must not be empty")

    def __getitem__(self: Self, indices: MultiIndices | BoolArray) -> Self:
        """Index the container and return a new container with sliced index.

        Args:
            indices: Position indices or boolean mask applied to data fields.

        Returns:
            A new container instance containing the indexed data and the
                corresponding entries of the ``index`` field.
        """
        obj = super().__getitem__(indices)
        if isinstance(indices, tuple):
            index = self.index[indices[0]]
        elif isinstance(indices, Array) and indices.dtype == bool:
            xp = self.__array_namespace__()
            index = xp.reshape(self.index, (self.index.size,) + (1,) * (indices.ndim - 1))
            index = xp.broadcast_to(index, indices.shape)[indices]
        else:
            index = self.index[indices]
        return type(self)(**(obj.to_dict() | {'index': index}))

    def __iter__(self: Self) -> Iterator[Self]:
        """Iterate over grouped items using unique indices.

        Yields:
            A new container instance corresponding to each unique value in
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
    def iloc(self: Self) -> ILocIndexer[Self]:
        """Indexer for integer-location based indexing.

        Returns:
            Helper that supports ``.iloc[...]`` style access.
        """
        return ILocIndexer(self)

    @property
    def loc(self: Self) -> LocIndexer[Self]:
        """Indexer for label/location based indexing.

        Returns:
            Helper that supports ``.loc[...]`` style access.
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

    def reshape(self: Self, shape: int | Sequence[int] | None=None) -> Self:
        """Reshape all data fields in the container to the given shape.

        The ``index`` field is preserved as-is.

        Args:
            shape: New shape to apply to all data fields. If an integer is
                provided, it is treated as a single-element tuple. If ``None``, the arrays
                are flattened.

        Returns:
            A new container instance with reshaped data fields.
        """
        obj = super().reshape(shape)
        if obj.shape[0] > prod(obj.shape):
            raise ValueError("Cannot reshape IndexedContainer: invalid shape for index")

        xp = self.__array_namespace__()
        sizes = xp.cumulative_prod(obj.shape)
        index_shape = xp.asarray(obj.shape)[sizes <= self.index.size]
        if prod(index_shape) != self.index.size:
            raise ValueError("Cannot reshape IndexedContainer: incompatible index size")

        old_index = self.index.reshape(index_shape)
        new_index = old_index.reshape((int(index_shape[0]), prod(index_shape[1:])))[:, 0]

        expanded_shape = (new_index.shape[0],) + (1,) * (len(index_shape) - 1) + new_index.shape[1:]
        expanded_index = xp.reshape(new_index, expanded_shape)
        if not xp.all(old_index == expanded_index):
            raise ValueError("Cannot reshape IndexedContainer: inconsistent index grouping")
        return type(self)(**(obj.to_dict() | {'index': new_index}))

    def take(self: Self, indices: IntSequence) -> Self:
        """Select elements by index groups and return the corresponding slice.

        Args:
            indices: Integer position or sequence of integer positions referring
                to groups (not raw row positions).

        Returns:
            A new container corresponding to the requested groups.
        """
        if isinstance(indices, int):
            indexer = self.index_array.get_index(indices)
        elif isinstance(indices, Array):
            indexer, _ = self.index_array.get_index(asnumpy(indices))
        else:
            indexer, _ = self.index_array.get_index(indices)
        return self[indexer]
