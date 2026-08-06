"""Container hierarchy for structured data in cbclib_v2.

Defines a four-level class hierarchy for holding typed, array-backed data:

* :class:`Container` — JSON-serialisable dataclass base (``to_dict`` /
  ``from_dict`` / ``replace``).
* :class:`DataContainer` — adds array-namespace awareness and
  ``to_numpy`` / ``to_jax`` / ``to_cupy`` conversions.
* :class:`ArrayContainer` — uniform-shape arrays; supports
  ``concat``, ``stack``, ``__getitem__``, and ``reshape``.
* :class:`IndexedContainer` — extends :class:`ArrayContainer` with an
  integer ``index`` field that groups rows into labelled frames; supports
  ``take``, ``loc``, and ``iloc`` accessors.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import InitVar, dataclass, fields
from math import prod
from typing import (Any, DefaultDict, Dict, Generic, Iterable, Iterator, List, Protocol, Sequence,
                    Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints, overload)
from typing_extensions import Self
import numpy as np
from .array_api import array_namespace, ascupy, asjax, asnumpy
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
    """Flatten a scalar, array, or nested sequence into a plain Python list.

    Handles strings, numpy scalars, arrays, and nested lists or tuples.
    Nested lists and tuples are flattened one level.

    Args:
        sequence: Input to convert.  May be a scalar, a 1-D array, a
            string, or a sequence whose elements are scalars, lists, or
            tuples.

    Returns:
        Flat Python list with all elements extracted.
    """
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
    return [index for index in indices if index < size]

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
    """JSON-serialisable dataclass base class.

    Every concrete subclass is a :func:`~dataclasses.dataclass`.
    :class:`Container` adds three operations on top of the standard dataclass
    machinery:

    * :meth:`to_dict` — deep-serialise to a plain ``dict``.
    * :meth:`from_dict` — reconstruct from a ``dict``, recursively
      deserialising nested :class:`Container` fields.
    * :meth:`replace` — return a modified copy (mirrors
      :func:`dataclasses.replace` but always returns the concrete subclass).

    The class is the root of the container hierarchy:
    :class:`Container` → :class:`DataContainer` → :class:`ArrayContainer`
    → :class:`IndexedContainer`.
    """

    def __reduce__(self) -> Tuple:
        return (self.__class__, tuple(getattr(self, field.name) for field in fields(self)))

    @classmethod
    def from_dict(cls: Type[Self], **values: Any) -> Self:
        """Reconstruct a container from a plain dictionary.

        Recursively deserialises nested :class:`Container` fields,
        ``Optional[Container]`` unions, and ``List[Container]`` /
        ``Tuple[Container, ...]`` collections.  Leaf fields are passed
        through unchanged.

        Args:
            **values: Keyword arguments matching the dataclass field names.

        Returns:
            New instance of the concrete subclass.
        """
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
        """Return a modified copy with selected fields replaced.

        Args:
            **kwargs: Field values to override.

        Returns:
            New instance of the same concrete type with the given fields
            replaced and all other fields copied from *self*.
        """
        return type(self)(**({f.name: getattr(self, f.name) for f in fields(self)} | kwargs))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the container to a plain dictionary.

        Nested :class:`Container` instances are converted recursively.
        ``list`` / ``tuple`` fields whose elements are :class:`Container`
        instances are converted element-wise.

        Returns:
            Mapping of field names to their serialised values.
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
    """Container base class with array-namespace awareness.

    Extends :class:`Container` by inspecting array-type fields to determine
    the active array backend (NumPy, JAX, or CuPy) and providing explicit
    device-transfer helpers.

    All array fields returned by :meth:`contents` are included in backend
    conversions; non-array fields (scalars, strings, nested containers) are
    carried over unchanged.
    """

    def __array_namespace__(self, api_version: str | None = None) -> AnyNamespace:
        """Return the array namespace shared by all array fields.

        Inspects the non-empty array fields via :meth:`contents` and
        delegates to :func:`~cbclib_v2.array_namespace`.  Falls back to
        :data:`~cbclib_v2.annotations.NumPy` when the container has no
        array fields.

        Returns:
            Active array namespace (NumPy, JaxNumPy, or CuPy).
        """
        contents = self.contents()

        if contents:
            return array_namespace(*contents.values())

        return NumPy

    @classmethod
    def is_empty(cls, data: Any) -> bool:
        """Return ``True`` when *data* is an array with zero elements."""
        if isinstance(data, Array):
            return data.size == 0
        return False

    def contents(self) -> Dict[str, Array]:
        """Return non-empty array fields as a ``{name: value}`` mapping.

        Only fields whose value passes the :meth:`is_empty` check are
        included.  Nested :class:`DataContainer` fields with non-empty
        contents are also included.

        Returns:
            Mapping from field name to array value for all initialised
            (non-empty) array fields.
        """
        data = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Array) and not self.is_empty(val):
                data[f.name] = val
            if isinstance(val, DataContainer) and len(val.contents()):
                data[f.name] = val
        return data

    def copy(self: Self) -> Self:
        xp = self.__array_namespace__()
        data = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Array):
                data[f.name] = xp.copy(val)
            elif isinstance(val, DataContainer):
                data[f.name] = val.copy()
        return self.replace(**data)

    def to_cupy(self: Self) -> Self:
        """Return a copy with all NumPy arrays converted to CuPy.

        Non-array fields are left unchanged.

        Returns:
            New container instance with CuPy arrays.
        """
        data = {}
        for attr, val in self.contents().items():
            if isinstance(val, Array):
                data[attr] = ascupy(val)
            if isinstance(val, DataContainer):
                data[attr] = val.to_cupy()
        return self.replace(**data)

    def to_jax(self: Self) -> Self:
        """Return a copy with all NumPy arrays converted to JAX arrays.

        Non-array fields are left unchanged.

        Returns:
            New container instance with JAX arrays.
        """
        data = {}
        for attr, val in self.contents().items():
            if isinstance(val, Array):
                data[attr] = asjax(val)
            if isinstance(val, DataContainer):
                data[attr] = val.to_jax()
        return self.replace(**data)

    def to_numpy(self: Self) -> Self:
        """Return a copy with all JAX or CuPy arrays converted to NumPy.

        Non-array fields are left unchanged.

        Returns:
            New container instance with NumPy arrays.
        """
        data = {}
        for attr, val in self.contents().items():
            if isinstance(val, Array):
                data[attr] = asnumpy(val)
            if isinstance(val, DataContainer):
                data[attr] = val.to_numpy()
        return self.replace(**data)

class ArrayContainer(DataContainer):
    """Container for dataclasses whose array fields share a common leading shape.

    Extends :class:`DataContainer` with field-wise ``concat``,
    ``stack``, integer/boolean ``__getitem__``, and ``reshape``.  The
    :attr:`shape` property returns the leading dimensions that are identical
    across all array fields.
    """

    @classmethod
    def is_empty(cls, data: Any) -> bool:
        """Return ``True`` when *data* is **not** an array (non-array fields are excluded)."""
        # Since ArrayContainer has a consistent leading shape
        # we can't treat empty arrays as empty containers.
        return not isinstance(data, Array)

    @classmethod
    def concat(cls: Type[Self], containers: Iterable[Self]) -> Self:
        """Concatenate a sequence of containers field-wise along axis 0.

        Args:
            containers: Non-empty iterable of container instances of the
                same concrete type.

        Returns:
            New container with all array fields concatenated.

        Raises:
            ValueError: If *containers* is empty.
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

        result = {}
        for key, vals in concatenated.items():
            if isinstance(vals[0], Array):
                result[key] = xp.concat(vals)
            if isinstance(vals[0], ArrayContainer):
                result[key] = type(vals[0]).concat(vals)
        return cls(**(defaults | result))

    @classmethod
    def stack(cls: Type[Self], containers: Iterable[Self], axis: int=0) -> Self:
        """Stack a sequence of containers along a new axis.

        Args:
            containers: Non-empty iterable of container instances of the
                same concrete type.
            axis: Axis along which to insert the new dimension.

        Returns:
            New container with all array fields stacked.

        Raises:
            ValueError: If *containers* is empty.
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

        result = {}
        for key, vals in stacked.items():
            if isinstance(vals[0], Array):
                result[key] = xp.stack(vals, axis=axis)
            if isinstance(vals[0], ArrayContainer):
                result[key] = type(vals[0]).stack(vals, axis=axis)
        return cls(**(defaults | result))

    @property
    def shape(self) -> Shape:
        """Common leading shape shared by all array fields.

        Returns the longest prefix of axis lengths that is identical across
        every array field returned by :meth:`contents`.

        Returns:
            Tuple of integers giving the common leading shape.

        Raises:
            ValueError: If no uniform shape prefix exists.
        """
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

    @property
    def size(self) -> int:
        """Total number of elements in the common leading shape.

        Returns:
            Product of the integers in :attr:`shape`.
        """
        return prod(self.shape)

    def __getitem__(self: Self, indices: MultiIndices | BoolArray) -> Self:
        """Index into the container, returning a new container of the same type.

        Only the fields returned by :meth:`contents` are indexed; other
        fields are copied unchanged.

        Args:
            indices: Indices or boolean mask applied to array fields.

        Returns:
            New container instance with the indexed array fields.
        """
        xp = self.__array_namespace__()
        data = {}
        for attr, val in self.contents().items():
            if isinstance(val, Array):
                data[attr] = xp.asarray(val[indices])
            if isinstance(val, ArrayContainer):
                data[attr] = val[indices]
        return self.replace(**data)

    def reshape(self: Self, shape: int | Sequence[int] | None=None) -> Self:
        """Reshape all array fields to *shape*.

        Args:
            shape: Target shape.  An integer is treated as a 1-tuple.
                ``None`` flattens all fields to 1-D.

        Returns:
            New container instance with reshaped array fields.
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
    """Split a sequence into *n_chunks* roughly equal parts.

    If the elements are :class:`ArrayContainer` subclasses the chunks are
    reassembled into container instances via
    :meth:`ArrayContainer.concat`.  For plain arrays a stacked array
    is yielded.  Otherwise a plain Python list is yielded per chunk.

    Args:
        containers: Sequence to split.  Elements may be containers, arrays,
            or arbitrary Python objects.
        n_chunks: Number of chunks (must be positive).

    Yields:
        Container instances, stacked arrays, or lists — one per chunk.
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
    """Wrapper around an integer index array backed by a C++ ``Indexer``.

    Stores a sorted integer array and exposes efficient group-lookup
    operations used by :class:`IndexedContainer`.  Internally delegates to
    :class:`~cbclib_v2._src.src.index.Indexer` for range-based lookups.

    The array is always stored as a contiguous 1-D NumPy ``int`` array;
    arrays on other devices are transferred to CPU at construction time.

    Attributes:
        index: C++ ``Indexer`` wrapping the raw integer array.
    """

    arr       : InitVar[IntArray]

    def __post_init__(self, arr: IntArray):
        xp = self.__namespace__ = NumPy
        arr = xp.asarray(asnumpy(arr), dtype=int)
        self.index = Indexer(xp.reshape(arr, (-1,) if arr.ndim == 0 else arr.shape))

    def __array_namespace__(self, api_version: str | None = None) -> AnyNamespace:
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
        """Raw 1-D NumPy integer array of index values."""
        return self.index.array

    @property
    def is_decreasing(self) -> bool:
        """``True`` if the index is strictly decreasing."""
        return self.index.is_decreasing

    @property
    def is_increasing(self) -> bool:
        """``True`` if the index is strictly increasing."""
        return self.index.is_increasing

    def get_index(self, key: IntSequence) -> Tuple[NDIntArray | slice, NDIntArray]:
        """Return row positions and reset indices for a key or sequence of keys.

        Args:
            key: A scalar index value or array of index values to look up.

        Returns:
            Tuple ``(row_selector, reset_index)`` where *row_selector* is a
            slice or integer array selecting the matching rows, and
            *reset_index* is a 0-based integer array for the selected rows.
        """
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
        """Return the sorted array of unique index values."""
        return self.index.unique()

    def reset(self) -> 'IndexArray':
        """Return a new :class:`IndexArray` with 0-based contiguous indices.

        Each unique value in the original index is replaced by its
        ordinal position (0, 1, 2, …).

        Returns:
            New :class:`IndexArray` with reset indices.
        """
        _, new_index = self.get_index(self.index.unique())
        return IndexArray(new_index)

class Indexed(Protocol):
    """Protocol for objects that carry an integer index and support group-wise access.

    Any object implementing this protocol can be used with
    :class:`GenericIndexer`, :class:`ILocIndexer`, and :class:`LocIndexer`.
    """
    index       : IntArray

    def __getitem__(self: I, indices: Indices | BoolArray) -> I: ...

    def replace(self: I, **kwargs: Any) -> I: ...

    def reset_index(self: I) -> I: ...

    def take(self: I, indices: IntSequence, reset_index: bool = False) -> I: ...

    def unique_index(self: I) -> IntArray: ...

@dataclass
class GenericIndexer(Generic[I]):
    """Group-wise indexer for :class:`Indexed` objects.

    Wraps an :class:`Indexed` object and provides ``__getitem__`` that
    selects rows by *index value* (not row position).

    Attributes:
        obj: The :class:`Indexed` object to index into.
    """
    obj         : I

    def __getitem__(self, indices: IntSequence) -> I:
        return self.obj.take(indices, reset_index=True)

@dataclass
class ILocIndexer(GenericIndexer[I]):
    """Integer-location indexer — selects groups by their ordinal position.

    ``obj.iloc[i]`` returns the group whose index value is
    ``obj.unique_index()[i]``.  Supports scalar integers, slices,
    integer arrays, and :class:`IndexArray` objects.
    """
    def __getitem__(self, indices: slice | IntSequence | IndexArray) -> I:
        xp = NumPy
        if isinstance(indices, IndexArray):
            idxs = self.obj.unique_index()[xp.asarray(indices)]
        elif isinstance(indices, int):
            idxs = self.obj.unique_index()[xp.atleast_1d(indices)]
        else:
            idxs = self.obj.unique_index()[indices]
        return super().__getitem__(idxs)

@dataclass
class LocIndexer(GenericIndexer[I]):
    """Label-based indexer — selects groups by their index value.

    ``obj.loc[v]`` returns all rows whose ``index`` equals *v*.  Supports
    scalar integers, integer arrays, slices over the index value range, and
    :class:`IndexArray` objects.
    """

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
    """Concatenate index arrays while preserving monotonicity.

    When the first element of a subsequent array is less than the last
    element of the previous one, the subsequent array is shifted upward so
    that the combined sequence remains non-decreasing.  This is used by
    :meth:`IndexedContainer.concat` to merge frame indices from
    multiple chunks without collisions.

    Args:
        arrays: Iterable of 1-D integer arrays to concatenate.
        xp: Array namespace for intermediate operations.

    Returns:
        Single concatenated integer array with preserved monotonicity.
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
    """Array container with an integer ``index`` field grouping rows into frames.

    Extends :class:`ArrayContainer` by treating the ``index`` field
    specially: it is excluded from :meth:`contents` (so it is not touched
    by field-wise operations) and handled explicitly by :meth:`concat`,
    :meth:`__getitem__`, :meth:`__iter__`, and :meth:`take`.

    Row order is always maintained sorted by ``index``; if the constructor
    receives an unsorted ``index``, all data fields are permuted accordingly.

    Attributes:
        index: Integer frame index for each row, shape ``(N,)``.
    """

    index       : IntArray

    def __post_init__(self):
        try:
            self._index = IndexArray(self.index)
        except ValueError:
            xp = self.__array_namespace__()
            indices = xp.argsort(self.index)
            for attr, val in self.contents().items():
                setattr(self, attr, val[indices])
            self.index = self.index[indices]
            self._index = IndexArray(self.index)

    @classmethod
    def concat(cls: Type[Self], containers: Iterable[Self]) -> Self:
        """Concatenate indexed containers while keeping indices unique.

        Data fields are concatenated field-wise; the combined ``index`` is
        built by :func:`concatenate_index` so that frame labels remain
        monotonically non-decreasing and do not collide across chunks.

        Args:
            containers: Non-empty iterable of container instances of the
                same concrete type.

        Returns:
            New container instance with concatenated data and adjusted
            index.
        """
        obj = super(IndexedContainer, cls).concat(containers)
        xp = obj.__array_namespace__()
        index = concatenate_index((container.index for container in containers), xp)
        return cls(**(obj.contents() | {'index': index}))

    @classmethod
    def stack(cls: Type[Self], containers: Iterable[Self], axis: int=0) -> Self:
        """Stack indexed containers along a new axis.

        Data fields are stacked field-wise; the ``index`` is taken from the
        first container.

        Args:
            containers: Non-empty iterable of container instances of the
                same concrete type.
            axis: Axis along which to insert the new dimension.

        Returns:
            New container instance with stacked data fields.

        Raises:
            ValueError: If *containers* is empty.
        """
        obj = super(IndexedContainer, cls).stack(containers, axis)
        for container in containers:
            return cls(**(obj.contents() | {'index': container.index}))

        raise ValueError("containers must not be empty")

    def __getitem__(self: Self, indices: MultiIndices | BoolArray) -> Self:
        """Index data fields and the corresponding rows of ``index``.

        Args:
            indices: Position indices or boolean mask applied to all fields.

        Returns:
            New container with the selected data rows and matching index
            entries.
        """
        xp = self.__array_namespace__()
        obj = super().__getitem__(indices)

        if isinstance(indices, tuple):
            index = self.index[indices[0]]
        elif isinstance(indices, Array) and indices.dtype == bool:
            index = xp.reshape(self.index, (self.index.size,) + (1,) * (indices.ndim - 1))
            index = xp.broadcast_to(index, indices.shape)[indices]
        else:
            index = self.index[indices]
        return type(self)(**(obj.contents() | {'index': xp.asarray(index)}))

    def __iter__(self: Self) -> Iterator[Self]:
        """Iterate over groups, yielding one container per unique index value.

        Yields:
            Container slice for each unique value in :attr:`index`.
        """
        for index in self._index.unique():
            indexer, _ = self._index.get_index(index)
            yield self[indexer]

    def __len__(self) -> int:
        """Return the number of unique index groups."""
        return self._index.unique().size

    @property
    def iloc(self: Self) -> ILocIndexer[Self]:
        """Integer-location indexer — select groups by ordinal position.

        Example:
            ``obj.iloc[0]`` returns the first group;
            ``obj.iloc[1:3]`` returns groups 1 and 2.
        """
        return ILocIndexer(self)

    @property
    def loc(self: Self) -> LocIndexer[Self]:
        """Label-based indexer — select groups by index value.

        Example:
            ``obj.loc[42]`` returns all rows with ``index == 42``.
        """
        return LocIndexer(self)

    def contents(self) -> Dict[str, Any]:
        """Return non-empty data fields, **excluding** the ``index`` field.

        The ``index`` attribute is intentionally omitted so that field-wise
        operations inherited from :class:`ArrayContainer` do not touch it.

        Returns:
            Mapping of data field names to values.
        """
        contents = super().contents()
        if 'index' in contents:
            del contents['index']
        return contents

    def reshape(self: Self, shape: int | Sequence[int] | None=None) -> Self:
        """Reshape all data fields and adjust ``index`` to match.

        The ``index`` is collapsed to 1-D by taking the first index value
        along every trailing axis, which requires all rows in a group to
        share the same index value.

        Args:
            shape: Target shape for data fields.  ``None`` flattens to 1-D.

        Returns:
            New container with reshaped data and adjusted index.

        Raises:
            ValueError: If the reshape is incompatible with the index
                grouping.
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
        return type(self)(**(obj.contents() | {'index': new_index}))

    def take(self: Self, indices: IntSequence, reset_index: bool = False) -> Self:
        """Select groups by index value and return the corresponding slice.

        Unlike ``__getitem__``, which operates on raw row positions,
        :meth:`take` looks up rows by their ``index`` value.

        Args:
            indices: A single index value or a sequence of index values to
                retrieve.

        Returns:
            New container containing all rows belonging to the requested
            index groups.
        """
        indexer, new_index = self._index.get_index(indices)
        if reset_index:
            return self[indexer].replace(index=new_index)
        return self[indexer]

    def reset_index(self: Self) -> Self:
        """Return a new container with reset indices and adjusted data fields.

        Each unique value in the original ``index`` is replaced by its
        ordinal position (0, 1, 2, …).  Data fields are permuted to match
        the new index order.

        Returns:
            New container with reset indices.
        """
        xp = self.__array_namespace__()
        return self.replace(index=xp.asarray(self._index.reset().array))

    def unique_index(self) -> IntArray:
        """Return the sorted array of unique index values.

        Returns:
            A flat array of unique index values.
        """
        xp = self.__array_namespace__()
        return xp.asarray(self._index.unique())
