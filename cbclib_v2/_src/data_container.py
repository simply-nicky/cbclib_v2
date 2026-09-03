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
from dataclasses import dataclass, fields
from math import prod
from typing import (Any, DefaultDict, Dict, Generic, Iterable, Iterator, List, Protocol, Sequence,
                    Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints, overload)
from typing_extensions import Self
import numpy as np
from .array_api import array_namespace, ascupy, asjax, asnumpy
from .annotations import (Array, AnyNamespace, BoolArray, DataclassInstance, Indices, IntArray,
                          IntSequence, MultiIndices, NumPy, RealSequence, Shape)

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

    def to_xp(self: Self, xp: AnyNamespace) -> Self:
        """Return a copy with all arrays converted to the given array namespace.

        Non-array fields are left unchanged.

        Args:
            xp: Target array namespace (NumPy, JAX, or CuPy).

        Returns:
            New container instance with arrays in the target namespace.
        """
        name = getattr(xp, "__name__", "")
        if name == "numpy" or name.startswith("array_api_compat.numpy"):
            return self.to_numpy()
        if name.startswith("jax.numpy"):
            return self.to_jax()
        if name == "cupy":
            return self.to_cupy()
        raise ValueError(f"Unsupported array namespace: {name}")

def normalise_indices(indices: Tuple, shape: Shape) -> Tuple[Indices, ...]:
    ellipsis_count = sum(index is Ellipsis for index in indices)
    if ellipsis_count > 1:
        raise IndexError("An index can only contain one ellipsis")

    consumed = 0
    for index in indices:
        if index is Ellipsis:
            continue
        if isinstance(index, Array) and index.dtype == bool:
            consumed += index.ndim
        else:
            consumed += 1

    missing = len(shape) - consumed
    if missing < 0:
        raise IndexError(f"Too many indices for container with shape {shape}")

    normalized: List[Indices] = []
    for index in indices:
        if index is Ellipsis:
            normalized.extend([slice(None)] * missing)
        else:
            normalized.append(index)

    if ellipsis_count == 0:
        normalized.extend([slice(None)] * missing)

    return tuple(normalized)

def validate_shape(contents: Dict[str, Array], shape: Shape) -> None:
    for name, value in contents.items():
        if value.shape[:len(shape)] != shape:
            raise ValueError(f"Field '{name}' has shape {value.shape} "
                             f"which is incompatible with leading shape {shape}")

class ArrayContainer(DataContainer):
    """Container for dataclasses whose array fields share a common leading shape.

    Extends :class:`DataContainer` with field-wise ``concat``,
    ``stack``, integer/boolean ``__getitem__``, and ``reshape``.  The
    :attr:`shape` property returns the leading dimensions that are identical
    across all array fields.
    """
    def __post_init__(self):
        validate_shape(self.contents(), self.shape)

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
        return containers[0].replace(**result)

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
        return containers[0].replace(**result)

    @property
    def shape(self) -> Shape:
        raise NotImplementedError("ArrayContainer subclasses must implement the 'shape' property")

    @property
    def ndim(self) -> int:
        """Number of dimensions in the common leading shape.

        Returns:
            Length of :attr:`shape`.
        """
        return len(self.shape)

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
        if not isinstance(indices, tuple):
            common_indices = normalise_indices((indices,), self.shape)
        else:
            common_indices = normalise_indices(indices, self.shape)

        xp = self.__array_namespace__()
        data = {}
        for attr, value in self.contents().items():
            if isinstance(value, ArrayContainer):
                data[attr] = value[common_indices]
            else:
                payload_ndim = value.ndim - len(self.shape)
                if payload_ndim < 0:
                    raise ValueError(
                        f"Field {attr!r} has fewer dimensions than container shape "
                        f"{self.shape}"
                    )
                field_indices = common_indices + (slice(None),) * payload_ndim
                data[attr] = xp.asarray(value[field_indices])
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
class IndexLookup(DataContainer):
    order           : IntArray
    unique          : IntArray
    offsets         : IntArray

    @classmethod
    def build(cls, array: IntArray) -> IndexLookup:
        xp = array_namespace(array)
        values = xp.reshape(array, -1)
        order = xp.argsort(values, stable=True)
        sorted_values = values[order]

        unique, offsets = xp.unique(sorted_values, return_index=True)

        offsets = xp.concat((offsets, xp.array([values.size], dtype=int)))
        return cls(order, unique, offsets)

    def get_index(self, keys: IntSequence) -> Tuple[IntArray, IntArray]:
        xp = self.__array_namespace__()
        targets = xp.atleast_1d(keys)
        if targets.size == 0:
            empty = xp.asarray([], dtype=int)
            return empty, empty
        if self.unique.size == 0:
            raise KeyError(f"Index values are not present: {targets.tolist()}")

        locations = xp.searchsorted(self.unique, targets)
        clipped = xp.minimum(locations, self.unique.size - 1)

        valid = (
            (locations < self.unique.size)
            & (self.unique[clipped] == targets)
        )
        if not xp.all(valid):
            missing = targets[~valid]
            raise KeyError(f"Index values are not present: {missing.tolist()}")

        starts = self.offsets[locations]
        stops = self.offsets[locations + 1]

        chunks = [self.order[start:stop] for start, stop in zip(starts, stops)]
        positions = xp.concat(chunks)
        reset = xp.repeat(xp.arange(targets.size), stops - starts)
        return positions, reset

    def reset_index(self) -> IntArray:
        xp = self.__array_namespace__()
        counts = self.offsets[1:] - self.offsets[:-1]
        sorted_index = xp.repeat(xp.arange(self.unique.size), counts)
        inverse_order = xp.argsort(self.order)
        return sorted_index[inverse_order]

class Indexed(Protocol):
    """Protocol for objects that carry an integer index and support group-wise access.

    Any object implementing this protocol can be used with
    :class:`GenericIndexer`, :class:`ILocIndexer`, and :class:`LocIndexer`.
    """
    index       : IntArray

    def __array_namespace__(self) -> AnyNamespace: ...

    def __getitem__(self: I, indices: Indices | BoolArray) -> I: ...

    def replace(self: I, **kwargs: Any) -> I: ...

    def reset_index(self: I) -> IntArray: ...

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
    reset_index : bool = False

    def __getitem__(self, indices: IntSequence) -> I:
        return self.obj.take(indices, self.reset_index)

@dataclass
class ILocIndexer(GenericIndexer[I]):
    """Integer-location indexer — selects groups by their ordinal position.

    ``obj.iloc[i]`` returns the group whose index value is
    ``obj.unique_index()[i]``.  Supports scalar integers, slices,
    integer arrays, and :class:`IndexArray` objects.
    """
    def __getitem__(self, indices: slice | IntSequence) -> I:
        xp = self.obj.__array_namespace__()
        if isinstance(indices, int):
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

    def __getitem__(self, indices: slice | IntSequence) -> I:
        xp = self.obj.__array_namespace__()
        if isinstance(indices, int):
            idxs = xp.atleast_1d(indices)
        elif isinstance(indices, slice):
            start, stop, step = indices.indices(self.obj.index.size)
            idxs = list(range(start, stop, step))
        else:
            idxs = indices
        return super().__getitem__(idxs)

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
        if self.index.ndim != self.ndim:
            raise ValueError(
                f"Index shape {self.index.shape} is incompatible with "
                f"leading shape {self.shape}"
            )
        if self.ndim and self.index.shape[0] != self.shape[0]:
            raise ValueError(
                f"Index shape {self.index.shape} is incompatible with "
                f"leading shape {self.shape}"
            )

        xp = self.__array_namespace__()
        try:
            self.index = xp.broadcast_to(self.index, self.shape)
        except ValueError as error:
            raise ValueError(
                f"Index shape {self.index.shape} is incompatible with "
                f"leading shape {self.shape}"
            ) from error

        super().__post_init__()
        self._indexer = None

    @property
    def indexer(self) -> IndexLookup:
        if self._indexer is None:
            self._indexer = IndexLookup.build(self.index.reshape(-1))
        return self._indexer

    def __iter__(self: Self) -> Iterator[Self]:
        """Iterate over groups, yielding one container per unique index value.

        Yields:
            Container slice for each unique value in :attr:`index`.
        """
        for index in self.indexer.unique:
            yield self.take(index, reset_index=False)

    def __len__(self) -> int:
        """Return the number of unique index groups."""
        return self.indexer.unique.size

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

    def take(self: Self, indices: IntSequence, reset_index: bool=False) -> Self:
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
        flat = self.reshape()
        positions, new_index = self.indexer.get_index(indices)
        result = flat[positions]

        if reset_index:
            return result.replace(index=new_index)
        return result

    def reset(self: Self) -> Self:
        """Return a copy of the container with the reset index.

        Returns:
            New container instance with the reset index.
        """
        return self.replace(index=self.reset_index())

    def reset_index(self: Self) -> IntArray:
        """Return a reset index where each unique value in the original ``index``
        is replaced by its ordinal position (0, 1, 2, …).

        Returns:
            Reset index array.
        """
        xp = self.__array_namespace__()
        return xp.reshape(self.indexer.reset_index(), self.shape)

    def unique_index(self) -> IntArray:
        """Return the sorted array of unique index values.

        Returns:
            A flat array of unique index values.
        """
        return self.indexer.unique
