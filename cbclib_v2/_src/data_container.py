"""Transforms are common image transformations. They can be chained together using
:class:`cbclib_v2.ComposeTransforms`. You pass a :class:`cbclib_v2.Transform` instance to a data
container :class:`cbclib_v2.CrystData`. All transform classes are inherited from the abstract
:class:`cbclib_v2.Transform` class.
"""
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import (Any, DefaultDict, Dict, Iterable, Iterator, List, Sequence, Set, Sized, Tuple,
                    Type, TypeVar, get_origin, get_type_hints, overload)
import numpy as np
import jax.numpy as jnp
from .annotations import (Array, ArrayNamespace, BoolArray, DataclassInstance, Indices, IntArray,
                          IntSequence, JaxArray, JaxNumPy, NDArray, NumPy, NDIntArray, RealSequence,
                          Scalar)

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

    def contents(self) -> Dict[str, Any]:
        """Return a list of the attributes stored in the container that are initialised.

        Returns:
            List of the attributes stored in the container.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)
                if not isinstance(getattr(self, f.name), Sized) or len(getattr(self, f.name))}

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

class Transform(DataContainer):
    """Abstract transform class."""

    def index_array(self, ss_idxs: NDIntArray,
                    fs_idxs: NDIntArray) -> Tuple[NDIntArray, NDIntArray]:
        raise NotImplementedError

    def forward(self, inp: NDArray) -> NDArray:
        """Return a transformed image.

        Args:
            inp : Input image.

        Returns:
            Transformed image.
        """
        ss_idxs, fs_idxs = np.indices(inp.shape[-2:])
        ss_idxs, fs_idxs = self.index_array(ss_idxs, fs_idxs)
        return inp[..., ss_idxs, fs_idxs]

    def forward_points(self, x: Array, y: Array) -> Tuple[Array, Array]:
        raise NotImplementedError

    def backward(self, inp: NDArray, out: NDArray) -> NDArray:
        ss_idxs, fs_idxs = np.indices(out.shape[-2:])
        ss_idxs, fs_idxs = self.index_array(ss_idxs, fs_idxs)
        out[..., ss_idxs, fs_idxs] = inp
        return out

    def backward_points(self, x: Array, y: Array) -> Tuple[Array, Array]:
        raise NotImplementedError

@dataclass
class Crop(Transform):
    """Crop transform. Crops a frame according to a region of interest.

    Args:
        roi : Region of interest. Comprised of four elements ``[y_min, y_max, x_min, x_max]``.
    """
    roi : List[int] | Tuple[int, int, int, int] | NDIntArray

    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, Crop):
            return self.roi[0] == obj.roi[0] and self.roi[1] == obj.roi[1] and \
                   self.roi[2] == obj.roi[2] and self.roi[3] == obj.roi[3]
        return NotImplemented

    def __ne__(self, obj: object) -> bool:
        if isinstance(obj, Crop):
            return self.roi[0] != obj.roi[0] or self.roi[1] != obj.roi[1] or \
                   self.roi[2] != obj.roi[2] or self.roi[3] != obj.roi[3]
        return NotImplemented

    def index_array(self, ss_idxs: NDIntArray,
                    fs_idxs: NDIntArray) -> Tuple[NDIntArray, NDIntArray]:
        """Filter the indices of a frame ``(ss_idxs, fs_idxs)`` according to the cropping
        transform.

        Args:
            ss_idxs: Slow axis indices of a frame.
            fs_idxs: Fast axis indices of a frame.

        Returns:
            A tuple of filtered frame indices ``(ss_idxs, fs_idxs)``.
        """
        return (ss_idxs[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]],
                fs_idxs[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]])

    def forward_points(self, x: Array, y: Array) -> Tuple[Array, Array]:
        """Transform detector coordinates.

        Args:
            x : A set of  x coordinates.
            y : A set of y coordinates.

        Returns:
            A tuple of transformed x and y coordinates.
        """
        return x - self.roi[2], y - self.roi[0]

    def backward_points(self, x: Array, y: Array) -> Tuple[Array, Array]:
        """Transform detector coordinates back.

        Args:
            x : A set of transformed x coordinates.
            y : A set of transformed y coordinates.

        Returns:
            A tuple of x and y coordinates.
        """
        return x + self.roi[2], y + self.roi[0]

@dataclass
class Downscale(Transform):
    """Downscale the image by a integer ratio.

    Args:
        scale : Downscaling integer ratio.
    """
    scale : int

    def index_array(self, ss_idxs: NDIntArray,
                    fs_idxs: NDIntArray) -> Tuple[NDIntArray, NDIntArray]:
        """Filter the indices of a frame ``(ss_idxs, fs_idxs)`` according to the downscaling
        transform.

        Args:
            ss_idxs: Slow axis indices of a frame.
            fs_idxs: Fast axis indices of a frame.

        Returns:
            A tuple of filtered frame indices ``(ss_idxs, fs_idxs)``.
        """
        return (ss_idxs[::self.scale, ::self.scale], fs_idxs[::self.scale, ::self.scale])

    def forward_points(self, x: Array, y: Array) -> Tuple[Array, Array]:
        """Transform detector coordinates.

        Args:
            x : A set of  x coordinates.
            y : A set of y coordinates.

        Returns:
            A tuple of transformed x and y coordinates.
        """
        return x // self.scale, y // self.scale

    def backward_points(self, x: Array, y: Array) -> Tuple[Array, Array]:
        """Transform detector coordinates back.

        Args:
            x : A set of transformed x coordinates.
            y : A set of transformed y coordinates.

        Returns:
            A tuple of x and y coordinates.
        """
        return x * self.scale, y * self.scale

@dataclass
class Mirror(Transform):
    """Mirror the data around an axis.

    Args:
        axis : Axis of reflection.
        shape : Shape of the input array.
    """
    axis: int
    shape: Tuple[int, int]

    def index_array(self, ss_idxs: NDIntArray,
                    fs_idxs: NDIntArray) -> Tuple[NDIntArray, NDIntArray]:
        """Filter the indices of a frame ``(ss_idxs, fs_idxs)`` according to the mirroring
        transform.

        Args:
            ss_idxs: Slow axis indices of a frame.
            fs_idxs: Fast axis indices of a frame.

        Returns:
            A tuple of filtered frame indices ``(ss_idxs, fs_idxs)``.
        """
        if self.axis == 0:
            return (ss_idxs[::-1], fs_idxs[::-1])
        if self.axis == 1:
            return (ss_idxs[:, ::-1], fs_idxs[:, ::-1])
        raise ValueError('Axis must equal to 0 or 1')

    def forward_points(self, x: Array, y: Array) -> Tuple[Array, Array]:
        """Transform detector coordinates.

        Args:
            x : A set of  x coordinates.
            y : A set of y coordinates.

        Returns:
            A tuple of transformed x and y coordinates.
        """
        if self.axis:
            return x, self.shape[0] - y
        return self.shape[1] - x, y

    def backward_points(self, x: Array, y: Array) -> Tuple[Array, Array]:
        """Transform detector coordinates back.

        Args:
            x : A set of transformed x coordinates.
            y : A set of transformed y coordinates.

        Returns:
            A tuple of x and y coordinates.
        """
        return self.forward_points(x, y)

@dataclass
class ComposeTransforms(Transform):
    """Composes several transforms together.

    Args:
        transforms: List of transforms.
    """
    transforms : List[Transform]

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.transforms) < 2:
            raise ValueError('Two or more transforms are needed to compose')

        self.transforms = [transform.replace() for transform in self.transforms]

    def __iter__(self) -> Iterator[Transform]:
        return self.transforms.__iter__()

    @overload
    def __getitem__(self, index: int) -> Transform: ...

    @overload
    def __getitem__(self, index: slice) -> List[Transform]: ...

    def __getitem__(self, index: int | slice) -> Transform | List[Transform]:
        return self.transforms[index]

    def index_array(self, ss_idxs: NDIntArray,
                    fs_idxs: NDIntArray) -> Tuple[NDIntArray, NDIntArray]:
        """Filter the indices of a frame ``(ss_idxs, fs_idxs)`` according to the composed transform.

        Args:
            ss_idxs: Slow axis indices of a frame.
            fs_idxs: Fast axis indices of a frame.

        Returns:
            A tuple of filtered frame indices ``(ss_idxs, fs_idxs)``.
        """
        for transform in self:
            ss_idxs, fs_idxs = transform.index_array(ss_idxs, fs_idxs)
        return ss_idxs, fs_idxs

    def forward_points(self, x: Array, y: Array) -> Tuple[Array, Array]:
        """Transform detector coordinates.

        Args:
            x : A set of  x coordinates.
            y : A set of y coordinates.

        Returns:
            A tuple of transformed x and y coordinates.
        """
        for transform in self:
            x, y = transform.forward_points(x, y)
        return x, y

    def backward_points(self, x: Array, y: Array) -> Tuple[Array, Array]:
        """Transform detector coordinates back.

        Args:
            x : A set of transformed x coordinates.
            y : A set of transformed y coordinates.

        Returns:
            A tuple of x and y coordinates.
        """
        for transform in list(self)[::-1]:
            x, y = transform.backward_points(x, y)
        return x, y
