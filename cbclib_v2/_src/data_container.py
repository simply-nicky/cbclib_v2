"""Transforms are common image transformations. They can be chained together using
:class:`cbclib.ComposeTransforms`. You pass a :class:`cbclib.Transform` instance to a data
container :class:`cbclib.CrystData`. All transform classes are inherited from the abstract
:class:`cbclib.Transform` class.
"""
from configparser import ConfigParser
from dataclasses import dataclass, fields
import json
import os
import re
from typing import (Any, Callable, ClassVar, Dict, Iterator, List, Tuple, Type, Union, TypeVar,
                    get_args, get_origin, overload)
import numpy as np
import jax.numpy as jnp
from .annotations import Array, DataclassInstance, ExpandedType, NDArray, JaxArray, NDIntArray

D = TypeVar("D", bound="DataContainer")

class DataContainer(DataclassInstance):
    """Abstract data container class based on :class:`dataclass`. Has :class:`dict` interface,
    and :func:`DataContainer.replace` to create a new obj with a set of data attributes replaced.
    """
    def contents(self) -> Dict[str, Any]:
        """Return a list of the attributes stored in the container that are initialised.

        Returns:
            List of the attributes stored in the container.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)
                if getattr(self, field.name) is not None}

    def replace(self: D, **kwargs: Any) -> D:
        """Return a new container object with a set of attributes replaced.

        Args:
            kwargs : A set of attributes and the values to to replace.

        Returns:
            A new container object with updated attributes.
        """
        return type(self)(**(self.to_dict() | kwargs))

    def to_dict(self) -> Dict[str, Any]:
        """Export the :class:`Sample` object to a :class:`dict`.

        Returns:
            A dictionary of :class:`Sample` object's attributes.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}

class BaseFormatter:
    aliases : ClassVar[Tuple[Type, ...]]

    @classmethod
    def is_instance(cls, t: ExpandedType) -> bool:
        if isinstance(t, tuple):
            return any(t[0] is alias for alias in cls.aliases)

        return any(t is alias for alias in cls.aliases)

class SimpleFormatter(BaseFormatter):
    @classmethod
    def format_string(cls, string: str) -> Any:
        return cls.aliases[0](string)

class Formatter(BaseFormatter):
    @classmethod
    def format_string(cls, string: str, dtype: Type) -> Any:
        raise NotImplementedError

class FloatFormatter(SimpleFormatter):
    aliases = (float, np.floating)

class IntFormatter(SimpleFormatter):
    aliases = (int, np.integer)

class BoolFormatter(SimpleFormatter):
    aliases = (bool,)

class StringFormatter(SimpleFormatter):
    aliases = (str,)

class ListFormatter(Formatter):
    aliases = (list,)

    @classmethod
    def format_string(cls, string: str, dtype: Type) -> List:
        is_list = re.search(r'^\[([\s\S]*)\]$', string)
        if is_list:
            return [dtype(p.strip('\'\"'))
                    for p in re.split(r'\s*,\s*', is_list.group(1)) if p]
        raise ValueError(f"Invalid string: '{string}'")

class TupleFormatter(Formatter):
    aliases = (tuple,)

    @classmethod
    def format_string(cls, string: str, dtype: Type) -> Tuple:
        is_tuple = re.search(r'^\(([\s\S]*)\)$', string)
        if is_tuple:
            return tuple(dtype(p.strip('\'\"'))
                         for p in re.split(r'\s*,\s*', is_tuple.group(1)) if p)
        raise ValueError(f"Invalid string: '{string}'")

class NDArrayFormatter(Formatter):
    aliases = (np.ndarray,)

    @classmethod
    def format_string(cls, string: str, dtype: Type) -> NDArray:
        is_list = re.search(r'^\[([\s\S]*)\]$', string)
        if is_list:
            return np.fromstring(is_list.group(1), dtype=dtype, sep=',')
        raise ValueError(f"Invalid string: '{string}'")

class JaxArrayFormatter(Formatter):
    aliases = (JaxArray,)

    @classmethod
    def format_string(cls, string: str, dtype: Type) -> JaxArray:
        is_list = re.search(r'^\[([\s\S]*)\]$', string)
        if is_list:
            return jnp.fromstring(is_list.group(1), dtype=dtype, sep=',')
        raise ValueError(f"Invalid string: '{string}'")

class StringFormatting:
    FormatterDict = Dict[str, Union[Type[SimpleFormatter], Type[Formatter]]]
    formatters : FormatterDict = {'ndarray': NDArrayFormatter,
                                  'list': ListFormatter,
                                  'tuple': TupleFormatter,
                                  'Array': JaxArrayFormatter,
                                  'float': FloatFormatter,
                                  'int': IntFormatter,
                                  'bool': BoolFormatter,
                                  'str': StringFormatter}
    @classmethod
    def expand_types(cls, t: Type) -> ExpandedType:
        origin = get_origin(t)
        if origin is None:
            return t
        return (origin, [cls.expand_types(arg) for arg in get_args(t)])

    @classmethod
    def list_types(cls, expanded_type: ExpandedType) -> List[ExpandedType]:
        def add_type(t: ExpandedType, types: List):
            if not isinstance(t, tuple):
                types.append(t)
            else:
                origin, args = t
                types.append((origin, args))
                if len(args) > 1:
                    for arg in args:
                        add_type(arg, types)

        types = []
        add_type(expanded_type, types)
        return types

    @classmethod
    def flatten_list(cls, nested_list: List) -> List:
        if not bool(nested_list):
            return nested_list

        if isinstance(nested_list[0], (list, tuple)):
            return cls.flatten_list(*nested_list[:1]) + list(cls.flatten_list(nested_list[1:]))

        return list(nested_list[:1]) + cls.flatten_list(nested_list[1:])

    @classmethod
    def get_dtype(cls, types: List) -> Type:
        formatters = [formatter for formatter in cls.formatters.values()
                      if issubclass(formatter, SimpleFormatter)]
        for formatter in formatters:
            for t in types:
                if formatter.is_instance(t):
                    return formatter.aliases[0]
        return float

    @classmethod
    def get_formatter(cls, t: Type) -> Callable[[str,], Any]:
        types = cls.list_types(cls.expand_types(t))
        for formatter in cls.formatters.values():
            for extended_type in types:
                if formatter.is_instance(extended_type):
                    if issubclass(formatter, SimpleFormatter):
                        return formatter.format_string

                    if isinstance(extended_type, tuple):
                        dtype = cls.get_dtype(cls.flatten_list(extended_type[1]))
                    else:
                        dtype = float

                    return lambda string: formatter.format_string(string, dtype)
        return str

    @classmethod
    def format_dict(cls, dct: Dict[str, Any], types: Dict[str, Type]) -> Dict[str, Any]:
        formatted_dct = {}
        for attr, val in dct.items():
            formatter = cls.get_formatter(types[attr])
            if isinstance(val, dict):
                formatted_dct[attr] = {k: formatter(v) for k, v in val.items()}
            if isinstance(val, str):
                formatted_dct[attr] = formatter(val)
        return formatted_dct

    @overload
    @classmethod
    def to_string(cls, node: Dict[str, Any]) -> Dict[str, str]:
        ...

    @overload
    @classmethod
    def to_string(cls, node: List[Any]) -> List[str]:
        ...

    @overload
    @classmethod
    def to_string(cls, node: Union[Any, Array]) -> str:
        ...

    @classmethod
    def to_string(cls, node: Union[Any, Dict[str, Any], List[Any], Array]
                  ) -> Union[str, List[str], Dict[str, str]]:
        if isinstance(node, dict):
            return {k: cls.to_string(v) for k, v in node.items()}
        if isinstance(node, list):
            return [cls.to_string(v) for v in node]
        if isinstance(node, (np.ndarray, JaxArray)):
            return np.array2string(np.array(node), separator=',')
        return str(node)

    @staticmethod
    def str_to_list(strings: Union[str, List[str]]) -> List[str]:
        """Convert `strings` to a list of strings.

        Args:
            strings : String or a list of strings

        Returns:
            List of strings.
        """
        if isinstance(strings, str):
            return [strings,]
        return list(str(string) for string in strings)

class Parser():
    fields : Dict[str, Union[str, List[str], Dict[str, str]]]

    def read_all(self, file: str) -> Dict[str, Any]:
        raise NotImplementedError

    def read(self, file: str) -> Dict[str, Any]:
        """Initialize the container object with an INI file ``file``.

        Args:
            file : Path to the ini file.

        Returns:
            A new container with all the attributes imported from the ini file.
        """
        parser = self.read_all(file)

        result: Dict[str, Any] = {}
        for section, attrs in self.fields.items():
            if isinstance(attrs, str):
                result[attrs] = dict(parser[section])
            elif isinstance(attrs, tuple):
                for attr in attrs:
                    result[attr] = parser[section][attr]
            elif isinstance(attrs, dict):
                for key, attr in attrs.items():
                    result[attr] = parser[section][key]
            else:
                raise TypeError(f"Invalid 'fields' values: {attrs}")

        return result

    def to_dict(self, obj: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for section, attrs in self.fields.items():
            if isinstance(attrs, str):
                result[section] = getattr(obj, attrs)
            if isinstance(attrs, tuple):
                result[section] = {attr: getattr(obj, attr) for attr in attrs}
            if isinstance(attrs, dict):
                result[section] = {key: getattr(obj, attr) for key, attr in attrs.items()}
        return result

    def write(self, file: str, obj: Any):
        raise NotImplementedError

@dataclass
class INIParser(Parser, DataContainer):
    """Abstract data container class based on :class:`dataclass` with an interface to read from
    and write to INI files.
    """
    fields : Dict[str, Union[str, Tuple[str, ...]]]
    types : Dict[str, Type]

    def read_all(self, file: str) -> Dict[str, Any]:
        if not os.path.isfile(file):
            raise ValueError(f"File {file} doesn't exist")

        ini_parser = ConfigParser()
        ini_parser.read(file)

        return {section: dict(ini_parser.items(section)) for section in ini_parser.sections()}

    def read(self, file: str) -> Dict[str, Any]:
        return StringFormatting.format_dict(super().read(file), self.types)

    def to_dict(self, obj: Any) -> Dict[str, str]:
        return StringFormatting.to_string(super().to_dict(obj))

    def write(self, file: str, obj: Any):
        """Save all the attributes stored in the container to an INI file ``file``.

        Args:
            file : Path to the ini file.
        """
        ini_parser = ConfigParser()
        for section, val in self.to_dict(obj).items():
            ini_parser[section] = val

        with np.printoptions(precision=None):
            with open(file, 'w') as out_file:
                ini_parser.write(out_file)

@dataclass
class JSONParser(Parser, DataContainer):
    fields: Dict[str, Union[str, Tuple[str, ...]]]

    def read_all(self, file: str) -> Dict[str, Any]:
        with open(file, 'r') as f:
            json_dict = json.load(f)

        return json_dict

    def write(self, file: str, obj: Any):
        with open(file, 'w') as out_file:
            json.dump(self.to_dict(obj), out_file, sort_keys=True, ensure_ascii=False, indent=4)

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
    roi : Union[List[int], Tuple[int, int, int, int], NDIntArray]

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
        if len(self.transforms) < 2:
            raise ValueError('Two or more transforms are needed to compose')

        self.transforms = [transform.replace() for transform in self.transforms]

    def __iter__(self) -> Iterator[Transform]:
        return self.transforms.__iter__()

    def __getitem__(self, idx: Union[int, slice]) -> Union[Transform, List[Transform]]:
        return self.transforms[idx]

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
