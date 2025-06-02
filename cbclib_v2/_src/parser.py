from configparser import ConfigParser
from dataclasses import dataclass
from enum import Enum
import json
import os
import re
from typing import Any, Callable, ClassVar, Dict, List, Tuple, Type, get_args, get_origin, overload
import numpy as np
import jax.numpy as jnp
from .data_container import DataContainer
from .annotations import Array, ExpandedType, JaxArray, NDArray

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
    FormatterDict = Dict[str, Type[SimpleFormatter] | Type[Formatter]]
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
    def to_string(cls, node: Dict[str, Any]) -> Dict[str, str]: ...

    @overload
    @classmethod
    def to_string(cls, node: List) -> List[str]: ...

    @overload
    @classmethod
    def to_string(cls, node: Array | Any) -> str: ...

    @classmethod
    def to_string(cls, node: Any | Dict[str, Any] | List | Array
                  ) -> str | List[str] | Dict[str, str]:
        if isinstance(node, dict):
            return {k: cls.to_string(v) for k, v in node.items()}
        if isinstance(node, list):
            return [cls.to_string(v) for v in node]
        if isinstance(node, (np.ndarray, JaxArray)):
            return np.array2string(np.array(node), separator=',')
        if isinstance(node, Enum):
            return str(node.value)
        return str(node)

class Parser():
    fields : Dict[str, str | List[str] | Dict[str, str]]

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
    fields : Dict[str, str | Tuple[str, ...]]
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
    fields: Dict[str, str | Tuple[str, ...]]

    def read_all(self, file: str) -> Dict[str, Any]:
        with open(file, 'r') as f:
            json_dict = json.load(f)

        return json_dict

    def to_dict(self, obj: Any) -> Dict[str, Any]:
        def array_to_list(**values: Any) -> Dict[str, Any]:
            result = {}
            for key, val in values.items():
                if isinstance(val, dict):
                    result[key] = array_to_list(**val)
                elif isinstance(val, (np.ndarray, JaxArray)):
                    result[key] = val.tolist()
                else:
                    result[key] = val
            return result

        return array_to_list(**super().to_dict(obj))

    def write(self, file: str, obj: Any):
        with open(file, 'w') as out_file:
            json.dump(self.to_dict(obj), out_file, sort_keys=True, ensure_ascii=False, indent=4)
