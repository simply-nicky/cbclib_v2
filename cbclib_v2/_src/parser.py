from configparser import ConfigParser
from enum import Enum
import dataclasses
import json
import os
import re
from typing import Any, Callable, ClassVar, Dict, List, Tuple, Type, get_args, get_origin, overload
import numpy as np
from .data_container import Container
from .annotations import (AnyType, Array, ArrayNamespace, DataclassInstance, ExpandedType, NumPy,
                          UnionType)

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

    @classmethod
    def format_string(cls, string: str) -> bool:
        return string in ['True', 'true', 'yes', 'y']

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

class ArrayFormatter(Formatter):
    aliases = (np.ndarray,)

    @classmethod
    def format_string(cls, string: str, dtype: Type, xp: ArrayNamespace) -> Array:
        is_list = re.search(r'^\[([\s\S]*)\]$', string)
        if is_list:
            return xp.fromstring(is_list.group(1), dtype=dtype, sep=',')
        raise ValueError(f"Invalid string: '{string}'")

class StringFormatting:
    FormatterDict = Dict[str, Type[SimpleFormatter] | Type[Formatter]]
    formatters : FormatterDict = {'ndarray': ArrayFormatter,
                                  'list': ListFormatter,
                                  'tuple': TupleFormatter,
                                  'float': FloatFormatter,
                                  'int': IntFormatter,
                                  'bool': BoolFormatter,
                                  'str': StringFormatter}
    @classmethod
    def expand_types(cls, t: AnyType) -> ExpandedType:
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
    def get_formatter(cls, t: AnyType, xp: ArrayNamespace) -> Callable[[str,], Any]:
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

                    if issubclass(formatter, ArrayFormatter):
                        return lambda string: formatter.format_string(string, dtype, xp)
                    return lambda string: formatter.format_string(string, dtype)
        return str

    @classmethod
    def format_dict(cls, dct: Dict[str, Any], types: Dict[str, AnyType] | AnyType,
                    xp: ArrayNamespace) -> Dict[str, Any]:
        formatted_dct = {}
        for attr, val in dct.items():
            if isinstance(val, dict) and isinstance(types, dict):
                formatted_dct[attr] = cls.format_dict(val, types[attr], xp)
            if isinstance(val, str):
                if isinstance(types, dict):
                    formatter = cls.get_formatter(types[attr], xp)
                else:
                    formatter = cls.get_formatter(types, xp)
                formatted_dct[attr] = formatter(val)
        return formatted_dct

    @overload
    @classmethod
    def to_string(cls, node: Dict) -> Dict: ...

    @overload
    @classmethod
    def to_string(cls, node: List) -> List: ...

    @overload
    @classmethod
    def to_string(cls, node: Array | Any) -> str: ...

    @classmethod
    def to_string(cls, node: Any | Dict | List | Array
                  ) -> str | List | Dict:
        if isinstance(node, dict):
            return {k: cls.to_string(v) for k, v in node.items()}
        if isinstance(node, list):
            return [cls.to_string(v) for v in node]
        if isinstance(node, np.ndarray):
            return np.array2string(np.array(node), separator=',')
        if isinstance(node, Enum):
            return str(node.value)
        return str(node)

FieldValues = str | Dict[str, Any] | Tuple[str, Dict[str, 'FieldValues']]
TypeValues = AnyType | Dict[str, AnyType]

def fields(cls: DataclassInstance | type[DataclassInstance], default: str | None
           ) -> Dict[str, FieldValues]:
    result = {}

    for field in dataclasses.fields(cls):
        origin = field.type
        if isinstance(field.type, UnionType):
            origin = get_args(origin)[0]

        while get_origin(origin) is not None:
            origin = get_origin(origin)

        if isinstance(origin, DataclassInstance):
            result[field.name] = (field.name, fields(origin, None))
        elif isinstance(origin, type) and issubclass(origin, dict):
            result[field.name] = field.name
        elif default is not None:
            if default not in result:
                result[default] = {}
            if not isinstance(result[default], dict):
                raise ValueError(f"Default value '{default}' is already used, please change it")
            result[default][field.name] = field.name
        else:
            result[field.name] = field.name

    return result

def type_hints(cls: DataclassInstance | type[DataclassInstance]) -> Dict[str, TypeValues]:
    result = {}
    for field in dataclasses.fields(cls):
        origin = field.type
        if isinstance(origin, DataclassInstance):
            result[field.name] = type_hints(origin)
        else:
            result[field.name] = field.type
    return result

class Parser():
    fields : Dict[str, FieldValues]

    def read_all(self, file: str) -> Dict[str, Any]:
        raise NotImplementedError

    def read(self, file: str) -> Dict[str, Any]:
        """Initialize the container object with an INI file ``file``.

        Args:
            file : Path to the ini file.

        Returns:
            A new container with all the attributes imported from the ini file.
        """
        def read_fields(flds: Dict, data: Dict) -> Dict:
            result: Dict[str, Any] = {}
            for section, attrs in flds.items():
                if isinstance(attrs, str):
                    result[attrs] = data[section]
                elif isinstance(attrs, dict):
                    # 'default' section with a dictionary of attributes
                    result.update(**read_fields(attrs, data[section]))
                elif isinstance(attrs, tuple):
                    # 'container' contains another container at 'container.attr'
                    attr, attr_fields = attrs
                    result[attr] = read_fields(attr_fields, data[section])
                else:
                    raise TypeError(f"Invalid 'fields' values: {attrs}")
            return result

        return read_fields(self.fields, self.read_all(file))

    def to_dict(self, obj: Any) -> Dict[str, Dict[str, Any]]:
        def extract_fields(obj: Any, flds: Dict) -> Dict:
            result: Dict[str, Any] = {}
            for section, attrs in flds.items():
                if isinstance(attrs, str):
                    result[section] = getattr(obj, attrs)
                if isinstance(attrs, dict):
                    result[section] = extract_fields(obj, attrs)
                if isinstance(attrs, tuple):
                    attr, attr_fields = attrs
                    result[section] = extract_fields(getattr(obj, attr), attr_fields)
            return result

        return extract_fields(obj, self.fields)

    def write(self, file: str, obj: Any):
        raise NotImplementedError

@dataclasses.dataclass
class INIParser(Parser, Container):
    """Abstract data container class based on :class:`dataclass` with an interface to read from
    and write to INI files.
    """
    fields  : Dict[str, FieldValues]
    types   : Dict[str, TypeValues]

    @classmethod
    def from_container(cls, container: DataclassInstance | type[DataclassInstance],
                       default: str='default') -> 'INIParser':
        return cls(fields(container, default), type_hints(container))

    def read_all(self, file: str) -> Dict[str, Any]:
        if not os.path.isfile(file):
            raise ValueError(f"File {file} doesn't exist")

        ini_parser = ConfigParser()
        ini_parser.read(file)

        return {section: dict(ini_parser.items(section)) for section in ini_parser.sections()}

    def read(self, file: str, xp: ArrayNamespace=NumPy) -> Dict[str, Any]:
        return StringFormatting.format_dict(super().read(file), self.types, xp)

    def to_dict(self, obj: Any) -> Dict[str, Dict[str, Any]]:
        return StringFormatting.to_string(super().to_dict(obj))

    def write(self, file: str, obj: Any):
        """Save all the attributes stored in the container to an INI file ``file``.

        Args:
            file : Path to the ini file.
        """
        ini_parser = ConfigParser()
        ini_parser.read_dict(self.to_dict(obj))

        with np.printoptions(precision=None):
            with open(file, 'w') as out_file:
                ini_parser.write(out_file)

@dataclasses.dataclass
class JSONParser(Parser, Container):
    fields  : Dict[str, FieldValues]

    @classmethod
    def from_container(cls, container: DataclassInstance | type[DataclassInstance],
                       default: str='default') -> 'JSONParser':
        return cls(fields(container, default))

    def read_all(self, file: str) -> Dict[str, Any]:
        with open(file, 'r') as f:
            json_dict = json.load(f)

        return json_dict

    def to_dict(self, obj: Any) -> Dict[str, Dict[str, Any]]:
        def array_to_list(**values: Any) -> Dict[str, Any]:
            result = {}
            for key, val in values.items():
                if isinstance(val, dict):
                    result[key] = array_to_list(**val)
                elif isinstance(val, np.ndarray):
                    result[key] = val.tolist()
                else:
                    result[key] = val
            return result

        return array_to_list(**super().to_dict(obj))

    def write(self, file: str, obj: Any):
        with open(file, 'w') as out_file:
            json.dump(self.to_dict(obj), out_file, sort_keys=True, ensure_ascii=False, indent=4)
