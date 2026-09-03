from configparser import ConfigParser
from enum import Enum
import dataclasses
import json
import os
import re
from typing import (Any, Callable, ClassVar, Dict, List, Tuple, Type, get_args, get_origin,
                    get_type_hints as typing_get_type_hints, overload)
import numpy as np
from .data_container import Container, resolved_type
from .annotations import AnyType, Array, AnyNamespace, ExpandedType, NDArray, NumPy, UnionType

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
    aliases = (NDArray,)

    @classmethod
    def format_string(cls, string: str, dtype: Type, xp: AnyNamespace) -> Array:
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
    def get_formatter(cls, t: AnyType, xp: AnyNamespace) -> Callable[[str,], Any]:
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
                    xp: AnyNamespace) -> Dict[str, Any]:
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
        if isinstance(node, NDArray):
            return np.array2string(np.array(node), separator=',')
        if isinstance(node, Enum):
            return str(node.value)
        return str(node)

ContainerType = type[Container]

@dataclasses.dataclass(frozen=True)
class FieldLocator:
    """Locate one parameter in a nested object.

    Args:
        field_name: Dot-separated attribute path relative to the root object.
    """
    field_name: str

    def __post_init__(self) -> None:
        if not self.field_name or any(not name for name in self.path):
            raise ValueError(f"Invalid field name: '{self.field_name}'")

    @property
    def path(self) -> Tuple[str, ...]:
        """Return the individual attribute names in the locator."""
        return tuple(self.field_name.split('.'))

    def getattr(self, obj: Any) -> Any:
        """Extract the located parameter from an object."""
        value = obj
        for field_name in self.path:
            value = getattr(value, field_name)
        return value

FieldValues = FieldLocator | Dict[str, 'FieldValues']
FieldInfo = Dict[str, FieldValues]
TypeValues = AnyType | Dict[str, AnyType]

def fields(cls: ContainerType, data: Dict[str, Any],
           default: str | None) -> FieldInfo:
    """Create the file-to-object field mapping for a container type.

    File parameter names remain dictionary keys. Leaf values are
    :class:`FieldLocator` objects containing the complete attribute path from
    the root container. Nested container fields retain their own file
    dictionaries, while non-container fields are grouped under ``default``
    when one is provided.

    Args:
        cls: Container type represented by the mapping.
        data: Serialized object dictionary. It is used to resolve polymorphic
            container member types.
        default: Optional file section for non-container fields. If omitted,
            those fields remain at the current file-dictionary level.

    Returns:
        Recursive field mapping preserving the generated file layout.
    """
    def find_fields(cls: ContainerType, data: Dict[str, Any], default: str | None,
                parent: Tuple[str, ...]) -> FieldInfo:
        result: FieldInfo = {}

        for field in dataclasses.fields(cls):
            origin = field.type
            if isinstance(field.type, UnionType):
                origin = get_args(origin)[0]

            while get_origin(origin) is not None:
                origin = get_origin(origin)

            if isinstance(origin, type) and issubclass(origin, Container):
                origin = resolved_type(origin, field.name, data)
                result[field.name] = find_fields(origin, data[field.name], None,
                                                 parent + (field.name,))
            elif isinstance(origin, type) and issubclass(origin, dict):
                result[field.name] = FieldLocator('.'.join(parent + (field.name,)))
            elif default is not None:
                if default not in result:
                    result[default] = {}
                default_fields = result[default]
                if not isinstance(default_fields, dict):
                    raise ValueError(f"Default value '{default}' is already used, please change it")
                default_fields[field.name] = FieldLocator('.'.join(parent + (field.name,)))
            else:
                result[field.name] = FieldLocator('.'.join(parent + (field.name,)))

        return result

    return find_fields(cls, data, default, ())

def get_type_hints(cls: Type[Any],
                   data: Dict[str, Any] | None=None) -> Dict[str, TypeValues]:
    """Return type hints with nested container members expanded recursively.

    Args:
        cls: Container type whose fields are inspected.
        data: Optional object dictionary used to resolve polymorphic container fields.

    Returns:
        Type hints arranged like the dictionary used to construct ``cls``.
    """
    result: Dict[str, TypeValues] = {}
    hints = typing_get_type_hints(cls)
    for field in dataclasses.fields(cls):
        field_type = hints[field.name]
        origin = field_type
        if isinstance(field_type, UnionType):
            origin = get_args(origin)[0]

        while get_origin(origin) is not None:
            origin = get_origin(origin)

        if isinstance(origin, type) and issubclass(origin, Container):
            child_data = None
            if data is not None:
                origin = resolved_type(origin, field.name, data)
                child_data = data[field.name]
            result[field.name] = get_type_hints(origin, child_data)
        else:
            result[field.name] = field_type
    return result

def read_fields(field_info: FieldInfo, data: Dict[str, Any]) -> Dict[str, Any]:
    """Select file fields and reconstruct their object dictionary.

    Dictionaries in ``field_info`` describe paths through the file data and
    are traversed without being copied into the result. This flattens a
    default file section as in the original parser protocol. At each leaf,
    the corresponding :class:`FieldLocator` supplies the destination path in
    the object dictionary, recreating nested container members. All leaves
    write into one shared result so separate file sections can contribute to
    the same nested object.

    Args:
        field_info: Recursive mapping from file names to object field locators.
        data: Nested dictionary read from the file.

    Returns:
        Constructor-shaped dictionary containing the selected object fields.

    Raises:
        ValueError: If a mapped file name is missing or locator paths conflict.
        TypeError: If a mapping value is neither a locator nor a dictionary.
    """
    result: Dict[str, Any] = {}

    def set_value(locator: FieldLocator, value: Any) -> None:
        node = result
        for field_name in locator.path[:-1]:
            child = node.setdefault(field_name, {})
            if not isinstance(child, dict):
                raise ValueError(f"Field locator '{locator.field_name}' conflicts with another "
                                 "field locator")
            node = child
        node[locator.path[-1]] = value

    def read_node(node_info: FieldInfo, node_data: Dict[str, Any]) -> None:
        for parameter, attrs in node_info.items():
            if parameter not in node_data:
                raise ValueError(f"Section '{parameter}' not found in the file")
            if isinstance(attrs, FieldLocator):
                set_value(attrs, node_data[parameter])
            elif isinstance(attrs, dict):
                read_node(attrs, node_data[parameter])
            else:
                raise TypeError(f"Invalid 'fields' values: {attrs}")

    read_node(field_info, data)
    return result

def extract_fields(field_info: FieldInfo, obj: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for section, attrs in field_info.items():
        if isinstance(attrs, FieldLocator):
            result[section] = attrs.getattr(obj)
        elif isinstance(attrs, dict):
            result[section] = extract_fields(attrs, obj)
        else:
            raise TypeError(f"Invalid 'fields' values: {attrs}")
    return result

@dataclasses.dataclass
class Parser():
    """Read and write selected object fields using a file mapping.

    Attributes:
        field_info: Recursive file-to-object mapping. Every key is a parameter
            or section name in the file dictionary. A :class:`FieldLocator`
            value maps that file parameter to one complete object attribute
            path, such as ``FieldLocator('lens.foc_pos')``. A dictionary value
            describes another level of the file dictionary and recursively
            contains the same two value variants. Thus file nesting is
            represented by dictionaries and object nesting by locators.
    """
    field_info      : FieldInfo

    @classmethod
    def from_container(cls, container: Container, default: str | None=None) -> 'Parser':
        raise NotImplementedError

    @classmethod
    def from_file(cls, file: str, container_type: ContainerType, default: str | None=None
                  ) -> 'Parser':
        raise NotImplementedError

    @classmethod
    def read_all(cls, file: str) -> Dict[str, Any]:
        raise NotImplementedError

    def read(self, file: str) -> Dict[str, Any]:
        """Initialize the container object with an INI file ``file``.

        Args:
            file : Path to the ini file.

        Returns:
            A new container with all the attributes imported from the ini file.
        """
        return read_fields(self.field_info, self.read_all(file))

    def to_dict(self, obj: Any) -> Dict[str, Dict[str, Any]]:
        return extract_fields(self.field_info, obj)

    def write(self, file: str, obj: Any):
        raise NotImplementedError

@dataclasses.dataclass
class INIParser(Parser, Container):
    """Abstract data container class based on :class:`dataclass` with an interface to read from
    and write to INI files.
    """
    type_info   : Dict[str, Any]

    @classmethod
    def from_container(cls, container: Container, default: str | None=None) -> 'INIParser':
        data = container.to_dict()
        field_info = fields(type(container), data, default)
        type_info = get_type_hints(type(container), data)
        return cls(field_info, type_info)

    @classmethod
    def from_file(cls, file: str, container_type: ContainerType, default: str | None=None
                  ) -> 'INIParser':
        data = cls.read_all(file)
        field_info = fields(container_type, data, default)
        type_info = get_type_hints(container_type, data)
        return cls(field_info, type_info)

    @classmethod
    def read_all(cls, file: str) -> Dict[str, Any]:
        if not os.path.isfile(file):
            raise ValueError(f"File {file} doesn't exist")

        ini_parser = ConfigParser()
        ini_parser.read(file)

        return {section: dict(ini_parser.items(section)) for section in ini_parser.sections()}

    def read(self, file: str, xp: AnyNamespace=NumPy) -> Dict[str, Any]:
        return StringFormatting.format_dict(super().read(file), self.type_info, xp)

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
    @classmethod
    def from_container(cls, container: Container, default: str | None=None) -> 'JSONParser':
        data = container.to_dict()
        field_info = fields(type(container), data, default)
        return cls(field_info)

    @classmethod
    def from_file(cls, file: str, container_type: ContainerType, default: str | None=None
                  ) -> 'JSONParser':
        data = cls.read_all(file)
        field_info = fields(container_type, data, default)
        return cls(field_info)

    @classmethod
    def read_all(cls, file: str) -> Dict[str, Any]:
        with open(file, 'r') as f:
            json_dict = json.load(f)

        return json_dict

    def to_dict(self, obj: Any) -> Dict[str, Dict[str, Any]]:
        def array_to_list(**values: Any) -> Dict[str, Any]:
            result = {}
            for key, val in values.items():
                if isinstance(val, dict):
                    result[key] = array_to_list(**val)
                elif isinstance(val, NDArray):
                    result[key] = val.tolist()
                else:
                    result[key] = val
            return result

        return array_to_list(**super().to_dict(obj))

    def write(self, file: str, obj: Any):
        with open(file, 'w') as out_file:
            json.dump(self.to_dict(obj), out_file, sort_keys=True, ensure_ascii=False, indent=4)

def get_extension(file_or_extension: str) -> str:
    if file_or_extension == 'ini':
        return 'ini'
    if file_or_extension == 'json':
        return 'json'

    ext = os.path.splitext(file_or_extension)[1].lower()
    if ext == '.ini':
        return 'ini'
    if ext == '.json':
        return 'json'
    raise ValueError(f"Unsupported file or extension format: {file_or_extension}")

def from_container(file_or_extension: str, container: Container, default: str | None=None
                   ) -> Parser:
    ext = get_extension(file_or_extension)
    if ext == 'ini':
        return INIParser.from_container(container, default)
    if ext == 'json':
        return JSONParser.from_container(container, default)
    raise ValueError(f"Unsupported file format: {file_or_extension}")

def from_file(file: str, container_type: ContainerType, default: str | None=None) -> Parser:
    ext = get_extension(file)
    if ext == 'ini':
        return INIParser.from_file(file, container_type, default)
    if ext == 'json':
        return JSONParser.from_file(file, container_type, default)
    raise ValueError(f"Unsupported file format: {file}")

def read_all(file: str) -> Dict[str, Any]:
    ext = get_extension(file)
    if ext == 'ini':
        return INIParser.read_all(file)
    if ext == 'json':
        return JSONParser.read_all(file)
    raise ValueError(f"Unsupported file format: {file}")
