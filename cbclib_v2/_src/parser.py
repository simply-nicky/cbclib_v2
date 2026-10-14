from configparser import ConfigParser
from enum import Enum
import dataclasses
from inspect import getmembers
import json
import os
from typing import (Any, Dict, Tuple, Type, get_args, get_origin,
                    get_type_hints as typing_get_type_hints, overload)
import numpy as np
from .data_container import Container, resolved_type
from .annotations import AnyType, Array, AnyNamespace, NDArray, NumPy, UnionType
from .array_api import asnumpy

class StringFormatting:
    """Encode INI values as JSON and restore decoded values to their annotated types."""

    @classmethod
    def format_value(cls, value: Any, value_type: AnyType, xp: AnyNamespace) -> Any:
        """Restore tuples and backend arrays from a decoded JSON value."""
        if value is None:
            return value
        args = get_args(value_type)
        if isinstance(value_type, UnionType):
            value_type = next((arg for arg in args
                               if (get_origin(arg) or arg) is NDArray), args[0])
            return cls.format_value(value, value_type, xp)
        origin = get_origin(value_type) or value_type
        if origin in get_args(Array):
            return xp.asarray(value)
        if origin in (tuple, list):
            if not args:
                return origin(value)
            if origin is tuple and args[-1] is not Ellipsis:
                return tuple(cls.format_value(item, item_type, xp)
                             for item, item_type in zip(value, args))
            return origin(cls.format_value(item, args[0], xp) for item in value)
        if origin is dict and args:
            return {key: cls.format_value(item, args[1], xp) for key, item in value.items()}
        return value

    @classmethod
    def format_dict(cls, dct: Dict[str, Any], types: Dict[str, AnyType] | AnyType,
                    xp: AnyNamespace) -> Dict[str, Any]:
        """Reconstruct annotated types from a decoded, mapped parameter dictionary."""
        formatted_dct = {}
        for attr, value in dct.items():
            value_type = types[attr] if isinstance(types, dict) else types
            if isinstance(value, dict) and isinstance(value_type, dict):
                formatted_dct[attr] = cls.format_dict(value, value_type, xp)
            else:
                formatted_dct[attr] = cls.format_value(value, value_type, xp)
        return formatted_dct

    @classmethod
    def json_default(cls, value: Any) -> Any:
        """Represent array, NumPy scalar, and enum values using JSON-native values."""
        if isinstance(value, Array):
            return asnumpy(value).tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Enum):
            return value.value
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")

    @overload
    @classmethod
    def to_string(cls, node: Dict) -> Dict: ...

    @overload
    @classmethod
    def to_string(cls, node: Any) -> str | Dict: ...

    @classmethod
    def to_string(cls, node: Any) -> str | Dict:
        """Encode each INI value as JSON, preserving the file-section mapping."""
        if isinstance(node, dict):
            return {key: cls.to_string(value) for key, value in node.items()}
        return json.dumps(node, default=cls.json_default)

ContainerType = type[Container]

@dataclasses.dataclass(frozen=True)
class FieldLocator:
    """Locate one parameter in a nested object.

    Args:
        field_name: Dot-separated attribute path relative to the root object.
    """
    field_name: str

    def __post_init__(self):
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

    # Adding properties to the type hints dictionary
    for name, member in getmembers(cls):
        if isinstance(member, property) and member.fget is not None:
            property_hints = typing_get_type_hints(member.fget)
            if 'return' in property_hints:
                result[name] = property_hints['return']
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

    def set_value(locator: FieldLocator, value: Any):
        node = result
        for field_name in locator.path[:-1]:
            child = node.setdefault(field_name, {})
            if not isinstance(child, dict):
                raise ValueError(f"Field locator '{locator.field_name}' conflicts with another "
                                 "field locator")
            node = child
        node[locator.path[-1]] = value

    def read_node(node_info: FieldInfo, node_data: Dict[str, Any]):
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

        return {section: {name: json.loads(value) for name, value in ini_parser.items(section)}
                for section in ini_parser.sections()}

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

        with open(file, 'w') as out_file:
            ini_parser.write(out_file)

@dataclasses.dataclass
class JSONParser(Parser, Container):
    type_info : Dict[str, Any]

    @classmethod
    def from_container(cls, container: Container, default: str | None=None) -> 'JSONParser':
        data = container.to_dict()
        field_info = fields(type(container), data, default)
        return cls(field_info, get_type_hints(type(container), data))

    @classmethod
    def from_file(cls, file: str, container_type: ContainerType, default: str | None=None
                  ) -> 'JSONParser':
        data = cls.read_all(file)
        field_info = fields(container_type, data, default)
        return cls(field_info, get_type_hints(container_type, data))

    @classmethod
    def read_all(cls, file: str) -> Dict[str, Any]:
        with open(file, 'r') as f:
            json_dict = json.load(f)

        return json_dict

    def read(self, file: str, xp: AnyNamespace=NumPy) -> Dict[str, Any]:
        return StringFormatting.format_dict(super().read(file), self.type_info, xp)

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
            json.dump(self.to_dict(obj), out_file, default=StringFormatting.json_default,
                      sort_keys=True, ensure_ascii=False, indent=4)

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
