from typing import (Any, Callable, ClassVar, Dict, Hashable, Iterable, Mapping, Tuple,
                    Type, TypeVar, overload)
import dataclasses
from dataclasses import MISSING, Field, fields
from typing_extensions import dataclass_transform
from jax.tree_util import register_pytree_with_keys
from .data_container import DataContainer
from .annotations import KeyArray, PyTree

Generator = Callable[[KeyArray], Any]

class DynamicField(Field):
    random  : Generator | None

    def __init__(self, name, type, default, default_factory, random, init,
                 repr, hash, compare, metadata, kw_only):
        super().__init__(default, default_factory, init, repr, hash, compare,
                         metadata, kw_only)
        self.name, self.type = name, type
        self.random = None if random is MISSING else random

T = TypeVar('T')

# NOTE: Actual return type is 'Field[T]', but we want to help type checkers
# to understand the magic that happens at runtime.
@overload
def field(*, default: T, random: Any=..., static: bool=..., init: bool=...,
          repr: bool=..., hash: bool | None=..., compare: bool=...,
          metadata: Mapping[str, Any] | None=..., kw_only: bool=...) -> T:
    ...

@overload
def field(*, default_factory: Callable[[], T], random: Any=..., static: bool=...,
          init: bool=..., repr: bool=..., hash: bool | None=..., compare: bool=...,
          metadata: Mapping[str, Any] | None=..., kw_only: bool=...) -> T:
    ...

@overload
def field(*, random: Callable[[KeyArray], T], static: bool=..., init: bool=...,
          repr: bool=..., hash: bool | None=..., compare: bool=...,
          metadata: Mapping[str, Any] | None=..., kw_only: bool=...) -> T:
    ...

@overload
def field(*, static: bool=..., init: bool=..., repr: bool=...,
          hash: bool | None=..., compare: bool=...,
          metadata: Mapping[str, Any] | None=..., kw_only: bool=...) -> Any:
    ...

def add_dynamic_field(func: Callable):
    def wrapper(*args, random: Any=MISSING, static: bool=False,
                metadata: Mapping[str, Any] | None=None, **kwargs):
        if metadata is None:
            metadata = {}
        metadata = dict(metadata) | {'static': static, 'random': random}
        return Field(*args, metadata=metadata, **kwargs)

    return wrapper

def field(*, default: Any=MISSING, default_factory: Any=MISSING, random: Any=MISSING,
          static: bool=False, init: bool=True, repr: bool=True, hash: bool | None=None,
          compare: bool=True, metadata: Mapping[str, Any] | None=None,
          kw_only=MISSING) -> Any:
    """A field creator with a static indicator.

    The static flag indicates whether a field is a pytree or static.  Pytree fields are
    differentiated and traced.  Static fields are hashed and compared.
    """
    wrapper = add_dynamic_field(dataclasses.field)
    return wrapper(default=default, default_factory=default_factory, random=random,
                   static=static, init=init, repr=repr, hash=hash, compare=compare,
                   metadata=metadata, kw_only=kw_only)

S = TypeVar('S', bound='BaseState')

@dataclass_transform(field_specifiers=(field,))
class BaseState(DataContainer):
    __dynamic_fields__      : ClassVar[Dict[str, DynamicField]]
    __static_fields__       : ClassVar[Dict[str, Field]]

def dynamic_fields(class_or_instance: Type[BaseState] | BaseState) -> Tuple[DynamicField, ...]:
    return tuple(class_or_instance.__dynamic_fields__.values())

def static_fields(class_or_instance: Type[BaseState] | BaseState) -> Tuple[Field, ...]:
    return tuple(class_or_instance.__static_fields__.values())

class State(BaseState):
    def __init_subclass__(cls, random: bool=False, init: bool=True, repr: bool=True,
                          eq: bool=True, order: bool=False, unsafe_hash: bool=False,
                          frozen: bool=False, match_args: bool=True, kw_only: bool=False,
                          slots: bool=False):
        super().__init_subclass__()

        cls._process_properties()

        dataclasses.dataclass(
            init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash,
            frozen=frozen, match_args=match_args, kw_only=kw_only, slots=slots
        )(cls)

        cls.__dynamic_fields__ = {}
        cls.__static_fields__ = {}

        cls._sort_fields()
        cls._register_pytree()

        if random:
            generators, defaults = cls._process_fields()
            cls._create_random(generators, defaults)

    @classmethod
    def _process_properties(cls):
        for attribute in cls.__annotations__:
            default = getattr(cls, attribute, MISSING)
            if isinstance(default, property):
                setattr(cls, attribute, MISSING)

    @classmethod
    def _create_random(cls, generators: Dict[str, Generator], defaults: Dict[str, Any]):
        def random(cls: Type[S], key: KeyArray) -> S:
            return cls(**{attr: gen(key) for attr, gen in generators.items()},
                       **defaults)

        setattr(cls, 'random', classmethod(random))

    @classmethod
    def _process_fields(cls) -> Tuple[Dict[str, Generator], Dict[str, Any]]:
        defaults : Dict[str, Any] = {}
        for fld in static_fields(cls):
            if fld.default is MISSING:
                raise ValueError(f"default is missing for {fld.name}")
            defaults[fld.name] = fld.default

        generators : Dict[str, Generator] = {}
        for fld in dynamic_fields(cls):
            if fld.random is None:
                raise ValueError(f"random is missing for {fld.name}")
            generators[fld.name] = fld.random

        return generators, defaults

    @classmethod
    def _sort_fields(cls):
        for fld in fields(cls):
            if fld.metadata.get('static', False):
                if not isinstance(fld.default, Hashable):
                    raise ValueError(f"The default value for {fld.name} is not hashable")
                cls.__static_fields__[fld.name] = fld
            else:
                dynamic_field = DynamicField(
                    fld.name, fld.type, fld.default, fld.default_factory,
                    fld.metadata.get('random', None), fld.init, fld.repr,
                    fld.hash, fld.compare, fld.metadata, fld.kw_only
                )
                cls.__dynamic_fields__[fld.name] = dynamic_field


    @classmethod
    def _register_pytree(cls):
        # Register the class as a Jax PyTree.
        def flatten_with_keys(x: Any, /) -> Tuple[Iterable[Tuple[str, PyTree]], Hashable]:
            hashed = tuple(getattr(x, name) for name in cls.__static_fields__)
            trees = tuple((name, getattr(x, name)) for name in cls.__dynamic_fields__)
            return trees, hashed

        def unflatten(hashed: Hashable, trees: Iterable[PyTree], /) -> Any:
            if not isinstance(hashed, tuple):
                raise TypeError
            hashed_args = dict(zip(cls.__static_fields__, hashed))
            tree_args = dict(zip(cls.__dynamic_fields__, trees))
            return cls(**hashed_args, **tree_args)

        def flatten(x: Any, /) -> Tuple[Iterable[PyTree], Hashable]:
            hashed = tuple(getattr(x, name) for name in cls.__static_fields__)
            trees = tuple(getattr(x, name) for name in cls.__dynamic_fields__)
            return trees, hashed

        register_pytree_with_keys(cls, flatten_with_keys, unflatten, flatten)

    @classmethod
    def random(cls: Type[S], key: KeyArray) -> S:
        raise RuntimeError("the class is created without random")
