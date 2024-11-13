import copy
from functools import partial
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union,
                    cast, get_type_hints, overload)
from dataclasses import dataclass, is_dataclass, fields
import pandas as pd
import jax.numpy as jnp
from jax import jit, random, tree_util
from .dataclasses import jax_dataclass, field
from .geometry import euler_angles, euler_matrix, tilt_angles, tilt_matrix
from .._src.annotations import IntArray, KeyArray, RealArray, Shape
from .._src.data_container import DataclassInstance, Parser, INIParser, JSONParser

State = DataclassInstance
KNNQuery = Callable[[RealArray, IntArray, RealArray, IntArray], IntArray]

@dataclass
class Detector():
    x_pixel_size : float
    y_pixel_size : float

    def to_indices(self, x: RealArray, y: RealArray) -> Tuple[RealArray, RealArray]:
        return x / self.x_pixel_size, y / self.y_pixel_size

    def to_coordinates(self, i: RealArray, j: RealArray) -> Tuple[RealArray, RealArray]:
        return i * self.x_pixel_size, j * self.y_pixel_size

@jax_dataclass
class XtalCell():
    angles  : RealArray
    lengths : RealArray

    def __post_init__(self):
        if self.lengths.shape != self.angles.shape:
            raise ValueError("angles and lengths have incompatible shapes: "\
                             f"{self.angles.shape} and {self.lengths.shape}")

    @property
    def shape(self) -> Shape:
        return self.angles.shape

    @property
    def alpha(self) -> RealArray:
        return self.angles[..., 0]

    @property
    def beta(self) -> RealArray:
        return self.angles[..., 1]

    @property
    def gamma(self) -> RealArray:
        return self.angles[..., 2]

    def to_basis(self) -> 'XtalState':
        cos = jnp.cos(self.angles)
        sin = jnp.sin(self.gamma)
        v_ratio = jnp.sqrt(jnp.ones(self.shape[:-1]) - jnp.sum(cos**2, axis=-1) + \
                           2 * jnp.prod(cos, axis=-1))
        a_vec = jnp.broadcast_to(jnp.array([1, 0, 0]), self.shape)
        b_vec = jnp.stack((cos[..., 2], sin, jnp.zeros(self.shape[:-1])), axis=-1)
        c_vec = jnp.stack((cos[..., 1], (cos[..., 0] - cos[..., 1] * cos[..., 2]) / sin,
                           v_ratio / sin))
        return XtalState(self.lengths[..., None] * jnp.stack((a_vec, b_vec, c_vec), axis=-2))

@jax_dataclass
class XtalState():
    basis : RealArray

    @property
    def a(self) -> RealArray:
        return self.basis[..., 0, :]

    @property
    def b(self) -> RealArray:
        return self.basis[..., 1, :]

    @property
    def c(self) -> RealArray:
        return self.basis[..., 2, :]

    @property
    def num(self) -> int:
        return self.basis.size // 9

    @classmethod
    def parser(cls, ext: str='ini') -> Parser:
        types = {'a_vec': RealArray, 'b_vec': RealArray, 'c_vec': RealArray}
        if ext == 'ini':
            return INIParser({'basis': ('a_vec', 'b_vec', 'c_vec')},
                             types=cast(Dict[str, Type], types))
        if ext == 'json':
            return JSONParser({'basis': ('a_vec', 'b_vec', 'c_vec')})

        raise ValueError(f"Invalid format: {ext}")

    @classmethod
    def read(cls, file: str, ext: str='ini') -> 'XtalState':
        return cls(jnp.stack(list(cls.parser(ext).read(file).values())))

    @classmethod
    def import_spherical(cls, basis: RealArray) -> 'XtalState':
        """Return a new :class:`XtalState` object, initialised by a stacked matrix of three basis
        vectors written in spherical coordinate system.

        Args:
            mat : A matrix of three stacked basis vectors in spherical coordinate system.

        Returns:
            A new :class:`XtalState` object.
        """
        return cls(jnp.stack((basis[..., 0] * jnp.sin(basis[..., 1]) * jnp.cos(basis[..., 2]),
                              basis[..., 0] * jnp.sin(basis[..., 1]) * jnp.sin(basis[..., 2]),
                              basis[..., 0] * jnp.cos(basis[..., 1])), axis=-1))

    def lattice_constants(self) -> XtalCell:
        """Return a stack of unit cell vectors in spherical coordinate system.

        Returns:
            A matrix of three stacked unit cell vectors in spherical coordinate system.
        """
        lengths = jnp.sqrt(jnp.sum(self.basis**2, axis=-1))
        angles = jnp.stack([jnp.sum(self.b * self.c) / (lengths[..., 1] * lengths[..., 2]),
                            jnp.sum(self.c * self.a) / (lengths[..., 2] * lengths[..., 0]),
                            jnp.sum(self.a * self.b) / (lengths[..., 0] * lengths[..., 1])],
                           axis=-1)
        return XtalCell(angles=jnp.arccos(angles), lengths=lengths)

    def reciprocate(self) -> 'XtalState':
        """Calculate the basis of the reciprocal lattice.

        Returns:
            The basis of the reciprocal lattice.
        """
        a_rec = jnp.cross(self.b, self.c) / jnp.dot(jnp.cross(self.b, self.c), self.a)
        b_rec = jnp.cross(self.c, self.a) / jnp.dot(jnp.cross(self.c, self.a), self.b)
        c_rec = jnp.cross(self.a, self.b) / jnp.dot(jnp.cross(self.a, self.b), self.c)
        return XtalState(jnp.stack((a_rec, b_rec, c_rec)))

    def to_spherical(self) -> RealArray:
        """Return a stack of unit cell vectors in spherical coordinate system.

        Returns:
            A matrix of three stacked unit cell vectors in spherical coordinate system.
        """
        lengths = jnp.sqrt(jnp.sum(self.basis**2, axis=-1))
        return jnp.stack((lengths, jnp.arccos(self.basis[..., 2] / lengths),
                          jnp.arctan2(self.basis[..., 1], self.basis[..., 0])), axis=-1)

@jax_dataclass
class LensState():
    foc_pos         : RealArray
    pupil_roi       : Tuple[float, float, float, float] = field(static=True)

    @classmethod
    def parser(cls, ext: str='ini') -> Parser:
        if ext == 'ini':
            return INIParser({'exp_geom': ('foc_pos', 'pupil_roi')},
                             types=get_type_hints(cls))
        if ext == 'json':
            return JSONParser({'exp_geom': ('foc_pos', 'pupil_roi')})

        raise ValueError(f"Invalid format: {ext}")

    @classmethod
    def read(cls, file: str, ext: str='ini') -> 'LensState':
        return cls(**cls.parser(ext).read(file))

Params = Union[RealArray, float]
S = TypeVar('S', bound=State)

def init_from_bounds(state: S, default: Callable[[Params], Params],
                     bounds: Optional[Dict[str, Any]]=None) -> Callable[[KeyArray,], S]:
    def asdict(obj: DataclassInstance) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        _T = TypeVar("_T", Tuple, List, Dict)

        @overload
        def wrapper(obj: DataclassInstance, is_dynamic: bool) -> Dict[str, Any]: ...

        @overload
        def wrapper(obj: _T, is_dynamic: bool) -> _T: ...

        @overload
        def wrapper(obj: Any, is_dynamic: bool) -> Any: ...

        def wrapper(obj: Union[DataclassInstance, _T, Any], is_dynamic: bool
                          ) -> Union[Dict[str, Any], _T, Any]:
            if is_dataclass(obj):
                result = []
                for f in fields(obj):
                    value = wrapper(getattr(obj, f.name), is_dynamic)
                    is_pytree = f.metadata.get('pytree_node', True)
                    if isinstance(value, dict) or is_pytree == is_dynamic:
                        result.append((f.name, value))
                return dict(result)
            if isinstance(obj, tuple) and hasattr(obj, '_fields'):
                return type(obj)(*[wrapper(v, is_dynamic) for v in obj])
            if isinstance(obj, (list, tuple)):
                return type(obj)(wrapper(v, is_dynamic) for v in obj)
            if isinstance(obj, dict):
                return type(obj)((wrapper(k, is_dynamic), wrapper(v, is_dynamic))
                                 for k, v in obj.items())

            return copy.deepcopy(obj)

        if not is_dataclass(obj):
            raise TypeError("asdict() should be called on dataclass instances")
        return wrapper(obj, True), wrapper(obj, False)

    def flatten_state(state: State) -> Tuple[Dict, Dict]:
        return asdict(state)


    @overload
    def combine(first: Dict, second: Dict) -> Dict:
        ...

    @overload
    def combine(first: Optional[Any], second: Optional[Any]) -> Any:
        ...

    def combine(first: Union[Dict, Optional[Any]], second: Union[Dict, Optional[Any]]
                ) -> Union[Dict, Any]:
        if isinstance(first, dict) and isinstance(second, dict):
            result = {}
            for key, val in first.items():
                result[key] = combine(val, second.get(key, None))
            for key, val in second.items():
                result[key] = combine(first.get(key, None), val)

            return result

        if first is None:
            return second
        return first

    @overload
    def unflatten_state(state: S, params: Dict) -> S:
        ...

    @overload
    def unflatten_state(state: Params, params: Params) -> Params:
        ...

    def unflatten_state(state: Union[S, Params], params: Union[Dict, Params]) -> Union[S, Params]:
        if is_dataclass(state):
            attributes = {}
            for fld in fields(state):
                if not isinstance(params, dict) or fld.name not in params:
                    raise ValueError(f"No attribute '{fld.name}' in params: {params:s}")
                attributes[fld.name] = unflatten_state(getattr(state, fld.name), params[fld.name])
            return type(state)(**attributes)
        else:
            if isinstance(params, dict):
                raise ValueError(f"Invalid params: {params}")
            return params

    def generate_bounds(key: KeyArray, params: Union[Dict, Params],
                        bounds: Union[Dict, Params, None], default: Callable[[Params], Params]
                        ) -> Union[Dict, Params]:
        if isinstance(params, dict):
            result = {}
            for attr, val in params.items():
                if isinstance(bounds, dict) and attr in bounds:
                    result[attr] = generate_bounds(key, val, bounds[attr], default)
                else:
                    result[attr] = generate_bounds(key, val, None, default)
            return result

        if bounds is None or isinstance(bounds, dict):
            bound = default(params)
        else:
            bound = bounds

        key, subkey = random.split(key)
        return bound * random.uniform(subkey, jnp.shape(params), minval=-1, maxval=1)

    def init(rng) -> S:
        dynamic_params, static_params = flatten_state(state)
        bound_params = generate_bounds(rng, dynamic_params, bounds, default)
        dynamic_params = tree_util.tree_map(jnp.add, dynamic_params, bound_params)
        return unflatten_state(state, combine(dynamic_params, static_params))

    return init

@partial(jit, static_argnums=1)
def rand_rotation_matrix(rng: KeyArray, shape: Shape=()) -> RealArray:
    values = random.uniform(rng, shape=shape + (3,))
    theta = 2.0 * jnp.pi * values[..., 0]
    phi = 2.0 * jnp.pi * values[..., 1]
    r = jnp.sqrt(values[..., 2])
    V = jnp.stack((jnp.cos(phi) * r, jnp.sin(phi) * r, jnp.sqrt(1.0 - values[..., 2])), axis=-1)
    st = jnp.sin(theta)
    ct = jnp.cos(theta)
    R = jnp.stack((jnp.stack((ct, st, jnp.zeros(shape)), axis=-1),
                   jnp.stack((-st, ct, jnp.zeros(shape)), axis=-1),
                   jnp.broadcast_to(jnp.array([0, 0, 1]), shape + (3,))), axis=-2)
    V = 2 * V[..., None, :] * V[..., None] - jnp.broadcast_to(jnp.eye(3), shape + (3, 3))
    return jnp.sum(V[..., None] * R[..., None, :, :], axis=-2)

@jax_dataclass
class RotationState():
    matrix : RealArray
    mat_columns : ClassVar[Tuple[str, ...]] = ('Rxx', 'Rxy', 'Rxz',
                                               'Ryx', 'Ryy', 'Ryz',
                                               'Rzx', 'Rzy', 'Rzz')

    @classmethod
    def import_dataframe(cls, data: pd.Series) -> 'RotationState':
        """Initialize a new :class:`Sample` object with a :class:`pandas.Series` array. The array
        must contain the following columns:

        * `Rxx`, `Rxy`, `Rxz`, `Ryx`, `Ryy`, `Ryz`, `Rzx`, `Rzy`, `Rzz` : Rotational matrix.
        * `z` : z coordinate [m].

        Args:
            data : A :class:`pandas.Series` array.

        Returns:
            A new :class:`Sample` object.
        """
        matrix = jnp.asarray(data[list(cls.mat_columns)].to_numpy())
        return cls(jnp.reshape(matrix, (3, 3)))

    def rotate(self, vecs: RealArray) -> RealArray:
        return jnp.sum(self.matrix[..., None, :, :] * vecs[..., None], axis=-2)

    def to_euler(self) -> 'EulerState':
        r"""Calculate Euler angles with Bunge convention [EUL]_.

        Returns:
            A set of Euler angles with Bunge convention :math:`\phi_1, \Phi, \phi_2`.
        """
        return EulerState(euler_angles(self.matrix))

    def to_tilt(self) -> RealArray:
        r"""Calculate an axis of rotation and a rotation angle for a rotation matrix.

        Returns:
            A set of three angles :math:`\theta, \alpha, \beta`, a rotation angle :math:`\theta`,
            an angle between the axis of rotation and OZ axis :math:`\alpha`, and a polar angle
            of the axis of rotation :math:`\beta`.
        """
        if jnp.allclose(self.matrix, self.matrix.T):
            eigw, eigv = jnp.stack(jnp.linalg.eigh(self.matrix))
            axis = eigv[jnp.isclose(eigw, 1.0)]
            theta = jnp.arccos(0.5 * (jnp.trace(self.matrix) - 1.0))
            return jnp.array([theta, jnp.arccos(axis[0, 2]), jnp.arctan2(axis[0, 1], axis[0, 0])])
        return tilt_angles(self.matrix)

class Transform():
    def apply(self, xtal: XtalState, state: State) -> XtalState:
        raise NotImplementedError

class Rotation(Transform):
    def apply(self, xtal: XtalState, state: RotationState) -> XtalState:
        return XtalState(jnp.sum(state.matrix[..., None, :, :] * xtal.basis[..., None], axis=-2))

@jax_dataclass
class EulerState():
    angles : RealArray

    def to_rotation(self) -> RotationState:
        return RotationState(euler_matrix(self.angles))

class Euler(Transform):
    rotation : Rotation = Rotation()

    def apply(self, xtal: XtalState, state: EulerState) -> XtalState:
        return self.rotation.apply(xtal, state.to_rotation())

@jax_dataclass
class TiltState():
    angles : RealArray

    def to_rotation(self) -> RotationState:
        return RotationState(tilt_matrix(self.angles))

class Tilt():
    rotation : Rotation = Rotation()

    def apply(self, xtal: XtalState, state: TiltState) -> XtalState:
        return self.rotation.apply(xtal, state.to_rotation())

@jax_dataclass
class TiltAxisState:
    angles : RealArray
    axis : RealArray

    def to_tilt(self) -> TiltState:
        r = jnp.sqrt(jnp.sum(self.axis**2, axis=-1))
        theta = jnp.broadcast_to(jnp.arccos(self.axis[..., 2] / r), self.angles.shape)
        phi = jnp.broadcast_to(jnp.arctan2(self.axis[..., 1], self.axis[..., 0]), self.angles.shape)
        return TiltState(jnp.stack((self.angles, theta, phi), axis=-1))

class TiltAxis():
    tilt : Tilt = Tilt()

    def apply(self, xtal: XtalState, state: TiltAxisState) -> XtalState:
        return self.tilt.apply(xtal, state.to_tilt())

@jax_dataclass
class ChainTransform():
    transforms : Tuple[Transform, ...]

    def apply(self, xtal: XtalState, state: Tuple[State, ...]) -> XtalState:
        for s, transform in zip(state, self.transforms):
            xtal = transform.apply(xtal, s)
        return xtal

@jax_dataclass
class InternalState():
    xtal    : XtalState
    lens    : LensState
    z       : RealArray
