from typing import Callable, ClassVar, Dict, Iterator, Tuple, Type, TypeVar, cast, get_type_hints
from dataclasses import dataclass
import pandas as pd
import jax.numpy as jnp
from jax import random
from .geometry import euler_angles, euler_matrix, project_to_rect, tilt_angles, tilt_matrix
from .state import State, dynamic_fields, field, static_fields
from .._src.annotations import KeyArray, RealArray, RealSequence, Shape
from .._src.data_container import Parser, INIParser, JSONParser

S = TypeVar('S', bound=State)

def random_array(array: RealArray, span: RealSequence) -> Callable[[KeyArray], RealArray]:
    def rnd(key: KeyArray):
        bound = jnp.asarray(span)
        return array + random.uniform(key, array.shape, array.dtype,
                                      -0.5 * bound, 0.5 * bound)

    return rnd

def random_state(state: S, span: S) -> Callable[[KeyArray], S]:
    def rnd(key: KeyArray):
        dynamic = {}
        for fld in dynamic_fields(state):
            center = jnp.asarray(getattr(state, fld.name))
            bound = jnp.abs(jnp.asarray(getattr(span, fld.name)))
            dynamic[fld.name] = center + random.uniform(key, center.shape, center.dtype,
                                                        -0.5 * bound, 0.5 * bound)

        static = {fld.name: getattr(state, fld.name) for fld in static_fields(state)}

        return type(state)(**dynamic, **static)

    return rnd

def random_rotation(shape: Shape=()) -> Callable[[KeyArray], 'RotationState']:
    def rnd(key: KeyArray):
        values = random.uniform(key, shape=shape + (3,))
        theta = 2.0 * jnp.pi * values[..., 0]
        phi = 2.0 * jnp.pi * values[..., 1]
        r = jnp.sqrt(values[..., 2])
        V = jnp.stack((jnp.cos(phi) * r, jnp.sin(phi) * r, jnp.sqrt(1.0 - values[..., 2])), axis=-1)
        st = jnp.sin(theta)
        ct = jnp.cos(theta)
        R = jnp.stack((jnp.stack((ct, st, jnp.zeros(shape)), axis=-1),
                       jnp.stack((-st, ct, jnp.zeros(shape)), axis=-1),
                       jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), shape + (3,))), axis=-2)
        V = 2 * V[..., None, :] * V[..., None] - jnp.broadcast_to(jnp.eye(3), shape + (3, 3))
        return RotationState(jnp.sum(V[..., None] * R[..., None, :, :], axis=-2))

    return rnd

@dataclass
class Detector():
    x_pixel_size : float
    y_pixel_size : float

    def to_indices(self, x: RealArray, y: RealArray) -> Tuple[RealArray, RealArray]:
        return x / self.x_pixel_size, y / self.y_pixel_size

    def to_coordinates(self, i: RealArray, j: RealArray) -> Tuple[RealArray, RealArray]:
        return i * self.x_pixel_size, j * self.y_pixel_size

class XtalCell(State):
    angles  : RealArray
    lengths : RealArray

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
        a_vec = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), self.shape)
        b_vec = jnp.stack((cos[..., 2], sin, jnp.zeros(self.shape[:-1])), axis=-1)
        c_vec = jnp.stack((cos[..., 1], (cos[..., 0] - cos[..., 1] * cos[..., 2]) / sin,
                           v_ratio / sin), axis=-1)
        return XtalState(self.lengths[..., None] * jnp.stack((a_vec, b_vec, c_vec), axis=-2))

class XtalState(State):
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


    def __len__(self) -> int:
        return self.basis.size // 9

    def __iter__(self) -> Iterator['XtalState']:
        for basis in jnp.reshape(self.basis, (-1, 3, 3)):
            yield XtalState(basis[None])

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
    def import_spherical(cls, r: RealArray, theta: RealArray, phi: RealArray) -> 'XtalState':
        """Return a new :class:`XtalState` object, initialised by a stacked matrix of three basis
        vectors written in spherical coordinate system.

        Args:
            mat : A matrix of three stacked basis vectors in spherical coordinate system.

        Returns:
            A new :class:`XtalState` object.
        """
        return cls(jnp.stack((r * jnp.sin(theta) * jnp.cos(phi),
                              r * jnp.sin(theta) * jnp.sin(phi),
                              r * jnp.cos(theta)), axis=-1))

    def lattice_constants(self) -> XtalCell:
        """Return a stack of unit cell vectors in spherical coordinate system.

        Returns:
            A matrix of three stacked unit cell vectors in spherical coordinate system.
        """
        lengths = jnp.sqrt(jnp.sum(self.basis**2, axis=-1))
        angles = jnp.stack([jnp.sum(self.b * self.c, axis=-1) / (lengths[..., 1] * lengths[..., 2]),
                            jnp.sum(self.c * self.a, axis=-1) / (lengths[..., 2] * lengths[..., 0]),
                            jnp.sum(self.a * self.b, axis=-1) / (lengths[..., 0] * lengths[..., 1])],
                           axis=-1)
        return XtalCell(angles=jnp.arccos(angles), lengths=lengths)

    def orientation_matrix(self) -> 'RotationState':
        matrix = jnp.linalg.inv(self.lattice_constants().to_basis().basis) @ self.basis
        return RotationState(matrix)

    def reciprocate(self) -> 'XtalState':
        """Calculate the basis of the reciprocal lattice.

        Returns:
            The basis of the reciprocal lattice.
        """
        a_rec = jnp.cross(self.b, self.c) / jnp.sum(jnp.cross(self.b, self.c) * self.a, axis=-1)
        b_rec = jnp.cross(self.c, self.a) / jnp.sum(jnp.cross(self.c, self.a) * self.b, axis=-1)
        c_rec = jnp.cross(self.a, self.b) / jnp.sum(jnp.cross(self.a, self.b) * self.c, axis=-1)
        return XtalState(jnp.stack((a_rec, b_rec, c_rec), axis=-2))

    def to_spherical(self) -> Tuple[RealArray, RealArray, RealArray]:
        """Return a stack of unit cell vectors in spherical coordinate system.

        Returns:
            A matrix of three stacked unit cell vectors in spherical coordinate system.
        """
        lengths = jnp.sqrt(jnp.sum(self.basis**2, axis=-1))
        return (lengths, jnp.arccos(self.basis[..., 2] / lengths),
                jnp.arctan2(self.basis[..., 1], self.basis[..., 0]))

class LensState(State):
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

class Pupil(State):
    vmin    : RealArray
    vmax    : RealArray

    def center(self) -> RealArray:
        return 0.5 * (self.vmin + self.vmax)

    def diagonals(self) -> RealArray:
        return jnp.array([[self.vmin, self.vmax],
                          [[self.vmin[0], self.vmax[1]], [self.vmax[0], self.vmin[1]]]])

    def edges(self, offset: float=0.0) -> RealArray:
        vmin = self.vmin - jnp.clip(offset, 0.0, 1.0) * (self.vmax - self.vmin)
        vmax = self.vmax + jnp.clip(offset, 0.0, 1.0) * (self.vmax - self.vmin)
        return jnp.array([[[vmin[0], vmax[1]], [vmax[0], vmax[1]]],
                          [[vmax[0], vmax[1]], [vmax[0], vmin[1]]],
                          [[vmax[0], vmin[1]], [vmin[0], vmin[1]]],
                          [[vmin[0], vmin[1]], [vmin[0], vmax[1]]]])

    def project(self, kxy: RealArray) -> RealArray:
        return project_to_rect(kxy, self.vmin, self.vmax)

class RotationState(State):
    matrix : RealArray
    mat_columns : ClassVar[Tuple[str, ...]] = ('Rxx', 'Rxy', 'Rxz',
                                               'Ryx', 'Ryy', 'Ryz',
                                               'Rzx', 'Rzy', 'Rzz')

    def __iter__(self) -> Iterator['RotationState']:
        for matrix in jnp.reshape(self.matrix, (-1, 3, 3)):
            yield RotationState(matrix[None])

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

class EulerState(State):
    angles : RealArray

    def to_rotation(self) -> RotationState:
        return RotationState(euler_matrix(self.angles))

class TiltState(State):
    angles : RealArray

    def to_rotation(self) -> RotationState:
        return RotationState(tilt_matrix(self.angles))

class TiltOverAxisState(State):
    angles : RealArray
    axis : RealArray

    def axis_spherical(self) -> RealArray:
        r = jnp.sqrt(jnp.sum(self.axis**2, axis=-1))
        theta = jnp.broadcast_to(jnp.arccos(self.axis[..., 2] / r), self.angles.shape)
        phi = jnp.broadcast_to(jnp.arctan2(self.axis[..., 1], self.axis[..., 0]), self.angles.shape)
        return jnp.stack((theta, phi))

    def to_tilt(self) -> TiltState:
        theta, phi = self.axis_spherical()
        return TiltState(jnp.stack((self.angles, theta, phi), axis=-1))

class InternalState(State):
    lens    : LensState
    xtal    : XtalState
    z       : RealArray
