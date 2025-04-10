from typing import Callable, ClassVar, Dict, Iterator, Tuple, Type, TypeVar, cast, get_type_hints
import pandas as pd
from jax import random
from .geometry import euler_angles, euler_matrix, tilt_angles, tilt_matrix
from .state import State, dynamic_fields, field, static_fields
from .._src.annotations import ArrayNamespace, JaxNumPy, KeyArray, RealArray, RealSequence, Shape
from .._src.data_container import Parser, INIParser, JSONParser, array_namespace

S = TypeVar('S', bound=State)

def random_array(array: RealArray, span: RealSequence) -> Callable[[KeyArray], RealArray]:
    xp = array_namespace(array, span)
    def rnd(key: KeyArray):
        bound = xp.asarray(span)
        return array + xp.asarray(random.uniform(key, array.shape, array.dtype,
                                                 -0.5 * bound, 0.5 * bound))

    return rnd

def random_state(state: S, span: S) -> Callable[[KeyArray], S]:
    xp = array_namespace(state, span)
    def rnd(key: KeyArray):
        dynamic = {}
        for fld in dynamic_fields(state):
            center = xp.asarray(getattr(state, fld.name))
            bound = xp.abs(xp.asarray(getattr(span, fld.name)))
            rnd = xp.asarray(random.uniform(key, center.shape, center.dtype,
                                            -0.5 * bound, 0.5 * bound))
            dynamic[fld.name] = center + rnd

        static = {fld.name: getattr(state, fld.name) for fld in static_fields(state)}

        return type(state)(**dynamic, **static)

    return rnd

def random_rotation(shape: Shape=(), xp: ArrayNamespace = JaxNumPy) -> Callable[[KeyArray], 'RotationState']:
    def rnd(key: KeyArray):
        """Creates a random rotation matrix.
        """
        # from http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
        # and  http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
        values = xp.asarray(random.uniform(key, shape=shape + (3,)))
        theta = 2.0 * xp.pi * values[..., 0]
        phi = 2.0 * xp.pi * values[..., 1]
        r = xp.sqrt(values[..., 2])
        V = xp.stack((xp.cos(phi) * r, xp.sin(phi) * r, xp.sqrt(1.0 - values[..., 2])), axis=-1)
        st = xp.sin(theta)
        ct = xp.cos(theta)
        R = xp.stack((xp.stack((ct, st, xp.zeros(shape)), axis=-1),
                      xp.stack((-st, ct, xp.zeros(shape)), axis=-1),
                      xp.broadcast_to(xp.array([0.0, 0.0, 1.0]), shape + (3,))), axis=-2)
        V = 2 * V[..., None, :] * V[..., None] - xp.broadcast_to(xp.eye(3), shape + (3, 3))
        return RotationState(xp.sum(V[..., None] * R[..., None, :, :], axis=-2))

    return rnd

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
        xp = self.__array_namespace__()
        cos = xp.cos(self.angles)
        sin = xp.sin(self.gamma)
        v_ratio = xp.sqrt(xp.ones(self.shape[:-1]) - xp.sum(cos**2, axis=-1) + \
                           2 * xp.prod(cos, axis=-1))
        a_vec = xp.broadcast_to(xp.array([1.0, 0.0, 0.0]), self.shape)
        b_vec = xp.stack((cos[..., 2], sin, xp.zeros(self.shape[:-1])), axis=-1)
        c_vec = xp.stack((cos[..., 1], (cos[..., 0] - cos[..., 1] * cos[..., 2]) / sin,
                           v_ratio / sin), axis=-1)
        return XtalState(self.lengths[..., None] * xp.stack((a_vec, b_vec, c_vec), axis=-2))

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
        xp = self.__array_namespace__()
        for basis in xp.reshape(self.basis, (-1, 3, 3)):
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
    def read(cls, file: str, ext: str='ini', xp: ArrayNamespace = JaxNumPy) -> 'XtalState':
        return cls(xp.stack(list(cls.parser(ext).read(file).values())))

    @classmethod
    def import_spherical(cls, r: RealArray, theta: RealArray, phi: RealArray,
                         xp: ArrayNamespace = JaxNumPy) -> 'XtalState':
        """Return a new :class:`XtalState` object, initialised by a stacked matrix of three basis
        vectors written in spherical coordinate system.

        Args:
            mat : A matrix of three stacked basis vectors in spherical coordinate system.

        Returns:
            A new :class:`XtalState` object.
        """
        return cls(xp.stack((r * xp.sin(theta) * xp.cos(phi), r * xp.sin(theta) * xp.sin(phi),
                             r * xp.cos(theta)), axis=-1))

    @property
    def unit_cell(self) -> XtalCell:
        """Return a stack of unit cell vectors in spherical coordinate system.

        Returns:
            A matrix of three stacked unit cell vectors in spherical coordinate system.
        """
        xp = self.__array_namespace__()
        lengths = xp.sqrt(xp.sum(self.basis**2, axis=-1))
        angles = xp.stack([xp.sum(self.b * self.c, axis=-1) / (lengths[..., 1] * lengths[..., 2]),
                           xp.sum(self.c * self.a, axis=-1) / (lengths[..., 2] * lengths[..., 0]),
                           xp.sum(self.a * self.b, axis=-1) / (lengths[..., 0] * lengths[..., 1])],
                          axis=-1)
        return XtalCell(angles=xp.arccos(angles), lengths=lengths)

    @property
    def orientation_matrix(self) -> 'RotationState':
        xp = self.__array_namespace__()
        matrix = xp.linalg.inv(self.unit_cell.to_basis().basis) @ self.basis
        return RotationState(matrix)

    def reciprocate(self) -> 'XtalState':
        """Calculate the basis of the reciprocal lattice.

        Returns:
            The basis of the reciprocal lattice.
        """
        xp = self.__array_namespace__()
        a_rec = xp.cross(self.b, self.c) / xp.sum(xp.cross(self.b, self.c) * self.a, axis=-1)
        b_rec = xp.cross(self.c, self.a) / xp.sum(xp.cross(self.c, self.a) * self.b, axis=-1)
        c_rec = xp.cross(self.a, self.b) / xp.sum(xp.cross(self.a, self.b) * self.c, axis=-1)
        return XtalState(xp.stack((a_rec, b_rec, c_rec), axis=-2))

    def to_spherical(self) -> Tuple[RealArray, RealArray, RealArray]:
        """Return a stack of unit cell vectors in spherical coordinate system.

        Returns:
            A matrix of three stacked unit cell vectors in spherical coordinate system.
        """
        xp = self.__array_namespace__()
        lengths = xp.sqrt(xp.sum(self.basis**2, axis=-1))
        return (lengths, xp.arccos(self.basis[..., 2] / lengths),
                xp.arctan2(self.basis[..., 1], self.basis[..., 0]))

class LensState(State):
    foc_pos     : RealArray

    @property
    def pupil_min(self) -> RealArray:
        raise NotImplementedError

    @property
    def pupil_max(self) -> RealArray:
        raise NotImplementedError

class FixedPupilState(LensState):
    pupil_roi   : Tuple[float, float, float, float] = field(static=True)

    @property
    def pupil_min(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.array([self.pupil_roi[2], self.pupil_roi[0]])

    @property
    def pupil_max(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.array([self.pupil_roi[3], self.pupil_roi[1]])

    @property
    def pupil_center(self) -> RealArray:
        return 0.5 * (self.pupil_min + self.pupil_max)

    @classmethod
    def parser(cls, ext: str='ini') -> Parser:
        if ext == 'ini':
            return INIParser({'exp_geom': ('foc_pos', 'pupil_roi')},
                             types=get_type_hints(cls))
        if ext == 'json':
            return JSONParser({'exp_geom': ('foc_pos', 'pupil_roi')})

        raise ValueError(f"Invalid format: {ext}")

    @classmethod
    def read(cls, file: str, ext: str='ini') -> 'FixedPupilState':
        return cls(**cls.parser(ext).read(file))

class FixedApertureState(LensState):
    pupil_center    : RealArray
    aperture        : Tuple[float, float] = field(static=True)

    @property
    def pupil_min(self) -> RealArray:
        xp = self.__array_namespace__()
        return self.pupil_center - 0.5 * xp.array(self.aperture)

    @property
    def pupil_max(self) -> RealArray:
        xp = self.__array_namespace__()
        return self.pupil_center + 0.5 * xp.array(self.aperture)

class RotationState(State):
    matrix : RealArray
    mat_columns : ClassVar[Tuple[str, ...]] = ('Rxx', 'Rxy', 'Rxz',
                                               'Ryx', 'Ryy', 'Ryz',
                                               'Rzx', 'Rzy', 'Rzz')

    def __iter__(self) -> Iterator['RotationState']:
        xp = self.__array_namespace__()
        for matrix in xp.reshape(self.matrix, (-1, 3, 3)):
            yield RotationState(matrix[None])

    @classmethod
    def import_dataframe(cls, data: pd.Series, xp: ArrayNamespace = JaxNumPy) -> 'RotationState':
        """Initialize a new :class:`Sample` object with a :class:`pandas.Series` array. The array
        must contain the following columns:

        * `Rxx`, `Rxy`, `Rxz`, `Ryx`, `Ryy`, `Ryz`, `Rzx`, `Rzy`, `Rzz` : Rotational matrix.
        * `z` : z coordinate [m].

        Args:
            data : A :class:`pandas.Series` array.

        Returns:
            A new :class:`Sample` object.
        """
        matrix = xp.asarray(data[list(cls.mat_columns)].to_numpy())
        return cls(xp.reshape(matrix, (3, 3)))

    def to_euler(self) -> 'EulerState':
        r"""Calculate Euler angles with Bunge convention [EUL]_.

        Returns:
            A set of Euler angles with Bunge convention :math:`\phi_1, \Phi, \phi_2`.
        """
        xp = self.__array_namespace__()
        return EulerState(euler_angles(self.matrix, xp))

    def to_tilt(self) -> 'TiltState':
        r"""Calculate an axis of rotation and a rotation angle for a rotation matrix.

        Returns:
            A set of three angles :math:`\theta, \alpha, \beta`, a rotation angle :math:`\theta`,
            an angle between the axis of rotation and OZ axis :math:`\alpha`, and a polar angle
            of the axis of rotation :math:`\beta`.
        """
        xp = self.__array_namespace__()
        if xp.allclose(self.matrix, xp.swapaxes(self.matrix, -1, -2)):
            eigw, eigv = xp.stack(xp.linalg.eigh(self.matrix))
            axis = eigv[xp.isclose(eigw, 1.0)]
            theta = xp.arccos(0.5 * (xp.trace(self.matrix) - 1.0))
            alpha = xp.arccos(axis[0, 2])
            beta = xp.arctan2(axis[0, 1], axis[0, 0])
            return TiltState(xp.array([theta, alpha, beta]))
        return TiltState(tilt_angles(self.matrix, xp))

class EulerState(State):
    angles : RealArray

    def to_rotation(self) -> RotationState:
        xp = self.__array_namespace__()
        return RotationState(euler_matrix(self.angles, xp))

class TiltState(State):
    angles : RealArray

    def axis(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.stack([xp.sin(self.angles[..., 1]) * xp.cos(self.angles[..., 2]),
                         xp.sin(self.angles[..., 1]) * xp.sin(self.angles[..., 2]),
                         xp.cos(self.angles[..., 1])], axis=-1)

    def to_rotation(self) -> RotationState:
        xp = self.__array_namespace__()
        return RotationState(tilt_matrix(self.angles, xp))

    def to_tilt_over_axis(self) -> 'TiltOverAxisState':
        return TiltOverAxisState(self.angles[..., 0], self.axis())

class TiltOverAxisState(State):
    angles : RealArray
    axis : RealArray

    def alpha(self) -> RealArray:
        xp = self.__array_namespace__()
        r = xp.sqrt(xp.sum(self.axis**2, axis=-1))
        return xp.broadcast_to(xp.arccos(self.axis[..., 2] / r), self.angles.shape)

    def beta(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.broadcast_to(xp.arctan2(self.axis[..., 1], self.axis[..., 0]),
                               self.angles.shape)

    def to_tilt(self) -> TiltState:
        xp = self.__array_namespace__()
        return TiltState(xp.stack((self.angles, self.alpha(), self.beta()), axis=-1))

class InternalState(State):
    lens    : LensState
    xtal    : XtalState
    z       : RealArray
