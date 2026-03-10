from typing import Callable, Iterator, Tuple, Type, TypeVar, get_type_hints, overload
from typing_extensions import Self
import pandas as pd
from .geometry import euler_angles, euler_matrix, tilt_angles, tilt_matrix
from .._src.annotations import (AnyGenerator, AnyNamespace, BoolArray, Indices, IntArray, JaxNumPy,
                                RealArray, RealSequence, Shape)
from .._src.array_api import array_namespace, asnumpy
from .._src.data_container import ArrayContainer, Container, DataContainer, IndexedContainer
from .._src.parser import JSONParser, INIParser, Parser, get_extension
from .._src.state import State, dynamic_fields, field, static_fields

def random_array(array: RealArray, span: RealSequence) -> Callable[[AnyGenerator], RealArray]:
    xp = array_namespace(array, span)
    def random(rng: AnyGenerator):
        bound = xp.asarray(span)
        return array + xp.asarray(rng.uniform(-0.5 * bound, 0.5 * bound, array.shape),
                                  dtype=array.dtype)

    return random

S = TypeVar('S', bound=State)

def random_state(state: S, span: S) -> Callable[[AnyGenerator], S]:
    xp = array_namespace(state, span)
    def random(rng: AnyGenerator):
        dynamic = {}
        for fld in dynamic_fields(state):
            attr = getattr(state, fld.name)

            if isinstance(attr, State):
                dynamic[fld.name] = random_state(attr, getattr(span, fld.name))(rng)
            else:
                center = xp.asarray(attr)
                bound = xp.abs(xp.asarray(getattr(span, fld.name)))
                rnd = xp.asarray(rng.uniform(-0.5 * bound, 0.5 * bound, center.shape),
                                 dtype=center.dtype)
                dynamic[fld.name] = center + rnd

        static = {fld.name: getattr(state, fld.name) for fld in static_fields(state)}

        return type(state)(**dynamic, **static)

    return random

def random_rotation(shape: Shape=(), xp: AnyNamespace = JaxNumPy
                    ) -> Callable[[AnyGenerator], 'RotationState']:
    def random(rng: AnyGenerator):
        """Creates a random rotation matrix.
        """
        # from http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
        # and  http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
        values = xp.asarray(rng.uniform(size=shape + (3,)))
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

    return random

def random_euler(shape: Shape=(), xp: AnyNamespace=JaxNumPy
                 ) -> Callable[[AnyGenerator], 'EulerState']:
    def random(rng: AnyGenerator):
        angles = rng.uniform(xp.array([0.0, 0.0, 0.0]), xp.array([2 * xp.pi, xp.pi, 2 * xp.pi]),
                             size=shape + (3,))
        return EulerState(xp.asarray(angles))
    return random

class BaseCell(Container):
    angles  : RealArray | Tuple[Tuple[float, float, float], ...]
    lengths : RealArray | Tuple[Tuple[float, float, float], ...]

    @classmethod
    def parser(cls, file_or_extension: str='ini') -> Parser:
        ext = get_extension(file_or_extension)
        field_info = {'unit_cell': {'angles': 'angles', 'lengths': 'lengths'}}
        if ext == 'ini':
            return INIParser(field_info, get_type_hints(cls))
        if ext == 'json':
            return JSONParser(field_info)
        raise ValueError(f"Unsupported file or extension format: {file_or_extension}")

    def to_basis(self, xp: AnyNamespace=JaxNumPy) -> 'XtalState':
        gamma = xp.asarray(self.angles)[..., 2]
        cos = xp.cos(xp.asarray(self.angles))
        sin = xp.sin(gamma)
        v_ratio = xp.sqrt(xp.ones(cos.shape[:-1]) - xp.sum(cos**2, axis=-1) + \
                          2 * xp.prod(cos, axis=-1))
        a_vec = xp.broadcast_to(xp.array([1.0, 0.0, 0.0]), cos.shape)
        b_vec = xp.stack((cos[..., 2], sin, xp.zeros(cos.shape[:-1])), axis=-1)
        c_vec = xp.stack((cos[..., 1], (cos[..., 0] - cos[..., 1] * cos[..., 2]) / sin,
                         v_ratio / sin), axis=-1)
        vectors = xp.stack((a_vec, b_vec, c_vec), axis=-2)
        return XtalState(xp.asarray(xp.asarray(self.lengths)[..., None] * vectors, dtype=float))

class FixedXtalCell(BaseCell, State, eq=True, unsafe_hash=True):
    angles  : Tuple[Tuple[float, float, float], ...] = field(static=True)
    lengths : Tuple[Tuple[float, float, float], ...] = field(static=True)

    @classmethod
    def read(cls, file: str) -> 'FixedXtalCell':
        data = cls.parser(file).read(file)
        return cls(tuple(data['angles']), tuple(data['lengths']))

class XtalCell(BaseCell, ArrayContainer, State):
    angles  : RealArray
    lengths : RealArray

    @property
    def alpha(self) -> RealArray:
        return self.angles[..., 0]

    @property
    def beta(self) -> RealArray:
        return self.angles[..., 1]

    @property
    def gamma(self) -> RealArray:
        return self.angles[..., 2]

    @classmethod
    def read(cls, file: str, xp: AnyNamespace=JaxNumPy) -> 'XtalCell':
        data = cls.parser(file).read(file)
        return cls(xp.asarray(data['angles']), xp.asarray(data['lengths']))

    def to_basis(self) -> 'XtalState':
        return super().to_basis(self.__array_namespace__())

class XtalState(ArrayContainer, State):
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

    def __getitem__(self, indices: Indices | BoolArray) -> 'XtalState':
        return XtalState(self.basis.reshape((-1, 3, 3))[indices])

    def __len__(self) -> int:
        return self.basis.size // 9

    def __iter__(self) -> Iterator['XtalState']:
        xp = self.__array_namespace__()
        for basis in xp.reshape(self.basis, (-1, 3, 3)):
            yield XtalState(basis[None])

    @classmethod
    def parser(cls, file: str='ini') -> Parser:
        ext = get_extension(file)
        field_info = {'basis': {'a': 'a', 'b': 'b', 'c': 'c'}}
        if ext == 'ini':
            return INIParser(field_info, {'a': RealArray, 'b': RealArray, 'c': RealArray})
        if ext == 'json':
            return JSONParser(field_info)

        raise ValueError(f"Unsupported file or extension format: {file}")

    @classmethod
    def read(cls, file: str, xp: AnyNamespace = JaxNumPy) -> 'XtalState':
        data = cls.parser(file).read(file)
        return cls(xp.stack((data['a'], data['b'], data['c'])))

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace=JaxNumPy) -> 'XtalState':
        a = xp.stack((xp.asarray(df['a_x']), xp.asarray(df['a_y']), xp.asarray(df['a_z'])), axis=-1)
        b = xp.stack((xp.asarray(df['b_x']), xp.asarray(df['b_y']), xp.asarray(df['b_z'])), axis=-1)
        c = xp.stack((xp.asarray(df['c_x']), xp.asarray(df['c_y']), xp.asarray(df['c_z'])), axis=-1)
        return cls(xp.stack((a, b, c), axis=-2))

    @classmethod
    def import_spherical(cls, r: RealArray, theta: RealArray, phi: RealArray,
                         xp: AnyNamespace = JaxNumPy) -> 'XtalState':
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
        return XtalCell(angles=xp.acos(angles), lengths=lengths)

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
        a_rec = xp.linalg.cross(self.b, self.c) / xp.sum(xp.linalg.cross(self.b, self.c) * self.a, axis=-1)
        b_rec = xp.linalg.cross(self.c, self.a) / xp.sum(xp.linalg.cross(self.c, self.a) * self.b, axis=-1)
        c_rec = xp.linalg.cross(self.a, self.b) / xp.sum(xp.linalg.cross(self.a, self.b) * self.c, axis=-1)
        return XtalState(xp.stack((a_rec, b_rec, c_rec), axis=-2))

    def to_dataframe(self, index: IntArray | None) -> pd.DataFrame:
        xp = self.__array_namespace__()
        if index is None:
            index = xp.arange(len(self))
        return pd.DataFrame({'index': asnumpy(index),
                             'a_x': asnumpy(self.a[..., 0]), 'a_y': asnumpy(self.a[..., 1]),
                             'a_z': asnumpy(self.a[..., 2]),
                             'b_x': asnumpy(self.b[..., 0]), 'b_y': asnumpy(self.b[..., 1]),
                             'b_z': asnumpy(self.b[..., 2]),
                             'c_x': asnumpy(self.c[..., 0]), 'c_y': asnumpy(self.c[..., 1]),
                             'c_z': asnumpy(self.c[..., 2])})

    def to_spherical(self) -> Tuple[RealArray, RealArray, RealArray]:
        """Return a stack of unit cell vectors in spherical coordinate system.

        Returns:
            A matrix of three stacked unit cell vectors in spherical coordinate system.
        """
        xp = self.__array_namespace__()
        lengths = xp.sqrt(xp.sum(self.basis**2, axis=-1))
        return (lengths, xp.acos(self.basis[..., 2] / lengths),
                xp.atan2(self.basis[..., 1], self.basis[..., 0]))

class XtalList(IndexedContainer, State):
    index       : IntArray
    basis       : RealArray

    def to_xtals(self) -> XtalState:
        return XtalState(self.basis)

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace=JaxNumPy) -> 'XtalList':
        xtals = XtalState.import_dataframe(df, xp)
        return cls(xp.asarray(df['index']), xtals.basis)

    def to_dataframe(self) -> pd.DataFrame:
        xp = self.__array_namespace__()
        return self.to_xtals().to_dataframe(index=xp.asarray(self.index))

Float = float | RealArray

class BaseLens(Container):
    foc_pos     : Tuple[float, float, float] | RealArray
    pupil_roi   : Tuple[float, float, float, float] | RealArray

    @property
    def pupil_min(self) -> Tuple[Float, Float]:
        return (self.pupil_roi[2], self.pupil_roi[0])

    @property
    def pupil_max(self) -> Tuple[Float, Float]:
        return (self.pupil_roi[3], self.pupil_roi[1])

    @property
    def pupil_center(self) -> Tuple[Float, Float]:
        return (0.5 * (self.pupil_min[0] + self.pupil_max[0]),
                0.5 * (self.pupil_min[1] + self.pupil_max[1]))

    @classmethod
    def parser(cls, file_or_extension: str='ini') -> Parser:
        ext = get_extension(file_or_extension)
        field_info = {'geometry': {'foc_pos': 'foc_pos', 'pupil_roi': 'pupil_roi'}}
        if ext == 'ini':
            return INIParser(field_info, get_type_hints(cls))
        if ext == 'json':
            return JSONParser(field_info)
        raise ValueError(f"Unsupported file or extension format: {file_or_extension}")

    @classmethod
    def read(cls: Type[Self], file: str, xp: AnyNamespace=JaxNumPy) -> Self:
        raise NotImplementedError

class FixedLens(BaseLens, State, eq=True, unsafe_hash=True):
    foc_pos     : Tuple[float, float, float] = field(static=True)
    pupil_roi   : Tuple[float, float, float, float] = field(static=True)

    @classmethod
    def read(cls, file: str, xp: AnyNamespace=JaxNumPy) -> 'FixedLens':
        data = cls.parser(file).read(file)
        return cls(tuple(data['foc_pos']), tuple(data['pupil_roi']))

class FixedPupilLens(DataContainer, FixedLens):
    foc_pos     : RealArray
    pupil_roi   : Tuple[float, float, float, float] = field(static=True)

    @classmethod
    def read(cls, file: str, xp: AnyNamespace=JaxNumPy) -> 'FixedPupilLens':
        data = cls.parser(file).read(file)
        return cls(xp.asarray(data['foc_pos']), tuple(data['pupil_roi']))

class FixedApertureLens(BaseLens, DataContainer, State):
    foc_pos         : RealArray
    pupil_center    : RealArray
    aperture        : Tuple[float, float] = field(static=True)

    @property
    def pupil_roi(self) -> RealArray:
        xp = self.__array_namespace__()
        x, y = self.pupil_center
        ap_x, ap_y = self.aperture
        return xp.array([y - 0.5 * ap_y, y + 0.5 * ap_y, x - 0.5 * ap_x, x + 0.5 * ap_x])

    @classmethod
    def from_roi(cls, obj: FixedLens | FixedPupilLens, xp: AnyNamespace=JaxNumPy
                 ) -> 'FixedApertureLens':
        return cls(xp.asarray(obj.foc_pos), xp.asarray(obj.pupil_center),
                   (float(obj.pupil_max[0] - obj.pupil_min[0]),
                    float(obj.pupil_max[1] - obj.pupil_min[1])))

    @classmethod
    def read(cls, file: str, xp: AnyNamespace=JaxNumPy) -> 'FixedApertureLens':
        return cls.from_roi(FixedLens.read(file), xp)

    @classmethod
    def parser(cls, file: str) -> Parser:
        ext = get_extension(file)
        field_info = {'geometry': {'foc_pos': 'foc_pos', 'pupil_roi': 'pupil_roi'}}
        if ext == 'ini':
            return INIParser(field_info, {'foc_pos': RealArray, 'pupil_roi': RealArray})
        if ext == 'json':
            return JSONParser(field_info)

        raise ValueError(f"Unsupported file or extension format: {file}")

class RotationState(ArrayContainer, State):
    matrix : RealArray

    def __len__(self) -> int:
        return self.matrix.size // 9

    def __iter__(self) -> Iterator['RotationState']:
        xp = array_namespace(self)
        for matrix in xp.reshape(self.matrix, (-1, 3, 3)):
            yield RotationState(matrix[None])

    @overload
    def __matmul__(self, other: 'RotationState') -> 'RotationState': ...

    @overload
    def __matmul__(self, other: XtalState) -> XtalState: ...

    @overload
    def __matmul__(self, other: RealArray) -> RealArray: ...

    def __matmul__(self, other: 'RotationState | RealArray | XtalState'
                   ) -> 'RotationState | RealArray | XtalState':
        if isinstance(other, RotationState):
            return RotationState(self.matrix @ other.matrix)
        if isinstance(other, XtalState):
            return XtalState(other.basis @ self.matrix)
        return other @ self.matrix

    @overload
    def __rmatmul__(self, other: XtalState) -> XtalState: ...

    @overload
    def __rmatmul__(self, other: RealArray) -> RealArray: ...

    def __rmatmul__(self, other: RealArray | XtalState) -> RealArray | XtalState:
        if isinstance(other, XtalState):
            return XtalState(other.basis @ self.matrix)
        return other @ self.matrix

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace=JaxNumPy
                         ) -> 'RotationState':
        """Initialize a new :class:`Sample` object with a :class:`pandas.Series` array. The array
        must contain the following columns:

        * `Rxx`, `Rxy`, `Rxz`, `Ryx`, `Ryy`, `Ryz`, `Rzx`, `Rzy`, `Rzz` : Rotational matrix.
        * `z` : z coordinate [m].

        Args:
            data : A :class:`pandas.Series` array.

        Returns:
            A new :class:`Sample` object.
        """
        a = xp.stack((xp.asarray(df['Rxx']), xp.asarray(df['Rxy']), xp.asarray(df['Rxz'])), axis=-1)
        b = xp.stack((xp.asarray(df['Ryx']), xp.asarray(df['Ryy']), xp.asarray(df['Ryz'])), axis=-1)
        c = xp.stack((xp.asarray(df['Rzx']), xp.asarray(df['Rzy']), xp.asarray(df['Rzz'])), axis=-1)
        return cls(xp.stack((a, b, c), axis=-2))

    def to_dataframe(self, index: IntArray | None) -> pd.DataFrame:
        xp = self.__array_namespace__()
        if index is None:
            index = xp.arange(len(self))
        a, b, c = self.matrix[..., 0, :], self.matrix[..., 1, :], self.matrix[..., 2, :]
        return pd.DataFrame({'index': asnumpy(index),
                             'Rxx': asnumpy(a[..., 0]), 'Rxy': asnumpy(a[..., 1]), 'Rxz': asnumpy(a[..., 2]),
                             'Ryx': asnumpy(b[..., 0]), 'Ryy': asnumpy(b[..., 1]), 'Ryz': asnumpy(b[..., 2]),
                             'Rzx': asnumpy(c[..., 0]), 'Rzy': asnumpy(c[..., 1]), 'Rzz': asnumpy(c[..., 2])})

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
        if xp.allclose(self.matrix, xp.permute_dims(self.matrix, (*range(self.matrix.ndim - 2), -1, -2))):
            eigw, eigv = xp.linalg.eigh(self.matrix)
            axis = eigv[xp.isclose(eigw, 1.0)]
            theta = xp.acos(0.5 * (xp.trace(self.matrix) - 1.0))
            alpha = xp.acos(axis[0, 2])
            beta = xp.atan2(axis[0, 1], axis[0, 0])
            return TiltState(xp.array([theta, alpha, beta]))
        return TiltState(tilt_angles(self.matrix, xp))

class EulerState(ArrayContainer, State):
    """Represents rotation state using Euler angles (Bunge convention).

    This class stores rotation information as Euler angles following the ZXZ
    convention (also known as the Bunge convention) and provides conversion to
    rotation matrix representation.

    The Euler angles define a sequence of three elementary rotations:
        1. Rotation around z-axis by phi1: (x,y,z) → (u,v,z)
        2. Rotation around u-axis by Phi: (u,v,z) → (u,w,z1)
        3. Rotation around z1-axis by phi2: (u,w,z1) → (x1,y1,z1)

    Attributes:
        angles (RealArray): Array of Euler angles [phi1, Phi, phi2] defining
            the rotation, where:

            - phi1: First rotation angle around z-axis, range [0, 2 * pi)
            - Phi: Second rotation angle around u-axis, range [0, pi]
            - phi2: Third rotation angle around z1-axis, range [0, 2 * pi)

    Notes:
        This class inherits from both ArrayContainer and State, providing
        array-like functionality and state management capabilities.

    Examples:
        >>> euler = EulerState(angles=array([0.0, pi/2, pi/4]))
        >>> rotation = euler.to_rotation()
    """
    angles : RealArray

    @property
    def phi1(self) -> RealArray:
        return self.angles[..., 0]

    @property
    def Phi(self) -> RealArray:
        return self.angles[..., 1]

    @property
    def phi2(self) -> RealArray:
        return self.angles[..., 2]

    def to_rotation(self) -> RotationState:
        xp = self.__array_namespace__()
        return RotationState(euler_matrix(self.angles, xp))

class TiltState(ArrayContainer, State):
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

class TiltOverAxisState(ArrayContainer, State):
    angles : RealArray
    axis : RealArray

    @classmethod
    def from_point(cls, points: RealArray) -> 'TiltOverAxisState':
        xp = array_namespace(points)
        angles = xp.sqrt(xp.sum(points**2, axis=-1))
        return cls(4.0 * xp.tan(angles), points / angles[..., None])

    def alpha(self) -> RealArray:
        xp = self.__array_namespace__()
        r = xp.sqrt(xp.sum(self.axis**2, axis=-1))
        return xp.broadcast_to(xp.acos(self.axis[..., 2] / r), self.angles.shape)

    def beta(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.broadcast_to(xp.atan2(self.axis[..., 1], self.axis[..., 0]),
                               self.angles.shape)

    def to_point(self) -> RealArray:
        return self.angles[..., None] * self.axis

    def to_tilt(self) -> TiltState:
        xp = self.__array_namespace__()
        return TiltState(xp.stack((self.angles, self.alpha(), self.beta()), axis=-1))

class BaseSetup(Container):
    lens    : BaseLens
    z       : Tuple[float, ...] | RealArray

    @property
    def foc_pos(self) -> Tuple[float, float, float] | RealArray:
        return self.lens.foc_pos

    @property
    def pupil_roi(self) -> Tuple[float, float, float, float] | RealArray:
        return self.lens.pupil_roi

    @property
    def pupil_min(self) -> Tuple[Float, Float]:
        return self.lens.pupil_min

    @property
    def pupil_max(self) -> Tuple[Float, Float]:
        return self.lens.pupil_max

    @property
    def pupil_center(self) -> Tuple[Float, Float] | RealArray:
        return self.lens.pupil_center

    @classmethod
    def parser(cls, file: str) -> Parser:
        ext = get_extension(file)
        field_info = {'geometry': {'foc_pos': 'foc_pos', 'pupil_roi': 'pupil_roi', 'z': 'z'}}
        if ext == 'ini':
            return INIParser(field_info, get_type_hints(cls))
        if ext == 'json':
            return JSONParser(field_info)

        raise ValueError(f"Invalid format: {ext}")

class FixedPupilSetup(BaseSetup, DataContainer, State):
    lens    : FixedPupilLens
    z       : RealArray

    @classmethod
    def read(cls, file: str, xp: AnyNamespace=JaxNumPy) -> 'FixedPupilSetup':
        data = cls.parser(file).read(file)
        return cls(FixedPupilLens(xp.asarray(data['foc_pos']), tuple(data['pupil_roi'])),
                   xp.asarray(data['z']))

class FixedApertureSetup(BaseSetup, DataContainer, State):
    lens    : FixedApertureLens
    z       : RealArray

    @classmethod
    def read(cls, file: str, xp: AnyNamespace=JaxNumPy) -> 'FixedApertureSetup':
        data = cls.parser(file).read(file)
        lens = FixedLens(tuple(data['foc_pos']), tuple(data['pupil_roi']))
        return cls(FixedApertureLens.from_roi(lens), xp.asarray(data['z']))

class FixedSetup(BaseSetup, State, eq=True, unsafe_hash=True):
    lens    : FixedLens = field(static=True)
    z       : Tuple[float, ...] = field(static=True)

    @classmethod
    def read(cls, file: str) -> 'FixedSetup':
        data = cls.parser(file).read(file)
        return cls(FixedLens(tuple(data['foc_pos']), tuple(data['pupil_roi'])), tuple(data['z']))

class BaseState(DataContainer, BaseSetup):
    xtal    : XtalState
    setup   : BaseSetup

    @property
    def lens(self) -> BaseLens:
        return self.setup.lens

    @property
    def z(self) -> Tuple[float, ...] | RealArray:
        return self.setup.z
