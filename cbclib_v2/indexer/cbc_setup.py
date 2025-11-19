from typing import Callable, Iterator, Tuple, Type, TypeVar, get_type_hints
import pandas as pd
from jax import random
from .geometry import euler_angles, euler_matrix, tilt_angles, tilt_matrix
from .._src.state import State, dynamic_fields, field, static_fields
from .._src.annotations import (ArrayNamespace, BoolArray, Indices, IntArray, JaxNumPy, KeyArray, NDRealArray, RealArray,
                                RealSequence, Shape)
from .._src.data_container import ArrayContainer, Container, DataContainer, IndexedContainer, array_namespace
from .._src.parser import JSONParser, INIParser, Parser, get_extension, get_parser

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
            attr = getattr(state, fld.name)

            if isinstance(attr, State):
                dynamic[fld.name] = random_state(attr, getattr(span, fld.name))(key)
            else:
                center = xp.asarray(attr)
                bound = xp.abs(xp.asarray(getattr(span, fld.name)))
                rnd = xp.asarray(random.uniform(key, center.shape, center.dtype,
                                                -0.5 * bound, 0.5 * bound))
                dynamic[fld.name] = center + rnd

        static = {fld.name: getattr(state, fld.name) for fld in static_fields(state)}

        return type(state)(**dynamic, **static)

    return rnd

def random_rotation(shape: Shape=(), xp: ArrayNamespace = JaxNumPy
                    ) -> Callable[[KeyArray], 'RotationState']:
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

class XtalCell(ArrayContainer, State):
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

    @classmethod
    def parser(cls, file_or_extension: str='ini') -> Parser:
        return get_parser(file_or_extension, cls, 'unit_cell')

    @classmethod
    def read(cls, file: str, xp: ArrayNamespace=JaxNumPy) -> 'XtalCell':
        data = cls.parser(file).read(file)
        return cls(xp.asarray(data['angles']), xp.asarray(data['lengths']))

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
    def parser(cls, file_or_extension: str='ini') -> Parser:
        ext = get_extension(file_or_extension)
        if ext == 'ini':
            return INIParser({'basis': {'a': 'a', 'b': 'b', 'c': 'c'}},
                             types={'a': NDRealArray, 'b': NDRealArray, 'c': NDRealArray})
        if ext == 'json':
            return JSONParser({'basis': {'a': 'a', 'b': 'b', 'c': 'c'}})

        raise ValueError(f"Unsupported file or extension format: {file_or_extension}")

    @classmethod
    def read(cls, file: str, xp: ArrayNamespace = JaxNumPy) -> 'XtalState':
        data = cls.parser(file).read(file)
        return cls(xp.stack((data['a'], data['b'], data['c'])))

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame, xp: ArrayNamespace=JaxNumPy) -> 'XtalState':
        a = xp.stack((df['a_x'].to_numpy(), df['a_y'].to_numpy(), df['a_z'].to_numpy()), axis=-1)
        b = xp.stack((df['b_x'].to_numpy(), df['b_y'].to_numpy(), df['b_z'].to_numpy()), axis=-1)
        c = xp.stack((df['c_x'].to_numpy(), df['c_y'].to_numpy(), df['c_z'].to_numpy()), axis=-1)
        return cls(xp.stack((a, b, c), axis=-2))

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

    def to_dataframe(self, index: IntArray | None) -> pd.DataFrame:
        xp = self.__array_namespace__()
        if index is None:
            index = xp.arange(len(self))
        return pd.DataFrame({'index': index,
                             'a_x': self.a[..., 0], 'a_y': self.a[..., 1], 'a_z': self.a[..., 2],
                             'b_x': self.b[..., 0], 'b_y': self.b[..., 1], 'b_z': self.b[..., 2],
                             'c_x': self.c[..., 0], 'c_y': self.c[..., 1], 'c_z': self.c[..., 2]})

    def to_spherical(self) -> Tuple[RealArray, RealArray, RealArray]:
        """Return a stack of unit cell vectors in spherical coordinate system.

        Returns:
            A matrix of three stacked unit cell vectors in spherical coordinate system.
        """
        xp = self.__array_namespace__()
        lengths = xp.sqrt(xp.sum(self.basis**2, axis=-1))
        return (lengths, xp.arccos(self.basis[..., 2] / lengths),
                xp.arctan2(self.basis[..., 1], self.basis[..., 0]))

class XtalList(IndexedContainer, State):
    index       : IntArray
    basis       : RealArray

    def to_xtals(self) -> XtalState:
        return XtalState(self.basis)

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame, xp: ArrayNamespace=JaxNumPy) -> 'XtalList':
        xtals = XtalState.import_dataframe(df, xp)
        return cls(df['index'].to_numpy(), xtals.basis)

    def to_dataframe(self) -> pd.DataFrame:
        xp = self.__array_namespace__()
        return self.to_xtals().to_dataframe(index=xp.asarray(self.index))


L = TypeVar("L", bound='BaseLens')
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
        return get_parser(file_or_extension, cls, 'geometry')

    @classmethod
    def read(cls: Type[L], file: str, xp: ArrayNamespace=JaxNumPy) -> L:
        raise NotImplementedError

class FixedLens(BaseLens, State, eq=True, unsafe_hash=True):
    foc_pos     : Tuple[float, float, float] = field(static=True)
    pupil_roi   : Tuple[float, float, float, float] = field(static=True)

    @classmethod
    def read(cls, file: str, xp: ArrayNamespace=JaxNumPy) -> 'FixedLens':
        data = cls.parser(file).read(file)
        return cls(tuple(data['foc_pos']), tuple(data['pupil_roi']))

class FixedPupilLens(DataContainer, FixedLens):
    foc_pos     : RealArray
    pupil_roi   : Tuple[float, float, float, float] = field(static=True)

    @classmethod
    def read(cls, file: str, xp: ArrayNamespace=JaxNumPy) -> 'FixedPupilLens':
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
    def from_roi(cls, obj: FixedLens | FixedPupilLens, xp: ArrayNamespace=JaxNumPy
                 ) -> 'FixedApertureLens':
        return cls(xp.asarray(obj.foc_pos), xp.asarray(obj.pupil_center),
                   (float(obj.pupil_max[0] - obj.pupil_min[0]),
                    float(obj.pupil_max[1] - obj.pupil_min[1])))

    @classmethod
    def read(cls, file: str, xp: ArrayNamespace=JaxNumPy) -> 'FixedApertureLens':
        return cls.from_roi(FixedLens.read(file), xp)

    @classmethod
    def parser(cls, file_or_extension: str='ini') -> Parser:
        ext = get_extension(file_or_extension)
        if ext == 'ini':
            return INIParser({'geometry': {'foc_pos': 'foc_pos', 'pupil_roi': 'pupil_roi'}},
                             {'foc_pos': RealArray, 'pupil_roi': RealArray})
        if ext == 'json':
            return JSONParser({'geometry': {'foc_pos': 'foc_pos', 'pupil_roi': 'pupil_roi'}})

        raise ValueError(f"Unsupported file or extension format: {file_or_extension}")

class RotationState(ArrayContainer, State):
    matrix : RealArray

    def __len__(self) -> int:
        return self.matrix.size // 9

    def __iter__(self) -> Iterator['RotationState']:
        xp = array_namespace(self)
        for matrix in xp.reshape(self.matrix, (-1, 3, 3)):
            yield RotationState(matrix[None])

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame, xp: ArrayNamespace=JaxNumPy) -> 'RotationState':
        """Initialize a new :class:`Sample` object with a :class:`pandas.Series` array. The array
        must contain the following columns:

        * `Rxx`, `Rxy`, `Rxz`, `Ryx`, `Ryy`, `Ryz`, `Rzx`, `Rzy`, `Rzz` : Rotational matrix.
        * `z` : z coordinate [m].

        Args:
            data : A :class:`pandas.Series` array.

        Returns:
            A new :class:`Sample` object.
        """
        a = xp.stack((df['Rxx'].to_numpy(), df['Rxy'].to_numpy(), df['Rxz'].to_numpy()), axis=-1)
        b = xp.stack((df['Ryx'].to_numpy(), df['Ryy'].to_numpy(), df['Ryz'].to_numpy()), axis=-1)
        c = xp.stack((df['Rzx'].to_numpy(), df['Rzy'].to_numpy(), df['Rzz'].to_numpy()), axis=-1)
        return cls(xp.stack((a, b, c), axis=-2))

    def to_dataframe(self, index: IntArray | None) -> pd.DataFrame:
        xp = self.__array_namespace__()
        if index is None:
            index = xp.arange(len(self))
        a, b, c = self.matrix[..., 0, :], self.matrix[..., 1, :], self.matrix[..., 2, :]
        return pd.DataFrame({'index': index,
                             'Rxx': a[..., 0], 'Rxy': a[..., 1], 'Rxz': a[..., 2],
                             'Ryx': b[..., 0], 'Ryy': b[..., 1], 'Ryz': b[..., 2],
                             'Rzx': c[..., 0], 'Rzy': c[..., 1], 'Rzz': c[..., 2]})

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

class EulerState(ArrayContainer, State):
    angles : RealArray

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
        return xp.broadcast_to(xp.arccos(self.axis[..., 2] / r), self.angles.shape)

    def beta(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.broadcast_to(xp.arctan2(self.axis[..., 1], self.axis[..., 0]),
                               self.angles.shape)

    def to_point(self) -> RealArray:
        return self.angles[..., None] * self.axis

    def to_tilt(self) -> TiltState:
        xp = self.__array_namespace__()
        return TiltState(xp.stack((self.angles, self.alpha(), self.beta()), axis=-1))

S = TypeVar('S', bound='BaseSetup')

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
    def parser(cls, ext: str='ini') -> Parser:
        if ext == 'ini':
            return INIParser({'geometry': {'foc_pos': 'foc_pos', 'pupil_roi': 'pupil_roi',
                                           'z': 'z'}},
                             types=get_type_hints(cls))
        if ext == 'json':
            return JSONParser({'geometry': {'foc_pos': 'foc_pos', 'pupil_roi': 'pupil_roi',
                                            'z': 'z'}})

        raise ValueError(f"Invalid format: {ext}")

class FixedPupilSetup(BaseSetup, DataContainer, State):
    lens    : FixedPupilLens
    z       : RealArray

    @classmethod
    def read(cls, file: str, xp: ArrayNamespace=JaxNumPy) -> 'FixedPupilSetup':
        data = cls.parser(file).read(file)
        return cls(FixedPupilLens(xp.asarray(data['foc_pos']), tuple(data['pupil_roi'])),
                   xp.asarray(data['z']))

class FixedApertureSetup(BaseSetup, DataContainer, State):
    lens    : FixedApertureLens
    z       : RealArray

    @classmethod
    def read(cls, file: str, xp: ArrayNamespace=JaxNumPy) -> 'FixedApertureSetup':
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
