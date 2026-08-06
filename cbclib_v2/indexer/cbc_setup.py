from typing import Callable, Generic, Iterator, Sequence, Tuple, Type, TypeVar, overload
from typing_extensions import Self
import pandas as pd
from .._src.annotations import (AnyGenerator, AnyNamespace, BoolArray, Indices, IntArray, JaxNumPy,
                                RealArray, RealSequence, Shape)
from .._src.array_api import (array_namespace, asnumpy, euler_angles, euler_matrix, safe_divide,
                              safe_sqrt, tilt_angles, tilt_matrix)
from .._src.data_container import ArrayContainer, DataContainer, IndexedContainer
from .._src.parser import (FieldInfo, FieldLocator, JSONParser, INIParser, Parser, get_extension,
                           get_type_hints)
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

StaticAngles = Tuple[Tuple[float, float, float], ...]
StaticLengths = Tuple[Tuple[float, float, float], ...]
AnyAngles = TypeVar('AnyAngles', bound=RealArray | StaticAngles)
AnyLengths = TypeVar('AnyLengths', bound=RealArray | StaticLengths)

class BaseCell(Generic[AnyAngles, AnyLengths]):
    angles  : AnyAngles
    lengths : AnyLengths

    @classmethod
    def parser(cls, file_or_extension: str='ini') -> Parser:
        ext = get_extension(file_or_extension)
        field_info: FieldInfo = {'unit_cell': {'angles': FieldLocator('angles'),
                                               'lengths': FieldLocator('lengths')}}
        if ext == 'ini':
            return INIParser(field_info, get_type_hints(cls))
        if ext == 'json':
            return JSONParser(field_info)
        raise ValueError(f"Unsupported file or extension format: {file_or_extension}")

    def to_basis(self, xp: AnyNamespace=JaxNumPy) -> 'XtalState':
        gamma = xp.asarray(self.angles)[..., 2]
        cos = xp.cos(xp.asarray(self.angles))
        sin = xp.sin(gamma)
        v_ratio = safe_sqrt(xp.ones(cos.shape[:-1]) - xp.sum(cos**2, axis=-1)
                            + 2 * xp.prod(cos, axis=-1), xp)
        a_vec = xp.broadcast_to(xp.array([1.0, 0.0, 0.0]), cos.shape)
        b_vec = xp.stack((cos[..., 2], sin, xp.zeros(cos.shape[:-1])), axis=-1)
        c_vec = xp.stack((cos[..., 1],
                          safe_divide(cos[..., 0] - cos[..., 1] * cos[..., 2], sin, xp),
                          safe_divide(v_ratio, sin, xp)), axis=-1)
        vectors = xp.stack((a_vec, b_vec, c_vec), axis=-2)
        return XtalState(xp.asarray(xp.asarray(self.lengths)[..., None] * vectors, dtype=float))

class FixedXtalCell(BaseCell[StaticAngles, StaticLengths], State, eq=True, unsafe_hash=True):
    angles  : StaticAngles = field(static=True)
    lengths : StaticLengths = field(static=True)

    @classmethod
    def read(cls, file: str) -> 'FixedXtalCell':
        data = cls.parser(file).read(file)
        return cls(tuple(data['angles']), tuple(data['lengths']))

class XtalCell(BaseCell[RealArray, RealArray], ArrayContainer, State):
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

def xtal_parser(file_or_extension: str='ini') -> Parser:
    ext = get_extension(file_or_extension)
    field_info: FieldInfo = {'basis': {'a': FieldLocator('a'), 'b': FieldLocator('b'),
                                       'c': FieldLocator('c')}}
    if ext == 'ini':
        return INIParser(field_info, {'a': RealArray, 'b': RealArray, 'c': RealArray})
    if ext == 'json':
        return JSONParser(field_info)

    raise ValueError(f"Unsupported file or extension format: {file_or_extension}")

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
    def read(cls, file: str, xp: AnyNamespace=JaxNumPy) -> 'XtalState':
        data = xtal_parser(file).read(file)
        return cls(xp.stack((data['a'], data['b'], data['c'])))

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series,
                         xp: AnyNamespace=JaxNumPy) -> 'XtalState':
        a = xp.stack((xp.asarray(df['a_x']), xp.asarray(df['a_y']), xp.asarray(df['a_z'])), axis=-1)
        b = xp.stack((xp.asarray(df['b_x']), xp.asarray(df['b_y']), xp.asarray(df['b_z'])), axis=-1)
        c = xp.stack((xp.asarray(df['c_x']), xp.asarray(df['c_y']), xp.asarray(df['c_z'])), axis=-1)
        return cls(xp.stack((a, b, c), axis=-2))

    @classmethod
    def import_spherical(cls, r: RealArray, theta: RealArray, phi: RealArray,
                         xp: AnyNamespace=JaxNumPy) -> 'XtalState':
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
        return RotationState(xp.asarray(matrix))

    def reciprocate(self) -> 'XtalState':
        """Calculate the basis of the reciprocal lattice.

        Returns:
            The basis of the reciprocal lattice.
        """
        xp = self.__array_namespace__()
        bc, ca, ab = (xp.linalg.cross(self.b, self.c), xp.linalg.cross(self.c, self.a),
                      xp.linalg.cross(self.a, self.b))
        a_rec = bc / xp.sum(bc * self.a, axis=-1)
        b_rec = ca / xp.sum(ca * self.b, axis=-1)
        c_rec = ab / xp.sum(ab * self.c, axis=-1)
        return XtalState(xp.stack((a_rec, b_rec, c_rec), axis=-2))

    def to_dataframe(self, index: IntArray | None=None) -> pd.DataFrame:
        xp = self.__array_namespace__()
        if index is None:
            index = xp.arange(len(self))
        return pd.DataFrame({'index': asnumpy(index),
                             'a_x': asnumpy(self.a[..., 0]),
                             'a_y': asnumpy(self.a[..., 1]),
                             'a_z': asnumpy(self.a[..., 2]),
                             'b_x': asnumpy(self.b[..., 0]),
                             'b_y': asnumpy(self.b[..., 1]),
                             'b_z': asnumpy(self.b[..., 2]),
                             'c_x': asnumpy(self.c[..., 0]),
                             'c_y': asnumpy(self.c[..., 1]),
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

    def write(self, file: str):
        xtal_parser(file).write(file, self)

Float = float | RealArray
StaticFoc = Tuple[float, float, float]
StaticPupil = Tuple[float, float, float, float]
AnyFoc = TypeVar('AnyFoc', bound=StaticFoc | RealArray)
AnyPupil = TypeVar('AnyPupil', bound=StaticPupil | RealArray)

class ResolvedLens(State, ArrayContainer):
    foc_pos     : RealArray
    pupil_roi   : RealArray

    def __post_init__(self):
        self.foc_pos = self.foc_pos.reshape((-1, 3))
        self.pupil_roi = self.pupil_roi.reshape((-1, 4))

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace=JaxNumPy
                         ) -> 'ResolvedLens':
        foc_pos = xp.stack((xp.asarray(df['foc_x']),
                            xp.asarray(df['foc_y']),
                            xp.asarray(df['foc_z'])),
                           axis=-1)
        pupil_roi = xp.stack((xp.asarray(df['pupil_y0']), xp.asarray(df['pupil_y1']),
                              xp.asarray(df['pupil_x0']), xp.asarray(df['pupil_x1'])),
                             axis=-1)
        return cls(foc_pos, pupil_roi)

    @property
    def pupil_y0(self) -> RealArray:
        return self.pupil_roi[..., 0]

    @property
    def pupil_y1(self) -> RealArray:
        return self.pupil_roi[..., 1]

    @property
    def pupil_x0(self) -> RealArray:
        return self.pupil_roi[..., 2]

    @property
    def pupil_x1(self) -> RealArray:
        return self.pupil_roi[..., 3]

    @property
    def pupil_min(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.stack((self.pupil_x0, self.pupil_y0), axis=-1)

    @property
    def pupil_max(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.stack((self.pupil_x1, self.pupil_y1), axis=-1)

    @property
    def pupil_center(self) -> RealArray:
        xp = self.__array_namespace__()
        x = 0.5 * (self.pupil_x0 + self.pupil_x1)
        y = 0.5 * (self.pupil_y0 + self.pupil_y1)
        return xp.stack((x, y), axis=-1)

    def broadcast(self, size: int) -> 'ResolvedLens':
        xp = self.__array_namespace__()
        return ResolvedLens(xp.broadcast_to(self.foc_pos, (size, 3)),
                            xp.broadcast_to(self.pupil_roi, (size, 4)))

    def collapse(self) -> 'ResolvedLens':
        xp = self.__array_namespace__()
        foc_pos = xp.mean(self.foc_pos, axis=0, keepdims=True)
        pupil_roi = xp.mean(self.pupil_roi, axis=0, keepdims=True)
        return ResolvedLens(foc_pos, pupil_roi)

    def to_dataframe(self, index: IntArray) -> pd.DataFrame:
        return pd.DataFrame({'index': asnumpy(index),
                             'foc_x': asnumpy(self.foc_pos[..., 0]),
                             'foc_y': asnumpy(self.foc_pos[..., 1]),
                             'foc_z': asnumpy(self.foc_pos[..., 2]),
                             'pupil_y0': asnumpy(self.pupil_y0),
                             'pupil_y1': asnumpy(self.pupil_y1),
                             'pupil_x0': asnumpy(self.pupil_x0),
                             'pupil_x1': asnumpy(self.pupil_x1)})

def lens_parser(cls: 'Type[BaseLens]', file_or_extension: str='ini') -> Parser:
    ext = get_extension(file_or_extension)
    field_info: FieldInfo = {'geometry': {'foc_pos': FieldLocator('foc_pos'),
                                          'pupil_roi': FieldLocator('pupil_roi')}}
    if ext == 'ini':
        return INIParser(field_info, get_type_hints(cls))
    if ext == 'json':
        return JSONParser(field_info)
    raise ValueError(f"Unsupported file or extension format: {file_or_extension}")

class BaseLens(Generic[AnyFoc, AnyPupil]):
    foc_pos     : AnyFoc
    pupil_roi   : AnyPupil

    def __len__(self) -> int:
        if isinstance(self.foc_pos, tuple):
            if isinstance(self.pupil_roi, tuple):
                return 1
            return self.pupil_roi.size // 4
        return self.foc_pos.size // 3

    @classmethod
    def from_parameters(cls: Type[Self], foc_pos: Sequence[float], pupil_roi: Sequence[float],
                        xp: AnyNamespace) -> Self:
        raise NotImplementedError

    @classmethod
    def read(cls: Type[Self], file: str, xp: AnyNamespace=JaxNumPy) -> Self:
        data = lens_parser(cls, file).read(file)
        return cls.from_parameters(data['foc_pos'], data['pupil_roi'], xp)

    @classmethod
    def from_resolved(cls: Type[Self], resolved: 'ResolvedLens | ResolvedGeometry') -> Self:
        raise NotImplementedError

    def broadcast(self, size: int) -> Self:
        raise NotImplementedError

    def collapse(self) -> Self:
        raise NotImplementedError

    def resolve(self, xp: AnyNamespace) -> ResolvedLens:
        raise NotImplementedError

    def write(self, file: str) -> None:
        lens_parser(type(self), file).write(file, self.collapse())

class FixedLens(BaseLens[StaticFoc, StaticPupil], State, eq=True, unsafe_hash=True):
    foc_pos     : StaticFoc = field(static=True)
    pupil_roi   : StaticPupil = field(static=True)

    @classmethod
    def from_parameters(cls, foc_pos: Sequence[float], pupil_roi: Sequence[float],
                        xp: AnyNamespace) -> 'FixedLens':
        foc_pos = (float(foc_pos[0]), float(foc_pos[1]), float(foc_pos[2]))
        pupil_roi = (float(pupil_roi[0]), float(pupil_roi[1]),
                     float(pupil_roi[2]), float(pupil_roi[3]))
        return cls(foc_pos, pupil_roi)

    @classmethod
    def from_resolved(cls, resolved: 'ResolvedLens | ResolvedGeometry') -> 'FixedLens':
        if resolved.size != 1:
            resolved = resolved.collapse()

        foc_pos = (float(resolved.foc_pos[0, 0]),
                   float(resolved.foc_pos[0, 1]),
                   float(resolved.foc_pos[0, 2]))
        pupil_roi = (float(resolved.pupil_roi[0, 0]), float(resolved.pupil_roi[0, 1]),
                     float(resolved.pupil_roi[0, 2]), float(resolved.pupil_roi[0, 3]))
        return cls(foc_pos, pupil_roi)

    def broadcast(self, size: int) -> 'FixedLens':
        return self

    def collapse(self) -> 'FixedLens':
        return self

    def resolve(self, xp: AnyNamespace) -> ResolvedLens:
        return ResolvedLens(xp.asarray(self.foc_pos), xp.asarray(self.pupil_roi))

class FixedPupilLens(BaseLens[RealArray, StaticPupil], DataContainer, State):
    foc_pos     : RealArray
    pupil_roi   : StaticPupil = field(static=True)

    @classmethod
    def from_parameters(cls, foc_pos: Sequence[float], pupil_roi: Sequence[float],
                        xp: AnyNamespace) -> 'FixedPupilLens':
        pupil_roi = (float(pupil_roi[0]), float(pupil_roi[1]),
                     float(pupil_roi[2]), float(pupil_roi[3]))
        return cls(xp.array([foc_pos[0], foc_pos[1], foc_pos[2]]), pupil_roi)

    @classmethod
    def from_resolved(cls, resolved: 'ResolvedLens | ResolvedGeometry') -> 'FixedPupilLens':
        if resolved.size != 1:
            collapsed = resolved.collapse()
            foc_pos = resolved.foc_pos
        else:
            collapsed = resolved
            foc_pos = collapsed.foc_pos[0]

        pupil_roi = (float(collapsed.pupil_roi[0, 0]), float(collapsed.pupil_roi[0, 1]),
                     float(collapsed.pupil_roi[0, 2]), float(collapsed.pupil_roi[0, 3]))
        return cls(foc_pos, pupil_roi)

    def broadcast(self, size: int) -> 'FixedPupilLens':
        if len(self) != 1:
            raise ValueError("Cannot resize a lens with multiple focal positions or pupil ROIs.")
        xp = self.__array_namespace__()
        return FixedPupilLens(xp.broadcast_to(self.foc_pos, (size, 3)), self.pupil_roi)

    def collapse(self) -> 'FixedPupilLens':
        xp = self.__array_namespace__()
        foc_pos = xp.mean(xp.reshape(self.foc_pos, (-1, 3)), axis=0)
        return FixedPupilLens(foc_pos, self.pupil_roi)

    def resolve(self, xp: AnyNamespace) -> ResolvedLens:
        return ResolvedLens(xp.asarray(self.foc_pos), xp.asarray(self.pupil_roi))

class FixedApertureLens(BaseLens, DataContainer, State):
    foc_pos         : RealArray
    pupil_center    : RealArray
    aperture        : Tuple[float, float] = field(static=True)

    @property
    def pupil_roi(self) -> RealArray:
        xp = self.__array_namespace__()
        x, y = self.pupil_center[..., 0], self.pupil_center[..., 1]
        ap_x, ap_y = self.aperture
        return xp.stack((y - 0.5 * ap_y, y + 0.5 * ap_y,
                         x - 0.5 * ap_x, x + 0.5 * ap_x), axis=-1)

    @classmethod
    def from_parameters(cls, foc_pos: Sequence[float], pupil_roi: Sequence[float],
                        xp: AnyNamespace) -> 'FixedApertureLens':
        return cls.from_fixed(FixedLens.from_parameters(foc_pos, pupil_roi, xp), xp)

    @classmethod
    def from_fixed(cls, fixed: FixedLens, xp: AnyNamespace) -> 'FixedApertureLens':
        aperture = (fixed.pupil_roi[3] - fixed.pupil_roi[2],
                    fixed.pupil_roi[1] - fixed.pupil_roi[0])
        pupil_center = (0.5 * (fixed.pupil_roi[2] + fixed.pupil_roi[3]),
                        0.5 * (fixed.pupil_roi[0] + fixed.pupil_roi[1]))
        return cls(xp.asarray(fixed.foc_pos), xp.asarray(pupil_center), aperture)

    @classmethod
    def from_resolved(cls, resolved: 'ResolvedLens | ResolvedGeometry') -> 'FixedApertureLens':
        if resolved.size != 1:
            collapsed = resolved.collapse()
            foc_pos = resolved.foc_pos
            center = resolved.pupil_center
        else:
            collapsed = resolved
            foc_pos = collapsed.foc_pos[0]
            center = collapsed.pupil_center[0]

        aperture = (float(collapsed.pupil_roi[0, 3] - collapsed.pupil_roi[0, 2]),
                    float(collapsed.pupil_roi[0, 1] - collapsed.pupil_roi[0, 0]))
        return cls(foc_pos, center, aperture)

    def broadcast(self, size: int) -> 'FixedApertureLens':
        if len(self) != 1:
            raise ValueError("Cannot resize a lens with multiple focal positions or pupil ROIs.")
        xp = self.__array_namespace__()
        return FixedApertureLens(xp.broadcast_to(self.foc_pos, (size, 3)),
                                 xp.broadcast_to(self.pupil_center, (size, 2)), self.aperture)

    def collapse(self) -> 'FixedApertureLens':
        xp = self.__array_namespace__()
        foc_pos = xp.mean(xp.reshape(self.foc_pos, (-1, 3)), axis=0)
        pupil_center = xp.mean(xp.reshape(self.pupil_center, (-1, 2)), axis=0)
        return FixedApertureLens(foc_pos, pupil_center, self.aperture)

    def resolve(self, xp: AnyNamespace) -> ResolvedLens:
        return ResolvedLens(xp.asarray(self.foc_pos), xp.asarray(self.pupil_roi))

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
                             'Rxx': asnumpy(a[..., 0]), 'Rxy': asnumpy(a[..., 1]),
                             'Rxz': asnumpy(a[..., 2]),
                             'Ryx': asnumpy(b[..., 0]), 'Ryy': asnumpy(b[..., 1]),
                             'Ryz': asnumpy(b[..., 2]),
                             'Rzx': asnumpy(c[..., 0]), 'Rzy': asnumpy(c[..., 1]),
                             'Rzz': asnumpy(c[..., 2])})

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
        transposed = xp.permute_dims(self.matrix, (*range(self.matrix.ndim - 2), -1, -2))
        if xp.allclose(self.matrix, transposed):
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
        r = safe_sqrt(xp.sum(self.axis**2, axis=-1), xp)
        return xp.broadcast_to(xp.acos(safe_divide(self.axis[..., 2], r, xp)),
                               self.angles.shape)

    def beta(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.broadcast_to(xp.atan2(self.axis[..., 1], self.axis[..., 0]),
                               self.angles.shape)

    def to_point(self) -> RealArray:
        return self.angles[..., None] * self.axis

    def to_tilt(self) -> TiltState:
        xp = self.__array_namespace__()
        return TiltState(xp.stack((self.angles, self.alpha(), self.beta()), axis=-1))

class ResolvedGeometry(ResolvedLens):
    defocus : RealArray

    def __post_init__(self):
        super().__post_init__()
        self.defocus = self.defocus.reshape((-1,))

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace=JaxNumPy
                         ) -> 'ResolvedGeometry':
        lens = ResolvedLens.import_dataframe(df, xp)
        return cls(lens.foc_pos, lens.pupil_roi, xp.asarray(df['defocus']))

    def broadcast(self, size: int) -> 'ResolvedGeometry':
        xp = self.__array_namespace__()
        lens = super().broadcast(size)
        return ResolvedGeometry(lens.foc_pos, lens.pupil_roi,
                                xp.broadcast_to(self.defocus, (size,)))

    def collapse(self) -> 'ResolvedGeometry':
        xp = self.__array_namespace__()
        lens = super().collapse()
        defocus = xp.mean(self.defocus, axis=0, keepdims=True)
        return ResolvedGeometry(lens.foc_pos, lens.pupil_roi, defocus)

    def to_dataframe(self, index: IntArray) -> pd.DataFrame:
        df = super().to_dataframe(index=index)
        df['defocus'] = asnumpy(self.defocus)
        return df

def geometry_parser(cls: 'Type[BaseGeometry]', file: str) -> Parser:
    ext = get_extension(file)
    field_info: FieldInfo = {'geometry': {'foc_pos': FieldLocator('foc_pos'),
                                          'pupil_roi': FieldLocator('pupil_roi'),
                                          'defocus': FieldLocator('defocus')}}
    if ext == 'ini':
        return INIParser(field_info, get_type_hints(cls))
    if ext == 'json':
        return JSONParser(field_info)

    raise ValueError(f"Invalid format: {ext}")

AnyDF = TypeVar('AnyDF', bound=Tuple[float, ...] | RealArray)

class BaseGeometry(BaseLens, Generic[AnyFoc, AnyPupil, AnyDF]):
    defocus : AnyDF

    def __len__(self) -> int:
        if super().__len__() == 1:
            if isinstance(self.defocus, tuple):
                return 1
            return self.defocus.size
        return super().__len__()

    @classmethod
    def from_parameters(cls: Type[Self], foc_pos: Sequence[float], pupil_roi: Sequence[float],
                        defocus: Sequence[float], xp: AnyNamespace=JaxNumPy) -> Self:
        """Construct an experimental setup from effective geometry values.

        Args:
            foc_pos: Focal position ``(x, y, z)``.
            pupil_roi: Pupil bounds ``(y_min, y_max, x_min, x_max)``.
            defocus: Defocus distance along z-axis.
            xp: Array namespace for dynamic setup fields.

        Returns:
            Experimental setup instance.
        """
        raise NotImplementedError

    @classmethod
    def from_resolved(cls: Type[Self], resolved: ResolvedLens | ResolvedGeometry) -> Self:
        raise NotImplementedError

    @classmethod
    def read(cls: Type[Self], file: str, xp: AnyNamespace=JaxNumPy) -> Self:
        """Read setup geometry from a JSON or INI file."""
        data = geometry_parser(cls, file).read(file)
        return cls.from_parameters(data['foc_pos'], data['pupil_roi'],
                                   data['defocus'], xp)

    def broadcast(self: Self, size: int) -> Self:
        """Return a setup whose dynamic arrays have one row per pattern.

        Subclasses with dynamic geometry override this method. Static setups
        remain shared.
        """
        raise NotImplementedError

    def collapse(self: Self) -> Self:
        """Return a setup whose dynamic arrays are collapsed to a single row.

        Subclasses with dynamic geometry override this method. Static setups
        remain shared.
        """
        raise NotImplementedError

    def resolve(self, xp: AnyNamespace) -> ResolvedGeometry:
        """Return the resolved setup geometry.

        Args:
            xp: Array namespace for dynamic setup fields.

        Returns:
            A :class:`ResolvedGeometry` object.
        """
        resolved = super().resolve(xp)
        return ResolvedGeometry(resolved.foc_pos, resolved.pupil_roi,
                                xp.asarray(self.defocus))

    def write(self, file: str) -> None:
        """Write setup geometry to a JSON or INI file."""
        geometry_parser(type(self), file).write(file, self.collapse())

class FixedGeometry(BaseGeometry[StaticFoc, StaticPupil, Tuple[float, ...]], FixedLens):
    defocus : Tuple[float, ...] = field(static=True)

    @classmethod
    def from_parameters(cls, foc_pos: Sequence[float], pupil_roi: Sequence[float],
                        defocus: Sequence[float], xp: AnyNamespace=JaxNumPy) -> 'FixedGeometry':
        lens = FixedLens.from_parameters(foc_pos, pupil_roi, xp)
        return cls(lens.foc_pos, lens.pupil_roi, tuple(float(val) for val in defocus))

    @classmethod
    def from_resolved(cls, resolved: ResolvedLens | ResolvedGeometry) -> 'FixedGeometry':
        if resolved.size != 1:
            resolved = resolved.collapse()

        lens = FixedLens.from_resolved(resolved)
        if isinstance(resolved, ResolvedGeometry):
            defocus = tuple(float(val) for val in resolved.defocus)
        else:
            defocus = tuple(0.0 for _ in range(resolved.size))
        return cls(lens.foc_pos, lens.pupil_roi, defocus)

    def broadcast(self, size: int) -> 'FixedGeometry':
        return self

    def collapse(self) -> 'FixedGeometry':
        return self

class FixedPupilGeometry(BaseGeometry[RealArray, StaticPupil, RealArray], FixedPupilLens):
    defocus : RealArray

    @classmethod
    def from_parameters(cls, foc_pos: Sequence[float], pupil_roi: Sequence[float],
                        defocus: Sequence[float], xp: AnyNamespace=JaxNumPy
                        ) -> 'FixedPupilGeometry':
        lens = FixedPupilLens.from_parameters(foc_pos, pupil_roi, xp)
        return cls(lens.foc_pos, lens.pupil_roi, xp.asarray(defocus))

    @classmethod
    def from_resolved(cls, resolved: ResolvedLens | ResolvedGeometry) -> 'FixedPupilGeometry':
        lens = FixedPupilLens.from_resolved(resolved)
        if isinstance(resolved, ResolvedGeometry):
            defocus = resolved.defocus
        else:
            xp = resolved.__array_namespace__()
            defocus = xp.zeros(resolved.size, dtype=resolved.foc_pos.dtype)
        return cls(lens.foc_pos, lens.pupil_roi, defocus)

    def broadcast(self, size: int) -> 'FixedPupilGeometry':
        xp = self.__array_namespace__()

        lens = FixedPupilLens.broadcast(self, size)
        defocus = xp.broadcast_to(self.defocus, (size,))
        return self.replace(foc_pos=lens.foc_pos, pupil_roi=lens.pupil_roi, defocus=defocus)

    def collapse(self) -> 'FixedPupilGeometry':
        xp = self.__array_namespace__()

        lens = FixedPupilLens.collapse(self)
        defocus = xp.mean(xp.reshape(self.defocus, (-1,)), axis=0, keepdims=True)
        return self.replace(foc_pos=lens.foc_pos, pupil_roi=lens.pupil_roi, defocus=defocus)

class FixedApertureGeometry(BaseGeometry[RealArray, RealArray, RealArray], FixedApertureLens):
    defocus : RealArray

    @classmethod
    def from_parameters(cls, foc_pos: Sequence[float], pupil_roi: Sequence[float],
                        defocus: Sequence[float], xp: AnyNamespace=JaxNumPy
                        ) -> 'FixedApertureGeometry':
        lens = FixedApertureLens.from_parameters(foc_pos, pupil_roi, xp)
        return cls(lens.foc_pos, lens.pupil_center, lens.aperture, xp.asarray(defocus))

    @classmethod
    def from_resolved(cls, resolved: ResolvedLens | ResolvedGeometry) -> 'FixedApertureGeometry':
        lens = FixedApertureLens.from_resolved(resolved)
        if isinstance(resolved, ResolvedGeometry):
            defocus = resolved.defocus
        else:
            xp = resolved.__array_namespace__()
            defocus = xp.zeros(resolved.size, dtype=resolved.foc_pos.dtype)
        return cls(lens.foc_pos, lens.pupil_center, lens.aperture, defocus)

    def broadcast(self, size: int) -> 'FixedApertureGeometry':
        xp = self.__array_namespace__()

        lens = FixedApertureLens.broadcast(self, size)
        defocus = xp.broadcast_to(self.defocus, (size,))
        return self.replace(foc_pos=lens.foc_pos, pupil_center=lens.pupil_center, defocus=defocus)

    def collapse(self) -> 'FixedApertureGeometry':
        xp = self.__array_namespace__()

        lens = FixedApertureLens.collapse(self)
        defocus = xp.mean(xp.reshape(self.defocus, (-1,)), axis=0, keepdims=True)
        return self.replace(foc_pos=lens.foc_pos, pupil_center=lens.pupil_center, defocus=defocus)

class ResolvedSetup(State, DataContainer):
    xtal    : XtalState
    geometry: ResolvedLens | ResolvedGeometry

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace=JaxNumPy
                         ) -> 'ResolvedSetup':
        if 'defocus' in df:
            geometry = ResolvedGeometry.import_dataframe(df, xp)
        else:
            geometry = ResolvedLens.import_dataframe(df, xp)
        return cls(XtalState.import_dataframe(df, xp), geometry)

    def __getitem__(self, indices: Indices | BoolArray) -> 'ResolvedSetup':
        if len(self.xtal) == self.geometry.size:
            return ResolvedSetup(self.xtal[indices], self.geometry[indices])
        return ResolvedSetup(self.xtal[indices], self.geometry)

    def to_dataframe(self, index: IntArray | None=None) -> pd.DataFrame:
        if index is None:
            xp = self.__array_namespace__()
            index = xp.arange(len(self.xtal))

        df = self.xtal.to_dataframe(index=index)
        if self.geometry.size != len(self.xtal):
            geometry = self.geometry.broadcast(len(self.xtal))
        else:
            geometry = self.geometry

        return df.assign(**geometry.to_dataframe(index=index))

AnyXtal = TypeVar('AnyXtal', bound=XtalState)
AnyGeometry = TypeVar('AnyGeometry', bound=BaseLens | BaseGeometry)

class BaseSetup(DataContainer, Generic[AnyXtal, AnyGeometry]):
    xtal        : AnyXtal
    geometry    : AnyGeometry

    def resolve(self, xp: AnyNamespace) -> ResolvedSetup:
        return ResolvedSetup(self.xtal, self.geometry.resolve(xp))

    @classmethod
    def from_resolved(cls: Type[Self], resolved: ResolvedSetup) -> Self:
        raise NotImplementedError

# Some predefined experimental setups

class FixedSetup(BaseSetup[XtalState, FixedLens | FixedGeometry], State):
    xtal     : XtalState
    geometry : FixedLens | FixedGeometry

    @classmethod
    def from_resolved(cls, resolved: ResolvedSetup) -> 'FixedSetup':
        if isinstance(resolved.geometry, ResolvedGeometry):
            geometry = FixedGeometry.from_resolved(resolved.geometry)
        else:
            geometry = FixedLens.from_resolved(resolved.geometry)
        return cls(resolved.xtal, geometry)

class SerialFixedSetup(BaseSetup, State):
    cell     : XtalCell
    rotation : RotationState
    geometry : FixedLens | FixedGeometry

    @property
    def xtal(self) -> XtalState:
        return self.cell.to_basis() @ self.rotation

    @classmethod
    def from_resolved(cls, resolved: ResolvedSetup) -> 'SerialFixedSetup':
        if isinstance(resolved.geometry, ResolvedGeometry):
            geometry = FixedGeometry.from_resolved(resolved.geometry)
        else:
            geometry = FixedLens.from_resolved(resolved.geometry)
        return cls(resolved.xtal.unit_cell, resolved.xtal.orientation_matrix,
                   geometry)

class FixedPupilSetup(BaseSetup[XtalState, FixedPupilLens | FixedPupilGeometry], State):
    xtal     : XtalState
    geometry : FixedPupilLens | FixedPupilGeometry

    @classmethod
    def from_resolved(cls, resolved: ResolvedSetup) -> 'FixedPupilSetup':
        if isinstance(resolved.geometry, ResolvedGeometry):
            geometry = FixedPupilGeometry.from_resolved(resolved.geometry)
        else:
            geometry = FixedPupilLens.from_resolved(resolved.geometry)
        return cls(resolved.xtal, geometry)

class SerialFixedPupilSetup(BaseSetup, State):
    cell     : XtalCell
    rotation : RotationState
    geometry : FixedPupilLens | FixedPupilGeometry

    @property
    def xtal(self) -> XtalState:
        return self.cell.to_basis() @ self.rotation

    @classmethod
    def from_resolved(cls, resolved: ResolvedSetup) -> 'SerialFixedPupilSetup':
        if isinstance(resolved.geometry, ResolvedGeometry):
            geometry = FixedPupilGeometry.from_resolved(resolved.geometry)
        else:
            geometry = FixedPupilLens.from_resolved(resolved.geometry)
        return cls(resolved.xtal.unit_cell, resolved.xtal.orientation_matrix, geometry)

class FixedApertureSetup(BaseSetup[XtalState, FixedApertureLens | FixedApertureGeometry], State):
    xtal     : XtalState
    geometry : FixedApertureLens | FixedApertureGeometry

    @classmethod
    def from_resolved(cls, resolved: ResolvedSetup) -> 'FixedApertureSetup':
        if isinstance(resolved.geometry, ResolvedGeometry):
            geometry = FixedApertureGeometry.from_resolved(resolved.geometry)
        else:
            geometry = FixedApertureLens.from_resolved(resolved.geometry)
        return cls(resolved.xtal, geometry)

class SerialFixedApertureSetup(BaseSetup, State):
    cell     : XtalCell
    rotation : RotationState
    geometry : FixedApertureLens | FixedApertureGeometry

    @property
    def xtal(self) -> XtalState:
        return self.cell.to_basis() @ self.rotation

    @classmethod
    def from_resolved(cls, resolved: ResolvedSetup) -> 'SerialFixedApertureSetup':
        if isinstance(resolved.geometry, ResolvedGeometry):
            geometry = FixedApertureGeometry.from_resolved(resolved.geometry)
        else:
            geometry = FixedApertureLens.from_resolved(resolved.geometry)
        return cls(resolved.xtal.unit_cell, resolved.xtal.orientation_matrix, geometry)

class IndexingResult(State, IndexedContainer):
    index : IntArray
    xtal  : XtalState
