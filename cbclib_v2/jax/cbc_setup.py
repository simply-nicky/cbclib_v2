from functools import partial
from typing import (Any, Callable, ClassVar, Dict, Optional, Tuple, Type, TypeVar, Union, cast,
                    get_type_hints, overload)
from dataclasses import asdict, is_dataclass, fields
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import jit, random, tree_util
from .dataclasses import jax_dataclass, field
from .geometry import (det_to_k, euler_angles, euler_matrix, k_to_det, k_to_smp, source_lines,
                       tilt_angles, tilt_matrix)
from .primitives import line_distances
from ..src import draw_line_image, draw_line_mask, draw_line_table
from ..data_container import DataclassInstance, DataContainer, Parser, INIParser, JSONParser
from ..annotations import (BoolArray, Indices, IntArray, KeyArray, RealArray, Pattern,
                           PatternWithHKL, PatternWithHKLID, Shape)

State = DataclassInstance

@jax_dataclass
class Streaks(DataContainer):
    """Detector streak lines container. Provides an interface to draw a pattern for a set of
    lines.

    Args:
        x0 : x coordinates of the first point of a line.
        y0 : y coordinates of the first point of a line.
        x1 : x coordinates of the second point of a line.
        y1 : y coordinates of the second point of a line.
        length: Line's length in pixels.
        h : First Miller index.
        k : Second Miller index.
        l : Third Miller index.
        hkl_id : Bragg reflection index.
    """
    x0          : RealArray
    y0          : RealArray
    x1          : RealArray
    y1          : RealArray
    idxs        : IntArray = field(default_factory=lambda: jnp.array([], dtype=int))
    length      : RealArray = field(default_factory=lambda: jnp.array([]))
    mask        : BoolArray = field(default_factory=lambda: jnp.array([], dtype=bool))
    h           : Optional[IntArray] = field(default=None)
    k           : Optional[IntArray] = field(default=None)
    l           : Optional[IntArray] = field(default=None)
    hkl_id      : Optional[IntArray] = field(default=None)

    def __post_init__(self):
        if self.idxs.shape != self.x0.shape:
            self.idxs = jnp.zeros(self.x0.shape, dtype=int)
        if self.length.shape != self.x0.shape:
            self.length = jnp.sqrt((self.x1 - self.x0)**2 + (self.y1 - self.y0)**2)
        if self.mask.shape != self.x0.shape:
            self.mask = jnp.ones(self.x0.shape, dtype=bool)

    @property
    def hkl(self) -> Optional[IntArray]:
        if self.h is None or self.k is None or self.l is None:
            return None
        return jnp.stack((self.h, self.k, self.l), axis=1)

    def __len__(self) -> int:
        return self.length.shape[0]

    def mask_streaks(self, idxs: Union[Indices, BoolArray]) -> 'Streaks':
        """Return a new streaks container with a set of streaks discarded.

        Args:
            idxs : A set of indices of the streaks to discard.

        Returns:
            A new :class:`cbclib.Streaks` container.
        """
        return Streaks(**{attr: self[attr][idxs] for attr in self.contents()})

    def offset(self, x: float, y: float) -> 'Streaks':
        return self.replace(x0=self.x0 - x, x1=self.x1 - x, y0=self.y0 - y, y1=self.y1 - y)

    def pattern_dict(self, width: float, shape: Shape, kernel: str='rectangular',
                     num_threads: int=1) -> Union[Pattern, PatternWithHKL, PatternWithHKLID]:
        """Draw a pattern in the :class:`dict` format.

        Args:
            width : Lines width in pixels.
            shape : Detector grid shape.
            kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
                are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

        Returns:
            A pattern in dictionary format.
        """
        table = draw_line_table(lines=self.to_lines(width), shape=shape, idxs=self.idxs,
                                kernel=kernel, num_threads=num_threads)
        ids, idxs = np.array(list(table)).T
        normalised_shape = (np.prod(shape[:-2], dtype=int),) + shape[-2:]
        frames, y, x = jnp.unravel_index(idxs, normalised_shape)

        if self.hkl is not None:
            vals = np.array(list(table.values()))
            h, k, l = self.hkl[ids].T

            if self.hkl_id is not None:
                return PatternWithHKLID(ids, frames, y, x, vals, h, k, l, self.hkl_id[ids])
            return PatternWithHKL(ids, frames, y, x, vals, h, k, l)
        return Pattern(ids, frames, y, x)

    def pattern_dataframe(self, width: float, shape: Shape, kernel: str='rectangular',
                          num_threads: int=1) -> pd.DataFrame:
        """Draw a pattern in the :class:`pandas.DataFrame` format.

        Args:
            width : Lines width in pixels.
            shape : Detector grid shape.
            kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
                are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

            reduce : Discard the pixel data with reflection profile values equal to
                zero.

        Returns:
            A pattern in :class:`pandas.DataFrame` format.
        """
        return pd.DataFrame(self.pattern_dict(width=width, shape=shape, kernel=kernel,
                                              num_threads=num_threads)._asdict())

    def pattern_image(self, width: float, shape: Tuple[int, int], kernel: str='gaussian',
                      num_threads: int=1) -> RealArray:
        """Draw a pattern in the :class:`numpy.ndarray` format.

        Args:
            width : Lines width in pixels.
            shape : Detector grid shape.
            kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
                are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

        Returns:
            A pattern in :class:`numpy.ndarray` format.
        """
        return draw_line_image(self.to_lines(width), shape=shape, idxs=self.idxs, kernel=kernel,
                               num_threads=num_threads)

    def pattern_mask(self, width: float, shape: Tuple[int, int], max_val: int=1,
                     kernel: str='rectangular', num_threads: int=1) -> IntArray:
        """Draw a pattern mask.

        Args:
            width : Lines width in pixels.
            shape : Detector grid shape.
            max_val : Mask maximal value.
            kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
                are available:

                * 'biweigth' : Quartic (biweight) kernel.
                * 'gaussian' : Gaussian kernel.
                * 'parabolic' : Epanechnikov (parabolic) kernel.
                * 'rectangular' : Uniform (rectangular) kernel.
                * 'triangular' : Triangular kernel.

        Returns:
            A pattern mask.
        """
        return draw_line_mask(self.to_lines(width), shape=shape, idxs=self.idxs, max_val=max_val,
                              kernel=kernel, num_threads=num_threads)

    def to_dataframe(self) -> pd.DataFrame:
        """Export a streaks container into :class:`pandas.DataFrame`.

        Returns:
            A dataframe with all the data specified in :class:`cbclib.Streaks`.
        """
        return pd.DataFrame({attr: self[attr] for attr in self.contents()})

    def to_lines(self, width: Union[float, RealArray, None]=None) -> RealArray:
        """Export a streaks container into line parameters ``x0, y0, x1, y1, width``:

        * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
        * `width` : Line's width.

        Returns:
            An array of line parameters.
        """
        if width is None:
            lines = jnp.stack((self.x0, self.y0, self.x1, self.y1), axis=1)
        else:
            widths = jnp.broadcast_to(width, self.x0.shape)
            lines = jnp.stack((self.x0, self.y0, self.x1, self.y1, widths), axis=1)
        return jnp.where(self.mask[..., None], lines, 0)

    def trim(self) -> 'Streaks':
        return self.mask_streaks(self.mask)

@jax_dataclass
class MillerIndices():
    hkl     : IntArray
    hidxs   : IntArray
    bidxs   : IntArray

    @property
    def h(self) -> IntArray:
        return self.hkl[self.hidxs][..., 0]

    @property
    def k(self) -> IntArray:
        return self.hkl[self.hidxs][..., 1]

    @property
    def l(self) -> IntArray:
        return self.hkl[self.hidxs][..., 2]

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

class Xtal():
    def init_hkl(self, q_abs: Union[float, RealArray], state: XtalState) -> IntArray:
        constants = state.lattice_constants()
        lat_size = jnp.asarray(jnp.rint(q_abs / constants.lengths), dtype=int)
        lat_size = jnp.max(jnp.reshape(lat_size, (-1, 3)), axis=0)
        h_idxs = jnp.arange(-lat_size[0], lat_size[0] + 1)
        k_idxs = jnp.arange(-lat_size[1], lat_size[1] + 1)
        l_idxs = jnp.arange(-lat_size[2], lat_size[2] + 1)
        h_grid, k_grid, l_grid = jnp.meshgrid(h_idxs, k_idxs, l_idxs)
        hkl = jnp.stack((jnp.ravel(h_grid), jnp.ravel(k_grid), jnp.ravel(l_grid)), axis=1)
        hkl = jnp.compress(jnp.any(hkl, axis=1), hkl, axis=0)

        rec_vec = jnp.dot(hkl, state.basis)
        rec_abs = jnp.sqrt(jnp.sum(rec_vec**2, axis=-1))
        rec_abs = jnp.reshape(rec_abs, (hkl.shape[0], -1))
        return hkl[jnp.any(rec_abs < 0.3, axis=-1)]

    def init_miller(self, theta: Union[float, RealArray], hkl: IntArray, state: XtalState
                    ) -> MillerIndices:
        shape = (state.num, hkl.size // 3)
        hidxs = jnp.broadcast_to(jnp.arange(shape[1]), shape)
        bidxs = jnp.broadcast_to(jnp.arange(shape[0])[:, None], shape)
        miller = MillerIndices(hkl, hidxs, bidxs)

        rec_vec = self.rec_vectors(miller, state)
        rec_abs = jnp.sqrt((rec_vec**2).sum(axis=-1))
        rec_th = jnp.arccos(-rec_vec[..., 2] / rec_abs)
        src_th = rec_th - jnp.arccos(0.5 * rec_abs)
        idxs = jnp.where((jnp.abs(src_th) < theta))
        return MillerIndices(miller.hkl, miller.hidxs[idxs], miller.bidxs[idxs])

    def rec_vectors(self, miller: MillerIndices, state: XtalState) -> RealArray:
        return jnp.sum(state.basis[miller.bidxs] * miller.hkl[miller.hidxs][..., None], axis=-2)

    def init_streaks(self, x: RealArray, y: RealArray, mask: BoolArray, hkl_index: bool,
                     miller: MillerIndices) -> Streaks:
        result = {'idxs': miller.bidxs, 'x0': x[:, 0], 'y0': y[:, 0], 'x1': x[:, 1], 'y1': y[:, 1],
                  'h': miller.h, 'k': miller.k, 'l': miller.l, 'mask': mask}
        if hkl_index:
            result['hkl_id'] = miller.hidxs
        return Streaks(**result)

@jax_dataclass
class LensState():
    foc_pos         : RealArray
    pupil_roi       : RealArray

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

class Lens():
    def kin_center(self, state: LensState) -> RealArray:
        x, y = jnp.mean(state.pupil_roi[2:]), jnp.mean(state.pupil_roi[:2])
        return det_to_k(x, y, state.foc_pos, jnp.zeros(x.shape, dtype=int))

    def kin_max(self, state: LensState) -> RealArray:
        x, y = state.pupil_roi[3], state.pupil_roi[1]
        return det_to_k(x, y, state.foc_pos, jnp.zeros(x.shape, dtype=int))

    def kin_min(self, state: LensState) -> RealArray:
        x, y = state.pupil_roi[2], state.pupil_roi[0]
        return det_to_k(x, y, state.foc_pos, jnp.zeros(x.shape, dtype=int))

    def kin_to_sample(self, kin: RealArray, z: RealArray, idxs: IntArray, state: LensState
                      ) -> RealArray:
        """Project incident wave-vectors to the sample planes.

        Args:
            setup : Experimental setup.
            kin : An array of incident wave-vectors.
            idxs : Sample indices.

        Returns:
            Array of sample coordinates.
        """
        return k_to_smp(kin, z, state.foc_pos, idxs)

Params = Union[RealArray, float]
S = TypeVar('S', bound=State)

def init_from_bounds(state: S, bounds: Dict[str, Any], default: Callable[[Params], Params]
                     ) -> Callable[[KeyArray,], S]:
    def flatten_state(state: S) -> Dict:
        return asdict(state)

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

    def generate_bounds(rng: KeyArray, params: Union[Dict, Params],
                        bounds: Union[Dict, Params, None], default: Callable[[Params], Params]
                        ) -> Union[Dict, Params]:
        if isinstance(params, dict):
            result = {}
            for key, val in params.items():
                if isinstance(bounds, dict) and key in bounds:
                    result[key] = generate_bounds(rng, val, bounds[key], default)
                else:
                    result[key] = generate_bounds(rng, val, None, default)
            return result
        else:
            if bounds is None or isinstance(bounds, dict):
                bound = default(params)
            else:
                bound = bounds
            return bound * random.uniform(random.split(rng, num=1), jnp.shape(params),
                                          minval=-1, maxval=1)

    def init(rng) -> S:
        params = flatten_state(state)
        bound_params = generate_bounds(rng, params, bounds, default)
        return unflatten_state(state, tree_util.tree_map(jnp.add, params, bound_params))

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

class CBDModel():
    lens    : Lens = Lens()
    xtal    : Xtal = Xtal()

    def init(self, rng: KeyArray) -> State:
        raise NotImplementedError

    def to_internal(self, state: State) -> InternalState:
        raise NotImplementedError

    def kin_to_sample(self, kin: RealArray, idxs: IntArray, state: InternalState
                      ) -> RealArray:
        """Project incident wave-vectors to the sample planes.

        Args:
            setup : Experimental setup.
            kin : An array of incident wave-vectors.
            idxs : Sample indices.

        Returns:
            Array of sample coordinates.
        """
        return self.lens.kin_to_sample(kin, state.z, idxs, state.lens)

    def kout_to_detector(self, kout: RealArray, idxs: IntArray, rec_vec: RealArray,
                         state: InternalState) -> RealArray:
        smp_pos = self.kin_to_sample(kout - rec_vec, idxs, state)
        return k_to_det(kout, smp_pos, idxs)

    def init_miller(self, q_abs: float, state: InternalState) -> MillerIndices:
        hkl = self.xtal.init_hkl(q_abs, state.xtal)
        kz = jnp.asarray([self.lens.kin_min(state.lens)[..., 2],
                          self.lens.kin_max(state.lens)[..., 2]])
        return self.xtal.init_miller(jnp.arccos(jnp.min(kz)), hkl, state.xtal)

    def init_patterns(self, miller: MillerIndices, roi: Tuple[int, int, int, int], width: float,
                      pixel_size: Tuple[float, float], state: InternalState) -> RealArray:
        streaks = self.init_streaks(miller, False, pixel_size, state).offset(roi[2], roi[0])
        return line_distances(jnp.zeros((state.xtal.num, roi[1] - roi[0], roi[3] - roi[2])),
                              streaks.to_lines(width), streaks.idxs)

    def init_streaks(self, miller: MillerIndices, hkl_index: bool, pixel_size: Tuple[float, float],
                     state: InternalState) -> Streaks:
        rec_vec = self.xtal.rec_vectors(miller, state.xtal)
        kin = source_lines(rec_vec, kmin=self.lens.kin_min(state.lens),
                           kmax=self.lens.kin_max(state.lens))
        is_good = jnp.sum(kin, axis=(-2, -1)) > 0

        rec_vec = jnp.where(is_good[..., None], rec_vec, 0.0)
        pos = self.kout_to_detector(kin + rec_vec[..., None, :], miller.bidxs[..., None],
                                    rec_vec[..., None, :], state)
        pos = pos / jnp.array(pixel_size)
        return self.xtal.init_streaks(pos[..., 0], pos[..., 1], is_good, hkl_index, miller)

    def sample_position(self, state: InternalState) -> RealArray:
        return self.kin_to_sample(self.lens.kin_center(state.lens), jnp.zeros(1), state)
