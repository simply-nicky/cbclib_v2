import copy
from functools import partial
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union,
                    cast, get_type_hints, overload)
from dataclasses import dataclass, is_dataclass, fields
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import jit, random, tree_util
from .dataclasses import jax_dataclass, field
from .geometry import (det_to_k, euler_angles, euler_matrix, k_to_det, k_to_smp, kxy_to_k,
                       project_to_rect, tilt_angles, tilt_matrix, safe_divide)
from .primitives import build_and_knn_query
from ..src import draw_line_image, draw_line_mask, draw_line_table
from ..data_container import DataclassInstance, DataContainer, Parser, INIParser, JSONParser
from ..annotations import (BoolArray, Indices, IntArray, IntSequence, KeyArray, RealArray,
                           RealSequence, Pattern, PatternWithHKL, PatternWithHKLID, Shape)

State = DataclassInstance
KNNQuery = Callable[[RealArray, IntArray, RealArray, IntArray], IntArray]

@dataclass
class Detector():
    x_pixel_size : float
    y_pixel_size : float
    roi          : Tuple[float, float, float, float]

    def knn_query(self, k: int, num_threads: int=1) -> KNNQuery:
        return build_and_knn_query(self.roi, k, num_threads)

    def to_indices(self, x: RealArray, y: RealArray) -> Tuple[RealArray, RealArray]:
        return x / self.x_pixel_size, y / self.y_pixel_size

    def to_coordinates(self, i: RealArray, j: RealArray) -> Tuple[RealArray, RealArray]:
        return i * self.x_pixel_size, j * self.y_pixel_size

@dataclass
class Patterns(DataContainer):
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
    lines       : RealArray
    frames      : IntArray
    hkl         : Optional[IntArray] = None

    def __post_init__(self):
        if self.lines.shape[-1] != 4:
            raise ValueError(f"lines has an invalid shape: {self.lines.shape}")
        if self.frames.shape != self.shape:
            raise ValueError("lines and frames have incompatible shapes:"\
                             f" {self.lines.shape} and {self.frames.shape}")

    @property
    def length(self) -> RealArray:
        return jnp.sqrt((self.lines[..., 2] - self.lines[..., 0])**2 + \
                        (self.lines[..., 3] - self.lines[..., 1])**2)

    @property
    def shape(self) -> Shape:
        return self.lines.shape[:-1]

    @property
    def points(self) -> RealArray:
        return jnp.reshape(self.lines, self.shape + (2, 2))

    @property
    def x(self) -> RealArray:
        return jnp.stack((self.lines[..., 0], self.lines[..., 2]), axis=-1)

    @property
    def y(self) -> RealArray:
        return jnp.stack((self.lines[..., 1], self.lines[..., 3]), axis=-1)

    def mask_streaks(self, idxs: Union[Indices, BoolArray]) -> 'Patterns':
        """Return a new streaks container with a set of streaks discarded.

        Args:
            idxs : A set of indices of the streaks to discard.

        Returns:
            A new :class:`cbclib.Streaks` container.
        """
        return Patterns(**{attr: self[attr][idxs] for attr in self.contents()})

    def offset(self, x: float, y: float) -> 'Patterns':
        return self.replace(lines=self.lines + jnp.array([x, y, x, y]))

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
        table = draw_line_table(lines=self.to_lines(width=width), shape=shape, idxs=self.frames,
                                kernel=kernel, num_threads=num_threads)
        ids, idxs = np.array(list(table)).T
        normalised_shape = (np.prod(shape[:-2], dtype=int),) + shape[-2:]
        frames, y, x = jnp.unravel_index(idxs, normalised_shape)

        if self.hkl is not None:
            vals = np.array(list(table.values()))
            h, k, l = self.hkl[ids].T
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
        return draw_line_image(self.to_lines(width=width), shape=shape, idxs=self.frames,
                               kernel=kernel, num_threads=num_threads)

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
        return draw_line_mask(self.to_lines(width=width), shape=shape, idxs=self.frames,
                              max_val=max_val, kernel=kernel, num_threads=num_threads)

    def to_dataframe(self) -> pd.DataFrame:
        """Export a streaks container into :class:`pandas.DataFrame`.

        Returns:
            A dataframe with all the data specified in :class:`cbclib.Streaks`.
        """
        return pd.DataFrame({attr: self[attr] for attr in self.contents()})

    def to_lines(self, frames: Optional[IntSequence]=None,
                 width: Optional[RealSequence]=None) -> RealArray:
        """Export a streaks container into line parameters ``x0, y0, x1, y1, width``:

        * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
        * `width` : Line's width.

        Returns:
            An array of line parameters.
        """
        if width is None:
            lines = self.lines
        else:
            widths = jnp.broadcast_to(jnp.asarray(width), self.shape)
            lines = jnp.concatenate((self.lines, widths[..., None]), axis=-1)

        if frames is not None:
            lines = jnp.concat([lines[self.frames == frame] for frame in np.atleast_1d(frames)])
        return lines

@jax_dataclass
class MillerIndices():
    hkl     : IntArray
    index   : IntArray

    @property
    def h(self) -> IntArray:
        return self.hkl[..., 0]

    @property
    def k(self) -> IntArray:
        return self.hkl[..., 1]

    @property
    def l(self) -> IntArray:
        return self.hkl[..., 2]

    def collapse(self) -> 'MillerIndices':
        index = jnp.broadcast_to(self.index, self.hkl.shape[:-1])
        idxs = jnp.concatenate((self.hkl, index[..., None]), axis=-1)
        idxs = jnp.unique(jnp.reshape(idxs, (-1,) + idxs.shape[-1:]), axis=0)
        return MillerIndices(idxs[..., :3], idxs[..., 3])

    def floor(self) -> 'MillerIndices':
        return MillerIndices(jnp.array(jnp.floor(self.hkl), dtype=int), self.index)

    def offset(self, offsets: IntArray) -> 'MillerIndices':
        hkl = self.hkl
        shape = offsets.shape[:-1] + hkl.shape
        hkl = jnp.reshape(jnp.reshape(hkl, (-1, 3)) + offsets[..., None, :], shape)
        return MillerIndices(hkl, self.index)

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
    def hkl_in_aperture(self, theta: Union[float, RealArray], hkl: IntArray, state: XtalState
                        ) -> MillerIndices:
        index = jnp.broadcast_to(jnp.expand_dims(jnp.arange(state.num), axis=range(1, hkl.ndim)),
                                 (state.num,) + hkl.shape[:-1])
        hkl = jnp.broadcast_to(hkl[None, ...], (state.num,) + hkl.shape)

        rec_vec = self.hkl_to_q(MillerIndices(hkl, index), state)
        rec_abs = jnp.sqrt((rec_vec**2).sum(axis=-1))
        rec_th = jnp.arccos(-rec_vec[..., 2] / rec_abs)
        src_th = rec_th - jnp.arccos(0.5 * rec_abs)
        idxs = jnp.where((jnp.abs(src_th) < theta))
        return MillerIndices(hkl[idxs], index[idxs])

    def hkl_in_ball(self, q_abs: Union[float, RealArray], state: XtalState) -> IntArray:
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
        return hkl[jnp.any(rec_abs < q_abs, axis=-1)]

    def hkl_to_q(self, miller: MillerIndices, state: XtalState) -> RealArray:
        return jnp.sum(state.basis[miller.index] * miller.hkl[..., None], axis=-2)

    def q_to_hkl(self, q: RealArray, idxs: IntArray, state: XtalState) -> MillerIndices:
        hkl = jnp.sum(jnp.linalg.inv(state.basis)[idxs] * q[..., None], axis=-2)
        return MillerIndices(hkl, idxs)

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

@jax_dataclass
class Pupil():
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

class Lens():
    def project_to_pupil(self, kin: RealArray, pupil: Pupil) -> RealArray:
        kin = safe_divide(kin, jnp.sqrt(jnp.sum(kin**2, axis=-1))[..., None])
        return kxy_to_k(pupil.project(kin[..., :2]))

    def kin_center(self, state: LensState) -> RealArray:
        pupil_roi = jnp.asarray(state.pupil_roi)
        x, y = jnp.mean(pupil_roi[2:]), jnp.mean(pupil_roi[:2])
        return det_to_k(x, y, state.foc_pos, jnp.zeros(x.shape, dtype=int))

    def kin_max(self, state: LensState) -> RealArray:
        return det_to_k(jnp.array(state.pupil_roi[3]), jnp.array(state.pupil_roi[1]),
                        state.foc_pos, jnp.array(0))

    def kin_min(self, state: LensState) -> RealArray:
        return det_to_k(jnp.array(state.pupil_roi[2]), jnp.array(state.pupil_roi[0]),
                        state.foc_pos, jnp.array(0))

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

    # def in_laue(self, kin: RealArray, edges: RealArray, pupil: Pupil) -> RealArray:
    #     max_dist = normal_distance(pupil.edges().mean(axis=-2), edges[:, 0], edges[:, 1])
    #     diags = pupil.diagonals()
    #     pts = line_intersection(diags[..., 0, :], diags[..., 1, :],
    #                             kin[..., 0, None, :2], kin[..., 1, None, :2])
    #     dist = normal_distance(pts[..., None, :], edges[:, 0], edges[:, 1])
    #     idxs = jnp.argmin(jnp.abs(dist), axis=-1)
    #     dist = jnp.squeeze(jnp.take_along_axis(dist, idxs[..., None], axis=-1))
    #     return jnp.max(smooth_step(dist, 0.0, max_dist[idxs]), axis=-1)

    def pupil(self, state: LensState) -> Pupil:
        return Pupil(self.kin_min(state)[:2], self.kin_max(state)[:2])

    def zero_order(self, state: LensState):
        return k_to_det(self.kin_center(state), state.foc_pos, jnp.array(0))

Params = Union[RealArray, float]
S = TypeVar('S', bound=State)

def init_from_bounds(state: S, bounds: Dict[str, Any], default: Callable[[Params], Params]
                     ) -> Callable[[KeyArray,], S]:
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
