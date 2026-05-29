from __future__ import annotations
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum
from io import TextIOWrapper
from math import isnan, prod
import re
from types import TracebackType
from typing import (Any, ClassVar, DefaultDict, Dict, Generic, Iterator, List, Literal,
                    OrderedDict as OrderedDictType, Tuple, TypeVar, overload)
from .annotations import (Array, AnyNamespace, DataclassInstance, IntArray, NDArray, NumPy,
                          RealArray, Shape)
from .array_api import array_namespace, set_at
from .data_container import Container
from .streaks import StackedStreaks, Streaks
from ..indexer import Patterns

@dataclass
class CrystFELFile():
    filename        : str
    comment_sign    : ClassVar[str] = ';'

    def __post_init__(self):
        self.__file__ : TextIOWrapper | None = None

    def __bool__(self) -> bool:
        return self.__file__ is not None

    def __enter__(self) -> 'CrystFELFile':
        self.open()
        return self

    def __exit__(self, exc_type: BaseException | None, exc: BaseException | None,
                 traceback: TracebackType | None):
        self.close()

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        if self.__file__ is not None:
            for line in self.__file__.readlines():
                if line.startswith(self.comment_sign):
                    continue
                line_without_comments = line.strip().split(self.comment_sign)[0]
                line_items = re.split('([ \t])', string=line_without_comments)
                line_items = [item for item in line_items if item not in ('', ' ', '\t')]
                if len(line_items) < 3:
                    continue
                value = ''.join(line_items[2:])
                if line_items[1] != '=':
                    continue
                yield (line_items[0], value)

    def open(self):
        if not self:
            self.__file__ = open(self.filename, 'r')

    def close(self):
        if self:
            self.__file__ = None

Value = TypeVar("Value")

class AttributeParser(Generic[Value]):
    __value__           : Value

    def __bool__(self) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__value__.__repr__()})"

    def value(self) -> Value:
        return self.__value__

class SimpleParser:
    def parse(self, value: str):
        raise NotImplementedError

    def value(self) -> Any:
        raise NotImplementedError

class Parser:
    def parse(self, key: str, value: str):
        raise NotImplementedError

    def value(self) -> Any:
        raise NotImplementedError

@dataclass
class BitIntParser(AttributeParser[int], SimpleParser):
    __value__           : int | None = None

    def __bool__(self) -> bool:
        return self.__value__ is not None

    def parse(self, value: str):
        try:
            self.__value__ = int(value, base=0)
        except ValueError as exc:
            raise RuntimeError(f"Invalid int entry: {value}") from exc

@dataclass
class BoolParser(AttributeParser[bool], SimpleParser):
    __value__           : bool = False

    def __bool__(self) -> bool:
        return self.__value__

    def parse(self, value: str):
        if value == 'true':
            self.__value__ = True
        elif value.isdigit():
            self.__value__ = bool(int(value))
        else:
            raise RuntimeError(f"Invalid bool entry: {value}")

class Function(str, Enum):
    linear = 'linear'
    inverse = 'inverse'

@dataclass
class Unit():
    factor              : float = 1.0
    function            : str = 'linear'

    def __call__(self, value: float) -> float:
        if Function(self.function) == Function.linear:
            return self.factor * value
        if Function(self.function) == Function.inverse:
            return self.factor / value
        raise ValueError(f'Invalid function type: {self.function:s}')

@dataclass
class FloatParser(AttributeParser[float], SimpleParser):
    __value__           : float = float("nan")
    __units__           : Dict[str, Unit] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return not isnan(self.__value__)

    def parse(self, value: str):
        pattern = r'^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(' + r'|'.join(self.__units__) + r')?$'
        m = re.match(pattern, value)
        if m:
            self.__value__ = self.__units__.get(m.group(2), Unit())(float(m.group(1)))
        else:
            raise RuntimeError(f"Invalid float entry: {value}")

@dataclass
class IntParser(AttributeParser[int], SimpleParser):
    __value__           : int | None = None

    def __bool__(self) -> bool:
        return self.__value__ is not None

    def parse(self, value: str):
        if value.isdigit():
            self.__value__ = int(value)
        else:
            raise RuntimeError(f"Invalid int entry: {value}")

@dataclass
class StringParser(AttributeParser[str], SimpleParser):
    __value__           : str | None = None

    def __bool__(self) -> bool:
        return self.__value__ is not None

    def parse(self, value: str):
        self.__value__ = value

@dataclass
class DimensionsParser(AttributeParser[List[str | int | None]], Parser):
    __value__           : List[str | int | None] = field(default_factory=list)

    def __bool__(self) -> bool:
        return all(item is not None for item in self.__value__)

    def parse(self, key: str, value: str):
        try:
            dim_index = int(key[3])
        except IndexError as exc:
            raise RuntimeError("'dim' must be followed by a number, e.g. 'dim0')") from exc
        except ValueError as exc:
            raise RuntimeError(f"Invalid dimension number {key[3]}") from exc

        if dim_index > len(self.__value__) - 1:
            for _ in range(len(self.__value__), dim_index + 1):
                self.__value__.append(None)

        if value in ['ss', 'fs', '%']:
            self.__value__[dim_index] = value
        elif value.isdigit():
            self.__value__[dim_index] = int(value)
        else:
            raise RuntimeError(f"Invalid dim entry: {value}")

@dataclass
class DirectionParser(AttributeParser[Tuple[float, float, float] | None], SimpleParser):
    __value__           : Tuple[float, float, float] | None = None

    def __bool__(self) -> bool:
        return self.__value__ is not None

    def parse(self, value: str):
        items = [item for item in re.split("([+-])", string=value.strip()) if item]
        if items and items[0] not in ("+", "-"):
            items.insert(0, "+")
        items = [str().join((items[idx], items[idx + 1]))
                 for idx in range(0, len(items), 2)]
        if not items:
            raise RuntimeError(f"Invalid direction: {value}")

        x, y, z = None, None, 0.0
        for item in items:
            axis = item[-1]

            if item[:-1] == '+':
                value = '1.0'
            elif item[:-1] == '-':
                value = '-1.0'
            else:
                value = item[:-1]

            if axis == 'x':
                x = float(value)
            elif axis == 'y':
                y = float(value)
            elif axis == 'z':
                z = float(value)
            else:
                raise RuntimeError(f"Invalid symbol: {axis} (must be x, y, or z)")

        if x is None or y is None:
            raise RuntimeError(f'Invalid direction: {value}')

        self.__value__ = (x, y, z)

    def value(self) -> Dict[str, float]:
        if self.__value__ is None:
            return {'x': float('nan'), 'y': float('nan'), 'z': float('nan')}
        return {'x': self.__value__[0], 'y': self.__value__[1], 'z': self.__value__[2]}

P = TypeVar('P', bound='ParsingContainer')

class ParsingContainer(Parser, DataclassInstance):
    def __bool__(self) -> bool:
        return all(getattr(self, attr) for attr in self.parsers())

    def __repr__(self) -> str:
        attributes = []
        for attr in self.parsers():
            attributes.append(attr + '=' + getattr(self, attr).__repr__())
        return self.__class__.__name__ + '(' + ', '.join(attributes) + ')'

    @classmethod
    def attributes(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    def parsers(self) -> List[str]:
        return [attr for attr in self.attributes()
                if isinstance(getattr(self, attr), (Parser, SimpleParser))]

    def get(self, key: str, value: Any) -> Any:
        def find(key: str) -> Any | None:
            for attr in self.parsers():
                val = getattr(self, attr)
                if attr == key:
                    return val
                if isinstance(val, ParsingContainer):
                    result = val.get(key, value)
                    if result is not None:
                        return result
            return None

        result = find(key)
        return result if result is not None else value

    def parse(self, key: str, value: str):
        attr = self.get(key, None)
        if attr is None:
            raise RuntimeError(f'Invalid path: {key}')

        if isinstance(attr, SimpleParser):
            attr.parse(value)
        elif isinstance(attr, Parser):
            attr.parse(key, value)
        else:
            raise RuntimeError(f'Invalid attribute: {attr}')

    def value(self) -> Dict[str, Any]:
        value = {}
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, (Parser, SimpleParser)):
                value[f.name] = attr.value()
        return value

@dataclass
class CoordRegionParser(ParsingContainer):
    min_x               : FloatParser = field(default_factory=FloatParser)
    max_x               : FloatParser = field(default_factory=FloatParser)
    min_y               : FloatParser = field(default_factory=FloatParser)
    max_y               : FloatParser = field(default_factory=FloatParser)

@dataclass
class PixelRegionParser(ParsingContainer):
    min_fs              : IntParser = field(default_factory=IntParser)
    max_fs              : IntParser = field(default_factory=IntParser)
    min_ss              : IntParser = field(default_factory=IntParser)
    max_ss              : IntParser = field(default_factory=IntParser)

@dataclass
class BadPixelRegionParser(PixelRegionParser):
    panel               : StringParser = field(default_factory=StringParser)

@dataclass
class CornerParser(ParsingContainer):
    corner_x            : FloatParser = field(default_factory=FloatParser)
    corner_y            : FloatParser = field(default_factory=FloatParser)

    def value(self) -> Dict[str, float]:
        return {'x': self.corner_x.value(), 'y': self.corner_y.value()}

@dataclass
class MaskDataParser(ParsingContainer):
    # location of the mask array in the image data file
    mask_data           : StringParser = field(default_factory=StringParser)

    # filename to use for the mask data
    mask_file           : StringParser = field(default_factory=StringParser)

    # bit mask for good pixels
    mask_goodbits       : BitIntParser = field(default_factory=BitIntParser)

    # bit mask for bad pixels
    mask_badbits        : BitIntParser = field(default_factory=BitIntParser)

wl_units = {'A': Unit(1e-10), 'm': Unit()}
E_units = {'eV': Unit(), 'keV': Unit(1e-3)}
length_units = {'mm': Unit(1e-3), 'm': Unit()}

DEFAULT_WL = FloatParser(float('nan'), wl_units)
DEFAULT_PE = FloatParser(float('nan'), E_units)
DEFAULT_CLEN = FloatParser(float('nan'), length_units)

@dataclass
class PanelParser(ParsingContainer):
    # Beam parameters

    # wavelength of the radiation
    wavelength          : FloatParser = field(default_factory=lambda: DEFAULT_WL)
    # energy of a single photon
    photon_energy       : FloatParser = field(default_factory=lambda: DEFAULT_PE)
    # bandwidth of the radiation as a fraction of wavelength
    bandwidth           : FloatParser = field(default_factory=FloatParser)

    # Physical locations

    # overall z-pozition for the detector
    clen                : FloatParser = field(default_factory=lambda: DEFAULT_CLEN)

    # Data locations

    # location of the data in the data file
    data                : StringParser = field(default_factory=StringParser)
    # range of pixels in the data block that correspond to this panel
    region              : PixelRegionParser = field(default_factory=PixelRegionParser)

    # Pixel size

    # resolution of the detector in pixels per metre
    res                 : FloatParser = field(default_factory=FloatParser)

    # Physical panel locations

    # (x, y) position of the corner
    corner              : CornerParser = field(default_factory=CornerParser)
    # offset of the panel
    coffset             : FloatParser = field(default_factory=lambda: FloatParser(0.0))
    # vector of the fast scan direction
    fs                  : DirectionParser = field(default_factory=DirectionParser)
    # vector of the slow scan direction
    ss                  : DirectionParser = field(default_factory=DirectionParser)

    # Data dimensionality

    # dimension structure of the panel
    dim                 : DimensionsParser = field(default_factory=DimensionsParser)

    # Detector gain data

    # number of ADUs which will arise from one photon
    adu_per_photon      : FloatParser = field(default_factory=FloatParser)
    # saturation value for the detector
    max_adu             : FloatParser = field(default_factory=lambda: FloatParser(float('inf')))
    # location of the per-pixel saturation map in the data file
    saturation_map      : StringParser = field(default_factory=StringParser)
    # file containing a saturation map
    saturation_map_file : StringParser = field(default_factory=StringParser)

    # Bad regions

    # panel is ignored if true
    no_index            : BoolParser = field(default_factory=BoolParser)
    # mark a border of n pixels around the edge of the panel as bad
    mask_edge_pixels    : IntParser = field(default_factory=lambda: IntParser(0))

    # Mask data
    masks               : List[MaskDataParser] = field(default_factory=list)

    def __bool__(self) -> bool:
        return super().__bool__() and all(self.masks)

    def parse(self, key: str, value: str):
        if key.startswith('dim'):
            self.dim.parse(key, value)
        elif key.startswith('mask'):
            m = re.match(r'^mask(\d?)_(data|file|goodbits|badbits)$', key)
            if m:
                if not m.group(1):
                    index = 0
                else:
                    index = int(m.group(1))
                if index > 7:
                    raise RuntimeError(f'{key} is invalid, CrystFEL supports up to 8 masks')
                if index > len(self.masks) - 1:
                    for _ in range(len(self.masks), index + 1):
                        self.masks.append(MaskDataParser())
                self.masks[index].parse('mask_' + m.group(2), value)
        else:
            super().parse(key, value)

    def value(self) -> Dict:
        value = super().value()
        value['masks'] = [mask.value() for mask in self.masks]
        return value

RegionParser = BadPixelRegionParser | CoordRegionParser

@dataclass
class DetectorParser():
    bad_regions        : OrderedDictType[str, RegionParser] = field(default_factory=OrderedDict)
    panels             : OrderedDictType[str, PanelParser] = field(default_factory=OrderedDict)

    def check_bad_regions(self):
        for region_name, region in self.bad_regions.items():
            if isinstance(region, BadPixelRegionParser):
                panel_name = region.panel.value()
                if panel_name not in self.panels:
                    raise RuntimeError(f"Bad region '{region_name}' references "
                                       f"unknown panel '{panel_name}'")

def parse_crystfel_file(filename: str) -> DetectorParser:
    with CrystFELFile(filename) as file:
        detector = DetectorParser()
        default_panel = PanelParser()
        for attr, value in file:
            path = [item for item in re.split("(/)", attr) if item != "/"]
            if len(path) == 1:
                if attr.startswith(('group', 'rigid_group')):
                    pass
                else:
                    default_panel.parse(attr, value)
            elif len(path) == 2:
                if attr.startswith('bad'):
                    region_name, attr = path[0][3:], path[1]
                    if region_name not in detector.bad_regions:
                        if attr in BadPixelRegionParser.attributes():
                            detector.bad_regions[region_name] = BadPixelRegionParser()
                        elif attr in CoordRegionParser.attributes():
                            detector.bad_regions[region_name] = CoordRegionParser()
                        else:
                            raise RuntimeError(f"Invalid attribute '{attr}' " \
                                               f"in bad region '{region_name}'")

                    detector.bad_regions[region_name].parse(attr, value)
                else:
                    panel_name, attr = path
                    if panel_name not in detector.panels:
                        detector.panels[panel_name] = deepcopy(default_panel)
                    detector.panels[panel_name].parse(attr, value)
            else:
                raise RuntimeError(f"Invalid path: {attr}")
    detector.check_bad_regions()
    return detector

@dataclass
class Direction(Container):
    x                   : float
    y                   : float
    z                   : float

@dataclass
class PixelRegion(Container):
    min_fs              : int
    max_fs              : int
    min_ss              : int
    max_ss              : int

    @property
    def fs_slice(self) -> slice:
        return slice(self.min_fs, self.max_fs)

    @property
    def roi(self) -> Tuple[int, int, int, int]:
        return (self.min_ss, self.max_ss + 1, self.min_fs, self.max_fs + 1)

    @property
    def ss_slice(self) -> slice:
        return slice(self.min_ss, self.max_ss)

@dataclass
class BadPixelRegion(PixelRegion):
    panel               : str

@dataclass
class CoordRegion(Container):
    min_x               : float
    max_x               : float
    min_y               : float
    max_y               : float

Region = BadPixelRegion | CoordRegion

@dataclass
class Corner(Container):
    x                   : float
    y                   : float

@dataclass
class MaskData(Container):
    mask_data           : str
    mask_file           : str
    mask_goodbits       : int
    mask_badbits        : int

@dataclass
class Panel(Container):
    """A single detector panel parsed from a CrystFEL ``.geom`` file.

    Stores the physical geometry mapping raw (ss, fs) array coordinates to the
    CrystFEL lab frame (+x right, +y up, +z along the beam).  All positions
    and directions are in units of pixels unless otherwise noted.  Panels are
    normally accessed via :attr:`Detector.panels` or :meth:`Detector.panel`
    after calling :func:`read_crystfel`.

    See the `CrystFEL geometry reference
    <https://gitlab.desy.de/thomas.white/crystfel/-/blob/master/doc/man/crystfel_geometry.5.md>`_
    for the full description of each field.

    Attributes:
        wavelength: Radiation wavelength in metres.
        photon_energy: Photon energy in eV (accepts ``eV`` or ``keV`` units
            in the ``.geom`` file).
        clen: Camera length — overall z position of the detector in metres.
        data: HDF5 dataset path for this panel.
        region: Pixel bounding box selecting this panel from the raw array
            (``min_ss``, ``max_ss``, ``min_fs``, ``max_fs``).
        res: Detector resolution in pixels per metre.
        corner: Corner position in lab-frame pixel units (``corner_x``,
            ``corner_y``).
        coffset: Additional z offset of the panel in metres.
        fs: Fast-scan direction unit vector in lab-frame pixel units.
        ss: Slow-scan direction unit vector in lab-frame pixel units.
        dim: Ordered dimension labels (``'ss'``, ``'fs'``, or an integer
            module index) mapping raw array axes to image coordinates.
        masks: Per-panel mask dataset locations.

    Example:
        Read a CrystFEL geometry file from a ``.geom`` file and convert panel-local coordinates
        to lab-frame coordinates:

        >>> detector = read_crystfel('detector.geom')
        >>> panel = detector.panel(0)
        >>> x, y, z = panel.to_detector(0, 0)
    """
    # Beam parameters

    # wavelength of the radiation
    wavelength          : float
    # energy of a single photon
    photon_energy       : float
    # bandwidth of the radiation as a fraction of wavelength
    bandwidth           : float

    # Physical locations

    # overall z-pozition for the detector
    clen                : float

    # Data locations

    # location of the data in the data file
    data                : str
    # range of pixels in the data block that correspond to this panel
    region              : PixelRegion

    # Pixel size

    # resolution of the detector in pixels per metre
    res                 : float

    # Physical panel locations

    # (x, y) position of the corner
    corner              : Corner
    # offset of the panel
    coffset             : float
    # vector of the fast scan direction
    fs                  : Direction
    # vector of the slow scan direction
    ss                  : Direction

    # Data dimensionality

    # dimension structure of the panel
    dim                 : List[str | int]

    # Detector gain data

    # number of ADUs which will arise from one photon
    adu_per_photon      : float
    # saturation value for the detector
    max_adu             : float
    # location of the per-pixel saturation map in the data file
    saturation_map      : str
    # file containing a saturation map
    saturation_map_file : str

    # Bad regions

    # panel is ignored if true
    no_index            : bool
    # mark a border of n pixels around the edge of the panel as bad
    mask_edge_pixels    : int

    # Mask data
    masks               : List[MaskData]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of this panel in the raw data array, one entry per non-``'%'`` axis in :attr:`dim`."""
        return tuple(slice.stop - slice.start for slice in self.roi())

    @property
    def z_offset(self) -> float:
        """Z offset of the panel in pixels (``coffset * res``)."""
        return self.coffset * self.res

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Bounding box ``(x_min, y_min, x_max, y_max)`` in lab-frame pixel units."""
        x0, y0, _ = self.to_detector(0, 0)
        x1, y1, _ = self.to_detector(self.shape[0] - 1, self.shape[1] - 1)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def distance(self, *coordinates: IntArray | RealArray) -> RealArray:
        """Distance from each coordinate to the nearest point on this panel.

        Points inside the panel have distance zero.  Used by
        :class:`Detector` to assign coordinates to their nearest panel.

        Args:
            *coordinates: Coordinate arrays in raw data index space.  Pass
                ``(ss, fs)`` for a single-module detector or
                ``(module_id, ss, fs)`` for a multi-module detector.

        Returns:
            Array of distances in pixels, same shape as the input coordinates.
        """
        if len(coordinates) != len(self.shape):
            raise ValueError(f"The number of coordinates ({len(coordinates)}) "
                             f"must be equal to the number of dimensions ({len(self.shape)})")
        xp = array_namespace(*coordinates)
        ss, fs = coordinates[-2], coordinates[-1]
        dss = ss - xp.clip(ss, self.region.roi[0], self.region.roi[1])
        dfs = fs - xp.clip(fs, self.region.roi[2], self.region.roi[3])
        dist = xp.sqrt(dss**2 + dfs**2)

        roi = self.roi()
        if len(roi) == 3:
            index = coordinates[-3]
            index = xp.reshape(index, index.shape[:1] + (1,) * (dist.ndim - 1) + index.shape[1:])
            dist = xp.where(index == roi[0].start, dist, xp.inf)

        return dist

    def roi(self) -> Tuple[slice, ...]:
        """Slice tuple indexing this panel in the raw data array.

        Each entry corresponds to one non-``'%'`` axis in :attr:`dim`:
        ``'ss'`` and ``'fs'`` axes map to their ``min``/``max`` bounds from
        :attr:`region`; integer labels map to a length-1 slice at that index.

        Returns:
            Tuple of slices into the raw data array for this panel.

        Raises:
            ValueError: If :attr:`dim` is empty.
        """
        if not self.dim:
            raise ValueError('No panel dimension in the geometry data')

        rois = []
        for key in self.dim:
            if key == '%':
                continue

            if key == 'ss':
                rois.append(slice(self.region.roi[0], self.region.roi[1]))
            elif key == 'fs':
                rois.append(slice(self.region.roi[2], self.region.roi[3]))
            elif isinstance(key, int):
                rois.append(slice(key, key + 1))
            else:
                raise ValueError(f'Invalid panel dimension key: {key}')
        return tuple(rois)

    @overload
    def to_detector(self, ss: float, fs: float, half_pixel_shift: bool=True
                    ) -> Tuple[float, float, float]: ...

    @overload
    def to_detector(self, ss: Array, fs: Array, half_pixel_shift: bool=True
                    ) -> Tuple[RealArray, RealArray, RealArray]: ...

    def to_detector(self, ss: Array | float, fs: Array | float, half_pixel_shift: bool=True
                    ) -> Tuple[RealArray | float, RealArray | float, RealArray | float]:
        """Convert panel-local (ss, fs) coordinates to lab-frame (x, y, z).

        Applies the panel's ``ss``/``fs`` unit vectors and corner offset to
        map pixel positions to the CrystFEL lab frame (+x right, +y up,
        +z along the beam).

        Args:
            ss: Slow-scan pixel coordinate(s) relative to the panel corner.
            fs: Fast-scan pixel coordinate(s) relative to the panel corner.
            half_pixel_shift: Add a 0.5-pixel shift to place coordinates at
                pixel centres when ``True`` (default).

        Returns:
            Tuple ``(x, y, z)`` in lab-frame pixel units.

        Example:
            Convert the panel-local origin to lab-frame coordinates:

            >>> x, y, z = panel.to_detector(0, 0)
        """
        x = ss * self.ss.x + fs * self.fs.x + self.corner.x
        y = ss * self.ss.y + fs * self.fs.y + self.corner.y
        z = ss * self.ss.z + fs * self.fs.z + self.z_offset
        if half_pixel_shift:
            return x + 0.5, y + 0.5, z
        return x, y, z

@dataclass
class Assembler():
    ss              : IntArray
    fs              : IntArray

    @property
    def shape(self) -> Tuple[int, int]:
        return (int(self.ss.max()) + 1, int(self.fs.max()) + 1)

    @overload
    def __call__(self, frames: NDArray) -> NDArray: ...

    @overload
    def __call__(self, frames: Array) -> Array: ...

    def __call__(self, frames: Array) -> Array:
        xp = array_namespace(frames)
        frames = xp.reshape(frames, (-1,) + self.ss.shape)

        n_frames = frames.size // self.ss.size
        if n_frames > 1:
            result = xp.zeros((n_frames,) + self.shape)
        else:
            result = xp.zeros(self.shape)

        result[..., self.ss, self.fs] = frames
        return result

@dataclass
class Detector():
    """Full detector geometry parsed from a CrystFEL ``.geom`` file.

    Holds an ordered collection of :class:`Panel` objects and bad-pixel
    regions, and provides methods to assemble stacked module data into a
    single lab-frame image and to convert raw array coordinates to physical
    coordinates.  Normally created by :func:`read_crystfel`.

    Attributes:
        panels: Ordered mapping of panel name to :class:`Panel`.
        bad_regions: Ordered mapping of region name to bad-pixel or
            coordinate-space bad region.
        groups: Mapping of group name to list of panel names.

    Example:
        Read a CrystFEL geometry file from a ``.geom`` file:

        >>> detector = read_crystfel('detector.geom')

        Assemble stacked module data into a single image:

        >>> assembler = detector.assembler()
        >>> assembled = assembler(frames)

        Convert raw pixel indices to lab-frame coordinates in pixels:

        >>> ss = np.arange(512); fs = np.arange(512)
        >>> x, y, z = detector.to_detector(ss, fs)
    """
    panels          : OrderedDictType[str, Panel] = field(default_factory=OrderedDict)
    bad_regions     : OrderedDictType[str, Region] = field(default_factory=OrderedDict)
    groups          : Dict[str, List[str]] = field(default_factory=dict)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Overall bounding box ``(x_min, y_min, x_max, y_max)`` across all panels in lab-frame pixel units."""
        x, y = [], []
        for panel in self.panels.values():
            x0, y0, x1, y1 = panel.bounds
            x.extend([x0, x1])
            y.extend([y0, y1])
        return (min(x), min(y), max(x), max(y))

    @property
    def shape(self) -> Shape:
        """Shape of the raw detector data array inferred from panel ROIs."""
        shape : DefaultDict[int, List] = defaultdict(list)

        for panel in self.panels.values():
            for index, roi in enumerate(panel.roi()):
                shape[index].append(roi.stop)
        return tuple(max(shape[index]) for index in range(len(shape)))

    @property
    def num_modules(self) -> int:
        """Number of detector modules (product of all axes except the last two ss/fs axes)."""
        return prod(self.shape) // prod(self.shape[-2:])

    @property
    def pixel_size(self) -> float:
        """Pixel size in metres, taken from the first panel's resolution."""
        for panel in self.panels.values():
            return 1.0 / panel.res
        raise RuntimeError('No pixel resolution data in the panels')

    def assembler(self, xp: AnyNamespace=NumPy) -> Assembler:
        """Build an assembler that places stacked module data onto a single lab-frame image.

        Computes a rounded pixel map for all panels and returns an
        :class:`Assembler` callable.  Each call to the assembler takes
        stacked module data and writes each module's pixels to its correct
        position in the lab coordinate system.

        Args:
            xp: Array namespace used to build the pixel map; defaults to NumPy.

        Returns:
            :class:`Assembler` callable; pass stacked module data to get a
            single assembled image of shape ``(H, W)`` or ``(N, H, W)``.

        Example:
            Assemble stacked module data into a single image:

            >>> assembler = detector.assembler()
            >>> assembled = assembler(frames)
        """
        pix_x, pix_y, _ = self.pixel_map(xp=xp)
        pix_x = xp.asarray(xp.round(pix_x - pix_x.min()), dtype=int)
        pix_y = xp.asarray(xp.round(pix_y - pix_y.min()), dtype=int)
        return Assembler(pix_y, pix_x)

    def panel(self, module_id: int) -> Panel:
        """Return the panel for a given zero-based module index.

        Args:
            module_id: Zero-based index into the ordered :attr:`panels` mapping.

        Returns:
            :class:`Panel` at position ``module_id``.
        """
        return self.panels[list(self.panels.keys())[module_id]]

    def pixel_map(self, half_pixel_shift: bool=True, xp: AnyNamespace=NumPy):
        """Compute the (x, y, z) lab-frame coordinate map for all panels.

        Args:
            half_pixel_shift: Add a 0.5-pixel offset to place coordinates at
                pixel centres when ``True`` (default).
            xp: Array namespace; defaults to NumPy.

        Returns:
            Array of shape ``(3, *detector_shape)`` with x, y, z coordinates
            in lab-frame pixel units.
        """
        pixel_map = xp.zeros((3,) + self.shape)
        for panel in self.panels.values():
            roi = panel.roi()
            ss_grid, fs_grid = xp.meshgrid(xp.arange(panel.shape[-2]),
                                           xp.arange(panel.shape[-1]), indexing='ij')

            x, y, z = panel.to_detector(ss_grid, fs_grid, half_pixel_shift)
            pixel_map[(0, ...) + roi] = x
            pixel_map[(1, ...) + roi] = y
            pixel_map[(2, ...) + roi] = z
        return pixel_map

    def to_detector(self, *coordinates: IntArray | RealArray, half_pixel_shift: bool=True,
                    units: Literal['pixel', 'meter']='pixel', tolerance: float=1.0
                    ) -> Tuple[RealArray, RealArray, RealArray]:
        """Convert raw array coordinates to lab-frame (x, y, z).

        Each input point is assigned to the nearest panel by distance and
        transformed to CrystFEL lab-frame coordinates.  The output origin is
        the bottom-left corner of the detector bounding box.

        Args:
            *coordinates: Raw data array indices.  Pass ``(ss, fs)`` for a
                single-module detector or ``(module_id, ss, fs)`` for a
                multi-module detector.
            half_pixel_shift: Add a 0.5-pixel shift to place coordinates at
                pixel centres when ``True`` (default).
            units: ``'pixel'`` (default) returns coordinates in pixel units;
                ``'meter'`` scales by :attr:`pixel_size`.
            tolerance: Maximum pixel distance from a panel edge for a point
                to be assigned to that panel (default ``1.0``).

        Returns:
            Tuple ``(x, y, z)`` of arrays with the same shape as the input
            coordinates.

        Example:
            Convert raw pixel indices to lab-frame coordinates in pixels for
            a single-module detector:

            >>> x, y, z = detector.to_detector(ss, fs)

            And for a multi-module detector with module indices:

            >>> x, y, z = detector.to_detector(module_id, ss, fs, units='meter')
        """
        x_min, y_min = self.bounds[:2]
        xp = array_namespace(*coordinates)

        # We need to clip the coordinates to detector data shape
        ss = xp.clip(coordinates[-2], 0, self.shape[-2] - 1)
        fs = xp.clip(coordinates[-1], 0, self.shape[-1] - 1)
        coordinates = coordinates[:-2] + (ss, fs)

        x, y, z = xp.zeros(ss.shape), xp.zeros(ss.shape), xp.zeros(ss.shape)
        for panel in self.panels.values():
            mask = panel.distance(*coordinates) < tolerance
            mask = xp.all(mask, axis=tuple(range(ss.ndim, mask.ndim)))
            x_new, y_new, z_new = panel.to_detector(ss[mask] - panel.region.roi[0],
                                                    fs[mask] - panel.region.roi[2],
                                                    half_pixel_shift)
            x, y, z = set_at(x, mask, x_new), set_at(y, mask, y_new), set_at(z, mask, z_new)
        if units == 'pixel':
            unit = 1.0
        elif units == 'meter':
            unit = self.pixel_size
        else:
            raise ValueError(f'Invalid units: {units}')

        return unit * (x - x_min), unit * (y - y_min), unit * z

    def to_streaks(self, streaks: Streaks | StackedStreaks, half_pixel_shift: bool=True,
                   tolerance: float=1.0) -> Streaks:
        """Convert pixel-space streaks to lab-frame (x, y) coordinates.

        Assigns each streak to its nearest panel and applies the panel's
        ``ss``/``fs``-to-lab transform.  When a streak spans a panel boundary
        the panel closest to the streak's midpoint is used for the whole
        streak.

        Args:
            streaks: Pixel-space streaks (:class:`~cbclib_v2.Streaks` or
                :class:`~cbclib_v2.StackedStreaks`).
            half_pixel_shift: Add a 0.5-pixel shift to streak endpoints when
                ``True`` (default).
            tolerance: Maximum pixel distance for panel assignment
                (default ``1.0``).

        Returns:
            :class:`~cbclib_v2.Streaks` with endpoints in lab-frame pixel
            units.

        Example:
            Convert pixel-space streaks to lab-frame pixel coordinates:

            >>> lab_streaks = detector.to_streaks(pixel_streaks)
        """
        # We need to make sure that the distance between points stays the same
        # Sometimes a streak can span over multiple panels, so we need to choose one panel
        # for the whole streak
        if isinstance(streaks, StackedStreaks) and streaks.num_modules > 1:
            coordinates = (streaks.module_id, streaks.y, streaks.x)
        else:
            coordinates = (streaks.y, streaks.x)

        xp = streaks.__array_namespace__()
        distances = xp.stack([panel.distance(*coordinates) for panel in self.panels.values()])
        module_ids = xp.argmin(xp.sum(distances, axis=-1), axis=0)
        point_dists = xp.take_along_axis(distances, module_ids[None, ..., None], axis=0)[0]
        point_ids = xp.argmin(point_dists, axis=-1)
        other_ids = (point_ids + 1) % 2

        fs0, ss0 = xp.take_along_axis(streaks.points, point_ids[..., None, None],
                                      axis=-2)[..., 0, :].T
        fs1, ss1 = xp.take_along_axis(streaks.points, other_ids[..., None, None],
                                      axis=-2)[..., 0, :].T

        x0, y0, _ = self.to_detector(*(coordinates[:-2] + (ss0, fs0)),
                                     half_pixel_shift=half_pixel_shift, tolerance=tolerance)

        # I need to convert dfs and dss to dx and dy first
        panels_ss = xp.stack([xp.array([panel.ss.x, panel.ss.y])
                              for panel in self.panels.values()], axis=-1)
        panels_fs = xp.stack([xp.array([panel.fs.x, panel.fs.y])
                              for panel in self.panels.values()], axis=-1)

        ss_basis = xp.take_along_axis(panels_ss, module_ids[None, ...], axis=-1)
        fs_basis = xp.take_along_axis(panels_fs, module_ids[None, ...], axis=-1)
        dx, dy = ss_basis * (ss1 - ss0) + fs_basis * (fs1 - fs0)
        x1, y1 = x0 + dx, y0 + dy

        indices = xp.stack((point_ids, other_ids), axis=-1)
        x = xp.take_along_axis(xp.stack((x0, x1), axis=-1), indices, axis=-1)
        y = xp.take_along_axis(xp.stack((y0, y1), axis=-1), indices, axis=-1)
        return Streaks.import_xy(streaks.index, x, y)

    def to_patterns(self, streaks: Streaks) -> Patterns:
        """Convert lab-frame streaks to diffraction patterns in metres.

        Scales streak line coordinates by :attr:`pixel_size` to produce
        :class:`~cbclib_v2.indexer.Patterns` in physical units suitable for
        indexing.

        Args:
            streaks: Lab-frame streaks in pixel units (e.g. from
                :meth:`to_streaks`).

        Returns:
            :class:`~cbclib_v2.indexer.Patterns` with coordinates in metres.

        Example:
            Convert lab-frame pixel coordinates to metres for indexing:

            >>> patterns = detector.to_patterns(detector.to_streaks(pixel_streaks))
        """
        return Patterns(streaks.index, streaks.lines * self.pixel_size)

def read_crystfel(file: str) -> Detector:
    """Parse a CrystFEL ``.geom`` file and return a :class:`Detector`.

    The file format follows the `CrystFEL geometry convention
    <https://gitlab.desy.de/thomas.white/crystfel/-/blob/master/doc/man/crystfel_geometry.5.md>`_.

    Args:
        file: Path to the CrystFEL ``.geom`` geometry file.

    Returns:
        :class:`Detector` populated with panel geometry and bad-pixel regions.

    Example:
        Read a CrystFEL geometry file and print the number of modules and pixel size:

        >>> detector = read_crystfel('detector.geom')
        >>> print(detector.num_modules, detector.pixel_size)
    """
    parsed = parse_crystfel_file(file)
    detector = Detector()
    for name, panel in parsed.panels.items():
        detector.panels[name] = Panel.from_dict(**panel.value())
    for name, region in parsed.bad_regions.items():
        if isinstance(region, BadPixelRegionParser):
            detector.bad_regions[name] = BadPixelRegion(**region.value())
        elif isinstance(region, CoordRegionParser):
            detector.bad_regions[name] = CoordRegion(**region.value())
        else:
            raise RuntimeError(f'Invalid region type: {type(region)}')
    return detector
