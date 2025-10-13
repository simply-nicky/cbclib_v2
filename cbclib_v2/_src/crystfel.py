from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum
from io import TextIOWrapper
from math import isnan
import re
from types import TracebackType
from typing import (Any, ClassVar, Dict, Generic, Iterator, List,
                    OrderedDict as OrderedDictType, Tuple, TypeVar, overload)
from .annotations import Array, ArrayNamespace, DataclassInstance, IntArray, NumPy, RealArray
from .data_container import array_namespace

Value = TypeVar("Value")

class Attribute(Generic[Value]):
    __value__           : Value

    def __bool__(self) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__value__.__repr__()

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

@dataclass(repr=False)
class BitInt(Attribute[int], SimpleParser):
    __value__           : int | None = None

    def __bool__(self) -> bool:
        return self.__value__ is not None

    def parse(self, value: str):
        try:
            self.__value__ = int(value, base=0)
        except ValueError as exc:
            raise RuntimeError(f"Invalid int entry: {value}") from exc

@dataclass(repr=False)
class Bool(Attribute[bool], SimpleParser):
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

@dataclass(repr=False)
class Float(Attribute[float], SimpleParser):
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

@dataclass(repr=False)
class Int(Attribute[int], SimpleParser):
    __value__           : int | None = None

    def __bool__(self) -> bool:
        return self.__value__ is not None

    def parse(self, value: str):
        if value.isdigit():
            self.__value__ = int(value)
        else:
            raise RuntimeError(f"Invalid int entry: {value}")

@dataclass(repr=False)
class String(Attribute[str], SimpleParser):
    __value__           : str | None = None

    def __bool__(self) -> bool:
        return self.__value__ is not None

    def parse(self, value: str):
        self.__value__ = value

@dataclass(repr=False)
class Dimensions(Attribute[List[str | int | None]], Parser):
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

@dataclass(repr=False)
class Direction(Attribute[Tuple[float, float, float] | None], SimpleParser):
    __value__           : Tuple[float, float, float] | None = None

    @property
    def x(self) -> float:
        return self.__value__[0] if self.__value__ is not None else 0.0

    @property
    def y(self) -> float:
        return self.__value__[1] if self.__value__ is not None else 0.0

    @property
    def z(self) -> float:
        return self.__value__[2] if self.__value__ is not None else 0.0

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

        x, y, z = None, None, None
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

        if x is None or y is None or z is None:
            raise RuntimeError(f'Invalid direction: {value}')

        self.__value__ = (x, y, z)

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

@dataclass(repr=False)
class CoordRegion(ParsingContainer):
    min_x               : Float = Float()
    max_x               : Float = Float()
    min_y               : Float = Float()
    max_y               : Float = Float()

@dataclass(repr=False)
class PixelRegion(ParsingContainer):
    min_fs              : Int = Int()
    max_fs              : Int = Int()
    min_ss              : Int = Int()
    max_ss              : Int = Int()

    @property
    def roi(self) -> Tuple[int, int, int, int]:
        return (self.min_ss.value(), self.max_ss.value() + 1,
                self.min_fs.value(), self.max_fs.value() + 1)

@dataclass
class Corner(ParsingContainer):
    corner_x            : Float = Float()
    corner_y            : Float = Float()

    @property
    def x(self) -> float:
        return self.corner_x.value()

    @property
    def y(self) -> float:
        return self.corner_y.value()

@dataclass(repr=False)
class MaskData(ParsingContainer):
    # location of the mask array in the image data file
    mask_data           : String = String()

    # filename to use for the mask data
    mask_file           : String = String()

    # bit mask for good pixels
    mask_goodbits       : BitInt = BitInt()

    # bit mask for bad pixels
    mask_badbits        : BitInt = BitInt()

wl_units = {'A': Unit(1e-10), 'm': Unit()}
E_units = {'A': Unit(1e-10), 'eV': Unit(1.239842e-06, 'inverse'),
           'keV': Unit(1.239842e-03, 'inverse')}
length_units = {'mm': Unit(1e-3), 'm': Unit()}

@dataclass
class Panel(ParsingContainer):
    # Beam parameters

    # wavelength of the radiation
    wavelength          : Float = Float(float('nan'), wl_units)
    # energy of a single photon
    photon_energy       : Float = Float(float('nan'), E_units)
    # bandwidth of the radiation as a fraction of wavelength
    bandwidth           : Float = Float()

    # Physical locations

    # overall z-pozition for the detector
    clen                : Float = Float(float('nan'), length_units)

    # Data locations

    # location of the data in the data file
    data                : String = String()
    # range of pixels in the data block that correspond to this panel
    region              : PixelRegion = PixelRegion()

    # Pixel size

    # resolution of the detector in pixels per metre
    res                 : Float = Float()

    # Physical panel locations

    # (x, y) position of the corner
    corner              : Corner = Corner()
    # offset of the panel
    coffset             : Float = Float(0.0)
    # vector of the fast scan direction
    fs                  : Direction = Direction()
    # vector of teh slow scan direction
    ss                  : Direction = Direction()

    # Data dimensionality

    # dimension structure of the panel
    dim                 : Dimensions = Dimensions()

    # Detector gain data

    # number of ADUs which will arise from one photon
    adu_per_photon      : Float = Float()
    # saturation value for the detector
    max_adu             : Float = Float(float('inf'))
    # location of the per-pixel saturation map in the data file
    saturation_map      : String = String()
    # file containing a saturation map
    saturation_map_file : String = String()

    # Bad regions

    # panel is ignored if true
    no_index            : Bool = Bool()
    # mark a border of n pixels around the edge of the panel as bad
    mask_edge_pixels    : Int = Int(0)

    # Mask data
    masks               : List[MaskData] = field(default_factory=list)

    def __bool__(self) -> bool:
        return super().__bool__() and all(self.masks)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.region.roi[1] - self.region.roi[0],
                self.region.roi[3] - self.region.roi[2])

    @property
    def z_offset(self) -> float:
        return self.coffset.value() * self.res.value()

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        x0, y0, _ = self.to_detector(0, 0)
        x1, y1, _ = self.to_detector(self.shape[0] - 1, self.shape[1] - 1)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def distance(self, ss: RealArray, fs: RealArray) -> RealArray:
        xp = array_namespace(ss, fs)
        dss = ss - xp.clip(ss, self.region.roi[0], self.region.roi[1])
        dfs = fs - xp.clip(fs, self.region.roi[2], self.region.roi[3])
        return xp.sqrt(dss**2 + dfs**2)

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
                        self.masks.append(MaskData())
                self.masks[index].parse('mask_' + m.group(2), value)
        else:
            super().parse(key, value)

    @overload
    def to_detector(self, ss: float, fs: float, half_pixel_shift: bool=True
                    ) -> Tuple[float, float, float]: ...

    @overload
    def to_detector(self, ss: RealArray, fs: RealArray, half_pixel_shift: bool=True
                    ) -> Tuple[RealArray, RealArray, RealArray]: ...

    def to_detector(self, ss: RealArray | float, fs: RealArray | float, half_pixel_shift: bool=True
                    ) -> Tuple[RealArray | float, RealArray | float, RealArray | float]:
        x = ss * self.ss.x + fs * self.fs.x + self.corner.x
        y = ss * self.ss.y + fs * self.fs.y + self.corner.y
        z = ss * self.ss.z + fs * self.fs.z + self.z_offset
        if half_pixel_shift:
            return x + 0.5, y + 0.5, z
        return x, y, z

Region = PixelRegion | CoordRegion

@dataclass
class PixelIndices():
    ss              : IntArray
    fs              : IntArray

    @property
    def shape(self) -> Tuple[int, int]:
        return (int(self.ss.max()) + 1, int(self.fs.max()) + 1)

    def __call__(self, frames: Array) -> Array:
        xp = array_namespace(frames)
        result = xp.zeros(frames.shape[:-2] + self.shape)
        result[..., self.ss, self.fs] = frames
        return result

@dataclass
class Detector():
    panels          : OrderedDictType[str, Panel] = field(default_factory=OrderedDict)
    bad_regions     : OrderedDictType[str, Region] = field(default_factory=OrderedDict)
    groups          : Dict[str, List[str]] = field(default_factory=dict)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        x, y = [], []
        for panel in self.panels.values():
            x0, y0, x1, y1 = panel.bounds
            x.extend([x0, x1])
            y.extend([y0, y1])
        return (min(x), min(y), max(x), max(y))

    @property
    def shape(self) -> Tuple[int, int]:
        max_ss, max_fs = [], []
        for name, panel in self.panels.items():
            if not panel.region:
                raise ValueError(f'No panel data locations in the geometry data for panel {name}')
            max_ss.append(panel.region.roi[1])
            max_fs.append(panel.region.roi[3])
        return (max(max_ss), max(max_fs))

    def indices(self, xp: ArrayNamespace=NumPy) -> PixelIndices:
        pix_x, pix_y, _ = self.pixel_map(xp=xp)
        pix_x = xp.asarray(xp.round(pix_x - pix_x.min()), dtype=int)
        pix_y = xp.asarray(xp.round(pix_y - pix_y.min()), dtype=int)
        return PixelIndices(pix_y, pix_x)

    def pixel_map(self, half_pixel_shift: bool=True, xp: ArrayNamespace=NumPy):
        pixel_map = xp.zeros((3,) + self.shape)
        for name, panel in self.panels.items():
            if not panel.region:
                raise ValueError(f'No panel data locations in the geometry data for panel {name}')
            roi = panel.region.roi
            ss_grid, fs_grid = xp.meshgrid(xp.arange(panel.shape[0]),
                                           xp.arange(panel.shape[1]), indexing='ij')

            x, y, z = panel.to_detector(ss_grid, fs_grid, half_pixel_shift)
            pixel_map[0, roi[0]:roi[1], roi[2]:roi[3]] = x
            pixel_map[1, roi[0]:roi[1], roi[2]:roi[3]] = y
            pixel_map[2, roi[0]:roi[1], roi[2]:roi[3]] = z
        return pixel_map

    def to_detector(self, ss: RealArray, fs: RealArray, half_pixel_shift: bool=True,
                    tolerance: float=1.0) -> Tuple[RealArray, RealArray, RealArray]:
        x_min, y_min = self.bounds[:2]
        xp = array_namespace(ss, fs)
        if ss.shape != fs.shape:
            raise ValueError('ss and fs have incompatible shapes')
        x, y, z = xp.zeros(ss.shape), xp.zeros(ss.shape), xp.zeros(ss.shape)
        for panel in self.panels.values():
            mask = panel.distance(ss, fs) < tolerance
            mask = xp.all(mask, axis=tuple(range(1, mask.ndim)))
            x[mask], y[mask], z[mask] = panel.to_detector(ss[mask] - panel.region.roi[0],
                                                          fs[mask] - panel.region.roi[2],
                                                          half_pixel_shift)
        return x - x_min, y - y_min, z

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

def read_crystfel(filename: str) -> Detector:
    with CrystFELFile(filename) as file:
        detector = Detector()
        default_panel = Panel()
        for attr, value in file:
            path = [item for item in re.split("(/)", attr) if item != "/"]
            if len(path) == 1:
                if attr.startswith('group'):
                    pass
                else:
                    default_panel.parse(attr, value)
            elif len(path) == 2:
                if attr.startswith('bad'):
                    region_name, attr = path[0][3:], path[1]
                    if region_name not in detector.bad_regions:
                        if attr in PixelRegion.attributes():
                            detector.bad_regions[region_name] = PixelRegion()
                        elif attr in CoordRegion.attributes():
                            detector.bad_regions[region_name] = CoordRegion()
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
    return detector
