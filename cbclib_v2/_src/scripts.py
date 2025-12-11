from multiprocessing import Pool
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Literal, Sequence, Tuple, Type,
                    TypeVar, cast, overload)
from dataclasses import InitVar, dataclass, field
import numpy as np
from tqdm.auto import tqdm
from .annotations import Array, ArrayNamespace, Indices, IntArray, NumPy, RealArray, ROI
from .crystfel import Detector
from .cxi_protocol import CXIIndices, CXIStore
from .data_container import Container, D, IndexArray, array_namespace, list_indices, split
from .data_processing import CrystData, CrystMetadata, PCAProjection
from .label import Structure2D, Structure3D
from .parser import Parser, get_parser
from .streaks import StackedStreaks, Streaks
from .src.median import median
from ..indexer.cbc_data import MillerWithRLP, Patterns
from ..indexer.cbc_indexing import CBDIndexer
from ..indexer.cbc_setup import BaseSetup, TiltOverAxisState, XtalList, XtalState

P = TypeVar("P", bound='BaseParameters')

class BaseParameters(Container):
    @classmethod
    def parser(cls, file_or_extension: str='ini') -> Parser:
        return get_parser(file_or_extension, cls, 'parameters')

    @classmethod
    def read(cls: Type[P], file: str) -> P:
        return cls.from_dict(**cls.parser(file).read(file))

@dataclass
class ROIParameters(BaseParameters):
    xmin    : int = 0
    xmax    : int = 0
    ymin    : int = 0
    ymax    : int = 0

    def to_roi(self) -> ROI:
        return (self.ymin, self.ymax, self.xmin, self.xmax)

    def size(self) -> int:
        return max((self.ymax - self.ymin) * (self.xmax - self.xmin), 0)

@dataclass
class StructureParameters(BaseParameters):
    radius          : int
    rank            : int

    @overload
    def to_structure(self, kind: Literal['2d']) -> Structure2D: ...

    @overload
    def to_structure(self, kind: Literal['3d']) -> Structure3D: ...

    def to_structure(self, kind: Literal['2d', '3d']) -> Structure2D | Structure3D:
        if kind == '2d':
            return Structure2D(self.radius, self.rank)
        if kind == '3d':
            return Structure3D(self.radius, self.rank)
        raise ValueError(f"Invalid kind keyword: {kind}")

@dataclass
class MaskParameters(BaseParameters):
    method  : Literal['all-bad', 'no-bad', 'range', 'snr', 'std']
    vmin    : int = 0
    vmax    : int = 65535
    snr_max : float = 3.0
    std_max : float = 0.0
    std_min : float = 0.0

@dataclass
class BackgroundParameters(BaseParameters):
    method  : Literal['mean-poisson', 'median-poisson', 'robust-mean-scale', 'robust-mean-poisson']
    r0      : float = 0.0
    r1      : float = 0.5
    n_iter  : int = 12
    lm      : float = 9.0

@dataclass
class MetadataParameters(BaseParameters):
    mask        : MaskParameters
    background  : BackgroundParameters
    roi         : ROIParameters = field(default_factory=ROIParameters)
    num_threads : int = 1

def create_background(data: CrystData, params: BackgroundParameters, num_threads: int=1,
                      xp: ArrayNamespace=NumPy) -> CrystData:
    if params.method == 'mean-poisson':
        data.whitefield = xp.mean(data.data * data.mask, axis=0)
        data = data.update_std(method='poisson')

    if params.method == 'median-poisson':
        data = data.update_metadata(method='median-poisson')

    if params.method == 'robust-mean-scale':
        data = data.update_metadata(method='robust-mean-scale', r0=params.r0,
                                    r1=params.r1, n_iter=params.n_iter,
                                    lm=params.lm, num_threads=num_threads)

    if params.method == 'robust-mean-poisson':
        data = data.update_metadata(method='robust-mean-poisson', r0=params.r0,
                                    r1=params.r1, n_iter=params.n_iter,
                                    lm=params.lm, num_threads=num_threads)
    return data

def create_metadata(frames: RealArray, params: MetadataParameters) -> CrystMetadata:
    xp = array_namespace(frames)
    data = CrystData(data=frames)

    if params.mask.method == 'all-bad':
        if params.roi.size() == 0:
            raise ValueError("No ROI is provided")
        data = data.update_mask(method='all-bad', roi=params.roi.to_roi())
        data = create_background(data, params.background, params.num_threads, xp)

    elif params.mask.method == 'no-bad':
        data = create_background(data, params.background, params.num_threads, xp)

    elif params.mask.method == 'range':
        data = data.update_mask(method='range', vmin=params.mask.vmin, vmax=params.mask.vmax)
        data = create_background(data, params.background, params.num_threads, xp)

    elif params.mask.method == 'snr':
        data = data.update_mask(method='snr', snr_max=params.mask.snr_max)
        data = create_background(data, params.background, params.num_threads, xp)

    elif params.mask.method == 'std':
        data = create_background(data, params.background, params.num_threads, xp)
        mask = np.ones(data.std.shape, dtype=bool)
        if params.mask.std_min > 0.0:
            mask &= data.std > params.mask.std_min
        if params.mask.std_max > 0.0:
            mask &= data.std < params.mask.std_max
        data = data.import_mask(mask)

    else:
        raise ValueError(f"The mask method keyword is invalid: {params.mask.method}")

    return CrystMetadata(mask=data.mask, std=data.std, whitefield=data.whitefield)

@dataclass
class ScalingParameters(Container):
    method      : Literal['interpolate', 'lsq', 'no-scale', 'robust-lsq']
    good_fields : Tuple[int] = (0,)
    r0          : float = 0.5
    r1          : float = 0.99
    n_iter      : int = 3
    lm          : float = 9.0

@dataclass
class CrystMetafile(Container):
    metapath    : InitVar[str]
    metadata    : CrystMetadata = field(default_factory=CrystMetadata)
    frames      : Dict[str, IndexArray] = field(default_factory=dict)
    ss_idxs     : Indices = slice(None)
    fs_idxs     : Indices = slice(None)

    def __post_init__(self, metapath: str):
        self.metafile = CXIStore(metapath, CrystMetadata.default_protocol())

    @property
    def center_frames(self) -> IntArray:
        if self.metadata.is_empty(self.metadata.data_frames):
            return np.array([], dtype=int)
        return median(self.metadata.data_frames, axis=-1)

    def load(self, attr: str, indices: Indices=slice(None)) -> List[int]:
        xp = self.metadata.__array_namespace__()

        data_indices = self.metafile.indices(attr)
        if len(data_indices) == 0:
            raise ValueError(f"No data found for attribute '{attr}' in the metafile")

        indices = list_indices(indices, len(data_indices))
        if len(indices) > len(data_indices):
            indices = indices[:len(data_indices)]
        new_frames = xp.atleast_1d(data_indices.index[indices])
        default = IndexArray(xp.zeros((0,) + (1,) * (new_frames.ndim - 1), dtype=int))
        frames = self.frames.get(attr, default)

        to, where = frames.insert_index(new_frames)

        if to.size * where.size > 0:
            new_data = self.metafile.load(attr, data_indices[where], ss_idxs=self.ss_idxs,
                                            fs_idxs=self.fs_idxs, verbose=False)
            shape = (frames.size,) + new_data.shape[1:]
            old_data = xp.reshape(getattr(self.metadata, attr), shape)
            setattr(self.metadata, attr, xp.insert(old_data, to, new_data, axis=0))

            frames = xp.insert(xp.asarray(frames), to, new_frames[where], axis=0)
            self.frames[attr] = IndexArray(frames)

        return indices

    def load_all(self, attr: str):
        if attr not in self.metadata.contents():
            data = self.metafile.load(attr, ss_idxs=self.ss_idxs, fs_idxs=self.fs_idxs,
                                      verbose=False)
            setattr(self.metadata, attr, data)

    def import_data(self, data: RealArray, frames: IntArray | int=np.array([], dtype=int),
                    whitefield: RealArray=np.array([])) -> CrystData:
        self.load_all('flatfield')
        self.load_all('mask')
        self.load_all('std')

        return self.metadata.import_data(data, frames, whitefield)

    def interpolate(self, frames: IntArray, num_threads: int=1) -> RealArray:
        self.load('data_frames')

        xp = self.metadata.__array_namespace__()
        frames = xp.searchsorted(self.center_frames, frames)
        all_frames = xp.unique(xp.concatenate([frames, frames + 1]))
        all_frames = all_frames[all_frames < self.center_frames.shape[0]]

        self.load('whitefield', all_frames)
        return self.metadata.interpolate(frames, num_threads)

    def projection(self, data: RealArray, good_fields: Sequence[int]=(0,),
                   method: str="robust-lsq", r0: float=0.0, r1: float=0.5, n_iter: int=12,
                   lm: float=9.0, num_threads: int=1) -> PCAProjection:
        self.load_all('flatfield')
        good_fields = self.load('eigen_field', good_fields)

        indices, _ = self.frames['eigen_field'].get_index(good_fields)

        projection = self.metadata.projection(data, indices, method, r0, r1, n_iter, lm,
                                              num_threads)
        return PCAProjection(good_fields, projection.projection)

    def project(self, projection: PCAProjection) -> RealArray:
        self.load('eigen_field', (projection.good_fields))

        indices, _ = self.frames['eigen_field'].get_index(projection.good_fields)

        return self.metadata.project(PCAProjection(indices, projection.projection))

def scale_background(frames: IntArray, images: Array, metafile: CrystMetafile,
                     params: ScalingParameters, num_threads: int=1):
    if images.ndim < 2:
        raise ValueError("Image array must be at least 2 dimensional")

    if params.method == 'no-scale':
        return metafile.import_data(images, frames)

    if params.method == 'interpolate':
        whitefields = metafile.interpolate(frames, num_threads=num_threads)
        return metafile.import_data(images, frames, whitefields)

    if params.method in ['robust-lsq', 'lsq']:
        projection = metafile.projection(images, params.good_fields, params.method, params.r0,
                                         params.r1, params.n_iter, params.lm, num_threads)
        whitefields = metafile.project(projection)
        return metafile.import_data(images, frames, whitefields)

    raise ValueError(f'Invalid method keyword: {params.method}')

@dataclass
class RegionParameters(Container):
    structure   : StructureParameters
    vmin        : float
    npts        : int

@dataclass
class RegionFinderConfig(BaseParameters):
    regions     : RegionParameters
    scaling     : ScalingParameters
    center      : Tuple[float, float] | None = None
    roi         : ROIParameters = field(default_factory=ROIParameters)
    std_min     : float = 0.0
    num_threads : int = 1

AllStreaks = StackedStreaks | Streaks

def detect_regions(frames: IntArray, images: Array, metafile: CrystMetafile,
                   params: RegionFinderConfig, parallel: bool=True) -> AllStreaks:
    num_threads = params.num_threads if parallel else 1

    data = scale_background(frames, images, metafile, params.scaling, num_threads)

    data = data.update_snr(params.std_min)
    det_obj = data.region_detector(params.regions.structure.to_structure('2d'))
    regions = det_obj.detect_regions(params.regions.vmin, params.regions.npts, num_threads)
    return det_obj.detect_streaks(regions)

@overload
def concentric_only(streaks: Streaks, center: Tuple[float, float],
                    detector: Detector | None=None) -> Streaks: ...

@overload
def concentric_only(streaks: StackedStreaks, center: Tuple[float, float],
                    detector: Detector | None=None) -> StackedStreaks: ...

def concentric_only(streaks: StackedStreaks | Streaks, center: Tuple[float, float],
                    detector: Detector | None=None) -> AllStreaks:
    if isinstance(streaks, StackedStreaks):
        if detector is None:
            err_txt = "Detector geometry must be provided to filter out non-concentric streaks"
            raise ValueError(err_txt)

        if detector.num_modules > 1:
            x, y, _ = detector.to_detector(streaks.module_id, streaks.y, streaks.x)
        else:
            x, y, _ = detector.to_detector(streaks.y, streaks.x)
        mask = Streaks.import_xy(streaks.index, x, y).concentric_only(center[0], center[1])
        return streaks[mask]
    return streaks[streaks.concentric_only(center[0], center[1])]

@dataclass
class PeakParameters(RegionParameters):
    rank        : int | None = None

@dataclass
class StreakParameters(Container):
    structure   : StructureParameters
    xtol        : float
    vmin        : float
    min_size    : int
    nfa         : int
    lookahead   : int

@dataclass
class StreakFinderConfig(BaseParameters):
    peaks       : PeakParameters
    streaks     : StreakParameters
    scaling     : ScalingParameters
    center      : Tuple[float, float] | None = None
    std_min     : float = 0.0
    num_threads : int = 1

def detect_streaks(frames: IntArray, images: Array, metafile: CrystMetafile,
                   params: StreakFinderConfig, parallel: bool=True) -> StackedStreaks | Streaks:
    num_threads = params.num_threads if parallel else 1

    data = scale_background(frames, images, metafile, params.scaling, num_threads)

    data = data.update_snr(params.std_min)
    det_obj = data.streak_detector(params.streaks.structure.to_structure('2d'))
    peaks = det_obj.detect_peaks(params.peaks.vmin, params.peaks.npts,
                                 params.peaks.structure.to_structure('2d'),
                                 rank=params.peaks.rank, num_threads=num_threads)
    return det_obj.detect_streaks(peaks, params.streaks.xtol, params.streaks.vmin,
                                  params.streaks.min_size, params.streaks.lookahead,
                                  nfa=params.streaks.nfa, num_threads=num_threads)

FinderConfig = RegionFinderConfig | StreakFinderConfig
DetectionKind = Literal['regions', 'stacked-streaks', 'streaks']
DetectorFunc = Callable[[IntArray, Array, CrystMetafile, FinderConfig], AllStreaks]

@dataclass
class Detectors:
    detector : Detector | None = None
    parallel : bool = True

    def detect_streaks(self, frames: IntArray, images: Array, metafile: CrystMetafile,
                       params: StreakFinderConfig) -> AllStreaks:
        streaks = detect_streaks(frames, images, metafile, params, self.parallel)
        if params.center is not None:
            streaks = concentric_only(streaks, params.center, self.detector)
        return streaks

    def detect_regions(self, frames: IntArray, images: Array, metafile: CrystMetafile,
                       params: RegionFinderConfig) -> AllStreaks:
        streaks = detect_regions(frames, images, metafile, params, self.parallel)
        if params.center is not None:
            streaks = concentric_only(streaks, params.center, self.detector)
        return streaks

    def get_detector(self, kind: DetectionKind) -> DetectorFunc:
        detectors : dict[DetectionKind, DetectorFunc] = {
            'regions': cast(DetectorFunc, self.detect_regions),
            'streaks': cast(DetectorFunc, self.detect_streaks),
        }

        if kind not in detectors:
            raise ValueError(f'Invalid detection kind: {kind}')

        return detectors[kind]

OptIndices = CXIIndices | None
AnyIndices = OptIndices | Tuple[OptIndices, Indices, Indices]

def get_indices(file: CXIStore, indices: AnyIndices=None) -> Tuple[CXIIndices, Indices, Indices]:
    if indices is None:
        idxs, ss_idxs, fs_idxs = file.indices('data'), slice(None), slice(None)
    elif isinstance(indices, (list, tuple)):
        if len(indices) != 3:
            raise ValueError(f'a tuple of indices has an invalid size: {len(indices)} != 3')

        idxs, ss_idxs, fs_idxs = indices
        if idxs is None:
            idxs = file.indices('data')
    else:
        ss_idxs, fs_idxs = slice(None), slice(None)

    return idxs, ss_idxs, fs_idxs

PreProcessor = Callable[[Array], Array]

def run_detection(file: CXIStore, metapath: str, params: FinderConfig,
                  kind: DetectionKind, indices: AnyIndices=None, chunksize: int=1,
                  detector: Detector | None=None, pre_processor: PreProcessor | None=None,
                  xp: ArrayNamespace=NumPy) -> AllStreaks:
    if 'data' not in file.attributes():
        raise ValueError("No data found in the files")

    idxs, ss_idxs, fs_idxs = get_indices(file, indices)
    metafile = CrystMetafile(metapath, ss_idxs=ss_idxs, fs_idxs=fs_idxs)
    detect = Detectors(detector).get_detector(kind)

    streaks: List[AllStreaks] = []
    for index in tqdm(split(idxs, chunksize), total=int(xp.ceil(len(idxs) / chunksize))):
        data = file.load('data', idxs=index, ss_idxs=ss_idxs, fs_idxs=fs_idxs, verbose=False)
        if pre_processor is not None:
            data = pre_processor(data)
        pattern = detect(xp.atleast_1d(index.index), data, metafile, params)
        streaks.append(pattern.replace(index=pattern.index + int(index.index)))

    if streaks and isinstance(streaks[0], StackedStreaks):
        return StackedStreaks.concatenate(cast(List[StackedStreaks], streaks))
    return Streaks.concatenate(cast(List[Streaks], streaks))

streaks_worker : 'StreaksWorker'

@dataclass
class StreaksWorker():
    data_file       : CXIStore
    metapath        : InitVar[str]
    params          : FinderConfig
    kind            : DetectionKind
    detector        : InitVar[Detector | None] = None
    ss_idxs         : Indices = slice(None)
    fs_idxs         : Indices = slice(None)
    pre_processor   : PreProcessor | None = None

    def __post_init__(self, metapath, detector):
        self.detect = Detectors(detector=detector, parallel=False).get_detector(self.kind)
        self.metafile = CrystMetafile(metapath, ss_idxs=self.ss_idxs, fs_idxs=self.fs_idxs)

    def __call__(self, index: CXIIndices) -> AllStreaks:
        data = self.data_file.load('data', idxs=index, ss_idxs=self.ss_idxs,
                                    fs_idxs=self.fs_idxs, verbose=False)
        xp = array_namespace(data)
        if self.pre_processor is not None:
            data = self.pre_processor(data)
        streaks = self.detect(xp.ravel(index.index), data, self.metafile, self.params)
        return streaks.replace(index=xp.full(streaks.shape[0], int(index.index)))

    @classmethod
    def initialize(cls, data_file: CXIStore, metapath: str, params: FinderConfig,
                   kind: DetectionKind, detector: Detector | None = None,
                   ss_idxs: Indices=slice(None), fs_idxs: Indices=slice(None),
                   pre_processor: PreProcessor | None=None) -> 'StreaksWorker':
        return cls(data_file, metapath, params, kind, detector, ss_idxs, fs_idxs, pre_processor)

    @classmethod
    def initializer(cls, data_file: CXIStore, metapath: str, params: StreakFinderConfig,
                    kind: DetectionKind, detector: Detector | None = None,
                    ss_idxs: Indices=slice(None), fs_idxs: Indices=slice(None),
                    pre_processor: PreProcessor | None=None):
        global streaks_worker
        streaks_worker = cls.initialize(data_file, metapath, params, kind, detector,
                                        ss_idxs, fs_idxs, pre_processor)

    @staticmethod
    def run(index: CXIIndices) -> AllStreaks:
        return streaks_worker(index)

def pool_detection(file: CXIStore, metapath: str, params: FinderConfig,
                   kind: DetectionKind, indices: AnyIndices=None,
                   detector: Detector | None=None, pre_processor: PreProcessor | None=None
                   ) -> AllStreaks:
    if 'data' not in file.attributes():
        raise ValueError("No data found in the files")

    idxs, ss_idxs, fs_idxs = get_indices(file, indices)
    initargs = (file, metapath, params, kind, detector, ss_idxs, fs_idxs, pre_processor)

    streaks : List[AllStreaks] = []
    if params.num_threads > 1:
        with Pool(processes=params.num_threads, initializer=StreaksWorker.initializer,
                  initargs=initargs) as pool:
            for pattern in tqdm(pool.imap(StreaksWorker.run, idxs), total=len(idxs)):
                streaks.append(pattern)
    else:
        worker = StreaksWorker.initialize(*initargs)
        for args in tqdm(idxs, total=len(idxs)):
            streaks.append(worker(args))

    if streaks and isinstance(streaks[0], StackedStreaks):
        return StackedStreaks.concatenate(cast(List[StackedStreaks], streaks))
    return Streaks.concatenate(cast(List[Streaks], streaks))

@dataclass
class IndexingConfig(BaseParameters):
    shape           : Tuple[int, int, int]
    width           : float
    num_threads     : int
    threshold       : float
    n_max           : int
    vicinity        : StructureParameters
    connectivity    : StructureParameters

    def __post_init__(self):
        if not isinstance(self.shape, tuple):
            self.shape = (self.shape[0], self.shape[1], self.shape[2])

def index_patterns(candidates: MillerWithRLP, patterns: Patterns, indexer: CBDIndexer,
                  params: IndexingConfig, state: BaseSetup, parallel: bool=True
                  ) -> Tuple[IntArray, TiltOverAxisState]:
    num_threads = params.num_threads if parallel else 1
    xp = array_namespace(candidates, patterns)
    centers = patterns.sample(xp.full(patterns.shape[0], 0.5))
    points = indexer.points_to_kout(centers, state, xp)
    rotograms = indexer.index(candidates, patterns, points, state)
    rotomap = indexer.rotomap(params.shape, rotograms, params.width, num_threads)
    peaks = indexer.to_peaks(rotomap, params.threshold, params.n_max)
    return indexer.refine_peaks(peaks, rotomap, params.vicinity.to_structure('3d'),
                                params.connectivity.to_structure('3d'), num_threads)

def indexing_candidates(indexer: CBDIndexer, patterns: Patterns, xtal: XtalState, state: BaseSetup,
                        xp: ArrayNamespace=NumPy) -> Iterator[MillerWithRLP]:
    q1, q2 = indexer.patterns_to_q(patterns, state, xp)
    q_max = xp.max((xp.sqrt(xp.sum(q1.q**2, axis=-1)), xp.sqrt(xp.sum(q2.q**2, axis=-1))))
    hkl = indexer.xtal.hkl_in_ball(q_max, xtal, xp)
    return indexer.xtal.hkl_range(patterns.index_array.unique(), hkl, xtal, xp)

def run_indexing(patterns: Patterns, xtals: XtalState, state: BaseSetup, params: IndexingConfig,
             chunksize: int=1, xp: ArrayNamespace=NumPy) -> XtalList:
    indexer = CBDIndexer()
    rlp_iterator = indexing_candidates(indexer, patterns, xtals, state, xp)
    solutions: List[XtalList] = []
    total = len(patterns) // chunksize + (len(patterns) % chunksize > 0)

    if len(xtals) == 1:
        iterator = zip(split(patterns, chunksize), split(rlp_iterator, chunksize))

        for pattern, candidates in tqdm(iterator, total=total):
            idxs, tilts = index_patterns(candidates, pattern, indexer, params, state)
            solution = indexer.solutions(xtals, idxs, tilts, pattern)
            solutions.append(solution)
    elif len(xtals) == len(patterns):
        iterator = zip(split(patterns, chunksize), split(rlp_iterator, chunksize),
                       split(xtals, chunksize))

        for pattern, candidates, xtal in tqdm(iterator, total=total):
            idxs, tilts = index_patterns(candidates, pattern, indexer, params, state)
            solution = indexer.solutions(xtal, idxs, tilts, pattern)
            solutions.append(solution)
    else:
        raise ValueError(f'Number of crystals ({len(xtals):d}) and patterns ({len(patterns):d}) '\
                         'are inconsistent')

    return XtalList.concatenate(solutions)

indexing_worker : 'IndexingWorker'

@dataclass
class IndexingWorker():
    state   : BaseSetup
    params  : IndexingConfig
    xtal    : XtalState
    indexer : CBDIndexer

    def __call__(self, args: Tuple[Dict[str, Any], Dict[str, Any]]) -> XtalList:
        candidates, patterns = MillerWithRLP(**args[0]), Patterns(**args[1])
        idxs, tilts = index_patterns(candidates, patterns, self.indexer, self.params, self.state,
                                     False)
        initial = self.xtal if len(self.xtal) == 1 else self.xtal[idxs]
        return self.indexer.solutions(initial, idxs, tilts, patterns)

    @classmethod
    def initializer(cls, state: BaseSetup, params: IndexingConfig, xtal: XtalState,
                    indexer: CBDIndexer):
        global indexing_worker
        indexing_worker = cls(state, params, xtal, indexer)

    @staticmethod
    def run(args: Tuple[Dict[str, Any], Dict[str, Any]]) -> XtalList:
        return indexing_worker(args)

def dict_range(containers: Iterable[D]) -> Iterator[Dict[str, Any]]:
    for container in containers:
        yield container.to_dict()

def pool_indexing(patterns: Patterns, xtals: XtalState, state: BaseSetup, params: IndexingConfig,
                  xp: ArrayNamespace=NumPy) -> XtalList:
    indexer = CBDIndexer()
    rlp_iterator = indexing_candidates(indexer, patterns, xtals, state, xp)
    solutions : List[XtalList] = []
    with Pool(processes=params.num_threads, initializer=IndexingWorker.initializer,
              initargs=(state, params, xtals, indexer)) as pool:
        iterator = zip(dict_range(rlp_iterator), dict_range(patterns))
        for solution in tqdm(pool.imap(IndexingWorker.run, iterator), total=len(patterns)):
            solutions.append(solution)

    return XtalList.concatenate(solutions)
