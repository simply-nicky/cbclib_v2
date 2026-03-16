from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Iterator, List, Literal, Tuple, Type, overload
from dataclasses import InitVar, dataclass, field
from typing_extensions import Self
from tqdm.auto import tqdm
from .annotations import Array, AnyNamespace, IntArray, NDArray, NumPy, RealArray, ROI
from .array_api import default_api, Platform
from .config import get_cpu_config
from .crystfel import Detector
from .cxi_protocol import H5Handler, LoadWorker, TrainIndices
from .data_container import Container, array_namespace
from .data_processing import CrystData, CrystMetadata
from .functions import Structure
from .parser import from_container, from_file
from .streaks import StackedStreaks, Streaks
from ..indexer.cbc_data import MillerWithRLP, Patterns
from ..indexer.cbc_indexing import CBDIndexer
from ..indexer.cbc_setup import BaseSetup, TiltOverAxisState, XtalList, XtalState

class BaseParameters(Container):
    @classmethod
    def read(cls: Type[Self], file: str) -> Self:
        parser = from_file(file, cls, 'parameters')
        return cls.from_dict(**parser.read(file))

    def write(self, file: str) -> None:
        parser = from_container(file, self, 'parameters')
        parser.write(file, self)

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
    connectivity    : int

    def to_structure(self, rank: int=2) -> Structure:
        return Structure([self.radius,] * rank, self.connectivity)

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

def create_background(data: CrystData, params: BackgroundParameters, xp: AnyNamespace=NumPy
                      ) -> CrystData:
    if params.method == 'mean-poisson':
        data.whitefield = xp.mean(data.data * data.mask, axis=0)
        data = data.update_std(method='poisson')

    if params.method == 'median-poisson':
        data = data.update_metadata(method='median-poisson')

    if params.method == 'robust-mean-scale':
        data = data.update_metadata(method='robust-mean-scale', r0=params.r0,
                                    r1=params.r1, n_iter=params.n_iter, lm=params.lm)

    if params.method == 'robust-mean-poisson':
        data = data.update_metadata(method='robust-mean-poisson', r0=params.r0,
                                    r1=params.r1, n_iter=params.n_iter, lm=params.lm)
    return data

def create_metadata(frames: RealArray, params: MetadataParameters) -> CrystMetadata:
    xp = array_namespace(frames)
    data = CrystData(data=frames)

    if params.mask.method == 'all-bad':
        if params.roi.size() == 0:
            raise ValueError("No ROI is provided")
        data = data.update_mask(method='all-bad', roi=params.roi.to_roi())
        data = create_background(data, params.background, xp)

    elif params.mask.method == 'no-bad':
        data = create_background(data, params.background, xp)

    elif params.mask.method == 'range':
        data = data.update_mask(method='range', vmin=params.mask.vmin, vmax=params.mask.vmax)
        data = create_background(data, params.background, xp)

    elif params.mask.method == 'snr':
        data = data.update_mask(method='snr', snr_max=params.mask.snr_max)
        data = create_background(data, params.background, xp)

    elif params.mask.method == 'std':
        data = create_background(data, params.background, xp)
        mask = xp.ones(data.std.shape, dtype=bool)
        if params.mask.std_min > 0.0:
            mask &= data.std > params.mask.std_min
        if params.mask.std_max > 0.0:
            mask &= data.std < params.mask.std_max
        data = data.import_mask(mask)

    else:
        raise ValueError(f"The mask method keyword is invalid: {params.mask.method}")

    return CrystMetadata(mask=data.mask, std=data.std, flatfield=data.whitefield)

@dataclass
class ScalingParameters(Container):
    method      : Literal['lsq', 'no-scale', 'robust-lsq']
    good_fields : Tuple[int, ...] = (0,)
    r0          : float = 0.5
    r1          : float = 0.99
    n_iter      : int = 3
    lm          : float = 9.0

    def metadata(self, metapath: str, xp: AnyNamespace=NumPy) -> CrystMetadata:
        handler = H5Handler(CrystMetadata.default_protocol())
        idxs = handler.indices(metapath, 'eigen_field')
        eigen_field = handler.load('eigen_field', idxs[self.good_fields],
                                   verbose=False, xp=xp)

        flatfield = handler.load('flatfield', handler.indices(metapath, 'flatfield'),
                                 verbose=False, xp=xp)
        mask = handler.load('mask', handler.indices(metapath, 'mask'), verbose=False, xp=xp)
        std = handler.load('std', handler.indices(metapath, 'std'), verbose=False, xp=xp)
        return CrystMetadata(flatfield=flatfield, mask=mask, std=std,
                             eigen_field=eigen_field)

def scale_background(frames: IntArray | int, images: Array, metadata: CrystMetadata,
                     params: ScalingParameters) -> CrystData:
    if images.ndim < 2:
        raise ValueError("Image array must be at least 2 dimensional")

    if params.method == 'no-scale':
        return metadata.import_data(images, frames)

    if params.method in ['robust-lsq', 'lsq']:
        projection = metadata.projection(images, params.good_fields, params.method, params.r0,
                                         params.r1, params.n_iter, params.lm)
        whitefields = metadata.project(projection)
        return metadata.import_data(images, frames, whitefields)

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

AllStreaks = StackedStreaks | Streaks

def detect_regions(frames: IntArray | int, images: Array, metadata: CrystMetadata,
                   params: RegionFinderConfig) -> AllStreaks:
    data = scale_background(frames, images, metadata, params.scaling)

    data = data.update_snr(params.std_min)
    det_obj = data.region_detector(params.regions.structure.to_structure())
    regions = det_obj.detect_regions(params.regions.vmin, params.regions.npts)
    streaks = det_obj.detect_streaks(regions)
    if isinstance(frames, int):
        return streaks.replace(index=streaks.index + frames)
    return streaks.replace(index=frames[streaks.index])

@overload
def concentric_only(streaks: Streaks, center: Tuple[float, float],
                    detector: Detector | None=None) -> Streaks: ...

@overload
def concentric_only(streaks: StackedStreaks, center: Tuple[float, float],
                    detector: Detector | None=None) -> StackedStreaks: ...

def concentric_only(streaks: StackedStreaks | Streaks, center: Tuple[float, float],
                    detector: Detector | None=None) -> AllStreaks:
    if detector is None:
        if isinstance(streaks, StackedStreaks):
            raise ValueError("Detector must be provided to apply the detector geometry")
        return streaks[streaks.concentric_only(center[0], center[1])]

    if detector.num_modules > 1:
        x, y, _ = detector.to_detector(streaks.module_id, streaks.y, streaks.x)
    else:
        x, y, _ = detector.to_detector(streaks.y, streaks.x)
    mask = Streaks.import_xy(streaks.index, x, y).concentric_only(center[0], center[1])
    return streaks[mask]


@dataclass
class PeakParameters(RegionParameters):
    radius      : int | None = None

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

def detect_streaks(frames: IntArray | int, images: Array, metadata: CrystMetadata,
                   params: StreakFinderConfig) -> StackedStreaks | Streaks:
    data = scale_background(frames, images, metadata, params.scaling)

    data = data.update_snr(params.std_min)
    det_obj = data.streak_detector(params.streaks.structure.to_structure())
    peaks = det_obj.detect_peaks(params.peaks.vmin, params.peaks.npts,
                                 params.peaks.structure.to_structure())
    streaks = det_obj.detect_streaks(peaks, params.streaks.xtol, params.streaks.vmin,
                                     params.streaks.min_size, params.streaks.lookahead,
                                     nfa=params.streaks.nfa)
    if isinstance(frames, int):
        return streaks.replace(index=streaks.index + frames)
    return streaks.replace(index=frames[streaks.index])

DetectionFunc = Callable[[IntArray | int, Array, CrystMetadata], AllStreaks]
FinderConfig = RegionFinderConfig | StreakFinderConfig

def detect_patterns(params: FinderConfig) -> DetectionFunc:
    if isinstance(params, StreakFinderConfig):
        return partial(detect_streaks, params=params)
    if isinstance(params, RegionFinderConfig):
        return partial(detect_regions, params=params)
    raise ValueError(f'Invalid parameters type: {type(params)}')

def run_detection(loader: LoadWorker[NDArray], indices: TrainIndices, metapath: str,
                  params: FinderConfig, platform: Platform = 'cpu',
                  detector: Detector | None=None) -> AllStreaks:
    xp = default_api(platform)
    metadata = params.scaling.metadata(metapath, xp)
    detect = detect_patterns(params)

    streaks = []
    for frame, index in tqdm(zip(indices.index(), indices), total=len(indices),
                             desc='Detecting patterns'):
        data = xp.asarray(loader(index))
        pattern = detect(frame, data, metadata)
        if params.center is not None:
            pattern = concentric_only(pattern, params.center, detector)
        streaks.append(pattern)

    if streaks and isinstance(streaks[0], StackedStreaks):
        return StackedStreaks.concatenate(streaks)
    return Streaks.concatenate(streaks)

streaks_worker : 'StreaksWorker'

@dataclass
class StreaksWorker(LoadWorker[AllStreaks]):
    loader          : LoadWorker[NDArray]
    metapath        : InitVar[str]
    params          : FinderConfig
    platform        : InitVar[Platform]
    detector        : Detector | None

    def __post_init__(self, metapath, platform):
        self.xp = default_api(platform)
        self.metafile = self.params.scaling.metadata(metapath, self.xp)
        self.detect = detect_patterns(self.params)

    def __call__(self, indices: Tuple[int, Any]) -> AllStreaks:
        frame, index = indices
        data = self.xp.asarray(self.loader(index))
        pattern = self.detect(frame, data, self.metafile)
        if self.params.center is not None:
            pattern = concentric_only(pattern, self.params.center, self.detector)
        return pattern

    @classmethod
    def initializer(cls, loader: LoadWorker[NDArray], metapath: str, params: StreakFinderConfig,
                    platform: Platform, detector: Detector | None):
        global streaks_worker
        streaks_worker = cls(loader, metapath, params, platform, detector)

    @staticmethod
    def run(index: Tuple[int, Any]) -> AllStreaks:
        return streaks_worker(index)

def pool_detection(loader: LoadWorker[NDArray], indices: TrainIndices, metapath: str,
                   params: FinderConfig, platform: Platform = 'cpu', detector: Detector | None=None
                   ) -> AllStreaks:
    num_threads = get_cpu_config().effective_num_threads()
    initargs = (loader, metapath, params, platform, detector)

    streaks = []
    if platform == 'cpu' and num_threads > 1:
        with Pool(processes=num_threads, initializer=StreaksWorker.initializer,
                  initargs=initargs) as pool:
            for pattern in tqdm(pool.imap(StreaksWorker.run, zip(indices.index(), indices)),
                                total=len(indices)):
                streaks.append(pattern)
    else:
        worker = StreaksWorker(*initargs)
        for frame, index in tqdm(zip(indices.index(), indices), total=len(indices)):
            streaks.append(worker((frame, index)))

    if streaks and isinstance(streaks[0], StackedStreaks):
        return StackedStreaks.concatenate(streaks)
    return Streaks.concatenate(streaks)

@dataclass
class IndexingConfig(BaseParameters):
    shape           : Tuple[int, int, int]
    width           : float
    threshold       : float
    n_max           : int
    vicinity        : StructureParameters
    connectivity    : StructureParameters

    def __post_init__(self):
        if not isinstance(self.shape, tuple):
            self.shape = (self.shape[0], self.shape[1], self.shape[2])

def index_patterns(candidates: MillerWithRLP, patterns: Patterns, indexer: CBDIndexer,
                   params: IndexingConfig, state: BaseSetup
                   ) -> Tuple[IntArray, TiltOverAxisState]:
    xp = array_namespace(candidates, patterns)
    centers = patterns.sample(xp.full(patterns.shape[0], 0.5))
    points = indexer.points_to_kout(centers, state, xp)
    rotograms = indexer.index(candidates, patterns, points, state)
    rotomap = indexer.rotomap(params.shape, rotograms, patterns.index, params.width)
    peaks = indexer.to_peaks(rotomap, params.threshold, params.n_max)
    return indexer.refine_peaks(peaks, rotomap, params.vicinity.to_structure(3),
                                params.connectivity.to_structure(3))

def indexing_candidates(indexer: CBDIndexer, patterns: Patterns, xtal: XtalState, state: BaseSetup,
                        xp: AnyNamespace=NumPy) -> Iterator[MillerWithRLP]:
    q1, q2 = indexer.patterns_to_q(patterns, state, xp)
    q_max = xp.max((xp.sqrt(xp.sum(q1.q**2, axis=-1)), xp.sqrt(xp.sum(q2.q**2, axis=-1))))
    hkl = indexer.xtal.hkl_in_ball(q_max, xtal, xp)
    return indexer.xtal.hkl_range(patterns.index_array.unique(), hkl, xtal, xp)

def run_indexing(patterns: Patterns, xtals: XtalState, state: BaseSetup, params: IndexingConfig,
                 xp: AnyNamespace=NumPy) -> XtalList:
    indexer = CBDIndexer()
    rlp_iterator = indexing_candidates(indexer, patterns, xtals, state, xp)
    solutions: List[XtalList] = []

    if len(xtals) == 1:
        iterator = zip(patterns, rlp_iterator)

        for pattern, candidates in tqdm(iterator, total=len(patterns)):
            idxs, tilts = index_patterns(candidates, pattern, indexer, params, state)
            solution = indexer.solutions(xtals, idxs, tilts, pattern)
            solutions.append(solution)
    elif len(xtals) == len(patterns):
        iterator = zip(patterns, rlp_iterator, xtals)

        for pattern, candidates, xtal in tqdm(iterator, total=len(patterns)):
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

    def __call__(self, args: Tuple[MillerWithRLP, Patterns]) -> XtalList:
        candidates, patterns = args
        idxs, tilts = index_patterns(candidates, patterns, self.indexer, self.params, self.state)
        initial = self.xtal if len(self.xtal) == 1 else self.xtal[idxs]
        return self.indexer.solutions(initial, idxs, tilts, patterns)

    @classmethod
    def initializer(cls, state: BaseSetup, params: IndexingConfig, xtal: XtalState,
                    indexer: CBDIndexer):
        global indexing_worker
        indexing_worker = cls(state, params, xtal, indexer)

    @staticmethod
    def run(args: Tuple[MillerWithRLP, Patterns]) -> XtalList:
        return indexing_worker(args)

def pool_indexing(patterns: Patterns, xtals: XtalState, state: BaseSetup, params: IndexingConfig,
                  platform: Platform = 'cpu', xp: AnyNamespace=NumPy) -> XtalList:
    num_threads = get_cpu_config().effective_num_threads()
    indexer = CBDIndexer()
    rlp_iterator = indexing_candidates(indexer, patterns, xtals, state, xp)

    solutions : List[XtalList] = []
    if platform == 'cpu' and num_threads > 1:
        with Pool(processes=num_threads, initializer=IndexingWorker.initializer,
                  initargs=(state, params, xtals, indexer)) as pool:
            iterator = zip(rlp_iterator, patterns)
            for solution in tqdm(pool.imap(IndexingWorker.run, iterator), total=len(patterns)):
                solutions.append(solution)
    else:
        worker = IndexingWorker(state, params, xtals, indexer)
        for candidates, pattern in tqdm(zip(rlp_iterator, patterns), total=len(patterns)):
            solutions.append(worker((candidates, pattern)))

    return XtalList.concatenate(solutions)
