from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Tuple, Type, TypeVar, overload
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from .annotations import Array, ArrayNamespace, BoolArray, IntArray, NumPy, ReadOut, RealArray, ROI
from .cxi_protocol import FileStore
from .data_container import ArrayContainer, Container, D, IndexArray, array_namespace, split
from .data_processing import CrystData, RegionDetector, Streaks
from .label import Structure2D, Structure3D, Regions2D
from .parser import JSONParser, INIParser, Parser
from ..indexer.cbc_data import MillerWithRLP, Patterns
from ..indexer.cbc_indexing import CBDIndexer
from ..indexer.cbc_setup import BaseSetup, TiltOverAxisState, XtalList, XtalState

P = TypeVar("P", bound='BaseParameters')

class BaseParameters(Container):
    @classmethod
    def parser(cls, ext: str='ini') -> Parser:
        if ext == 'ini':
            return INIParser.from_container(cls, default='parameters')
        if ext == 'json':
            return JSONParser.from_container(cls, default='parameters')
        raise ValueError(f"Invalid format: {ext}")

    @classmethod
    def read(cls: Type[P], file: str, ext: str='ini') -> P:
        return cls.from_dict(**cls.parser(ext).read(file))

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
class CrystMetadata(ArrayContainer):
    mask        : BoolArray
    std         : RealArray
    whitefield  : RealArray

@dataclass
class MaskParameters(BaseParameters):
    method  : Literal['all-bad', 'no-bad', 'range', 'snr', 'std']
    vmin    : int = 0
    vmax    : int = 65535
    snr_max : float = 3.0
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

def create_metadata(frames: RealArray, params: MetadataParameters) -> CrystMetadata:
    xp = array_namespace(frames)
    data = CrystData(data=frames)

    if params.mask.method == 'all-bad':
        if params.roi.size() == 0:
            raise ValueError("No ROI is provided")
        data = data.update_mask(method='all-bad', roi=params.roi.to_roi())

    if params.mask.method == 'range':
        data = data.update_mask(method='range', vmin=params.mask.vmin, vmax=params.mask.vmax)

    if params.background.method == 'mean-poisson':
        data.whitefield = xp.mean(data.data * data.mask, axis=0)
        data = data.update_std(method='poisson')

    if params.background.method == 'median-poisson':
        data.whitefield = xp.median(data.data * data.mask, axis=0)
        data = data.update_std(method='poisson')

    if params.background.method == 'robust-mean-scale':
        data = data.update_whitefield(method='robust-mean-scale', r0=params.background.r0,
                                      r1=params.background.r1, n_iter=params.background.n_iter,
                                      lm=params.background.lm, num_threads=params.num_threads)

    if params.background.method == 'robust-mean-poisson':
        data = data.update_whitefield(method='robust-mean', r0=params.background.r0,
                                      r1=params.background.r1, n_iter=params.background.n_iter,
                                      lm=params.background.lm, num_threads=params.num_threads)
        data = data.update_std(method='poisson')

    if params.mask.method == 'snr':
        data = data.update_mask(method='snr', snr_max=params.mask.snr_max)

    if params.mask.method == 'std':
        data = data.import_mask(data.std > params.mask.std_min)

    return CrystMetadata(data.mask, data.std, data.whitefield)

@dataclass
class RegionParameters(Container):
    structure   : StructureParameters
    vmin        : float
    npts        : int

@dataclass
class RegionFinderParameters(BaseParameters):
    regions     : RegionParameters
    num_threads : int

def find_regions(frames: Array, metadata: CrystMetadata, params: RegionFinderParameters
                 ) -> Tuple[RegionDetector, List[Regions2D]]:
    if frames.ndim < 2:
        raise ValueError("Frame array must be at least 2 dimensional")
    data = CrystData(data=frames.reshape((-1,) + frames.shape[-2:]), mask=metadata.mask,
                     std=metadata.std, whitefield=metadata.whitefield)
    data = data.scale_whitefield(method='median', num_threads=params.num_threads)
    data = data.update_snr()
    det_obj = data.region_detector(params.regions.structure.to_structure('2d'))
    regions = det_obj.detect_regions(params.regions.vmin, params.regions.npts, params.num_threads)
    return det_obj, regions

@dataclass
class PatternRecognitionParameters(RegionFinderParameters):
    threshold   : float

def pattern_recognition(metadata: CrystMetadata, params: PatternRecognitionParameters,
                        xp: ArrayNamespace=NumPy) -> Callable[[Array], ReadOut]:
    def pattern_goodness(frames: Array) -> Tuple[float, float]:
        det_obj, regions = find_regions(frames, metadata, params)
        masses = det_obj.total_mass(regions)[0]
        fits = det_obj.ellipse_fit(regions)[0]
        if fits.size:
            values = xp.tanh((fits[:, 0] / fits[:, 1] - params.threshold)) * masses
            positive, negative = xp.sum(values[values > 0]), -xp.sum(values[values < 0])
            return (float(positive), float(negative))
        return (0.0, 0.0)

    return pattern_goodness

@dataclass
class StreakParameters(Container):
    structure   : StructureParameters
    xtol        : float
    vmin        : float
    min_size    : int
    nfa         : int

@dataclass
class StreakFinderParameters(BaseParameters):
    peaks               : RegionParameters
    streaks             : StreakParameters
    center              : Tuple[float, float] | None = None
    roi                 : ROIParameters = field(default_factory=ROIParameters)
    scale_whitefield    : bool = False
    num_threads         : int = 1

def detect_streaks_script(frames: Array, metadata: CrystMetadata, params: StreakFinderParameters,
                          parallel: bool=True) -> Streaks:
    num_threads = params.num_threads if parallel else 1
    if frames.ndim < 2:
        raise ValueError("Frame array must be at least 2 dimensional")
    data = CrystData(data=frames.reshape((-1,) + frames.shape[-2:]), mask=metadata.mask,
                     std=metadata.std, whitefield=metadata.whitefield)
    if params.roi.size():
        data = data.crop(params.roi.to_roi())
    if params.scale_whitefield:
        data = data.scale_whitefield(method='median', num_threads=num_threads)
    data = data.update_snr()
    det_obj = data.streak_detector(params.streaks.structure.to_structure('2d'))
    peaks = det_obj.detect_peaks(params.peaks.vmin, params.peaks.npts,
                                 params.peaks.structure.to_structure('2d'), num_threads)
    detected = det_obj.detect_streaks(peaks, params.streaks.xtol, params.streaks.vmin,
                                      params.streaks.min_size, nfa=params.streaks.nfa,
                                      num_threads=num_threads)
    streaks = det_obj.to_streaks(detected)
    if params.center is not None:
        mask = streaks.concentric_only(params.center[0], params.center[1])
        streaks = streaks[mask]
    return streaks

PreProcessor = Callable[[Array,], Array]

def run_detect_streaks(file: FileStore, metadata: CrystMetadata, params: StreakFinderParameters,
                       frames: IntArray | None=None, chunksize: int=1,
                       pre_processor: PreProcessor | None=None) -> Streaks:
    xp = array_namespace(frames, metadata)
    if 'data' not in file.attributes():
        raise ValueError("No data found in the files")
    if frames is None:
        frames = xp.arange(file.size)

    streaks = []
    for frame in tqdm(split(frames, chunksize), total=int(xp.ceil(frames.shape[0] / chunksize))):
        data = file.load('data', idxs=frame, verbose=False)
        if pre_processor is not None:
            data = pre_processor(data)
        pattern = detect_streaks_script(data, metadata, params)
        streaks.append(pattern.replace(index=pattern.index + frame[0]))
    return Streaks.concatenate(streaks)

detect_worker : 'DetectionWorker'

@dataclass
class DetectionWorker():
    input_file      : FileStore
    metadata        : CrystMetadata
    params          : StreakFinderParameters
    pre_processor   : PreProcessor | None = None

    def __call__(self, args: IntArray) -> Tuple[IntArray, RealArray]:
        data = self.input_file.load('data', idxs=args, verbose=False)
        if self.pre_processor is not None:
            data = self.pre_processor(data)
        streaks = detect_streaks_script(data, self.metadata, self.params, False)
        xp = array_namespace(streaks)
        return (xp.full(streaks.shape[0], args), streaks.lines)

    @classmethod
    def initializer(cls, input_file: FileStore, metadata: CrystMetadata,
                    params: StreakFinderParameters, pre_processor: PreProcessor | None=None):
        global detect_worker
        detect_worker = cls(input_file, metadata, params, pre_processor)

    @staticmethod
    def run(args: Array) -> Tuple[IntArray, RealArray]:
        return detect_worker(args)

def run_detect_streaks_pool(file: FileStore, metadata: CrystMetadata,
                            params: StreakFinderParameters, frames: IntArray | None=None,
                            pre_processor: PreProcessor | None=None) -> Streaks:
    xp = array_namespace(frames, metadata)
    if 'data' not in file.attributes():
        raise ValueError("No data found in the files")
    if frames is None:
        frames = xp.arange(file.size)

    streaks = []
    if params.num_threads > 1:
        with Pool(processes=params.num_threads, initializer=DetectionWorker.initializer,
                initargs=(file, metadata, params, pre_processor)) as pool:
            for index, lines in tqdm(pool.imap(DetectionWorker.run, frames),
                                     total=frames.shape[0]):
                streaks.append(Streaks(index=IndexArray(index), lines=lines))
    else:
        worker = DetectionWorker(file, metadata, params, pre_processor)
        for frame in tqdm(frames, total=frames.size):
            index, lines = worker(frame)
            streaks.append(Streaks(index=IndexArray(index), lines=lines))
    return Streaks.concatenate(streaks)

@dataclass
class CBDIndexingParameters(BaseParameters):
    patterns_file   : str
    output_file     : str
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

def pre_indexing(candidates: MillerWithRLP, patterns: Patterns, indexer: CBDIndexer,
                 params: CBDIndexingParameters, state: BaseSetup, parallel: bool=True
                 ) -> Tuple[IntArray, TiltOverAxisState]:
    num_threads = params.num_threads if parallel else 1
    xp = array_namespace(candidates, patterns)
    centers = patterns.sample(xp.full(patterns.shape[0], 0.5))
    points = indexer.points_to_kout(centers, state, xp)
    rotograms = indexer.index(candidates, patterns, points, state)
    rotomap = indexer.rotomap(params.shape, rotograms, params.width, num_threads)
    peaks = indexer.to_peaks(rotomap, params.threshold)
    return indexer.refine_peaks(peaks, rotomap, params.vicinity.to_structure('3d'),
                                params.connectivity.to_structure('3d'), num_threads)

def indexing_candidates(indexer: CBDIndexer, patterns: Patterns, xtal: XtalState, state: BaseSetup,
                        xp: ArrayNamespace=NumPy) -> Iterator[MillerWithRLP]:
    q1, q2 = indexer.patterns_to_q(patterns, state, xp)
    q_max = xp.max((xp.sqrt(xp.sum(q1.q**2, axis=-1)), xp.sqrt(xp.sum(q2.q**2, axis=-1))))
    hkl = indexer.xtal.hkl_in_ball(q_max, xtal, xp)
    return indexer.xtal.hkl_range(patterns.index.unique(), hkl, xtal, xp)

def run_pre_indexing(patterns: Patterns, xtals: XtalState, state: BaseSetup,
                     params: CBDIndexingParameters, chunksize: int=1, xp: ArrayNamespace=NumPy
                     ) -> XtalList:
    indexer = CBDIndexer()
    rlp_iterator = indexing_candidates(indexer, patterns, xtals, state, xp)
    solutions: List[XtalList] = []
    total = len(patterns) // chunksize + (len(patterns) % chunksize > 0)

    if len(xtals) == 1:
        iterator = zip(split(patterns, chunksize), split(rlp_iterator, chunksize))

        for pattern, candidates in tqdm(iterator, total=total):
            idxs, tilts = pre_indexing(candidates, pattern, indexer, params, state)
            solution = indexer.solutions(xtals, idxs, tilts, pattern)
            solutions.append(solution)
    elif len(xtals) == len(patterns):
        iterator = zip(split(patterns, chunksize), split(rlp_iterator, chunksize),
                       split(xtals, chunksize))

        for pattern, candidates, xtal in tqdm(iterator, total=total):
            idxs, tilts = pre_indexing(candidates, pattern, indexer, params, state)
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
    params  : CBDIndexingParameters
    xtal    : XtalState
    indexer : CBDIndexer

    def __call__(self, args: Tuple[Dict[str, Any], Dict[str, Any]]) -> XtalList:
        candidates, patterns = MillerWithRLP(**args[0]), Patterns(**args[1])
        idxs, tilts = pre_indexing(candidates, patterns, self.indexer, self.params, self.state,
                                   False)
        initial = self.xtal if len(self.xtal) == 1 else self.xtal[idxs]
        return self.indexer.solutions(initial, idxs, tilts, patterns)

    @classmethod
    def initializer(cls, state: BaseSetup, params: CBDIndexingParameters, xtal: XtalState,
                    indexer: CBDIndexer):
        global indexing_worker
        indexing_worker = cls(state, params, xtal, indexer)

    @staticmethod
    def run(args: Tuple[Dict[str, Any], Dict[str, Any]]) -> XtalList:
        return indexing_worker(args)

def dict_range(containers: Iterable[D]) -> Iterator[Dict[str, Any]]:
    for container in containers:
        yield container.to_dict()

def run_pre_indexing_pool(patterns: Patterns, xtals: XtalState, state: BaseSetup,
                          params: CBDIndexingParameters, xp: ArrayNamespace=NumPy) -> XtalList:
    indexer = CBDIndexer()
    rlp_iterator = indexing_candidates(indexer, patterns, xtals, state, xp)
    solutions : List[XtalList] = []
    with Pool(processes=params.num_threads, initializer=IndexingWorker.initializer,
              initargs=(state, params, xtals, indexer)) as pool:
        iterator = zip(dict_range(rlp_iterator), dict_range(patterns))
        for solution in tqdm(pool.imap(IndexingWorker.run, iterator), total=len(patterns)):
            solutions.append(solution)

    return XtalList.concatenate(solutions)
