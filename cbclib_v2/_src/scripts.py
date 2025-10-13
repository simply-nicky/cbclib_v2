from multiprocessing import Pool
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Literal, Sequence, Sized, Tuple, Type,
                    TypeVar, overload)
from dataclasses import InitVar, dataclass, field
import numpy as np
from tqdm.auto import tqdm

from .annotations import (Array, ArrayNamespace, Indices, IntArray, IntSequence, NumPy, RealArray,
                          ROI)
from .crystfel import Detector
from .cxi_protocol import CXIStore, DataIndices, FileStore, Kinds
from .data_container import Container, D, IndexArray, array_namespace, split, to_list
from .data_processing import CrystData, CrystMetadata, RegionDetector, Streaks
from .label import Structure2D, Structure3D, Regions2D
from .parser import JSONParser, INIParser, Parser
from .src.median import median
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

    if params.mask.method == 'range':
        data = data.update_mask(method='range', vmin=params.mask.vmin, vmax=params.mask.vmax)

    if params.mask.method == 'snr':
        data = data.update_mask(method='snr', snr_max=params.mask.snr_max)

    data = create_background(data, params.background, params.num_threads, xp)

    if params.mask.method == 'std':
        mask = np.ones(data.std.shape, dtype=bool)
        if params.mask.std_min > 0.0:
            mask &= data.std > params.mask.std_min
        if params.mask.std_max > 0.0:
            mask &= data.std < params.mask.std_max
        data = data.import_mask(mask)

    return CrystMetadata(mask=data.mask, std=data.std, whitefield=data.whitefield)

@dataclass
class ScalingParameters(Container):
    method      : Literal['interpolate', 'lsq', 'no-scale', 'robust-lsq']
    num_fields  : int = 1
    r0          : float = 0.5
    r1          : float = 0.99
    n_iter      : int = 3
    lm          : float = 9.0

@dataclass
class CrystMetafile(Container):
    input_path  : InitVar[str]
    metadata    : CrystMetadata = field(default_factory=CrystMetadata)
    frames      : Dict[str, IndexArray] = field(default_factory=dict)
    ss_idxs     : Indices = slice(None)
    fs_idxs     : Indices = slice(None)

    def __post_init__(self, input_path: str):
        self.input_file = CXIStore(input_path, CrystMetadata.protocol)
        self.center_frames = median(self.input_file.load('data_frames', verbose=False), axis=-1)

    def load(self, attr: str, indices: int | slice | IntArray | Sequence[int]=slice(None)):
        if self.metadata.protocol.get_kind(attr) in (Kinds.sequence, Kinds.stack):
            xp = self.metadata.__array_namespace__()

            data_indices = self.input_file.indices(attr)
            new_frames = xp.atleast_1d(data_indices.indices[indices])
            frames = self.frames.get(attr, IndexArray(np.array([], dtype=int)))

            to, where = frames.insert_index(new_frames)

            if to.size * where.size > 0:
                new_data = self.input_file.load(attr, data_indices[where], ss_idxs=self.ss_idxs,
                                                fs_idxs=self.fs_idxs, verbose=False)
                shape = (frames.size,) + new_data.shape[1:]
                old_data = xp.reshape(getattr(self.metadata, attr), shape)
                setattr(self.metadata, attr, xp.insert(old_data, to, new_data, axis=0))

                self.frames[attr] = IndexArray(xp.insert(xp.asarray(frames), to, new_frames[where]))
        else:
            data = self.input_file.load(attr, ss_idxs=self.ss_idxs, fs_idxs=self.fs_idxs,
                                        verbose=False)
            setattr(self.metadata, attr, data)

    def import_data(self, data: RealArray, frames: IntArray | int=np.array([], dtype=int),
                    whitefield: RealArray=np.array([])) -> CrystData:
        self.load('mask')
        self.load('std')

        return self.metadata.import_data(data, frames, whitefield)

    def interpolate(self, frames: IntArray, num_threads: int=1) -> RealArray:
        xp = self.metadata.__array_namespace__()
        frames = xp.searchsorted(self.center_frames, frames)
        all_frames = xp.unique(xp.concatenate([frames, frames + 1]))
        all_frames = all_frames[all_frames < self.center_frames.shape[0]]

        self.load('whitefield', all_frames)
        self.load('data_frames', all_frames)
        return self.metadata.interpolate(frames, num_threads)

    def projection(self, data: RealArray, num_fields: int, method: str="robust-lsq",
                   r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0,
                   num_threads: int=1) -> RealArray:
        self.load('eigen_field', list(range(num_fields)))

        return self.metadata.projection(data, num_fields, method, r0, r1, n_iter, lm, num_threads)

    def project(self, projection: RealArray) -> RealArray:
        self.load('eigen_field', list(range(projection.shape[-1])))

        return self.metadata.project(projection)

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
        projection = metafile.projection(images, params.num_fields, params.method, params.r0,
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
class RegionFinderParameters(BaseParameters):
    regions     : RegionParameters
    scaling     : ScalingParameters
    roi         : ROIParameters = field(default_factory=ROIParameters)
    std_min     : float = 0.0
    num_threads : int = 1

def find_regions(frames: IntArray, images: Array, metafile: CrystMetafile,
                 params: RegionFinderParameters, parallel: bool=True
                 ) -> Tuple[RegionDetector, List[Regions2D]]:
    num_threads = params.num_threads if parallel else 1

    data = scale_background(frames, images, metafile, params.scaling, num_threads)

    data = data.update_snr(params.std_min)
    det_obj = data.region_detector(params.regions.structure.to_structure('2d'))
    regions = det_obj.detect_regions(params.regions.vmin, params.regions.npts, num_threads)
    return det_obj, regions

@dataclass
class PeakParameters(RegionParameters):
    rank        : int = 2

@dataclass
class StreakParameters(Container):
    structure   : StructureParameters
    xtol        : float
    vmin        : float
    min_size    : int
    nfa         : int
    lookahead   : int

@dataclass
class StreakFinderParameters(BaseParameters):
    peaks       : PeakParameters
    streaks     : StreakParameters
    scaling     : ScalingParameters
    center      : Tuple[float, float] | None = None
    std_min     : float = 0.0
    num_threads : int = 1

def detect_streaks_script(frames: IntArray, images: Array, metafile: CrystMetafile,
                          params: StreakFinderParameters, detector: Detector | None=None,
                          parallel: bool=True) -> Streaks:
    num_threads = params.num_threads if parallel else 1

    data = scale_background(frames, images, metafile, params.scaling, num_threads)

    data = data.update_snr(params.std_min)
    det_obj = data.streak_detector(params.streaks.structure.to_structure('2d'))
    peaks = det_obj.detect_peaks(params.peaks.vmin, params.peaks.npts,
                                 params.peaks.structure.to_structure('2d'),
                                 rank=params.peaks.rank, num_threads=num_threads)
    detected = det_obj.detect_streaks(peaks, params.streaks.xtol, params.streaks.vmin,
                                      params.streaks.min_size, params.streaks.lookahead,
                                      nfa=params.streaks.nfa, num_threads=num_threads)
    streaks = det_obj.to_streaks(detected)
    if params.center is not None:
        if detector is not None and data.shape[-2:] == detector.shape:
            x, y, _ = detector.to_detector(streaks.y, streaks.x)
            streaks = Streaks.import_xy(streaks.index, x, y)
        mask = streaks.concentric_only(params.center[0], params.center[1])
        streaks = streaks[mask]
    return streaks

PreProcessor = Callable[[Array,], Array]
OptIntSequence = IntSequence | None

def run_detect_streaks(file: FileStore, metapath: str, params: StreakFinderParameters,
                       indices: OptIntSequence | Tuple[OptIntSequence, Indices, Indices]=None,
                       chunksize: int=1, detector: Detector | None=None,
                       pre_processor: PreProcessor | None=None, xp: ArrayNamespace=NumPy
                       ) -> Streaks:
    if 'data' not in file.attributes():
        raise ValueError("No data found in the files")
    idxs = file.indices('data')

    if indices is None:
        frames, ss_idxs, fs_idxs = list(range(len(idxs))), slice(None), slice(None)
    elif isinstance(indices, Sized) and len(indices) == 3:
        frames, ss_idxs, fs_idxs = indices
        if frames is None:
            frames = list(range(len(idxs)))
        else:
            frames = to_list(frames)
    else:
        frames = to_list(indices)
        ss_idxs, fs_idxs = slice(None), slice(None)

    metafile = CrystMetafile(metapath, ss_idxs=ss_idxs, fs_idxs=fs_idxs)

    streaks = []
    for index, frame in tqdm(zip(split(idxs, chunksize), split(frames, chunksize)),
                             total=int(xp.ceil(len(idxs) / chunksize))):
        data = file.load('data', idxs=index, ss_idxs=ss_idxs, fs_idxs=fs_idxs, verbose=False)
        if pre_processor is not None:
            data = pre_processor(data)
        pattern = detect_streaks_script(xp.asarray(frame), data, metafile, params, detector)
        streaks.append(pattern.replace(index=pattern.index + frame[0]))
    return Streaks.concatenate(streaks)

detect_worker : 'DetectionWorker'

@dataclass
class DetectionWorker():
    input_file      : FileStore
    metapath        : InitVar[str]
    params          : StreakFinderParameters
    ss_idxs         : Indices = slice(None)
    fs_idxs         : Indices = slice(None)
    detector        : Detector | None = None
    pre_processor   : PreProcessor | None = None

    def __post_init__(self, metapath):
        self.metafile = CrystMetafile(metapath, ss_idxs=self.ss_idxs, fs_idxs=self.fs_idxs)

    def __call__(self, args: Tuple[DataIndices, Any]) -> Streaks:
        index, frame = args
        data = self.input_file.load('data', idxs=index, ss_idxs=self.ss_idxs,
                                    fs_idxs=self.fs_idxs, verbose=False)
        xp = array_namespace(data)
        if self.pre_processor is not None:
            data = self.pre_processor(data)
        streaks = detect_streaks_script(xp.atleast_1d(frame), data, self.metafile, self.params,
                                        self.detector, False)
        return streaks.replace(index=xp.full(streaks.shape[0], frame))

    @classmethod
    def initialize(cls, input_file: FileStore, metapath: str, params: StreakFinderParameters,
                   ss_idxs: Indices=slice(None), fs_idxs: Indices=slice(None),
                   detector: Detector | None=None, pre_processor: PreProcessor | None=None
                   ) -> 'DetectionWorker':
        return cls(input_file, metapath, params, ss_idxs, fs_idxs, detector, pre_processor)

    @classmethod
    def initializer(cls, input_file: FileStore, metapath: str, params: StreakFinderParameters,
                    ss_idxs: Indices=slice(None), fs_idxs: Indices=slice(None),
                    detector: Detector | None=None, pre_processor: PreProcessor | None=None):
        global detect_worker
        detect_worker = cls.initialize(input_file, metapath, params, ss_idxs, fs_idxs, detector,
                                       pre_processor)

    @staticmethod
    def run(args: Tuple[DataIndices, Any]) -> Streaks:
        return detect_worker(args)

def run_detect_streaks_pool(file: FileStore, metapath: str, params: StreakFinderParameters,
                            indices: OptIntSequence | Tuple[OptIntSequence, Indices, Indices]=None,
                            detector: Detector | None=None,  pre_processor: PreProcessor | None=None
                            ) -> Streaks:
    if 'data' not in file.attributes():
        raise ValueError("No data found in the files")
    idxs = file.indices('data')

    if indices is None:
        frames, ss_idxs, fs_idxs = list(range(len(idxs))), slice(None), slice(None)
    elif isinstance(indices, Sized) and len(indices) == 3:
        frames, ss_idxs, fs_idxs = indices
        if frames is None:
            frames = list(range(len(idxs)))
        else:
            frames = to_list(frames)
    else:
        frames = to_list(indices)
        ss_idxs, fs_idxs = slice(None), slice(None)

    initargs = (file, metapath, params, ss_idxs, fs_idxs, detector, pre_processor)
    streaks = []
    if params.num_threads > 1:
        with Pool(processes=params.num_threads, initializer=DetectionWorker.initializer,
                  initargs=initargs) as pool:
            for pattern in tqdm(pool.imap(DetectionWorker.run, zip(idxs, frames)), total=len(idxs)):
                streaks.append(pattern)
    else:
        worker = DetectionWorker.initialize(*initargs)
        for args in tqdm(zip(idxs, frames), total=len(idxs)):
            streaks.append(worker(args))
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
