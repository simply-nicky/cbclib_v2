from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Tuple, overload
from dataclasses import dataclass
from tqdm.auto import tqdm
from .annotations import (ArrayNamespace, BoolArray, IntArray, NDArray, NDRealArray, NumPy,
                          ReadOut, RealArray, ROI)
from .data_container import ArrayContainer, Container, D, array_namespace, split
from .data_processing import CrystData, RegionDetector, StreakDetector, Streaks, Peaks
from .label import Structure2D, Structure3D, Regions2D
from ..indexer.cbc_data import MillerWithRLP, Patterns
from ..indexer.cbc_indexing import CBDIndexer, TiltOverAxis
from ..indexer.cbc_setup import BaseSetup, TiltOverAxisState, XtalState

@dataclass
class ROIParameters(Container):
    xmin    : int
    xmax    : int
    ymin    : int
    ymax    : int

    def to_roi(self) -> ROI:
        return (self.ymin, self.ymax, self.xmin, self.xmax)

@dataclass
class StructureParameters(Container):
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
class MaskParameters(Container):
    method  : Literal['all-bad', 'no-bad', 'range', 'snr', 'std']
    vmin    : int = 0
    vmax    : int = 65535
    snr_max : float = 3.0
    std_min : float = 0.0
    roi     : ROIParameters | None = None

@dataclass
class BackgroundParameters(Container):
    method  : Literal['mean-poisson', 'median-poisson', 'robust-mean-scale', 'robust-mean-poisson']
    r0      : float = 0.0
    r1      : float = 0.5
    n_iter  : int = 12
    lm      : float = 9.0

@dataclass
class MetadataParameters(Container):
    mask        : MaskParameters
    background  : BackgroundParameters
    num_threads : int

def create_metadata(frames: NDRealArray, params: MetadataParameters) -> CrystMetadata:
    xp = array_namespace(frames)
    data = CrystData(data=frames)

    if params.mask.method == 'all-bad':
        if params.mask.roi is None:
            raise ValueError("No ROI is provided")
        data = data.update_mask(method='all-bad', roi=params.mask.roi.to_roi())

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
class RegionFinderParameters(Container):
    regions     : RegionParameters
    num_threads : int

def find_regions(frames: NDArray, metadata: CrystMetadata, params: RegionFinderParameters
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
                        xp: ArrayNamespace=NumPy) -> Callable[[NDArray], ReadOut]:
    def pattern_goodness(frames: NDArray) -> Tuple[float, float]:
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
class StreakParameters(RegionParameters):
    xtol        : float
    min_size    : int
    nfa         : int

@dataclass
class StreakFinderParameters(Container):
    peaks       : RegionParameters
    streaks     : StreakParameters
    center      : Tuple[float, float] | None = None
    num_threads : int = 1

def find_streaks(frames: NDArray, metadata: CrystMetadata, params: StreakFinderParameters
                 ) -> Tuple[StreakDetector, Streaks, List[Peaks]]:
    if frames.ndim < 2:
        raise ValueError("Frame array must be at least 2 dimensional")
    data = CrystData(data=frames.reshape((-1,) + frames.shape[-2:]), mask=metadata.mask,
                     std=metadata.std, whitefield=metadata.whitefield)
    data = data.scale_whitefield(method='median', num_threads=params.num_threads)
    data = data.update_snr()
    det_obj = data.streak_detector(params.streaks.structure.to_structure('2d'))
    peaks = det_obj.detect_peaks(params.peaks.vmin, params.peaks.npts,
                                 params.peaks.structure.to_structure('2d'), params.num_threads)
    streaks = det_obj.detect_streaks(peaks, params.streaks.xtol, params.streaks.vmin,
                                     params.streaks.min_size, nfa=params.streaks.nfa,
                                     num_threads=params.num_threads)
    if params.center is not None:
        streaks = streaks.concentric_only(params.center[0], params.center[1], 0.33)
    return det_obj, streaks, peaks

@dataclass
class CBDIndexingParameters(Container):
    patterns_file   : str
    output_file     : str
    shape           : Tuple[int, int, int]
    width           : float
    num_threads     : int
    threshold       : float
    n_max           : int
    vicinity        : StructureParameters
    connectivity    : StructureParameters

def pre_indexing(candidates: MillerWithRLP, patterns: Patterns, indexer: CBDIndexer,
                 params: CBDIndexingParameters, state: BaseSetup, parallel: bool=True
                 ) -> Tuple[IntArray, TiltOverAxisState]:
    num_threads = params.num_threads if parallel else 1
    xp = array_namespace(state)
    centers = patterns.sample(xp.full(patterns.shape[0], 0.5))
    points = indexer.points_to_kout(centers, state)
    rotograms = indexer.index(candidates, patterns, points, state)
    rotomap = indexer.rotomap(params.shape, rotograms, params.width, num_threads)
    peaks = indexer.to_peaks(rotomap, params.threshold)
    return indexer.refine_peaks(peaks, rotomap, params.vicinity.to_structure('3d'),
                                params.connectivity.to_structure('3d'), num_threads)

def indexing_candidates(indexer: CBDIndexer, patterns: Patterns, xtal: XtalState, state: BaseSetup,
                        xp: ArrayNamespace=NumPy) -> Iterator[MillerWithRLP]:
    q1, q2 = indexer.patterns_to_q(patterns, state)
    q_max = xp.max((xp.sqrt(xp.sum(q1.q**2, axis=-1)), xp.sqrt(xp.sum(q2.q**2, axis=-1))))
    hkl = indexer.xtal.hkl_in_ball(q_max, xtal, xp)
    return indexer.xtal.hkl_range(patterns.indices(), hkl, xtal, xp)

def run_pre_indexing(patterns: Patterns, xtals: XtalState, state: BaseSetup,
                     params: CBDIndexingParameters, chunksize: int=1, xp: ArrayNamespace=NumPy
                     ) -> Tuple[IntArray, XtalState]:
    rlp_iterator = indexing_candidates(CBDIndexer(), patterns, xtals, state, xp)
    frames, solutions = [], []
    total = len(patterns) // chunksize + (len(patterns) % chunksize > 0)

    if len(xtals) == 1:
        iterator = zip(split(patterns, chunksize), split(rlp_iterator, chunksize))

        for pattern, candidates in tqdm(iterator, total=total):
            idxs, tilts = pre_indexing(candidates, pattern, CBDIndexer(), params, state)
            frames.append(pattern.indices()[idxs])
            solutions.append(TiltOverAxis().of_xtal(xtals, tilts))
    elif len(xtals) == len(patterns):
        iterator = zip(split(patterns, chunksize), split(rlp_iterator, chunksize),
                       split(xtals, chunksize))

        for pattern, candidates, xtal in tqdm(iterator, total=total):
            idxs, tilts = pre_indexing(candidates, pattern, CBDIndexer(), params, state)
            frames.append(pattern.indices()[idxs])
            solutions.append(TiltOverAxis().of_xtal(xtal[idxs], tilts))
    else:
        raise ValueError(f'Number of crystals ({len(xtals):d}) and patterns ({len(patterns):d}) '\
                         'are inconsistent')

    return xp.concatenate(frames), XtalState.concatenate(solutions)

worker : 'IndexingWorker'

@dataclass
class IndexingWorker():
    state   : BaseSetup
    params  : CBDIndexingParameters
    xtal    : XtalState

    def __call__(self, args: Tuple[Dict[str, Any], Dict[str, Any]]) -> Tuple[IntArray, RealArray]:
        candidates, patterns = MillerWithRLP(**args[0]), Patterns(**args[1])
        idxs, tilts = pre_indexing(candidates, patterns, CBDIndexer(),
                                   self.params, self.state, False)
        frames = patterns.indices()[idxs]
        if len(self.xtal) == 1:
            solutions = TiltOverAxis().of_xtal(self.xtal, tilts)
        else:
            solutions = TiltOverAxis().of_xtal(self.xtal[frames], tilts)
        return frames, solutions.basis

    @classmethod
    def initializer(cls, state: BaseSetup, params: CBDIndexingParameters, xtal: XtalState):
        global worker
        worker = cls(state, params, xtal)

    @staticmethod
    def run(args: Tuple[Dict[str, Any], Dict[str, Any]]) -> Tuple[IntArray, RealArray]:
        return worker(args)

def dict_range(containers: Iterable[D]) -> Iterator[Dict[str, Any]]:
    for container in containers:
        yield container.to_dict()

def run_pre_indexing_pool(patterns: Patterns, xtals: XtalState, state: BaseSetup,
                          params: CBDIndexingParameters, xp: ArrayNamespace=NumPy
                          ) -> Tuple[IntArray, XtalState]:
    rlp_iterator = indexing_candidates(CBDIndexer(), patterns, xtals, state, xp)
    frames, solutions = [], []
    with Pool(processes=params.num_threads, initializer=IndexingWorker.initializer,
              initargs=(state, params, xtals)) as pool:
        iterator = zip(dict_range(rlp_iterator), dict_range(patterns))
        for frame, solution in tqdm(pool.imap(IndexingWorker.run, iterator),
                                    total=len(patterns)):
            frames.append(frame)
            solutions.append(solution)

    return xp.concatenate(frames), XtalState(xp.concatenate(solutions))
