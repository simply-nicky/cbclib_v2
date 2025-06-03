from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterable, Iterator, Literal, Tuple, overload
from dataclasses import dataclass
from tqdm.auto import tqdm
from .annotations import (ArrayNamespace, IntArray, NDArray, NDBoolArray, NDRealArray, NumPy,
                          ReadOut, RealArray)
from .data_container import ArrayContainer, Container, D, array_namespace, split
from .data_processing import CrystData
from .label import Structure2D, Structure3D
from ..indexer.cbc_data import MillerWithRLP, Patterns
from ..indexer.cbc_indexing import CBDIndexer, TiltOverAxis
from ..indexer.cbc_setup import BaseSetup, TiltOverAxisState, XtalState

@dataclass
class StructureParameters(Container):
    radius          : int
    rank            : int

    @overload
    def to_structure(self, kind: Literal['2d']) -> Structure2D: ...

    @overload
    def to_structure(self, kind: Literal['3d']) -> Structure3D: ...

    def to_structure(self, kind: Literal['2d', '3d']
                     ) -> Structure2D | Structure3D:
        if kind == '2d':
            return Structure2D(self.radius, self.rank)
        if kind == '3d':
            return Structure3D(self.radius, self.rank)
        raise ValueError(f"Invalid kind keyword: {kind}")

@dataclass
class RegionFinderParameters(Container):
    structure   : StructureParameters
    ratio       : float
    sigma       : float
    vmin        : float
    npts        : int
    threshold   : float
    num_threads : int

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

@dataclass
class CrystMetadata(ArrayContainer):
    mask        : NDBoolArray
    std         : NDRealArray
    whitefield  : NDRealArray

def pattern_recognition(metadata: CrystMetadata, params: RegionFinderParameters,
                        xp: ArrayNamespace=NumPy) -> Callable[[NDArray], ReadOut]:
    def pattern_goodness(data: NDArray) -> Tuple[float, float]:
        cryst_data = CrystData(data=data[None], std=metadata.std, mask=metadata.mask,
                               whitefield=metadata.whitefield)
        cryst_data = cryst_data.scale_whitefield(method='median', num_threads=params.num_threads)
        cryst_data = cryst_data.update_snr()
        det_obj = cryst_data.region_detector(Structure2D(**params.structure.to_dict()))
        det_obj = det_obj.downscale(params.ratio, params.sigma, params.num_threads)
        regions = det_obj.detect_regions(params.vmin, params.npts, params.num_threads)
        masses = det_obj.total_mass(regions)[0]
        fits = det_obj.ellipse_fit(regions)[0]
        if fits.size:
            values = xp.tanh((fits[:, 0] / fits[:, 1] - params.threshold)) * masses
            positive, negative = xp.sum(values[values > 0]), -xp.sum(values[values < 0])
            return (float(positive), float(negative))
        return (0.0, 0.0)

    return pattern_goodness

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
