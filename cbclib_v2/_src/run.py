from dataclasses import dataclass
from functools import wraps
from multiprocessing.pool import Pool
import hashlib
import os
from pathlib import Path
import pickle
import re
from typing import Any, Callable, Dict, Generic, Iterator, List, Literal, Tuple, TypeVar, overload
from tqdm.auto import tqdm
from .annotations import AnyNamespace, Array, ArrayNamespace, CPArray, Indices, JaxArray, NDArray, NumPy
from .crystfel import Detector as Geometry, read_crystfel
from .cxi_protocol import (H5Files, H5Protocol, H5Handler, H5ReadWorker, LoadWorker, StackIndices,
                           TrainIndices)
from .data_container import Container, list_indices, split, to_list
from .scripts import BaseParameters

class IndexCacher:
    """Caching utility for run indices with file modification checking.

    Cache is stored in ~/.cache/cbclib_v2/ by default, but can be overridden
    via the CBCLIB_CACHE_DIR environment variable.
    """
    def __init__(self, run_id: int, config: 'RunConfig'):
        self.run_id = run_id
        self.config = config
        self.filenames = config.filenames(run_id)

    @classmethod
    def cache_dir(cls) -> Path:
        """Get cache directory, respecting environment override."""
        if cache_env := os.environ.get('CBCLIB_CACHE_DIR'):
            return Path(cache_env)
        return Path.home() / '.cache' / 'cbclib_v2'

    def cache_key(self) -> str:
        """Generate a unique cache key for this run config."""
        config_str = f"{self.config.facility}_{self.run_id}_{len(self.filenames)}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def cache_path(self) -> Path:
        """Get the cache file path for this run."""
        cache_dir = self.cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"run_{self.run_id}_{self.cache_key()}.pkl"

    def is_valid(self) -> bool:
        """Check if cache exists and is newer than all source HDF5 files."""
        cache_path = self.cache_path()
        if not cache_path.exists():
            return False

        cache_mtime = cache_path.stat().st_mtime
        for file in self.filenames:
            if Path(file).stat().st_mtime > cache_mtime:
                return False
        return True

    def load(self) -> TrainIndices:
        """Load indices from cache."""
        with open(self.cache_path(), 'rb') as f:
            return pickle.load(f)

    def save(self, indices: TrainIndices) -> None:
        """Save indices to cache."""
        cache_path = self.cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(indices, f)

def cache_indices(method: Callable) -> Callable:
    """Decorator for caching indices methods."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'cacher') or not isinstance(self.cacher, IndexCacher):
            err_txt = "The object must have a 'cacher' attribute of type IndexCacher " \
                      "to use cache_indices."
            raise AttributeError(err_txt)

        cacher = self.cacher

        if cacher.is_valid():
            return cacher.load()

        indices = method(self, *args, **kwargs)

        cacher.save(indices)
        return indices
    return wrapper

@dataclass
class RunConfig(BaseParameters):
    facility : Literal['XFEL', 'SwissFEL']

    def filenames(self, run_id: int) -> List[str]:
        raise NotImplementedError

    def geometry(self) -> Geometry:
        raise NotImplementedError

    @classmethod
    def type_resolver(cls, data: Dict[str, Any]) -> type['RunConfig']:
        facility = data.get('facility')
        if facility == 'XFEL':
            return XFELRunConfig
        if facility == 'SwissFEL':
            return SwissFELConfig
        raise ValueError(f"Unsupported facility type: {facility}")

IndicesType = TypeVar('IndicesType', bound='TrainIndices')

class BaseRun(Container, Generic[IndicesType]):
    run_id  : int
    config  : RunConfig

    def attributes(self):
        raise NotImplementedError

    def indices(self) -> IndicesType:
        raise NotImplementedError

    @overload
    def metadata(self, attr: str, keys: IndicesType, *, xp: ArrayNamespace[CPArray]) -> CPArray: ...

    @overload
    def metadata(self, attr: str, keys: IndicesType, *, xp: ArrayNamespace[JaxArray]) -> JaxArray: ...

    @overload
    def metadata(self, attr: str, keys: IndicesType, *, xp: ArrayNamespace[NDArray]) -> NDArray: ...

    @overload
    def metadata(self, attr: str, keys: IndicesType) -> NDArray: ...

    def metadata(self, attr: str, keys: IndicesType, *, xp: AnyNamespace=NumPy) -> Array:
        worker = self.meta_worker(attr)

        stack = []
        for index in iter(keys):
            stack.append(worker(index))

        return xp.asarray(NumPy.stack(stack, axis=0))

    @overload
    def data(self, keys: IndicesType, *, geometry: bool = False, n_processes: int = 1,
             verbose: bool = True, xp: ArrayNamespace[CPArray]) -> CPArray: ...

    @overload
    def data(self, keys: IndicesType, *, geometry: bool = False, n_processes: int = 1,
             verbose: bool = True, xp: ArrayNamespace[JaxArray]) -> JaxArray: ...

    @overload
    def data(self, keys: IndicesType, *, geometry: bool = False, n_processes: int = 1,
             verbose: bool = True, xp: ArrayNamespace[NDArray]) -> NDArray: ...

    @overload
    def data(self, keys: IndicesType, *, geometry: bool = False, n_processes: int = 1,
             verbose: bool = True) -> NDArray: ...

    def data(self, keys: IndicesType, *, geometry: bool = False, n_processes: int = 1,
             verbose: bool = True, xp: AnyNamespace=NumPy) -> Array:

        stack = []

        if n_processes > 1:
            try:
                pool, worker_class = self.pool(geometry)
            except NotImplementedError as exc:
                raise ValueError("Multiprocessing is not supported for this run type.") from exc

            with pool:
                for chunk in tqdm(pool.imap(worker_class.run, iter(keys)), total=len(keys),
                                  disable=not verbose, desc="Loading data"):
                    stack.append(chunk)

        else:
            worker = self.worker(geometry)
            for index in tqdm(keys, disable=not verbose, desc="Loading data"):
                stack.append(worker(index))

        if len(stack) == 1:
            return xp.asarray(stack[0])
        return xp.asarray(NumPy.stack(stack, axis=0))

    def pool(self, geometry: bool = False) -> Tuple[Pool, type[LoadWorker[NDArray]]]:
        raise NotImplementedError

    def visit_frames(self, keys: IndicesType, geometry: bool = False) -> Iterator[NDArray]:
        worker = self.worker(geometry)
        for index in keys:
            yield worker(index)

    def meta_worker(self, attr: str) -> LoadWorker[NDArray]:
        raise NotImplementedError

    def worker(self, geometry: bool = False) -> LoadWorker[NDArray]:
        raise NotImplementedError

@dataclass
class XFELRunConfig(RunConfig):
    data_dir        : str
    hdf5_protocol   : str
    file_pattern    : str
    geometry_file   : str
    num_modules     : int = 1
    starts_at       : int = 0

    def module_files(self, run_id: int) -> Iterator[List[str]]:
        data_dir = self.data_dir.format(run_id)

        for module_id in range(self.starts_at, self.starts_at + self.num_modules):
            pattern = self.file_pattern.format(run_id, module_id)
            module_files = []
            for path in os.listdir(data_dir):
                if re.match(pattern, path):
                    module_files.append(os.path.join(data_dir, path))
            yield module_files

    def filenames(self, run_id: int) -> List[str]:
        filenames = []
        for module_files in self.module_files(run_id):
            filenames.extend(module_files)
        return filenames

    def files(self, run_id: int) -> List[H5Files]:
        """Returns a list of H5Files objects, one for each module.

        Args:
            run_id: The ID of the run to load files for.

        Returns:
            A list where each element corresponds to all HDF5 files pertaining to a
            specific detector module.
        """
        files = []
        for module_files in self.module_files(run_id):
            files.append(H5Files(module_files))

        return files

    def geometry(self) -> Geometry:
        return read_crystfel(self.geometry_file)

    def protocol(self) -> H5Protocol:
        return H5Protocol.read(self.hdf5_protocol)

@dataclass
class FileStackIndices(TrainIndices):
    file_indices     : List[StackIndices]
    indices          : List[int] | None = None

    def __post_init__(self):
        # Validate that all StackIndices have the same number of frames
        n_frames_set = {len(file_indices) for file_indices in self.file_indices}
        if len(n_frames_set) > 1:
            raise ValueError("All StackIndices must have the same number of frames for stacking.")
        self.total = n_frames_set.pop()

    def __iter__(self) -> Iterator[Tuple[Tuple[str, Indices], ...]]:
        if self.indices is None:
            yield from zip(*self.file_indices)
        else:
            yield from zip(*(file_indices[self.indices] for file_indices in self.file_indices))

    def __getitem__(self, key: Indices) -> "FileStackIndices":
        if isinstance(key, slice):
            key_list = list_indices(key, len(self))
        else:
            key_list = to_list(key)
        if self.indices is not None:
            key_list = [self.indices[index] for index in key_list]
        return FileStackIndices(file_indices=self.file_indices, indices=key_list)

    def __len__(self) -> int:
        return self.total if self.indices is None else len(self.indices)

    def index(self) -> Iterator[int]:
        if self.indices is None:
            yield from range(self.total)
        else:
            yield from self.indices

    def split(self, num_chunks: int) -> Iterator["FileStackIndices"]:
        if self.indices is None:
            indices = list(range(self.total))
        else:
            indices = self.indices

        for chunk in split(indices, num_chunks):
            yield FileStackIndices(file_indices=self.file_indices, indices=chunk)

xfel_worker : Callable[[Tuple[Tuple[str, Indices], ...]], NDArray]

@dataclass
class XFELReadWorker(H5ReadWorker):
    data_path   : str | Tuple[str, ...]

    def __call__(self, indices: Tuple[Tuple[str, Indices], ...]) -> NDArray:
        data_arrays = []

        if isinstance(self.data_path, tuple):
            for data_path, index in zip(self.data_path, indices):
                data_arrays.append(self.load(data_path, index))
        else:
            for index in indices:
                data_arrays.append(self.load(self.data_path, index))
        return NumPy.stack(data_arrays, axis=0)

    @classmethod
    def initializer(cls, data_path: str | Tuple[str, ...], ss_indices: Indices,
                    fs_indices: Indices):
        global xfel_worker
        xfel_worker = cls(data_path, ss_indices, fs_indices)

    @staticmethod
    def run(index: Tuple[Tuple[str, Indices], ...]) -> NDArray:
        return xfel_worker(index)

@dataclass
class XFELReadWithGeomWorker(XFELReadWorker):
    geometry : Geometry

    def __post_init__(self):
        self.indices = self.geometry.indices()

    def __call__(self, indices: Tuple[Tuple[str, Indices], ...]) -> NDArray:
        data = super().__call__(indices)
        return self.indices(data)

    @classmethod
    def initializer(cls, data_path: str | Tuple[str, ...], ss_indices: Indices, fs_indices: Indices,
                    geometry: Geometry):
        global xfel_worker
        xfel_worker = cls(data_path, ss_indices, fs_indices, geometry)

XFELWorker = XFELReadWorker | XFELReadWithGeomWorker

@dataclass
class XFELRun(BaseRun[FileStackIndices]):
    run_id      : int
    config      : XFELRunConfig
    ss_idxs     : Indices | None = None
    fs_idxs     : Indices | None = None

    def __post_init__(self):
        self.handler = H5Handler(self.config.protocol())
        self.files = self.config.files(self.run_id)
        if 'data' not in self.handler.attributes():
            raise ValueError("Protocol must contain 'data' attribute for XFELRun.")
        self.data_paths : Dict[str, str | Tuple[str, ...]] = {}
        self.cacher = IndexCacher(self.run_id, self.config)

    def attributes(self) -> List[str]:
        return [attr for attr in self.handler.attributes() if attr != 'data']

    def data_path(self, attr: str = 'data') -> str | Tuple[str, ...]:
        if attr not in self.data_paths:
            data_paths = []
            for module_files in self.files:
                for file in module_files.visit_files():
                    data_path = self.handler.protocol.find_path(attr, file)
                    if data_path:
                        break
                if data_path:
                    if data_paths:
                        if data_paths[-1] != data_path:
                            data_paths.append(data_path)
                    else:
                        data_paths.append(data_path)

            if data_paths:
                self.data_paths[attr] = tuple(data_paths) if len(data_paths) > 1 else data_paths[0]

        if not self.data_paths[attr]:
            raise ValueError(f"Attribute '{attr}' not found in protocol for any module.")
        return self.data_paths[attr]

    @cache_indices
    def indices(self) -> FileStackIndices:
        return self._load_indices()

    def _load_indices(self) -> FileStackIndices:
        """Load indices from HDF5 files (uncached implementation)."""
        data_paths = []
        file_indices = []
        for module_files in self.files:
            indices = self.handler.indices(module_files, 'data')

            if data_paths:
                if data_paths[-1] != indices.data_path:
                    data_paths.append(indices.data_path)
            else:
                data_paths.append(indices.data_path)

            file_indices.append(indices.indices)

        self.data_paths['data'] = tuple(data_paths) if len(data_paths) > 1 else data_paths[0]
        return FileStackIndices(file_indices)

    def pool(self, geometry: bool = False) -> Tuple[Pool, type[XFELWorker]]:
        if geometry:
            init_args = (self.data_path(), self.ss_idxs, self.fs_idxs,
                        self.config.geometry())
            worker_class = XFELReadWithGeomWorker
        else:
            init_args = (self.data_path(), self.ss_idxs, self.fs_idxs)
            worker_class = XFELReadWorker

        pool = Pool(initializer=worker_class.initializer, initargs=init_args)
        return pool, worker_class

    def meta_worker(self, attr: str) -> XFELWorker:
        return XFELReadWorker(self.data_path(attr), None, None)

    def worker(self, geometry: bool = False) -> XFELWorker:
        if geometry:
            return XFELReadWithGeomWorker(self.data_path(), self.ss_idxs, self.fs_idxs,
                                          self.config.geometry())
        return XFELReadWorker(self.data_path(), self.ss_idxs, self.fs_idxs)

@dataclass
class SwissFELConfig(RunConfig):
    data_dir        : str
    hdf5_protocol   : str
    file_pattern    : str
    geometry_file   : str

    def filenames(self, run_id: int) -> List[str]:
        data_dir = self.data_dir.format(run_id)

        filenames = []
        for path in os.listdir(data_dir):
            if re.match(self.file_pattern, path):
                filenames.append(os.path.join(data_dir, path))
        return filenames

    def files(self, run_id: int) -> H5Files:
        return H5Files(self.filenames(run_id))

    def geometry(self) -> Geometry:
        return read_crystfel(self.geometry_file)

    def protocol(self) -> H5Protocol:
        return H5Protocol.read(self.hdf5_protocol)

sfel_worker : Callable[[Tuple[str, Indices]], NDArray]

@dataclass
class SFELReadWorker(H5ReadWorker):
    @classmethod
    def initializer(cls, data_path: str, ss_indices: Indices, fs_indices: Indices):
        global sfel_worker
        sfel_worker = cls(data_path, ss_indices, fs_indices)

    @staticmethod
    def run(index: Tuple[str, Indices]) -> NDArray:
        return sfel_worker(index)

@dataclass
class SFELReadWithGeomWorker(SFELReadWorker):
    geometry : Geometry

    def __post_init__(self):
        self.indices = self.geometry.indices()

    def __call__(self, indices: Tuple[str, Indices]) -> NDArray:
        data = super().__call__(indices)
        return self.indices(data)

    @classmethod
    def initializer(cls, data_path: str, ss_indices: Indices, fs_indices: Indices,
                    geometry: Geometry):
        global sfel_worker
        sfel_worker = cls(data_path, ss_indices, fs_indices, geometry)

SFELWorker = SFELReadWorker | SFELReadWithGeomWorker

@dataclass
class SwissFELRun(BaseRun[StackIndices]):
    run_id      : int
    config      : SwissFELConfig
    ss_idxs     : Indices | None = None
    fs_idxs     : Indices | None = None

    def __post_init__(self):
        self.handler = H5Handler(self.config.protocol())
        self.files = self.config.files(self.run_id)
        if 'data' not in self.handler.attributes():
            raise ValueError("Protocol must contain 'data' attribute for XFELRun.")
        self.data_paths : Dict[str, str] = {}
        self.cacher = IndexCacher(self.run_id, self.config)

    def data_path(self, attr: str = 'data') -> str:
        if attr not in self.data_paths:
            for file in self.files.visit_files():
                self.data_paths[attr] = self.handler.protocol.find_path(attr, file)
                if self.data_paths[attr]:
                    break

        if not self.data_paths[attr]:
            raise ValueError(f"Attribute '{attr}' not found in protocol for any module.")
        return self.data_paths[attr]

    def attributes(self) -> List[str]:
        return [attr for attr in self.handler.attributes() if attr != 'data']

    @cache_indices
    def indices(self) -> StackIndices:
        return self._load_indices()

    def _load_indices(self) -> StackIndices:
        """Load indices from HDF5 files (uncached implementation)."""
        indices = self.handler.indices(self.files, 'data')
        self.data_paths['data'] = indices.data_path
        return indices.indices

    def pool(self, geometry: bool = False) -> Tuple[Pool, type[SFELWorker]]:
        if geometry:
            init_args = (self.data_path(), self.ss_idxs, self.fs_idxs,
                         self.config.geometry())
            worker_class = SFELReadWithGeomWorker
        else:
            init_args = (self.data_path(), self.ss_idxs, self.fs_idxs)
            worker_class = SFELReadWorker

        pool = Pool(initializer=worker_class.initializer, initargs=init_args)
        return pool, worker_class

    def meta_worker(self, attr: str) -> SFELWorker:
        return SFELReadWorker(self.data_path(attr), None, None)

    def worker(self, geometry: bool = False) -> SFELWorker:
        if geometry:
            return SFELReadWithGeomWorker(self.data_path(), self.ss_idxs, self.fs_idxs,
                                          self.config.geometry())
        return SFELReadWorker(self.data_path(), self.ss_idxs, self.fs_idxs)

@overload
def open_run(run_id: int, config: XFELRunConfig) -> XFELRun: ...

@overload
def open_run(run_id: int, config: SwissFELConfig) -> SwissFELRun: ...

@overload
def open_run(run_id: int, config: RunConfig) -> BaseRun[TrainIndices]: ...

def open_run(run_id: int, config: RunConfig
             ) -> XFELRun | SwissFELRun | BaseRun[TrainIndices]:
    if isinstance(config, XFELRunConfig):
        return XFELRun(run_id, config)
    if isinstance(config, SwissFELConfig):
        return SwissFELRun(run_id, config)
    raise ValueError(f"Unsupported RunConfig type: {type(config)}")
