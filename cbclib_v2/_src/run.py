from __future__ import annotations
from dataclasses import dataclass, field
from functools import wraps
from multiprocessing.pool import Pool
import hashlib
import os
from pathlib import Path
import pickle
import re
from typing import (Any, Callable, Dict, Generic, Iterator, List, Literal, Tuple, Type, TypeVar,
                    cast, overload)
from tqdm.auto import tqdm
from .annotations import (AnyNamespace, Array, ArrayNamespace, CPArray, Indices, IntSequence,
                          JaxArray, NDArray, NumPy)
from .crystfel import Detector as Geometry, read_crystfel
from .cxi_protocol import (H5Files, H5Protocol, H5Handler, LoadWorker, StackIndices,
                           TrainIndexRecord, TrainIndices, WorkerType, load_dataset)
from .data_container import Container, list_indices, split, to_list
from .scripts import BaseParameters

Facility = Literal['LCLS', 'XFEL', 'SwissFEL']

@dataclass(frozen=True)
class RunLocator:
    """Structured identifier for a facility run.

    ``run_id`` is the public identifier users usually provide. ``variant`` is
    an optional facility-specific discriminator, such as an LCLS directory
    suffix, that must travel with the run ID for file lookup and caching.

    Attributes:
        run_id: Numeric run identifier.
        variant: Optional facility-specific discriminator. Used by
            :class:`~cbclib_v2.LCLSConfig` to choose the correct scan directory
            for a run when multiple run directories are present.
    """
    run_id     : int
    variant    : str | None = None

    @classmethod
    def coerce(cls, locator: 'int | RunLocator') -> 'RunLocator':
        """Return *locator* as a :class:`RunLocator`."""
        if isinstance(locator, int):
            return cls(locator)
        return locator

class IndexCacher:
    """Caching utility for run indices with file modification checking.

    Cache is stored in ~/.cache/cbclib_v2/ by default, but can be overridden
    via the CBCLIB_CACHE_DIR environment variable.
    """
    def __init__(self, locator: RunLocator, config: 'RunConfig'):
        self.locator = locator
        self.run_id = locator.run_id
        self.config = config
        self.filenames = config.filenames(locator)

    @classmethod
    def cache_dir(cls) -> Path:
        """Get cache directory, respecting environment override."""
        if cache_env := os.environ.get('CBCLIB_CACHE_DIR'):
            return Path(cache_env)
        return Path.home() / '.cache' / 'cbclib_v2'

    def cache_key(self) -> str:
        """Generate a unique cache key for this run config."""
        config_str = f"{self.config.facility}_{self.locator}_{len(self.filenames)}"
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

    def save(self, indices: TrainIndices):
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
    """Abstract base class for FEL facility run configurations.

    Subclasses provide facility-specific logic for locating HDF5 files,
    loading the detector geometry, and reading the HDF5 protocol.  Pass a
    concrete subclass to :func:`open_run` to obtain the matching
    :class:`BaseRun`.

    Attributes:
        facility: Facility identifier; one of ``'LCLS'``, ``'XFEL'``,
            or ``'SwissFEL'``.

    Example:
        Load a run using an XFEL configuration JSON file:

        >>> config = XFELConfig.read('config.json')
        >>> run = open_run(42, config)
    """
    facility : Facility = field(kw_only=True)

    def num_files(self, key: int | RunLocator) -> int:
        """Return the number of HDF5 files in the run."""
        raise NotImplementedError

    def filenames(self, locator: int | RunLocator) -> List[str]:
        """This method should return a list of all HDF5 file paths for the given run ID."""
        raise NotImplementedError

    def scan_dir(self, locator: int | RunLocator) -> str:
        """This method should return the directory path to scan for HDF5 files for the
        given run ID."""
        raise NotImplementedError

    def geometry(self) -> Geometry:
        """This method should return the detector geometry for the run."""
        raise NotImplementedError

    @classmethod
    def type_resolver(cls, data: Dict[str, Any]) -> type['RunConfig']:
        facility = data.get('facility')
        if facility == 'XFEL':
            return XFELConfig
        if facility == 'SwissFEL':
            return SwissFELConfig
        if facility == 'LCLS':
            return LCLSConfig
        raise ValueError(f"Unsupported facility type: {facility}")

T_Indices = TypeVar('T_Indices', bound='TrainIndices')
T_RunId = TypeVar('T_RunId')
T_Config = TypeVar('T_Config', bound='RunConfig')

class BaseRun(Container, Generic[T_RunId, T_Indices, T_Config]):
    """Abstract base class for reading detector data from a FEL run.

    A run wraps one or more HDF5 files belonging to a single experiment run
    and exposes a uniform interface for loading detector frames and per-frame
    metadata arrays. Concrete subclasses are created by :func:`open_run`.

    Attributes:
        run_id: Numeric run identifier, or an ordered collection of identifiers
            for a :class:`RunList`.
        config: :class:`RunConfig` that describes file locations and protocol.

    Example:
        Open a run and load the first 10 frames with geometry applied:

        >>> run = open_run(42, config)
        >>> indices = run.indices()
        >>> frames = run.data(indices[:10], geometry=True)
    """
    run_id  : T_RunId
    config  : T_Config

    def attributes(self):
        """This method should return a list of available metadata attributes as defined in the run's
        protocol."""
        raise NotImplementedError

    def indices(self) -> T_Indices:
        """This method should return a :class:`TrainIndices` object containing the indices of frames
        available in the run."""
        raise NotImplementedError

    @overload
    def metadata(self, attr: str, keys: T_Indices, *, xp: ArrayNamespace[CPArray]) -> CPArray: ...

    @overload
    def metadata(self, attr: str, keys: T_Indices, *, xp: ArrayNamespace[JaxArray]
                 ) -> JaxArray: ...

    @overload
    def metadata(self, attr: str, keys: T_Indices, *, xp: ArrayNamespace[NDArray]) -> NDArray: ...

    @overload
    def metadata(self, attr: str, keys: T_Indices) -> NDArray: ...

    def metadata(self, attr: str, keys: T_Indices, *, xp: AnyNamespace=NumPy) -> Array:
        """Load a scalar or sequence metadata attribute for a set of frames.

        Args:
            attr: Attribute name as defined in the run's protocol.
            keys: Frame index set returned by :meth:`indices`.
            xp: Array namespace for the returned array; defaults to NumPy.

        Returns:
            Array of shape ``(n_frames, ...)`` containing the attribute values.

        Example:
            Loading pulse IDs for the first 50 frames:

            >>> pulse_ids = run.metadata('pulse_id', indices[:50])
        """
        worker = self.meta_worker(attr)

        stack = []
        for index in iter(keys):
            stack.append(worker(index))

        return xp.asarray(NumPy.stack(stack, axis=0))

    @overload
    def data(self, keys: T_Indices, *, geometry: bool = False, n_processes: int = 1,
             verbose: bool = True, xp: ArrayNamespace[CPArray]) -> CPArray: ...

    @overload
    def data(self, keys: T_Indices, *, geometry: bool = False, n_processes: int = 1,
             verbose: bool = True, xp: ArrayNamespace[JaxArray]) -> JaxArray: ...

    @overload
    def data(self, keys: T_Indices, *, geometry: bool = False, n_processes: int = 1,
             verbose: bool = True, xp: ArrayNamespace[NDArray]) -> NDArray: ...

    @overload
    def data(self, keys: T_Indices, *, geometry: bool = False, n_processes: int = 1,
             verbose: bool = True) -> NDArray: ...

    def data(self, keys: T_Indices, *, geometry: bool = False, n_processes: int = 1,
             verbose: bool = True, xp: AnyNamespace=NumPy) -> Array:
        """Load detector frames for a set of indices.

        Args:
            keys: Frame index set returned by :meth:`indices`.
            geometry: Apply the CrystFEL geometry to assemble detector panels
                into a single image when ``True``.
            n_processes: Number of parallel worker processes.
            verbose: Show a progress bar when ``True``.
            xp: Array namespace for the returned array; defaults to NumPy.

        Returns:
            Array of shape ``(n_frames, *frame_shape)``.

        Example:
            Load the first 100 frames with geometry applied using 4 worker processes:

            >>> indices = run.indices()
            >>> frames = run.data(indices[:100], geometry=True, n_processes=4)
        """
        stack = []

        if n_processes > 1:
            try:
                pool, worker = self.pool(geometry, n_processes)
            except NotImplementedError as exc:
                raise ValueError("Multiprocessing is not supported for this run type.") from exc

            with pool:
                for chunk in tqdm(pool.imap(worker, iter(keys)), total=len(keys),
                                  disable=not verbose, desc="Loading data"):
                    stack.append(chunk)

        else:
            worker = self.worker(geometry)
            for index in tqdm(keys, disable=not verbose, desc="Loading data"):
                stack.append(worker(index))

        return xp.asarray(NumPy.stack(stack, axis=0))

    def pool(self, geometry: bool = False, processes: int | None = None
             ) -> Tuple[Pool, Callable[..., NDArray]]:
        """
        The method should return a multiprocessing pool and a worker for loading data frames.
        The worker function should accept an index from :meth:`indices` and return the
        corresponding data array for that index.

        Args:
            geometry: Whether to apply the CrystFEL geometry in the worker function.
            processes: Number of worker processes to create; defaults to the number of CPU cores.

        Returns:
            A tuple containing the multiprocessing pool and a worker for loading data frames.

        Example:
            Example of using the pool to load data in parallel:

            >>> pool, worker = run.pool(geometry=True)
            >>> with pool:
            ...     results = pool.map(worker, run.indices())
        """
        raise NotImplementedError

    def visit_frames(self, keys: T_Indices, geometry: bool = False) -> Iterator[NDArray]:
        """Iterate over detector frames one at a time without stacking.

        Useful for large datasets where loading all frames into memory at once
        is not practical.

        Args:
            keys: Frame index set returned by :meth:`indices`.
            geometry: Apply the CrystFEL geometry to each frame when ``True``.

        Yields:
            Individual detector frames as NumPy arrays.

        Example:
            Apply a processing function to each frame in the first 50 frames with geometry:

            >>> for frame in run.visit_frames(indices, geometry=True):
            ...     process(frame)
        """
        worker = self.worker(geometry)
        for index in keys:
            yield worker(index)

    def meta_worker(self, attr: str) -> LoadWorker[NDArray]:
        """Return a worker for loading metadata. A worker is a callable that takes an index from
        :meth:`indices` and returns the corresponding metadata array for the specified attribute.

        Args:
            attr: The attribute to load.

        Returns:
            An XFELWorker instance for loading the specified metadata attribute.

        Example:
            Example of using the metadata worker:

            >>> meta_worker = run.meta_worker('pulse_id')
            >>> pulse_ids = []
            >>> for index in run.indices()[:10]:
            ...     pulse_ids.append(meta_worker(index))
        """
        raise NotImplementedError

    def worker(self, geometry: bool = False) -> LoadWorker[NDArray]:
        """Return a worker for loading data frames. A worker is a callable that takes an index from
        :meth:`indices` and returns the corresponding data array for the specified attribute.

        Args:
            geometry: Whether to apply the CrystFEL geometry in the worker function.

        Returns:
            An XFELWorker instance for loading data frames.

        Example:
            Example of using the data worker:

            >>> data_worker = run.worker(geometry=True)
            >>> frames = []
            >>> for index in run.indices()[:10]:
            ...     frames.append(data_worker(index))
        """
        raise NotImplementedError

T_Files = TypeVar('T_Files')

class FELConfig(RunConfig, Generic[T_Files]):
    def files(self, locator: int | RunLocator) -> T_Files:
        """Return a list of H5Files objects for the given run ID.

        Args:
            locator: The run locator or run ID to load files for.

        Returns:
            A list of H5Files objects, one for each module in the run.
        """
        raise NotImplementedError

    def protocol(self) -> H5Protocol:
        """Return the HDF5 protocol for the run.

        Returns:
            An H5Protocol object representing the HDF5 protocol.
        """
        raise NotImplementedError

T_Path = TypeVar('T_Path')
T_Index = TypeVar('T_Index')

class BaseFELReadWorker(LoadWorker[NDArray], Generic[T_Path, T_Index]):
    data_path   : T_Path
    ss_indices  : Indices | None
    fs_indices  : Indices | None

    def __init__(self, data_path: T_Path, ss_indices: Indices | None,
                 fs_indices: Indices | None): ...

    def __call__(self, indices: T_Index) -> NDArray:
        raise NotImplementedError

    @classmethod
    def initializer(cls, data_path: T_Path, ss_indices: Indices, fs_indices: Indices):
        raise NotImplementedError

    @staticmethod
    def run(index: T_Index) -> NDArray:
        raise NotImplementedError

class BaseFELReadWithGeomWorker(LoadWorker[NDArray], Generic[T_Path, T_Index]):
    @classmethod
    def worker_class(cls) -> Type[BaseFELReadWorker[T_Path, T_Index]]:
        raise NotImplementedError

    def __init__(self, data_path: T_Path, ss_indices: Indices | None, fs_indices: Indices | None,
                 geometry: Geometry):
        self.worker = self.worker_class()(data_path, ss_indices, fs_indices)
        self.assembler = geometry.assembler()

    def __call__(self, indices: T_Index) -> NDArray:
        data = self.worker(indices)
        return self.assembler(data)

    @classmethod
    def initializer(cls, data_path: T_Path, ss_indices: Indices, fs_indices: Indices,
                    geometry: Geometry):
        raise NotImplementedError

    @classmethod
    def run(cls, index: T_Index) -> NDArray:
        return cls.worker_class().run(index)

T_FELConfig = TypeVar('T_FELConfig', bound='FELConfig')

class BaseFELRun(BaseRun[int, T_Indices, T_FELConfig], Generic[T_Path, T_Indices, T_FELConfig]):
    ss_idxs     : Indices | None = None
    fs_idxs     : Indices | None = None

    def __post_init__(self):
        if 'data' not in self.handler.attributes():
            raise ValueError(
                f"Protocol must contain 'data' attribute for {self.__class__.__name__}."
            )
        self.data_paths : Dict[str, T_Path] = {}
        self.cacher = IndexCacher(self.run_locator, self.config)

    @classmethod
    def worker_class(cls) -> Type[BaseFELReadWorker[T_Path, Any]]:
        raise NotImplementedError

    @classmethod
    def worker_geom_class(cls) -> Type[BaseFELReadWithGeomWorker[T_Path, Any]]:
        raise NotImplementedError

    @property
    def handler(self) -> H5Handler:
        """Return the H5Handler for the run's protocol."""
        return H5Handler(self.config.protocol())

    @property
    def run_locator(self) -> RunLocator:
        return RunLocator(self.run_id)

    def attributes(self) -> List[str]:
        """Return a list of attributes available in the HDF5 protocol, excluding 'data'.

        Returns:
            A list of attribute names as strings.
        """
        return [attr for attr in self.handler.attributes() if attr != 'data']

    def data_path(self, attr: str = 'data') -> T_Path:
        """Return the HDF5 data path for the given attribute.

        Args:
            attr: The attribute name to locate the data path for.

        Returns:
            The HDF5 data path as a string or a tuple of strings.
        """
        raise NotImplementedError

    def _load_indices(self) -> T_Indices:
        """Load indices from HDF5 files (uncached implementation)."""
        raise NotImplementedError

    @cache_indices
    def indices(self) -> T_Indices:
        """Return a table of frame indices for the run.

        Returns:
            An index table containing how to load each frame from the HDF5 files,
            including file paths and pixel indices.
        """
        return self._load_indices()

    def pool(self, geometry: bool=False, processes: int | None=None
             ) -> Tuple[Pool, WorkerType]:
        """Return a multiprocessing Pool and worker class for loading data.

        Args:
            geometry: Whether to apply the CrystFEL geometry in the worker function.
            processes: The number of worker processes to use. If None, defaults to the
                number of CPU cores.

        Returns:
            A tuple containing the Pool object and the worker class to use for loading data.

        Example:
            Example of using the pool to load data in parallel:

            >>> pool, worker = run.pool(geometry=True)
            >>> with pool:
            ...     results = pool.map(worker, run.indices())
        """
        if geometry:
            init_args = (self.data_path(), self.ss_idxs, self.fs_idxs,
                         self.config.geometry())
            worker_class = self.worker_geom_class()
        else:
            init_args = (self.data_path(), self.ss_idxs, self.fs_idxs)
            worker_class = self.worker_class()

        pool = Pool(processes=processes, initializer=worker_class.initializer, initargs=init_args)
        return pool, worker_class.run

    def meta_worker(self, attr: str) -> LoadWorker[NDArray]:
        """Return a worker for loading metadata. A worker is a callable that takes an index from
        :meth:`indices` and returns the corresponding data array for the specified attribute.

        Args:
            attr: The attribute to load.

        Returns:
            An XFELWorker instance for loading the specified attribute.

        Example:
            Example of using the metadata worker:

            >>> meta_worker = run.meta_worker('pulse_id')
            >>> pulse_ids = []
            >>> for index in run.indices()[:10]:
            ...     pulse_ids.append(meta_worker(index))
        """
        return self.worker_class()(self.data_path(attr), None, None)

    def worker(self, geometry: bool = False) -> LoadWorker[NDArray]:
        """Return a worker for loading data frames. A worker is a callable that takes an index from
        :meth:`indices` and returns the corresponding data array for the specified attribute.

        Args:
            geometry: Whether to apply the CrystFEL geometry in the worker function.

        Returns:
            An XFELWorker instance for loading data frames.

        Example:
            Example of using the data worker:

            >>> data_worker = run.worker(geometry=True)
            >>> frames = []
            >>> for index in run.indices()[:10]:
            ...     frames.append(data_worker(index))
        """
        if geometry:
            return self.worker_geom_class()(self.data_path(), self.ss_idxs, self.fs_idxs,
                                            self.config.geometry())
        return self.worker_class()(self.data_path(), self.ss_idxs, self.fs_idxs)

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

    def __getitem__(self, key: slice | int | IntSequence) -> "FileStackIndices":
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

    def records(self) -> Iterator[TrainIndexRecord]:
        for index, file_indices in zip(self.index(), self):
            filenames, file_idxs = zip(*file_indices)
            yield TrainIndexRecord(index, tuple(filenames), tuple(file_idxs))

    def split(self, num_chunks: int) -> Iterator["FileStackIndices"]:
        if self.indices is None:
            indices = list(range(self.total))
        else:
            indices = self.indices

        for chunk in split(indices, num_chunks):
            yield FileStackIndices(file_indices=self.file_indices, indices=chunk)

@dataclass
class XFELConfig(FELConfig[List[H5Files]]):
    """Run configuration for European XFEL experiments.

    Locates per-module HDF5 files by scanning a per-run data directory for
    filenames that match a module-indexed pattern, then assembles them into
    an :class:`XFELRun` via :func:`open_run`.

    Attributes:
        data_dir: Format string for the per-run data directory; the run ID
            is substituted with ``str.format`` (e.g.
            ``'/gpfs/exfel/exp/SPB/202302/p004456/proc/r{0:04d}'``).
        hdf5_protocol: Path to the JSON or INI file read by
            :meth:`H5Protocol.read`.
        file_pattern: Python format string converted to a regex; positional
            fields are filled with ``(run_id, module_id)`` in that order.
            Regex metacharacters must be double-escaped (e.g.
            ``'CORR-R{0:04d}-JNGFR{1:02d}-S(\\d{{5}})\\.h5'``).
        geometry_file: Path to the CrystFEL ``.geom`` geometry file.
        num_modules: Number of detector modules to read (default ``1``).
        starts_at: Index of the first detector module; modules
            ``[starts_at, starts_at + num_modules)`` are loaded
            (default ``0``).

    Example:
        Open a run with an XFEL configuration JSON file and load the first 10 frames with
        geometry applied:

        >>> config = XFELConfig.read('xfel_config.json')
        >>> run = open_run(100, config)
        >>> frames = run.data(run.indices()[:10], geometry=True)
    """
    data_dir        : str
    hdf5_protocol   : str
    file_pattern    : str
    geometry_file   : str
    num_modules     : int = 1
    starts_at       : int = 0
    facility        : Facility = field(default='XFEL', kw_only=True)

    def num_files(self, locator: int | RunLocator) -> int:
        for files in self.module_files(locator):
            return len(files)
        return 0

    def scan_dir(self, locator: int | RunLocator) -> str:
        """Return the directory path to scan for HDF5 files for the given run ID.

        Args:
            locator: The run locator or run ID to locate the data directory for.

        Returns:
            The directory path as a string.
        """
        locator = RunLocator.coerce(locator)
        if locator.variant is not None:
            raise ValueError("Run variants are not supported for XFEL runs.")
        return self.data_dir.format(locator.run_id)

    def module_files(self, locator: int | RunLocator) -> Iterator[List[str]]:
        """Yield lists of HDF5 file paths for each detector module in the run.

        Args:
            locator: The run locator or run ID to locate files for.

        Yields:
            Lists of file paths corresponding to each detector module, in order.
        """
        locator = RunLocator.coerce(locator)
        data_dir = self.scan_dir(locator)

        for module_id in range(self.starts_at, self.starts_at + self.num_modules):
            pattern = self.file_pattern.format(locator.run_id, module_id)
            module_files = []
            for path in os.listdir(data_dir):
                if re.match(pattern, path):
                    module_files.append(os.path.join(data_dir, path))
            yield module_files

    def filenames(self, locator: int | RunLocator) -> List[str]:
        """Return a list of all HDF5 file paths for the given run ID.

        Args:
            locator: The run locator or run ID to locate files for.

        Returns:
            A list of file paths as strings.
        """
        filenames = []
        for module_files in self.module_files(locator):
            filenames.extend(module_files)
        return filenames

    def files(self, locator: int | RunLocator) -> List[H5Files]:
        """Returns a list of H5Files objects, one for each module.

        Args:
            locator: The run locator or run ID to load files for.

        Returns:
            A list where each element corresponds to all HDF5 files pertaining to a
            specific detector module.
        """
        files = []
        for module_files in self.module_files(locator):
            files.append(H5Files(module_files))

        return files

    def geometry(self) -> Geometry:
        """Return the CrystFEL detector geometry for the run.

        Returns:
            A Geometry object representing the detector layout.
        """
        return read_crystfel(self.geometry_file)

    def protocol(self) -> H5Protocol:
        """Return the HDF5 protocol for the run.

        Returns:
            An H5Protocol object representing the HDF5 protocol.
        """
        return H5Protocol.read(self.hdf5_protocol)

xfel_worker : Callable[[Tuple[Tuple[str, Indices], ...]], NDArray]
XFELPathType = str | Tuple[str, ...]
XFELIndexType = Tuple[Tuple[str, Indices], ...]

@dataclass
class XFELReadWorker(BaseFELReadWorker[XFELPathType, XFELIndexType]):
    data_path   : XFELPathType
    ss_indices  : Indices | None
    fs_indices  : Indices | None

    def __call__(self, indices: XFELIndexType) -> NDArray:
        data_arrays = []

        if isinstance(self.data_path, tuple):
            for data_path, index in zip(self.data_path, indices):
                data_arrays.append(self.load(data_path, index))
        else:
            for index in indices:
                data_arrays.append(self.load(self.data_path, index))
        return NumPy.stack(data_arrays, axis=0)

    def load(self, data_path: str, index: Tuple[str, Indices]) -> NDArray:
        return load_dataset(data_path, index, self.ss_indices, self.fs_indices)

    @classmethod
    def initializer(cls, data_path: XFELPathType, ss_indices: Indices,
                    fs_indices: Indices):
        global xfel_worker
        xfel_worker = cls(data_path, ss_indices, fs_indices)

    @staticmethod
    def run(index: XFELIndexType) -> NDArray:
        return xfel_worker(index)

class XFELReadWithGeomWorker(BaseFELReadWithGeomWorker[XFELPathType, XFELIndexType]):
    @classmethod
    def worker_class(cls) -> Type[XFELReadWorker]:
        return XFELReadWorker

    @classmethod
    def initializer(cls, data_path: XFELPathType, ss_indices: Indices, fs_indices: Indices,
                    geometry: Geometry):
        global xfel_worker
        xfel_worker = cls(data_path, ss_indices, fs_indices, geometry)

@dataclass
class XFELRun(BaseFELRun[XFELPathType, FileStackIndices, XFELConfig]):
    """Detector run for European XFEL multi-module experiments.

    Reads data from the per-module HDF5 files described by an
    :class:`XFELConfig`. Module arrays are stacked along a new leading
    axis for each frame, so the returned shape is
    ``(n_frames, n_modules, *frame_shape)`` when ``geometry=False``.

    Attributes:
        run_id: Numeric run identifier.
        config: :class:`XFELConfig` with file locations and protocol.
        ss_idxs: Slow-scan (row) pixel indices for ROI selection; ``None``
            loads full rows.
        fs_idxs: Fast-scan (column) pixel indices for ROI selection; ``None``
            loads full columns.

    Example:
        Open a run with a :class:`XFELConfig` and load the first 20 frames with geometry
        applied using 4 worker processes:

        >>> run = open_run(100, config)
        >>> indices = run.indices()
        >>> frames = run.data(indices[:20], geometry=True, n_processes=4)
    """
    run_id      : int
    config      : XFELConfig
    ss_idxs     : Indices | None = None
    fs_idxs     : Indices | None = None

    @classmethod
    def worker_class(cls) -> Type[XFELReadWorker]:
        return XFELReadWorker

    @classmethod
    def worker_geom_class(cls) -> Type[XFELReadWithGeomWorker]:
        return XFELReadWithGeomWorker

    def data_path(self, attr: str='data') -> XFELPathType:
        """Return the HDF5 data path for the given attribute.

        Args:
            attr: The attribute name to locate the data path for.

        Returns:
            The data path as a string or a tuple of strings if multiple paths are found.
        """
        if attr not in self.data_paths:
            data_paths = []
            for module_files in self.config.files(self.run_locator):
                data_path = ''
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

    def _load_indices(self) -> FileStackIndices:
        """Load indices from HDF5 files (uncached implementation)."""
        data_paths = []
        file_indices = []
        for module_files in self.config.files(self.run_locator):
            indices = self.handler.indices(module_files, 'data')

            if data_paths:
                if data_paths[-1] != indices.data_path:
                    data_paths.append(indices.data_path)
            else:
                data_paths.append(indices.data_path)

            file_indices.append(indices.indices)

        self.data_paths['data'] = tuple(data_paths) if len(data_paths) > 1 else data_paths[0]
        return FileStackIndices(file_indices)

@dataclass
class SwissFELConfig(FELConfig[H5Files]):
    """Run configuration for SwissFEL experiments.

    Locates HDF5 data files by scanning a per-run data directory for
    filenames that match a regex pattern, then assembles them into a
    :class:`SwissFELRun` via :func:`open_run`.

    Attributes:
        data_dir: Format string for the per-run data directory; the run ID is
            substituted with ``str.format`` (e.g.
            ``'/sf/bernina/data/p19000/raw/r{0:04d}'``).
        hdf5_protocol: Path to the JSON or INI file read by
            :meth:`H5Protocol.read`.
        file_pattern: Regular expression matched directly against filenames in
            ``data_dir`` to select HDF5 files for the run
            (e.g. ``'run_\\d{6}\\.h5'``).
        geometry_file: Path to the CrystFEL ``.geom`` geometry file.

    Example:
        Open a run config from a JSON file and load the first 10 frames with geometry applied:

        >>> config = SwissFELConfig.read('sfel_config.json')
        >>> run = open_run(50, config)
        >>> frames = run.data(run.indices()[:10], geometry=True)
    """
    data_dir        : str
    hdf5_protocol   : str
    file_pattern    : str
    geometry_file   : str
    facility        : Facility = field(default='SwissFEL', kw_only=True)

    def num_files(self, locator: int | RunLocator) -> int:
        """Return the number of HDF5 files in the run.

        Args:
            locator: The run locator or run ID to locate files for.

        Returns:
            The number of HDF5 files as an integer.
        """
        return len(self.filenames(locator))

    def scan_dir(self, locator: int | RunLocator) -> str:
        """Return the directory path to scan for HDF5 files for the given run ID.

        Args:
            locator: The run locator or run ID to locate the data directory for.

        Returns:
            The directory path as a string.
        """
        locator = RunLocator.coerce(locator)
        if locator.variant is not None:
            raise ValueError("Run variants are not supported for SwissFEL runs.")
        return self.data_dir.format(locator.run_id)

    def filenames(self, locator: int | RunLocator) -> List[str]:
        """Return a list of all HDF5 file paths for the given run ID.

        Args:
            locator: The run locator or run ID to locate files for.

        Returns:
            A list of file paths as strings.
        """
        locator = RunLocator.coerce(locator)
        data_dir = self.scan_dir(locator)

        filenames = []
        for path in os.listdir(data_dir):
            if re.match(self.file_pattern, path):
                filenames.append(os.path.join(data_dir, path))
        return filenames

    def files(self, locator: int | RunLocator) -> H5Files:
        """Return an H5Files object containing all HDF5 files for the given run ID.

        Args:
            locator: The run locator or run ID to load files for.

        Returns:
            An H5Files object containing all HDF5 files for the run.
        """
        return H5Files(self.filenames(locator))

    def geometry(self) -> Geometry:
        """Return the CrystFEL detector geometry for the run.

        Returns:
            A Geometry object representing the detector layout.
        """
        return read_crystfel(self.geometry_file)

    def protocol(self) -> H5Protocol:
        """Return the HDF5 protocol for the run.

        Returns:
            An H5Protocol object representing the HDF5 protocol.
        """
        return H5Protocol.read(self.hdf5_protocol)

fel_worker : Callable[[Tuple[str, Indices]], NDArray]

@dataclass
class FELReadWorker(BaseFELReadWorker[str, Tuple[str, Indices]]):
    data_path   : str
    ss_indices  : Indices | None
    fs_indices  : Indices | None

    def __call__(self, indices: Tuple[str, Indices]) -> NDArray:
        return load_dataset(self.data_path, indices, self.ss_indices, self.fs_indices)

    @classmethod
    def initializer(cls, data_path: str, ss_indices: Indices, fs_indices: Indices):
        global fel_worker
        fel_worker = cls(data_path, ss_indices, fs_indices)

    @staticmethod
    def run(index: Tuple[str, Indices]) -> NDArray:
        return fel_worker(index)

class FELReadWithGeomWorker(BaseFELReadWithGeomWorker[str, Tuple[str, Indices]]):
    @classmethod
    def worker_class(cls) -> Type[FELReadWorker]:
        return FELReadWorker

    @classmethod
    def initializer(cls, data_path: str, ss_indices: Indices, fs_indices: Indices,
                    geometry: Geometry):
        global fel_worker
        fel_worker = cls(data_path, ss_indices, fs_indices, geometry)

@dataclass
class SwissFELRun(BaseFELRun[str, StackIndices, SwissFELConfig]):
    """Detector run for SwissFEL experiments.

    Reads data from a single-detector HDF5 file collection described by a
    :class:`SwissFELConfig`.

    Attributes:
        run_id: Numeric run identifier.
        config: :class:`SwissFELConfig` with file locations and protocol.
        ss_idxs: Slow-scan (row) pixel indices for ROI selection; ``None``
            loads full rows.
        fs_idxs: Fast-scan (column) pixel indices for ROI selection; ``None``
            loads full columns.

    Example:
        >>> run = open_run(50, config)
        >>> indices = run.indices()
        >>> frames = run.data(indices[:20], geometry=True)
    """
    run_id      : int
    config      : SwissFELConfig
    ss_idxs     : Indices | None = None
    fs_idxs     : Indices | None = None

    @classmethod
    def worker_class(cls) -> Type[FELReadWorker]:
        return FELReadWorker

    @classmethod
    def worker_geom_class(cls) -> Type[FELReadWithGeomWorker]:
        return FELReadWithGeomWorker

    def data_path(self, attr: str='data') -> str:
        """Return the HDF5 data path for the given attribute.

        Args:
            attr: The attribute name to locate the data path for.

        Returns:
            The data path as a string.
        """
        if attr not in self.data_paths:
            for file in self.config.files(self.run_locator).visit_files():
                self.data_paths[attr] = self.handler.protocol.find_path(attr, file)
                if self.data_paths[attr]:
                    break

        if not self.data_paths[attr]:
            raise ValueError(f"Attribute '{attr}' not found in protocol for any module.")
        return self.data_paths[attr]

    def _load_indices(self) -> StackIndices:
        """Load indices from HDF5 files (uncached implementation)."""
        indices = self.handler.indices(self.config.files(self.run_locator), 'data')
        self.data_paths['data'] = indices.data_path
        return indices.indices

@dataclass
class LCLSConfig(FELConfig[H5Files]):
    """Run configuration for LCLS experiments.

    Locates HDF5 data files by scanning a per-run data directory for
    filenames that match a regex pattern, then assembles them into a
    :class:`LCLSRun` via :func:`open_run`.

    Attributes:
        data_dir: Format string for the per-run data directory; the run ID is
            substituted with ``str.format`` (e.g.
            ``'/sf/bernina/data/p19000/raw/r{0:04d}'``).
        hdf5_protocol: Path to the JSON or INI file read by
            :meth:`H5Protocol.read`.
        file_pattern: Regular expression matched directly against filenames in
            ``data_dir`` to select HDF5 files for the run
            (e.g. ``'run_\\d{6}\\.h5'``).
        geometry_file: Path to the CrystFEL ``.geom`` geometry file.
        variant: Optional run variant, interpreted as an LCLS directory suffix.

    Example:
        Open a run config from a JSON file and load the first 10 frames with geometry applied:

        >>> config = LCLSConfig.read('lcls_config.json')
        >>> run = open_run(50, config)
        >>> frames = run.data(run.indices()[:10], geometry=True)
    """
    data_dir        : str
    hdf5_protocol   : str
    file_pattern    : str
    geometry_file   : str
    facility        : Facility = field(default='LCLS', kw_only=True)

    def num_files(self, locator: int | RunLocator) -> int:
        """Return the number of HDF5 files in the run.

        Args:
            locator: The run locator or run ID to locate files for.

        Returns:
            The number of HDF5 files as an integer.
        """
        return len(self.filenames(locator))

    def scan_dir(self, locator: int | RunLocator) -> str:
        """Return the directory path to scan for HDF5 files for the given run ID.

        Args:
            locator: The run locator or run ID to locate the data directory for.
                ``locator.variant`` is used as an optional directory suffix
                (e.g. '-proc').

        Returns:
            The directory path as a string.
        """
        locator = RunLocator.coerce(locator)
        data_dir = self.data_dir.format(locator.run_id)
        parent_dir = os.path.dirname(data_dir)
        scan_folder = os.path.basename(data_dir)
        dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
                if d.startswith(scan_folder)]
        if len(dirs) == 0:
            raise ValueError(f"No directories found matching pattern: {data_dir}")
        if len(dirs) > 1:
            if locator.variant is None:
                raise ValueError(f"Multiple directories found matching pattern: {data_dir},"
                                 f" specify a run variant to disambiguate.")
            matches = [d for d in dirs if d.endswith(locator.variant)]
            if not matches:
                raise ValueError(f"No directories found matching pattern: {data_dir}"
                                 f" with variant {locator.variant!r}.")
            if len(matches) > 1:
                raise ValueError(f"Multiple directories found matching pattern: {data_dir}"
                                 f" with variant {locator.variant!r}.")
            return matches[0]
        return dirs[0]

    def filenames(self, locator: int | RunLocator) -> List[str]:
        """Return a list of all HDF5 file paths for the given run ID.

        Args:
            locator: The run locator or run ID to locate files for.

        Returns:
            A list of file paths as strings.
        """
        locator = RunLocator.coerce(locator)
        data_dir = self.scan_dir(locator)

        filenames = []
        for path in os.listdir(data_dir):
            if re.match(self.file_pattern.format(locator.run_id), path):
                filenames.append(os.path.join(data_dir, path))
        return filenames

    def files(self, locator: int | RunLocator) -> H5Files:
        """Return an H5Files object containing all HDF5 files for the given run ID.

        Args:
            locator: The run locator or run ID to load files for.

        Returns:
            An H5Files object containing all HDF5 files for the run.
        """
        return H5Files(self.filenames(locator))

    def geometry(self) -> Geometry:
        """Return the CrystFEL detector geometry for the run."""
        return read_crystfel(self.geometry_file)

    def protocol(self) -> H5Protocol:
        """Return the HDF5 protocol for the run."""
        return H5Protocol.read(self.hdf5_protocol)

@dataclass
class LCLSRun(BaseFELRun[str, StackIndices, LCLSConfig]):
    """Detector run for LCLS experiments.

    Reads data from a single-detector HDF5 file collection described by a
    :class:`LCLSConfig`.

    Attributes:
        run_id: Numeric run identifier.
        config: :class:`LCLSConfig` with file locations and protocol.
        ss_idxs: Slow-scan (row) pixel indices for ROI selection; ``None``
            loads full rows.
        fs_idxs: Fast-scan (column) pixel indices for ROI selection; ``None``
            loads full columns.

    Example:
        >>> run = open_run(50, config)
        >>> indices = run.indices()
        >>> frames = run.data(indices[:20], geometry=True)
    """
    run_id      : int
    config      : LCLSConfig
    ss_idxs     : Indices | None = None
    fs_idxs     : Indices | None = None
    variant     : str | None = None

    @classmethod
    def worker_class(cls) -> Type[FELReadWorker]:
        return FELReadWorker

    @classmethod
    def worker_geom_class(cls) -> Type[FELReadWithGeomWorker]:
        return FELReadWithGeomWorker

    @property
    def run_locator(self) -> RunLocator:
        return RunLocator(self.run_id, self.variant)

    def data_path(self, attr: str='data') -> str:
        """Return the HDF5 data path for the given attribute.

        Args:
            attr: The attribute name to locate the data path for.

        Returns:
            The data path as a string.
        """
        if attr not in self.data_paths:
            for file in self.config.files(self.run_locator).visit_files():
                self.data_paths[attr] = self.handler.protocol.find_path(attr, file)
                if self.data_paths[attr]:
                    break

        if not self.data_paths[attr]:
            raise ValueError(f"Attribute '{attr}' not found in protocol for any module.")
        return self.data_paths[attr]

    def _load_indices(self) -> StackIndices:
        """Load indices from HDF5 files (uncached implementation)."""
        indices = self.handler.indices(self.config.files(self.run_locator), 'data')
        self.data_paths['data'] = indices.data_path
        return indices.indices

AnyRun = XFELRun | SwissFELRun | LCLSRun
RunListIndex = Tuple[int, Any]

@dataclass
class RunListIndices(TrainIndices):
    """Flatten frame indices from an ordered collection of runs.

    Iteration qualifies each native child index with the position of its run,
    allowing :class:`RunList` workers to dispatch without interpreting the
    facility-specific index representation.

    Attributes:
        run_indices: Native frame indices for each run in collection order.
        indices: Optional global frame positions selecting from the flattened
            collection.
    """
    run_indices     : List[TrainIndices]
    indices         : List[int] | None = None

    def __post_init__(self):
        self.offsets = [0]
        for indices in self.run_indices:
            self.offsets.append(self.offsets[-1] + len(indices))
        self.total = self.offsets[-1]

    def _selected_positions(self) -> Iterator[Tuple[int, int, int]]:
        """Yield global, run, and local positions for the selected frames."""
        run_index = 0
        previous = -1
        for index in self.indices or []:
            if index < 0:
                index += self.total
            if index < 0 or index >= self.total:
                raise IndexError(f"Index {index} is out of bounds for length {self.total}")

            if index < previous:
                run_index = int(NumPy.searchsorted(self.offsets, index, side='right') - 1)
            else:
                while index >= self.offsets[run_index + 1]:
                    run_index += 1

            yield index, run_index, index - self.offsets[run_index]
            previous = index

    def __iter__(self) -> Iterator[RunListIndex]:
        if self.indices is None:
            for run_index, indices in enumerate(self.run_indices):
                for native_index in indices:
                    yield run_index, native_index
        else:
            for _, run_index, local_index in self._selected_positions():
                native_index = next(iter(self.run_indices[run_index][local_index]))
                yield run_index, native_index

    def __getitem__(self, key: Indices) -> "RunListIndices":
        if isinstance(key, slice):
            key_list = list_indices(key, len(self))
        else:
            key_list = to_list(cast(IntSequence, key))
        positions = list(self.index())
        return RunListIndices(self.run_indices, [positions[index] for index in key_list])

    def __len__(self) -> int:
        return self.total if self.indices is None else len(self.indices)

    def index(self) -> Iterator[int]:
        if self.indices is None:
            yield from range(self.total)
        else:
            yield from self.indices

    def records(self) -> Iterator[TrainIndexRecord]:
        if self.indices is None:
            for run_index, indices in enumerate(self.run_indices):
                for local_index, record in enumerate(indices.records()):
                    index = self.offsets[run_index] + local_index
                    yield TrainIndexRecord(index, record.filename, record.file_index)
        else:
            for index, run_index, local_index in self._selected_positions():
                record = next(self.run_indices[run_index][local_index].records())
                yield TrainIndexRecord(index, record.filename, record.file_index)

    def split(self, num_chunks: int) -> Iterator["RunListIndices"]:
        positions = list(self.index())
        for chunk in split(positions, num_chunks):
            yield RunListIndices(self.run_indices, cast(List[int], chunk))

run_list_worker : RunListWorker[Any]
T_Output = TypeVar('T_Output')

@dataclass
class RunListWorker(LoadWorker[T_Output]):
    """Dispatch run-qualified indices to the corresponding child worker."""
    workers     : List[LoadWorker[T_Output]]

    def __call__(self, index: RunListIndex) -> T_Output:
        run_index, native_index = index
        return self.workers[run_index](native_index)

    @classmethod
    def initializer(cls, workers: List[LoadWorker[T_Output]]):
        global run_list_worker
        run_list_worker = cls(workers)

    @staticmethod
    def run(index: RunListIndex) -> T_Output:
        return run_list_worker(index)

@dataclass
class RunList(BaseRun[range | List[int], RunListIndices, RunConfig]):
    """Expose an ordered collection of facility runs through the run interface.

    Each child run retains ownership of file discovery, index caching, geometry,
    and HDF5 loading. This class only flattens their frame indices and dispatches
    run-qualified work to the corresponding child.

    Attributes:
        run_id: Non-empty list or range of numeric run identifiers.
        config: Shared facility configuration used to open every child run.
        variant: Optional facility-specific discriminator applied to every run.
    """
    run_id      : range | List[int]
    config      : RunConfig
    variant     : str | None = None
    runs        : Tuple[BaseRun[int, Any, Any], ...] = field(init=False)

    def __post_init__(self):
        if len(self.run_id) == 0:
            raise ValueError("RunList requires at least one run ID.")
        self.run_id = list(self.run_id)
        self.runs = tuple(open_run(run_id, self.config, variant=self.variant)
                          for run_id in self.run_id)

    def attributes(self) -> List[str]:
        """Return metadata attributes exposed by the shared run configuration."""
        return self.runs[0].attributes()

    def indices(self) -> RunListIndices:
        """Return the flattened, run-qualified frame indices."""
        return RunListIndices([run.indices() for run in self.runs])

    def pool(self, geometry: bool = False, processes: int | None = None
             ) -> Tuple[Pool, Callable[[RunListIndex], NDArray]]:
        """Return a pool dispatching frame loads to child run workers."""
        workers = [run.worker(geometry) for run in self.runs]
        pool = Pool(processes=processes, initializer=RunListWorker.initializer,
                    initargs=(workers,))
        return pool, RunListWorker.run

    def meta_worker(self, attr: str) -> RunListWorker[NDArray]:
        """Return a worker dispatching metadata loads to child runs."""
        return RunListWorker([run.meta_worker(attr) for run in self.runs])

    def worker(self, geometry: bool = False) -> RunListWorker[NDArray]:
        """Return a worker dispatching detector frame loads to child runs."""
        return RunListWorker([run.worker(geometry) for run in self.runs])


@overload
def open_run(run_id: int, config: XFELConfig) -> XFELRun: ...

@overload
def open_run(run_id: int, config: SwissFELConfig) -> SwissFELRun: ...

@overload
def open_run(run_id: int, config: LCLSConfig, *, variant: str | None = None) -> LCLSRun: ...

@overload
def open_run(run_id: range | List[int], config: RunConfig, *, variant: str | None = None
             ) -> RunList: ...

@overload
def open_run(run_id: int, config: RunConfig, *, variant: str | None = None
             ) -> BaseRun[int, Any, Any]: ...

def open_run(run_id: int | range | List[int], config: RunConfig, *, variant: str | None = None
             ) -> BaseRun[int, Any, Any] | AnyRun | RunList:
    """Create a run object from a run ID and facility configuration.

    Dispatches on the concrete type of *config* and returns the matching
    :class:`BaseRun` subclass: :class:`XFELRun` for :class:`XFELConfig`,
    :class:`SwissFELRun` for :class:`SwissFELConfig`, and :class:`LCLSRun`
    for :class:`LCLSConfig`.

    Args:
        run_id: Numeric identifier, list of identifiers, or range of runs to open.
        config: Facility-specific run configuration.
        variant: Optional facility-specific run discriminator. For LCLS, this is used
            as the run directory suffix.

    Returns:
        A facility-specific run for one identifier, or :class:`RunList` for a
        list or range of identifiers.

    Raises:
        ValueError: If *config* is not a recognized :class:`RunConfig` subclass.

    Example:
        Open a run with an XFEL configuration JSON file and load the first 10 frames with
        geometry applied:

        >>> config = XFELConfig.read('xfel_config.json')
        >>> run = open_run(100, config)
        >>> frames = run.data(run.indices()[:10], geometry=True)
    """
    if isinstance(run_id, (list, range)):
        return RunList(run_id, config, variant)
    if isinstance(config, LCLSConfig):
        return LCLSRun(run_id, config, variant=variant)
    if variant is not None:
        raise ValueError(f"Run variants are not supported for {config.facility} runs.")
    if isinstance(config, XFELConfig):
        return XFELRun(run_id, config)
    if isinstance(config, SwissFELConfig):
        return SwissFELRun(run_id, config)
    raise ValueError(f"Unsupported RunConfig type: {type(config)}")
