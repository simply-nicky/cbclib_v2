from argparse import ArgumentParser
from dataclasses import dataclass
import json
from multiprocessing import cpu_count
import os
import re
from shlex import quote
import stat
import tempfile
from typing import Any, ClassVar, Iterator, List, Literal, Tuple, Type
import h5py
from jax import config as jax_config
import pandas as pd
from tqdm.auto import tqdm
from .. import cuda
from ..cuda import Allocator
from .._src.annotations import AnyNamespace, IntArray, JaxNumPy, NumPy
from .._src.array_api import asnumpy, default_api, default_rng, Platform
from .._src.config import CPUConfig
from .._src.crystfel import Detector
from .._src.data_container import compute_index
from .._src.data_processing import CrystMetadata
from .._src.cxi_protocol import H5Handler, TrainIndices, write_hdf
from .._src.parser import read_all
from .._src.run import BaseRun, RunConfig, open_run
from .._src.scripts import (BaseParameters, FinderConfig, IndexingConfig, MetadataParameters,
                            PostRefineConfig, RefineConfig, RegionFinderConfig,
                            StreakFinderConfig)
from .._src.scripts import create_metadata, pool_detection, pool_indexing, scale_background
from .._src.streaks import StackedStreaks, Streaks
from ..indexer import (BaseSetup, FixedGeometry, FixedLens, LinePoints, Miller, Patterns,
                       ResolvedSetup, XtalCell, XtalState)
from ..scaler import FullState, ScalerModel, ScalerState
from .slurm_manager import SLURMScript, ScriptSpec

@dataclass
class SystemConfig(BaseParameters):
    """Compute backend and thread-count configuration.

    Corresponds to the ``"system"`` section of ``scan.json`` (see
    :doc:`/workflows`).  Controls the compute backend and OpenMP thread
    count for every CLI and SLURM batch-pipeline step.

    Attributes:
        platform: Compute backend — ``'cpu'`` or ``'gpu'``.
        cuda_allocator: Unified GPU allocator mode for cbclib CUDA kernels,
            CuPy, and JAX/XLA. Supported values are:

            * ``"default"`` — use each backend's default allocator. This is
              the safest mode and the CLI default.
            * ``"cuda_malloc_async"`` — request CUDA stream-ordered
              allocation across all three backends. Intended for mixed
              cbclib/CuPy/JAX GPU workloads and requires a compatible GPU
              node plus matching driver/runtime support.

        num_threads: Number of OpenMP threads.  ``0`` or negative values
            are replaced by :func:`multiprocessing.cpu_count` at
            initialisation.
    """
    platform        : Platform
    cuda_allocator  : Allocator = 'default'
    num_threads     : int = 0

    def __post_init__(self):
        if self.num_threads <= 0:
            self.num_threads = cpu_count()
        if self.platform not in ['cpu', 'gpu']:
            raise ValueError(f"Invalid platform: {self.platform}")
        if self.cuda_allocator not in ('default', 'cuda_malloc_async'):
            raise ValueError(f"Invalid CUDA allocator: {self.cuda_allocator}")

    def apply(self) -> None:
        """Apply system-level runtime configuration for this CLI process.

        For GPU runs this must happen before data loading or any GPU backend
        initialises its allocator state.  Unsupported async allocator setup
        raises immediately.  CPU runs skip allocator configuration.

        Raises:
            RuntimeError: If ``platform == 'gpu'`` and allocator setup fails.
        """
        if self.platform == 'cpu':
            return
        cuda.set_allocator(self.cuda_allocator, strict=True)

    def cpu_config(self) -> CPUConfig:
        """Return a :class:`~cbclib_v2.CPUConfig` context manager for this thread count."""
        return CPUConfig(num_threads=self.num_threads)

    def array_api(self) -> AnyNamespace:
        """Return the array namespace matching :attr:`platform` (NumPy or CuPy)."""
        return default_api(self.platform)

    def jax_api(self) -> AnyNamespace:
        """Return JAX NumPy configured to use :attr:`platform` as its default backend."""
        jax_config.update("jax_platform_name", self.platform)
        return JaxNumPy

@dataclass
class DetectConfig(BaseParameters):
    """Hit-finding thresholds and output directories.

    Corresponds to the ``"detect"`` section of ``scan.json`` (see
    :doc:`/workflows`).  Used by ``cbclib_cli detect`` and the SLURM
    detection array job to decide which frames count as hits and where to
    write results.

    A frame is classified as a *hit* when the number of detected streaks (or
    regions) exceeds :attr:`hit_threshold`.

    Attributes:
        hit_threshold: Minimum number of detections per frame required to
            count as a hit.
        streaks_dir: Root directory for per-chunk streak detection output
            files.
        regions_dir: Root directory for per-chunk region detection output
            files.
    """

    hit_threshold   : int
    streaks_dir     : str
    regions_dir     : str

@dataclass
class MetadataConfig(BaseParameters):
    """Parameters for the background-whitefield computation step.

    Corresponds to the ``"metadata"`` section of ``scan.json`` (see
    :doc:`/workflows`).  Used by ``cbclib_cli metadata`` to control how many
    frames are sampled and where the resulting HDF5 metadata file is written.

    Attributes:
        n_frames: Number of frames randomly sampled from the run to
            average into the background whitefield.
        output_dir: Directory where the metadata HDF5 file is written.
    """
    n_frames        : int
    output_dir      : str

@dataclass
class MetaListConfig(BaseParameters):
    """Parameters for the PCA metalist computation step.

    Corresponds to the ``"metalist"`` section of ``scan.json`` (see
    :doc:`/workflows`).  Used by ``cbclib_cli metalist`` and the SLURM
    metalist array job to build the collection of background estimates that
    drives PCA-based background subtraction.

    Attributes:
        n_frames: Number of consecutive frames averaged per background
            estimate.
        spacing: Number of events between consecutive background-estimate
            centres.
        output_dir: Root directory for per-chunk metalist HDF5 files.
    """
    n_frames        : int
    spacing         : int
    output_dir      : str

@dataclass
class SetupConfig(BaseParameters):
    """Crystal geometry and unit-cell file paths.

    Corresponds to the ``"setup"`` section of ``scan.json`` (see
    :doc:`/workflows`).  Used by ``cbclib_cli index`` / ``cbclib_cli refine``
    and the corresponding SLURM jobs to locate detector geometry, crystal
    unit-cell information, indexed orientations, and refined solutions.

    Attributes:
        setup_file: Path to the detector geometry JSON file (read by
            :class:`~cbclib_v2.indexer.FixedGeometry`).
        unit_file: Path to the crystal unit-cell JSON file (read by
            :class:`~cbclib_v2.indexer.XtalCell`).
        reflections_dir: Directory where reflection lists are written.
        solutions_dir: Directory where per-chunk refined solutions are written.
        xtals_dir: Directory where per-chunk indexing results are written.
    """
    setup_file      : str
    unit_file       : str
    reflections_dir : str
    solutions_dir   : str
    xtals_dir       : str

    def unit_cell(self, xp: AnyNamespace=NumPy) -> XtalCell:
        """Load the crystal unit cell from :attr:`unit_file`.

        Args:
            xp: Array namespace for the loaded arrays.

        Returns:
            :class:`~cbclib_v2.indexer.XtalCell` instance.

        Raises:
            ValueError: If :attr:`unit_file` is empty.
        """
        if self.unit_file == str():
            raise ValueError("No crystal file provided")
        return XtalCell.read(self.unit_file, xp)

    def xtal(self, xp: AnyNamespace=NumPy) -> XtalState:
        """Return the crystal unit cell as a reciprocal-basis :class:`~cbclib_v2.indexer.XtalState`.

        Args:
            xp: Array namespace for the loaded arrays.

        Returns:
            :class:`~cbclib_v2.indexer.XtalState` with basis vectors.
        """
        return self.unit_cell(xp).to_basis()

    def geometry(self) -> FixedLens | FixedGeometry:
        """Load the fixed detector geometry from :attr:`setup_file`.

        Returns:
            :class:`~cbclib_v2.indexer.FixedGeometry` instance.

        Raises:
            ValueError: If :attr:`setup_file` is empty.
        """
        if self.setup_file == str():
            raise ValueError("No setup file provided")
        if 'defocus' in read_all(self.setup_file):
            return FixedGeometry.read(self.setup_file)
        return FixedLens.read(self.setup_file)

@dataclass
class ScanFiles(BaseParameters):
    """Scan-specific file and directory naming utilities.

    Provides helpers that translate a ``(scan_num, image_kind)`` pair into
    the canonical file and directory names used throughout the pipeline.

    Attributes:
        scan_num: Run number identifying the scan.
        image_kind: Image layout — ``'full'`` for assembled lab-frame
            images, ``'stacked'`` for per-module stacks.
    """
    scan_num            : int
    image_kind          : Literal['full', 'stacked']
    dir_pattern         : ClassVar[str] = 'scan_{scan_num:d}_{kind}'
    file_pattern        : ClassVar[str] = 'scan_{scan_num:d}_{kind}'

    def list_files(self, dir: str) -> Iterator[str]:
        """Yield paths of scan files found in *dir*.

        Matches files whose name follows the canonical
        ``scan_{num}_{kind}[_f{index}][_{suffix}]{ext}`` pattern.

        Args:
            dir: Directory to search.

        Yields:
            Absolute paths of matching files.
        """
        path = self.file_pattern.format(scan_num=self.scan_num, kind=self.image_kind)
        filename, extension = os.path.splitext(path)
        pattern = filename
        pattern += r'(_f[0-9]{4})?'
        pattern += r'(_([^.]+))?'
        pattern += re.escape(extension)

        for filename in os.listdir(dir):
            if re.match(pattern, filename):
                yield os.path.join(dir, filename)

    def list_dirs(self, dir: str) -> Iterator[str]:
        """Yield paths of scan subdirectories found in *dir*.

        Matches directories whose name follows the canonical
        ``scan_{num}_{kind}[_{suffix}]`` pattern.

        Args:
            dir: Parent directory to search.

        Yields:
            Absolute paths of matching subdirectories.
        """
        pattern = self.dir_pattern.format(scan_num=self.scan_num, kind=self.image_kind)
        pattern += r'(_([^.]+))?'

        for filename in os.listdir(dir):
            path = os.path.join(dir, filename)
            if os.path.isdir(path) and re.match(pattern, filename):
                yield path

    def scan_dir(self, suffix: str=str()) -> str:
        """Return the canonical scan directory name (without a parent path).

        Args:
            suffix: Optional suffix appended after an underscore.

        Returns:
            Directory name string, e.g. ``'scan_373_stacked'``.
        """
        dir = self.dir_pattern.format(scan_num=self.scan_num, kind=self.image_kind)
        if suffix:
            dir += f'_{suffix}'
        return dir

    def scan_subdir(self, dir: str, suffix: str=str()) -> str:
        """Return the absolute path to the canonical scan subdirectory inside *dir*.

        Args:
            dir: Parent directory.
            suffix: Optional suffix for the subdirectory name.

        Returns:
            Absolute path string.
        """
        return os.path.join(dir, self.scan_dir(suffix))

    def scan_file(self, file_index: int | None=None, /, *, suffix: str=str(), extension: str='.h5',
                  dir: str | None=None, make_dirs: bool=False) -> str:
        """Build the canonical output file path for this scan.

        The returned path follows the pattern
        ``{dir}/scan_{num}_{kind}[_f{index:04d}][_{suffix}]{extension}``.

        Args:
            file_index: Optional zero-based chunk index appended as
                ``_f{index:04d}``.  ``None`` omits the chunk suffix.
            suffix: Optional string appended after the chunk index.
            extension: File extension including the leading dot.
                Defaults to ``'.h5'``.
            dir: Parent directory.  When provided, the file is placed
                inside :meth:`scan_subdir` (if *file_index* is given) or
                directly in *dir* (if *file_index* is ``None``).

        Returns:
            File path string.
        """
        def get_filename(file_index: int | None, suffix: str, extension: str) -> str:
            filename = self.file_pattern.format(scan_num=self.scan_num, kind=self.image_kind)
            if file_index is not None:
                filename += f'_f{file_index:04d}'
            if suffix:
                filename += f'_{suffix}'
            return filename + extension

        if dir is None:
            file = get_filename(file_index, suffix, extension)
        elif file_index is None:
            filename = get_filename(None, suffix, extension)
            file = os.path.join(dir, filename)
        else:
            file = os.path.join(self.scan_subdir(dir, suffix),
                                get_filename(file_index, str(), extension))

        if make_dirs:
            dir_path = os.path.dirname(file)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        return file

@dataclass
class ScanConfig(ScanFiles):
    """Top-level experiment configuration shared by all three workflows.

    Aggregates all per-step configurations and provides helpers for opening
    the run, locating metadata files, and building output paths.  Loaded from
    a JSON file via :meth:`~cbclib_v2.scripts.BaseParameters.read`.

    See :doc:`/workflows` for the JSON format and a worked example.

    Attributes:
        data: Facility data source configuration (HDF5 layout, geometry
            file, module count).
        setup: Crystal geometry and unit-cell file paths.
        detect: Hit-finding thresholds and output directories.
        metadata: Background-whitefield computation parameters.
        metalist: PCA metalist computation parameters.
        system: Compute backend and thread count.
    """
    data            : RunConfig
    setup           : SetupConfig
    detect          : DetectConfig
    metadata        : MetadataConfig
    metalist        : MetaListConfig
    system          : SystemConfig

    @property
    def apply_geometry(self) -> bool:
        """Whether to assemble per-module stacks into lab-frame images on load.

        ``True`` for ``image_kind == 'full'``; ``False`` for
        ``image_kind == 'stacked'``.
        """
        if self.image_kind == 'full':
            return True
        if self.image_kind == 'stacked':
            return False
        raise ValueError(f"Invalid image_kind keyword: {self.image_kind}")

    def __post_init__(self):
        if self.detect.streaks_dir == self.metadata.output_dir:
            raise ValueError("Streaks and metadata must be saved to different folders")
        if self.detect.regions_dir == self.metadata.output_dir:
            raise ValueError("Regions and metadata must be saved to different folders")
        if self.metadata.output_dir == self.metalist.output_dir:
            raise ValueError("Metadata and metalist must be saved to different folders")

    def find_metadata(self, file_index: int | None=None) -> str:
        """Return the path to the best available metadata file for *file_index*.

        Searches in priority order: per-chunk metalist file → combined
        metalist file → single metadata file.

        Args:
            file_index: Chunk index.  When provided, the per-chunk metalist
                file is tried first.

        Returns:
            Path to the first existing metadata or metalist HDF5 file.

        Raises:
            ValueError: If no metadata file is found for the given scan and
                chunk.
        """
        if file_index is not None:
            metalist_path = self.scan_file(file_index, dir=self.metalist.output_dir)
            if os.path.isfile(metalist_path):
                return metalist_path

        metalist_path = self.scan_file(dir=self.metalist.output_dir)
        if os.path.isfile(metalist_path):
            return metalist_path

        metadata_path = os.path.join(self.metadata.output_dir, self.scan_file(file_index))
        if os.path.isfile(metadata_path):
            return metadata_path

        err_txt = f"No metadata exists for the scan {self.scan_num}"
        if file_index is not None:
            err_txt += f" and file No. {file_index}"
        raise ValueError(err_txt)

    def run(self) -> BaseRun[TrainIndices]:
        """Open the facility run and return a :class:`~cbclib_v2.BaseRun`.

        Dispatches to the appropriate run class based on
        ``data.facility`` (e.g. :class:`~cbclib_v2.XFELRun` for EuXFEL,
        :class:`~cbclib_v2.SwissFELRun` for SwissFEL).

        Returns:
            Opened run object.
        """
        return open_run(self.scan_num, self.data)

@dataclass
class BaseScript:
    """Abstract base class for ``cbclib_cli`` pipeline scripts."""

    def get_chunk(self, indices: TrainIndices, chunk_id: int | None, n_chunks: int | None
                  ) -> TrainIndices:
        if chunk_id is not None and n_chunks is not None:
            print(f"Processing chunk No. {chunk_id}")
            return list(indices.split(n_chunks))[chunk_id]

        print("Processing the whole scan")
        return indices

    def get_patterns(self, hits_file: str, geometry: Detector, xp: AnyNamespace) -> Patterns:
        dataframe = pd.read_hdf(hits_file, 'data')
        if 'module_id' in dataframe.columns:
            num_modules = geometry.num_modules
            streaks = StackedStreaks.import_dataframe(dataframe, num_modules=num_modules, xp=xp)
        else:
            streaks = Streaks.import_dataframe(dataframe, xp=xp)
        assembled = geometry.to_streaks(streaks)
        return geometry.to_meters(assembled)

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        raise NotImplementedError

    @classmethod
    def parser_description(cls) -> str:
        raise NotImplementedError

    @classmethod
    def from_file(cls, *args, **kwargs):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

DetectionKind = Literal['streaks', 'regions']
FileKind = Literal['streaks', 'regions', 'xtals', 'solutions', 'reflections']
HitsDir = Literal['streaks', 'regions']
XtalDir = Literal['xtals', 'solutions']

@dataclass(frozen=True)
class CompileSchema:
    """Describe the tables and metadata belonging to one pipeline artifact.

    Attributes:
        required_tables: Tables that must occur in every chunk.
        optional_tables: Tables that may be absent from individual chunks.
        trace_tables: Optimisation traces requiring a source ``chunk_id`` column.
        preserve_config: Whether all chunks must carry the same configuration.
    """
    required_tables : Tuple[str, ...]
    optional_tables : Tuple[str, ...] = ()
    trace_tables    : Tuple[str, ...] = ()
    preserve_config : bool = False

    @property
    def tables(self) -> Tuple[str, ...]:
        """Return every table allowed by this artifact schema."""
        return self.required_tables + self.optional_tables

@dataclass(frozen=True)
class CompileChunk:
    """Hold one validated per-chunk artifact and its provenance.

    Attributes:
        chunk_id: Numeric identifier parsed from the source filename.
        path: Source artifact path.
        tables: Validated Pandas tables keyed by their HDF5 names.
        config: Parsed root configuration, when present.
        extra: Source-file provenance stored under the ``extra`` group.
    """
    chunk_id : int
    path     : str
    tables   : dict[str, pd.DataFrame | pd.Series]
    config   : Any | None
    extra    : dict[str, Any]

@dataclass
class CompileFiles(BaseScript):
    """Implements ``cbclib_cli compile``.

    Validates and concatenates per-chunk pipeline artifacts into one HDF5 file.
    Optimisation traces retain their chunk identity, compatible configurations
    are preserved at the root, and source provenance is written to ``extra``.

    Attributes:
        kind: ``'streaks'``, ``'regions'``, ``'xtals'``, ``'solutions'``, or
            ``'reflections'`` — selects which output directory to read from.
        scan_file: Path to the scan configuration JSON file.
        in_suffix: Suffix selecting the per-chunk input directory.
        out_suffix: Suffix appended to the compiled output filename.
    """
    kind            : FileKind
    scan_file       : str
    in_suffix       : str = str()
    out_suffix      : str = str()

    schemas : ClassVar[dict[FileKind, CompileSchema]] = {
        'streaks': CompileSchema(required_tables=('data', 'metadata')),
        'regions': CompileSchema(required_tables=('data', 'metadata')),
        'xtals': CompileSchema(required_tables=('data',), preserve_config=True),
        'solutions': CompileSchema(required_tables=('data', 'miller', 'stats'),
                                   optional_tables=('candidates',), trace_tables=('stats',),
                                   preserve_config=True),
        'reflections': CompileSchema(required_tables=('data', 'stats'),
                                     trace_tables=('stats',), preserve_config=True),
    }

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('kind', type=str, choices=FileKind.__args__,
                             help='Type of chunked files to compile')
        initial.add_argument('scan', type=str, help='Path to a scan parameters JSON file')
        initial.add_argument('--in-suffix', type=str, default=str(),
                             help='Suffix of the per-chunk detection files to read')
        initial.add_argument('--out-suffix', type=str, default=str(),
                             help='Suffix of the merged detection file to write')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Compile chunked files into a single table"

    @classmethod
    def from_file(cls, kind: FileKind, scan_file: str, in_suffix: str,
                  out_suffix: str) -> 'CompileFiles':
        return cls(kind, scan_file, in_suffix, out_suffix)

    @property
    def input_dir(self) -> str:
        """Return the directory where per-chunk detection files are read from."""
        scan = ScanConfig.read(self.scan_file)
        if self.kind == 'streaks':
            return scan.detect.streaks_dir
        if self.kind == 'regions':
            return scan.detect.regions_dir
        if self.kind == 'xtals':
            return scan.setup.xtals_dir
        if self.kind == 'solutions':
            return scan.setup.solutions_dir
        if self.kind == 'reflections':
            return scan.setup.reflections_dir
        raise ValueError(f"Invalid file kind: {self.kind}")

    @staticmethod
    def chunk_id(path: str) -> int:
        """Extract the chunk identifier from a canonical per-chunk filename."""
        match = re.search(r'_f([0-9]{4})(?:_|\.|$)', os.path.basename(path))
        if match is None:
            raise ValueError(f"Could not determine chunk ID from file: {path}")
        return int(match.group(1))

    @staticmethod
    def metadata_value(value: Any) -> Any:
        """Convert an HDF5 scalar or array into a manifest-compatible value."""
        if isinstance(value, bytes):
            return value.decode()
        if hasattr(value, 'item'):
            try:
                return CompileFiles.metadata_value(value.item())
            except ValueError:
                pass
        if hasattr(value, 'tolist'):
            return json.dumps(value.tolist())
        return value

    def load_chunk(self, path: str, schema: CompileSchema) -> CompileChunk:
        """Load and validate the tables and provenance of one chunk artifact."""
        with pd.HDFStore(path, mode='r') as store:
            keys = tuple(key.lstrip('/') for key in store.keys())

        missing = set(schema.required_tables).difference(keys)
        if missing:
            raise ValueError(f"Missing required tables {sorted(missing)} in file: {path}")
        unexpected = set(keys).difference(schema.tables)
        if unexpected:
            raise ValueError(f"Unexpected tables {sorted(unexpected)} in file: {path}")

        tables = {key: pd.read_hdf(path, key) for key in keys}
        for key, table in tables.items():
            if not isinstance(table, (pd.DataFrame, pd.Series)):
                raise ValueError(f"Table '{key}' has unsupported type in file: {path}")
        config = None
        extra = {}
        with h5py.File(path, mode='r') as input_file:
            if 'config' in input_file.attrs:
                raw_config = self.metadata_value(input_file.attrs['config'])
                try:
                    config = json.loads(raw_config)
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError(f"Invalid configuration in file: {path}") from error
            if 'extra' in input_file:
                group = input_file['extra']
                if not isinstance(group, h5py.Group):
                    raise ValueError(f"Expected 'extra' to be an HDF5 group in file: {path}")
                for name, item in group.items():
                    if not isinstance(item, h5py.Dataset):
                        raise ValueError(f"Expected scalar provenance at extra/{name} in: {path}")
                    extra[name] = self.metadata_value(item[()])

        if schema.preserve_config and config is None:
            raise ValueError(f"Missing configuration in file: {path}")
        return CompileChunk(self.chunk_id(path), path, tables, config, extra)

    @staticmethod
    def validate_chunks(chunks: List[CompileChunk], schema: CompileSchema) -> Any | None:
        """Ensure chunk identifiers and artifact configurations are consistent."""
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError(f"Duplicate chunk IDs found: {chunk_ids}")
        if not schema.preserve_config:
            return None

        config = chunks[0].config
        for chunk in chunks[1:]:
            if chunk.config != config:
                raise ValueError(f"Configuration in {chunk.path} does not match "
                                 f"{chunks[0].path}")
        return config

    @staticmethod
    def compile_table(chunks: List[CompileChunk], key: str, trace: bool
                      ) -> pd.DataFrame | pd.Series | None:
        """Concatenate one semantic table across all chunks."""
        tables = []
        table_type: type[pd.DataFrame] | type[pd.Series] | None = None
        for chunk in chunks:
            if key not in chunk.tables:
                continue
            table = chunk.tables[key]
            if table_type is None:
                table_type = type(table)
            elif not isinstance(table, table_type):
                raise ValueError(f"Table '{key}' has inconsistent types across chunks")
            if trace:
                if not isinstance(table, pd.DataFrame):
                    raise ValueError(f"Trace table '{key}' must be a DataFrame")
                if 'chunk_id' in table.columns:
                    raise ValueError(f"Trace table '{key}' already contains a chunk_id column")
                table = table.assign(chunk_id=chunk.chunk_id)
            tables.append(table)
        if not tables:
            return None
        return pd.concat(tables, ignore_index=True)

    @staticmethod
    def manifest(chunks: List[CompileChunk]) -> pd.DataFrame:
        """Build the per-chunk source and provenance manifest."""
        rows = []
        for chunk in chunks:
            row = {'chunk_id': chunk.chunk_id, 'source_file': os.path.abspath(chunk.path)}
            reserved = set(row).intersection(chunk.extra)
            if reserved:
                raise ValueError(f"Provenance fields use reserved names: {sorted(reserved)}")
            row.update(chunk.extra)
            rows.append(row)
        return pd.DataFrame(rows).sort_values('chunk_id').reset_index(drop=True)

    def run(self) -> None:
        schema = self.schemas[self.kind]

        scan = ScanConfig.read(self.scan_file)
        print(f'Assembling {self.kind} for a scan {scan.scan_num:d}...')

        scan_dir = scan.scan_subdir(self.input_dir, self.in_suffix)
        scan_files = list(sorted(scan.list_files(scan_dir)))

        print(f'Found {len(scan_files)} {self.kind} files for run {scan.scan_num:d}')
        if not scan_files:
            raise ValueError(f"No {self.kind} files found under {scan_dir}")

        chunks = [self.load_chunk(path, schema) for path in scan_files]
        chunks.sort(key=lambda chunk: chunk.chunk_id)
        config = self.validate_chunks(chunks, schema)

        output_path = scan.scan_file(suffix=self.out_suffix, dir=self.input_dir)
        print(f'Writing the detected {self.kind} to the file: {output_path}')
        output_dir = os.path.dirname(output_path) or os.curdir
        mode_source = output_path if os.path.exists(output_path) else chunks[0].path
        output_mode = stat.S_IMODE(os.stat(mode_source).st_mode)
        file_descriptor, temporary_path = tempfile.mkstemp(prefix='.compile-', suffix='.h5',
                                                            dir=output_dir)
        os.close(file_descriptor)
        try:
            mode = 'w'
            for key in schema.tables:
                table = self.compile_table(chunks, key, key in schema.trace_tables)
                if table is None:
                    continue
                table.to_hdf(temporary_path, key=key, mode=mode)
                mode = 'a'
            self.manifest(chunks).to_hdf(temporary_path, key='extra', mode=mode)

            if config is not None:
                with h5py.File(temporary_path, mode='a') as output_file:
                    output_file.attrs['config'] = json.dumps(config)
            os.chmod(temporary_path, output_mode)
            os.replace(temporary_path, output_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

@dataclass
class CreateMetadata(BaseScript):
    """Implements ``cbclib_cli metadata``.

    Randomly samples frames from the run, computes the background
    whitefield, and writes the result to an HDF5 metadata file used by all
    subsequent detection steps.

    Attributes:
        scan: Scan configuration.
        params: Metadata computation parameters (mask and background method).
    """
    scan        : ScanConfig
    params      : MetadataParameters

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('scan', type=str,
                             help='Path to a scan parameters JSON file')
        initial.add_argument('parameters', type=str,
                             help='Path to a metadata parameters JSON file')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Calculate CBD metadata needed for streak detection"

    @classmethod
    def from_file(cls, scan_file: str, params_file: str) -> 'CreateMetadata':
        scan = ScanConfig.read(scan_file)
        params = MetadataParameters.read(params_file)
        return cls(scan, params)

    def run(self):
        print("Configuring the script...")
        xp = self.scan.system.array_api()
        rng = default_rng(xp=xp)

        run = self.scan.run()

        print("Looking for data...")
        indices = run.indices()
        frames = rng.choice(len(indices), self.scan.metadata.n_frames, replace=False)
        indices = indices[frames]

        print(f"Loading {self.scan.metadata.n_frames:d} frames...")
        images = run.data(indices, geometry=self.scan.apply_geometry, xp=xp)

        print("Generating metadata...")
        with self.scan.system.cpu_config():
            metadata = create_metadata(images, self.params)

        output_file = os.path.join(self.scan.metadata.output_dir, self.scan.scan_file())
        print(f"Saving to {output_file}")
        write_hdf(metadata, output_file, H5Handler(metadata.protocol))

@dataclass
class CreateMetaList(BaseScript):
    """Implements ``cbclib_cli metalist``.

    Computes background whitefields at regularly-spaced points across a data
    chunk, runs PCA on the collection, and writes the eigen-fields and
    supporting arrays to a per-chunk HDF5 metalist file.

    Attributes:
        scan: Scan configuration.
        params: Metadata computation parameters.
        chunk_id: Zero-based index of this chunk (``None`` = whole scan).
        n_chunks: Total number of chunks (``None`` = whole scan).
    """
    scan        : ScanConfig
    params      : MetadataParameters
    chunk_id    : int | None
    n_chunks    : int | None
    n_out       : int | None

    def __post_init__(self):
        if self.n_chunks is not None:
            if self.chunk_id is None:
                raise ValueError("chunk_id must be provided when n_chunks is specified")
            self.chunk_id = compute_index(self.chunk_id, self.n_chunks)

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('scan', type=str,
                             help='Path to a scan parameters JSON file')
        initial.add_argument('parameters', type=str,
                             help='Path to a metadata parameters JSON file')
        initial.add_argument('--chunk_id', '-id', type=int, help='ID of the chunk to process')
        initial.add_argument('--n_chunks', '-n', type=int, help='Total number of chunks')
        initial.add_argument('--n_out', '-no', type=int,
                             help='Number of background estimates to compute')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Calculate a list of CBD metadata"

    @classmethod
    def from_file(cls, scan_file: str, params_file: str, chunk_id: int | None, n_chunks: int | None,
                  n_out: int | None = None
                  ) -> 'CreateMetaList':
        scan = ScanConfig.read(scan_file)
        params = MetadataParameters.read(params_file)
        return cls(scan, params, chunk_id, n_chunks, n_out)

    def run(self):
        print("Configuring the script...")
        xp = self.scan.system.array_api()

        print("Looking for data...")
        run = self.scan.run()
        chunk = self.get_chunk(run.indices(), self.chunk_id, self.n_chunks)
        print(f"Starting to process {len(chunk):d} frames...")

        offsets = xp.arange(0, self.scan.metalist.n_frames) - self.scan.metalist.n_frames // 2
        spacing = min(self.scan.metalist.spacing, len(chunk) - len(offsets))

        n_out = self.n_out or (len(chunk) - len(offsets)) // spacing
        centers = xp.linspace(-int(offsets[0]), len(chunk) - int(offsets[-1]) - 1,
                              n_out, dtype=int)
        batches = xp.asarray(centers[:, None] + offsets, dtype=int)
        print(f"Creating a metadata list of {batches.shape[0]:d} points...")

        output_path = self.scan.scan_file(self.chunk_id, dir=self.scan.metalist.output_dir,
                                          make_dirs=True)
        print(f"The results will be saved to {output_path}")

        handler = H5Handler(CrystMetadata.default_protocol())
        mask, var = xp.ones(1, dtype=bool), xp.zeros(1)

        whitefields = []
        with self.scan.system.cpu_config():
            for index, batch in tqdm(enumerate(batches), total=batches.shape[0],
                                     desc='Generating the list'):
                images = run.data(chunk[batch], geometry=self.scan.apply_geometry, verbose=False,
                                  xp=xp)
                metadata = create_metadata(images, self.params)
                mask = mask & metadata.mask
                var = var + metadata.std**2
                whitefields.append(metadata.flatfield)
                with h5py.File(output_path, 'a') as out_file:
                    handler.save('whitefields', metadata.flatfield, out_file, mode='insert',
                                 idxs=index)

            print("Performing PC analysis...")
            metadata = CrystMetadata(whitefields=xp.stack(whitefields, axis=0),
                                     mask=mask, std=xp.sqrt(var / batches.shape[0]))
            metadata = metadata.pca()

        print("Saving the rest...")
        write_hdf(metadata, output_path, handler, 'eigen_field', 'eigen_value', 'flatfield',
                  'mask', 'std', file_mode='a')

@dataclass
class DetectHits(BaseScript):
    """Implements ``cbclib_cli detect``.

    Loads a data chunk, runs streak or region detection via
    :func:`~cbclib_v2.scripts.pool_detection`, filters by
    :attr:`~DetectConfig.hit_threshold`, and writes per-hit streak data (or
    a frame-only CSV) to the output directory.

    Attributes:
        scan: Scan configuration.
        params: Detection configuration (:class:`~cbclib_v2.scripts.StreakFinderConfig`
            or :class:`~cbclib_v2.scripts.RegionFinderConfig`).
        chunk_id: Zero-based chunk index (``None`` = whole scan).
        n_chunks: Total number of chunks (``None`` = whole scan).
        frames_only: When ``True``, save only the list of hit frame indices
            (CSV) instead of the full streak table.
    """
    scan        : ScanConfig
    params      : FinderConfig
    chunk_id    : int | None
    n_chunks    : int | None
    frames_only : bool
    out_suffix  : str

    def __post_init__(self):
        if self.n_chunks is not None:
            if self.chunk_id is None:
                raise ValueError("chunk_id must be provided when n_chunks is specified")
            self.chunk_id = compute_index(self.chunk_id, self.n_chunks)

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('kind', type=str, choices=['streaks', 'regions'],
                             help='Kind of detection to perform')
        initial.add_argument('scan', type=str,
                             help='Path to a scan parameters JSON file')
        initial.add_argument('parameters', type=str,
                             help='Path to a streak finder parameters JSON file')
        initial.add_argument('--chunk_id', '-id', type=int, help='Index of the chunk to process')
        initial.add_argument('--n_chunks', '-n', type=int, help='Number of chunks to process')
        initial.add_argument('--frames-only', action='store_true',
                             help='Only save the list of frames with hits')
        initial.add_argument('--out-suffix', '-os', type=str, default=str(),
                             help='Suffix of the detection files to write')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Detect streaks in CBD patterns"

    @classmethod
    def from_file(cls, kind: DetectionKind, scan_file: str, params_file: str, chunk_id: int | None,
                  n_chunks: int | None, frames_only: bool, out_suffix: str) -> 'DetectHits':
        scan = ScanConfig.read(scan_file)
        if kind == 'streaks':
            params = StreakFinderConfig.read(params_file)
        elif kind == 'regions':
            params = RegionFinderConfig.read(params_file)
        else:
            raise ValueError(f"Invalid detection kind: {kind}")
        return cls(scan, params, chunk_id, n_chunks, frames_only, out_suffix)

    @property
    def kind(self) -> DetectionKind:
        """Detection kind derived from the type of :attr:`params`."""
        if isinstance(self.params, StreakFinderConfig):
            return 'streaks'
        if isinstance(self.params, RegionFinderConfig):
            return 'regions'
        raise ValueError(f"Invalid parameters type for detection: {type(self.params)}")

    @staticmethod
    def meta_dataframe(run: BaseRun[TrainIndices], hits: TrainIndices) -> pd.DataFrame:
        """Build a per-hit-frame metadata table for a detection chunk."""

        def serialize(value: str | int | tuple[str, ...] | tuple[int, ...]
                           ) -> str | int:
            """Return a scalar HDF5 table value for a frame provenance field."""
            if isinstance(value, tuple):
                return json.dumps(value)
            return value

        pulse_ids = run.metadata('pulse_id', hits)
        if pulse_ids.ndim > 1:
            pulse_ids = pulse_ids[:, 0]

        dataframe = {'index': [], 'filename': [], 'file_index': [], 'pulse_id': pulse_ids}
        for record in hits.records():
            dataframe['index'].append(serialize(record.index))
            dataframe['filename'].append(serialize(record.filename))
            dataframe['file_index'].append(serialize(record.file_index))
        return pd.DataFrame(dataframe)

    def run(self):
        xp = self.scan.system.array_api()

        print("Looking for data...")
        run = self.scan.run()
        chunk = self.get_chunk(run.indices(), self.chunk_id, self.n_chunks)
        frames = xp.asarray(list(chunk.index()))

        print(f"Starting to process {len(chunk):d} frames...")

        metadata_path = self.scan.find_metadata(self.chunk_id)
        print(f"Using the metadata saved at {metadata_path}")

        loader = run.worker(self.scan.apply_geometry)
        detector = None if self.scan.apply_geometry else self.scan.data.geometry()
        with self.scan.system.cpu_config():
            streaks = pool_detection(loader, chunk, metadata_path, self.params,
                                     self.scan.system.platform, detector)

        indices, counts = xp.unique_counts(streaks.index)
        hit_indices = indices[counts > self.scan.detect.hit_threshold]
        hits = streaks.take(hit_indices)
        print(f"{hit_indices.size:d} hits were found.")

        if len(hits) > 0:
            if self.kind == 'streaks':
                output_dir = self.scan.detect.streaks_dir
            else:
                output_dir = self.scan.detect.regions_dir

            if self.frames_only:
                print("Frames only requested, skipping saving the full hits data.")
                output_path = self.scan.scan_file(self.chunk_id, extension='.csv',
                                                  suffix=self.out_suffix, dir=output_dir,
                                                  make_dirs=True)
                dataframe = pd.DataFrame({'frame': asnumpy(frames[hit_indices])})
                dataframe.to_csv(output_path, index=False)
                print(f"The results were saved to {output_path}")
            else:
                print("Preparing the file...")
                df = hits.to_dataframe(frames)
                metadata = self.meta_dataframe(run, chunk[hit_indices])

                output_path = self.scan.scan_file(self.chunk_id, suffix=self.out_suffix,
                                                  dir=output_dir, make_dirs=True)
                print(f"The results will be saved to {output_path}")

                df.to_hdf(output_path, key='data')
                metadata.to_hdf(output_path, key='metadata')

@dataclass
class IndexingScript(BaseScript):
    """Implements ``cbclib_cli index``.

    Loads a per-chunk streak file, assembles patterns, and indexes them via
    :func:`~cbclib_v2.scripts.pool_indexing`.  Crystal orientations can be
    seeded from a prior indexing run or derived from the unit cell.

    Attributes:
        scan: Scan configuration.
        params: Indexing configuration.
        xtals: Path to an HDF5 file with initial crystal orientations.
            Empty string → use the unit cell from :attr:`~ScanConfig.setup`.
        hits_dir: Logical directory with detected streaks to read from: ``'streaks'``
            or ``'regions'``.
        streaks_dir: Logical directory with detected streaks to read from:
            ``'streaks'`` or ``'regions'``.
        in_suffix: Suffix of the detection files to read.
        out_suffix: Suffix appended to the output directory name.
        chunk_id: Zero-based chunk index (``None`` = whole scan).
    """
    scan        : ScanConfig
    params      : IndexingConfig
    xtals       : str
    hits_dir    : HitsDir
    in_suffix   : str
    out_suffix  : str
    chunk_id    : int | None

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('scan', type=str,
                             help='Path to a scan parameters JSON file')
        initial.add_argument('parameters', type=str,
                             help='Path to an indexing parameters JSON file')
        initial.add_argument('--xtals', '-x', type=str, default=str(),
                             help='Path to a crystal orientations H5 file')
        initial.add_argument('--hits-dir', type=str, choices=['streaks', 'regions'],
                             default='streaks',
                             help='Logical directory with detected streaks to read from')
        initial.add_argument('--in-suffix', '-is', type=str, default=str(),
                             help='Suffix of the detection files to read')
        initial.add_argument('--out-suffix', '-os', type=str, default=str(),
                             help='Suffix of the indexing files to write')
        initial.add_argument('--chunk_id', '-id', type=int, help='ID of the chunk to process')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Index detected streaks in CBD patterns"

    @classmethod
    def from_file(cls, scan_file: str, params_file: str, xtals: str, hits_dir: HitsDir,
                  in_suffix: str, out_suffix: str, chunk_id: int | None) -> 'IndexingScript':
        scan = ScanConfig.read(scan_file)
        params = IndexingConfig.read(params_file)
        return cls(scan, params, xtals, hits_dir, in_suffix, out_suffix, chunk_id)

    @property
    def hits_directory(self) -> str:
        """Return the configured detection directory selected by :attr:`hits_dir`."""
        if self.hits_dir == 'streaks':
            return self.scan.detect.streaks_dir
        if self.hits_dir == 'regions':
            return self.scan.detect.regions_dir
        raise ValueError(f"Invalid hits_dir: {self.hits_dir}")

    @property
    def xtal_file(self) -> str:
        """Return the path to the crystal orientations file to read."""
        if self.xtals:
            return self.xtals
        return self.scan.setup.unit_file

    def run(self):
        print("Configuring the script...")
        xp = self.scan.system.array_api()

        geometry = self.scan.data.geometry()
        hits_file = self.scan.scan_file(self.chunk_id, suffix=self.in_suffix,
                                        dir=self.hits_directory)
        if not os.path.isfile(hits_file):
            print(f"No streaks file found at {hits_file}")
            return

        print(f"Loading detected streaks from {hits_file}...")
        patterns = self.get_patterns(hits_file, geometry, xp)
        frames = patterns.unique_index()

        if self.xtals:
            print(f"Loading crystal orientations from {self.xtal_file}...")
            df = pd.read_hdf(self.xtals, 'data')
            xtals = XtalState.import_dataframe(df, xp=xp)
        else:
            print("No crystal orientations provided, "\
                  f"using the unit cell from {self.xtal_file}...")
            xtals = self.scan.setup.xtal(xp=xp)
        geometry = self.scan.setup.geometry()

        print(f"Indexing {len(patterns):d} patterns...")
        with self.scan.system.cpu_config():
            result = pool_indexing(patterns.to_points(), xtals, geometry, self.params,
                                   self.scan.system.platform, xp)

        output_path = self.scan.scan_file(self.chunk_id, dir=self.scan.setup.xtals_dir,
                                          suffix=self.out_suffix, make_dirs=True)

        print(f"Saving the results to {output_path}...")
        extra = {'xtal_file': self.xtal_file, 'hits_file': hits_file,
                 'setup_file': self.scan.setup.setup_file}
        self.params.save(output_path, mode='w', extra=extra)
        result.to_dataframe(frames).to_hdf(output_path, key='data')

@dataclass
class RefineScript(BaseScript):
    """Implements ``cbclib_cli refine``.

    Loads a per-chunk indexed crystal orientations file, refines the orientations
    against the detected streaks, and writes the refinement result to an HDF5
    file.  The output stores champion orientations under ``data``, all refined
    candidates under ``candidates``, the optimisation trace under ``stats``, and
    input-file provenance under ``files``. The refinement configuration is
    retained in the root HDF5 metadata.

    Attributes:
        scan: Scan configuration.
        params: Refinement configuration.
        hits_dir: Logical directory with detected streaks to read from: ``'streaks'``
            or ``'regions'``.
        xtal_dir: Logical crystal orientation directory to read from: ``'xtals'``
            or ``'solutions'``.
        in_suffix: Suffix of the orientation files to read.
        out_suffix: Suffix of the refinement files to write.
        chunk_id: Zero-based chunk index (``None`` = whole scan).
    """
    scan        : ScanConfig
    params      : RefineConfig
    hits_dir    : HitsDir
    xtal_dir    : XtalDir
    in_suffix   : str
    out_suffix  : str
    chunk_id    : int | None

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('scan', type=str,
                             help='Path to a scan parameters JSON file')
        initial.add_argument('parameters', type=str,
                             help='Path to a refinement parameters JSON file')
        initial.add_argument('--hits-dir', type=str, choices=['streaks', 'regions'],
                             default='streaks',
                             help='Logical directory with detected streaks to read from')
        initial.add_argument('--xtal-dir', type=str, choices=['xtals', 'solutions'],
                             default='xtals',
                             help='Logical crystal orientation directory to read from')
        initial.add_argument('--in-suffix', '-is', type=str, default=str(),
                             help='Suffix of the orientation files to read')
        initial.add_argument('--out-suffix', '-os', type=str, default=str(),
                             help='Suffix of the refinement files to write')
        initial.add_argument('--chunk_id', '-id', type=int, help='ID of the chunk to process')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Refine indexed crystal orientations and scattering geometry in CBD patterns"

    @classmethod
    def from_file(cls, scan_file: str, params_file: str, hits_dir: HitsDir, xtal_dir: XtalDir,
                  in_suffix: str, out_suffix: str, chunk_id: int | None) -> 'RefineScript':
        scan = ScanConfig.read(scan_file)
        params = RefineConfig.read(params_file)
        return cls(scan, params, hits_dir, xtal_dir, in_suffix, out_suffix, chunk_id)

    @property
    def hits_directory(self) -> str:
        """Return the configured detection directory selected by :attr:`hits_dir`."""
        if self.hits_dir == 'streaks':
            return self.scan.detect.streaks_dir
        if self.hits_dir == 'regions':
            return self.scan.detect.regions_dir
        raise ValueError(f"Invalid hits_dir: {self.hits_dir}")

    @property
    def setup_file(self) -> str:
        """Return the path to the setup file used for indexing and refinement."""
        if self.xtal_dir == 'xtals':
            return self.scan.setup.setup_file
        if self.xtal_dir == 'solutions':
            return self.scan.scan_file(self.chunk_id, dir=self.scan.setup.solutions_dir,
                                       suffix=self.in_suffix)
        raise ValueError(f"Invalid xtal_dir: {self.xtal_dir}")

    @property
    def xtal_directory(self) -> str:
        """Return the configured orientation directory selected by :attr:`xtal_dir`."""
        if self.xtal_dir == 'xtals':
            return self.scan.setup.xtals_dir
        if self.xtal_dir == 'solutions':
            return self.scan.setup.solutions_dir
        raise ValueError(f"Invalid xtal_dir: {self.xtal_dir}")

    def import_xtals(self, patterns: Patterns, df: pd.DataFrame | pd.Series,
                     xp: AnyNamespace) -> Tuple[IntArray, LinePoints, BaseSetup]:
        frames, xtals = XtalState.import_dataframe(df, index=True, xp=xp)
        target = patterns.take(frames, reset_index=True)
        initial = self.params.setup.import_xtal(xtals, self.setup_file)
        return frames, target.to_points(), initial

    def import_solutions(self, patterns: Patterns, df: pd.DataFrame | pd.Series,
                         xp: AnyNamespace) -> Tuple[IntArray, LinePoints, BaseSetup]:
        frames, resolved = ResolvedSetup.import_dataframe(df, index=True, xp=xp)
        target = patterns.loc[frames]
        initial = self.params.setup.import_resolved(resolved)
        return frames, target.to_points(), initial

    def run(self):
        print("Configuring the script...")
        xp = self.scan.system.jax_api()

        geometry = self.scan.data.geometry()
        in_file = self.scan.scan_file(self.chunk_id, dir=self.xtal_directory,
                                      suffix=self.in_suffix)
        if not os.path.isfile(in_file):
            print(f"No indexed crystal orientations file found at {in_file}")
            return

        print(f"Loading indexed crystal orientations from {in_file}...")
        df = pd.read_hdf(in_file, 'data')

        hits_file = self.scan.scan_file(self.chunk_id, dir=self.hits_directory)
        if not os.path.isfile(hits_file):
            print(f"No streaks file found at {hits_file}")
            return

        print(f"Loading detected streaks from {hits_file}...")
        patterns = self.get_patterns(hits_file, geometry, xp)

        if self.xtal_dir == 'xtals':
            frames, points, initial = self.import_xtals(patterns, df, xp)
        elif self.xtal_dir == 'solutions':
            frames, points, initial = self.import_solutions(patterns, df, xp)
        else:
            raise ValueError(f"Invalid xtal_dir: {self.xtal_dir}")

        print(f"Loaded {len(patterns)} patterns.")
        resolved = initial.resolve(xp)

        context = self.params.init_context(points, resolved)

        with self.scan.system.cpu_config():
            print(f"Refining {len(patterns)} patterns...")

            if self.xtal_dir == 'xtals':
                candidates, stats = context.refine(frames, initial, self.params)
                indices = candidates.champions_only()
                champions = candidates[indices]
                points = points.iloc[indices]

                context = self.params.init_context(points, champions.resolved)
            elif self.xtal_dir == 'solutions':
                champions, stats = context.refine(frames, initial, self.params)
                candidates = None
            else:
                raise ValueError(f"Invalid xtal_dir: {self.xtal_dir}")

        output_path = self.scan.scan_file(self.chunk_id, dir=self.scan.setup.solutions_dir,
                                          suffix=self.out_suffix, make_dirs=True)

        print(f"Saving refined crystal orientations to {output_path}...")
        extra = {'input_file': in_file, 'hits_file': hits_file, 'setup_file': self.setup_file}
        self.params.save(output_path, mode='w', extra=extra)
        stats.to_dataframe().to_hdf(output_path, key='stats', mode='a')

        if self.params.indexed_thr > 0.0:
            if candidates is None:
                candidates = champions

            fitness = context.pattern_fitness(self.params.indexed_thr, champions.resolved)
            is_champion = fitness > self.params.indexed_thr
            champions = champions[is_champion]
            points = points.take(points.unique_index()[is_champion])

            context = self.params.init_context(points, champions.resolved)
            print(f"Keeping {len(points)} patterns with fitness above " \
                  f"{self.params.indexed_thr:.2f}...")

        miller = context.miller(champions.resolved)
        miller.to_dataframe(champions.frames).to_hdf(output_path, key='miller', mode='a')
        champions.to_dataframe().to_hdf(output_path, key='data', mode='a')

        if candidates is not None:
            print(f"Saving all candidates to {output_path}...")
            candidates.to_dataframe().to_hdf(output_path, key='candidates', mode='a')

@dataclass
class PostRefineScript(BaseScript):
    """Implements ``cbclib_cli post_refine``.

    Loads a per-chunk refined crystal orientations file, performs post-refinement
    against the measured patterns, and writes the result to an HDF5 file.  The
    output stores champion orientations under ``data``, all refined candidates
    under ``candidates``, the optimisation trace under ``stats``, and input-file
    provenance under ``files``. The refinement configuration is retained in the
    root HDF5 metadata.

    Attributes:
        scan: Scan configuration.
        params: Post-refinement configuration.
        in_suffix: Suffix of the refinement files to read.
        out_suffix: Suffix of the post-refinement files to write.
        chunk_id: Zero-based chunk index (``None`` = whole scan).
    """
    scan        : ScanConfig
    params      : PostRefineConfig
    in_suffix   : str
    out_suffix  : str
    chunk_id    : int | None

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('scan', type=str,
                             help='Path to a scan parameters JSON file')
        initial.add_argument('parameters', type=str,
                             help='Path to a post-refinement parameters JSON file')
        initial.add_argument('--in-suffix', '-is', type=str, default=str(),
                             help='Suffix of the refinement files to read')
        initial.add_argument('--out-suffix', '-os', type=str, default=str(),
                             help='Suffix of the post-refinement files to write')
        initial.add_argument('--chunk_id', '-id', type=int, help='ID of the chunk to process')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Post-refine indexed crystal orientations and scattering geometry in CBD patterns"

    @classmethod
    def from_file(cls, scan_file: str, params_file: str, in_suffix: str, out_suffix: str,
                  chunk_id: int | None) -> 'PostRefineScript':
        scan = ScanConfig.read(scan_file)
        params = PostRefineConfig.read(params_file)
        return cls(scan, params, in_suffix, out_suffix, chunk_id)

    def run(self):
        print("Configuring the script...")
        xp = self.scan.system.jax_api()

        geometry = self.scan.data.geometry()
        in_file = self.scan.scan_file(self.chunk_id, dir=self.scan.setup.solutions_dir,
                                      suffix=self.in_suffix)
        if not os.path.isfile(in_file):
            print(f"No refined crystal orientations file found at {in_file}")
            return

        print(f"Loading refined crystal orientations from {in_file}...")
        df = pd.read_hdf(in_file, 'data')
        frames, resolved = ResolvedSetup.import_dataframe(df, index=True, xp=xp)

        scaler = ScalerModel()
        if self.params.miller == 'indexed':
            print(f"Loading indexed miller indices from {in_file}...")
            miller = Miller.import_dataframe(pd.read_hdf(in_file, 'miller'), frames, xp)
            miller = scaler.xtal.hkl_to_q(miller, resolved.xtal, xp)
        elif self.params.miller == 'all':
            q_abs = scaler.lens.max_resolution(xp.asarray(geometry.corners), resolved.geometry,
                                               xp)
            miller = self.params.all_hkl(q_abs, scaler, resolved, geometry, xp)
        else:
            raise ValueError(f"Invalid miller option: {self.params.miller}")

        metadata_path = self.scan.find_metadata(self.chunk_id)
        print(f"Using the metadata saved at {metadata_path}")
        metadata = self.params.scaling.metadata(metadata_path, xp)
        metadata = metadata.assemble(geometry.assembler(xp))

        print("Looking for data...")
        run = self.scan.run()

        print(f"Loading {frames.size} frames...")
        images = run.data(run.indices()[frames], geometry=True, xp=xp)

        print(f"Scaling background for {frames.size:d} frames...")
        cryst_data = scale_background(frames, images, metadata, self.params.scaling)

        extra = {'input_file': in_file, 'metadata_file': metadata_path}
        context = self.params.init_context(scaler, cryst_data, miller, resolved, geometry, xp)

        print(f"Refining scaling for {frames.size:d} patterns...")
        initial = ScalerState.default(len(context.data.reflections), len(resolved.xtal),
                                      self.params.sigma, xp)
        optimised, stats = context.refine_scaling(initial, resolved,
                                                  self.params.optimise.intensities)

        if self.params.optimise.setup:
            print(f"Post-refining {frames.size:d} patterns...")
            full = FullState(optimised, self.params.setup.import_resolved(resolved))
            optimised, resolved, post_stats = context.post_refine(full, self.params.optimise.setup)
            result = context.to_result(frames, optimised, resolved)

            output_path = self.scan.scan_file(self.chunk_id, dir=self.scan.setup.solutions_dir,
                                              suffix=self.out_suffix, make_dirs=True)

            print(f"Saving the refined crystal orientations to {output_path}...")
            self.params.save(output_path, mode='w', extra=extra)
            post_stats.to_dataframe().to_hdf(output_path, key='stats', mode='a')
            result.to_dataframe().to_hdf(output_path, key='data', mode='a')
            miller.to_dataframe(frames).to_hdf(output_path, key='miller', mode='a')
        else:
            print(f"Skipping post-refinement for {frames.size:d} patterns...")

        reflections = context.to_list(optimised, resolved)

        output_path = self.scan.scan_file(self.chunk_id, dir=self.scan.setup.reflections_dir,
                                          suffix=self.out_suffix, make_dirs=True)
        print(f"Saving the refined reflections to {output_path}...")

        self.params.save(output_path, mode='w', extra=extra)
        reflections.to_dataframe(frames).to_hdf(output_path, key='data', mode='a')
        stats.to_dataframe().to_hdf(output_path, key='stats', mode='a')

class SBatchScripts:
    """Factory for single ``sbatch`` job scripts.

    Each classmethod assembles a ``cbclib_cli <subcommand> …`` shell command,
    reads SLURM parameters from *script_file*, and returns a
    :class:`~cbclib_v2.slurm.SLURMScript` ready for submission via
    :meth:`~cbclib_v2.slurm.SLURMJobManager.submit`.
    """
    main        : ClassVar[str] = 'cbclib_cli'

    @classmethod
    def compile(cls, kind: DetectionKind, scan_file: str, script_file: str,
                in_suffix: str | None=None, out_suffix: str | None=None) -> SLURMScript:
        """Build a ``cbclib_cli compile`` script.

        Args:
            kind: ``'streaks'`` or ``'regions'``.
            scan_file: Path to the scan configuration JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            in_suffix: Suffix of the per-chunk detection files to read.
            out_suffix: Suffix of the merged detection file to write.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the compile step.
        """
        command = f"{cls.main} compile {quote(kind)} {quote(scan_file)}"
        if in_suffix is not None:
            command += f" --in-suffix {quote(in_suffix)}"
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="compile", command=command, parameters=script_spec)

    @classmethod
    def index(cls, scan_file: str, params_file: str, script_file: str,
              xtals: str | None=None, hits_dir: HitsDir='streaks',
              in_suffix: str | None=None, out_suffix: str | None=None,
              chunk_id: int | None=None) -> SLURMScript:
        """Build a ``cbclib_cli index`` script.

        Args:
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the indexing parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            xtals: Path to an initial crystal orientations HDF5 file
                (empty string → use unit cell).
            hits_dir: Logical directory with detected streaks to read from:
                ``'streaks'`` or ``'regions'``.
            in_suffix: Suffix of the detection files to read.
            out_suffix: Suffix of the indexing files to write.
            chunk_id: Optional chunk index for chunked processing.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the indexing step.
        """
        command = f"{cls.main} index {quote(scan_file)} {quote(params_file)} " \
                  f"--hits-dir {quote(hits_dir)} "
        if xtals is not None:
            command += f"--xtals {quote(xtals)} "
        if in_suffix is not None:
            command += f"--in-suffix {quote(in_suffix)} "
        if out_suffix is not None:
            command += f"--out-suffix {quote(out_suffix)} "
        if chunk_id is not None:
            command += f' --chunk_id {chunk_id:d}'
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="index", command=command, parameters=script_spec)

    @classmethod
    def metadata(cls, scan_file: str, params_file: str, script_file: str,
                 frames: List[int] | None=None) -> SLURMScript:
        """Build a ``cbclib_cli metadata`` script.

        Args:
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the metadata parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            frames: Optional explicit list of frame indices to use.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the metadata step.
        """
        command = f"{cls.main} metadata {quote(scan_file)} {quote(params_file)}"
        if frames is not None:
            command += f' --frames {frames}'
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="metadata", command=command, parameters=script_spec)

    @classmethod
    def metalist(cls, scan_file: str, params_file: str, script_file: str,
                 chunk_id: int | None=None, n_chunks: int | None=None,
                 n_out: int | None=None) -> SLURMScript:
        """Build a ``cbclib_cli metalist`` script.

        Args:
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the metadata parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            chunk_id: Optional chunk index.
            n_chunks: Optional total chunk count.
            n_out: Optional number of background estimates to compute.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the metalist step.
        """
        command = f"{cls.main} metalist {quote(scan_file)} {quote(params_file)}"
        if chunk_id is not None and n_chunks is not None:
            command += f' --chunk_id {chunk_id:d}'
            command += f' --n_chunks {n_chunks:d}'
        if n_out is not None:
            command += f' --n_out {n_out:d}'
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="metalist", command=command, parameters=script_spec)

    @classmethod
    def detect(cls, kind: DetectionKind, scan_file: str, params_file: str, script_file: str,
               chunk_id: int | None=None, n_chunks: int | None=None, frames_only: bool=False,
               out_suffix: str | None=None) -> SLURMScript:
        """Build a ``cbclib_cli detect`` script.

        Args:
            kind: ``'streaks'`` or ``'regions'``.
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the detection parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            chunk_id: Optional chunk index.
            n_chunks: Optional total chunk count.
            frames_only: Pass ``--frames-only`` to save only hit frame
                indices rather than the full streak table.
            out_suffix: Suffix of the detection files to write.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the detection step.
        """
        command = f"{cls.main} detect {quote(kind)} {quote(scan_file)} {quote(params_file)}"
        if chunk_id is not None and n_chunks is not None:
            command += f' --chunk_id {chunk_id:d}'
            command += f' --n_chunks {n_chunks:d}'
        if frames_only:
            command += ' --frames-only'
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="detect", command=command, parameters=script_spec)

    @classmethod
    def refine(cls, scan_file: str, params_file: str, script_file: str,
               hits_dir: HitsDir | None=None, xtal_dir: XtalDir | None=None,
               in_suffix: str | None=None, out_suffix: str | None=None,
               chunk_id: int | None=None) -> SLURMScript:
        """Build a ``cbclib_cli refine`` script.

        Args:
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the refinement parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            hits_dir: Logical directory with detected streaks to read from.
            xtal_dir: Logical crystal orientation directory to read from.
            in_suffix: Suffix of the orientation files to read.
            out_suffix: Suffix of the refinement files to write.
            chunk_id: Optional chunk index.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the refinement step.
        """
        command = f"{cls.main} refine {quote(scan_file)} {quote(params_file)}"
        if hits_dir is not None:
            command += f" --hits-dir {quote(hits_dir)}"
        if xtal_dir is not None:
            command += f" --xtal-dir {quote(xtal_dir)}"
        if in_suffix is not None:
            command += f" --in-suffix {quote(in_suffix)}"
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        if chunk_id is not None:
            command += f' --chunk_id {chunk_id:d}'
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="refine", command=command, parameters=script_spec)

    @classmethod
    def post_refine(cls, scan_file: str, params_file: str, script_file: str,
                    in_suffix: str | None=None, out_suffix: str | None=None,
                    chunk_id: int | None=None) -> SLURMScript:
        """Build a ``cbclib_cli post_refine`` script.

        Args:
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the post-refinement parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            in_suffix: Suffix of the refinement files to read.
            out_suffix: Suffix of the post-refinement files to write.
            chunk_id: Optional chunk index.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the post-refinement step.
        """
        command = f"{cls.main} post_refine {quote(scan_file)} {quote(params_file)}"
        if in_suffix is not None:
            command += f" --in-suffix {quote(in_suffix)}"
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        if chunk_id is not None:
            command += f' --chunk_id {chunk_id:d}'
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="post_refine", command=command, parameters=script_spec)

class SBatchArrayScripts:
    """Factory for ``sbatch --array`` job scripts.

    Like :class:`SBatchScripts` but each script injects
    ``FILE_INDEX=${SLURM_ARRAY_TASK_ID}`` so that the chunk index is taken
    from the SLURM task ID at runtime.  Use with
    :meth:`~cbclib_v2.slurm.SLURMJobManager.submit_array`.
    """
    main        : ClassVar[str] = 'cbclib_cli'

    @classmethod
    def index(cls, scan_file: str, params_file: str, script_file: str,
              xtals: str | None=None, hits_dir: HitsDir='streaks',
              in_suffix: str | None=None, out_suffix: str | None=None) -> SLURMScript:
        """Build a ``cbclib_cli index`` array script for *n_chunks* tasks.

        Args:
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the indexing parameters JSON.
            xtals: Path to an initial crystal orientations HDF5 file.
            hits_dir: Logical directory with detected streaks to read from:
                ``'streaks'`` or ``'regions'``.
            in_suffix: Suffix of the detection files to read.
            out_suffix: Suffix of the indexing files to write.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            n_chunks: Total number of array tasks.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` configured for array submission.
        """
        command = f"{cls.main} index {quote(scan_file)} {quote(params_file)} "\
                  f" --chunk_id ${{FILE_INDEX}} --hits-dir {quote(hits_dir)}"
        if xtals is not None:
            command += f" --xtals {quote(xtals)}"
        if in_suffix is not None:
            command += f" --in-suffix {quote(in_suffix)}"
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')
        return SLURMScript(job_name="index_array", command=command, parameters=script_spec)

    @classmethod
    def metalist(cls, scan_file: str, params_file: str, script_file: str, n_chunks: int
                 ) -> SLURMScript:
        """Build a ``cbclib_cli metalist`` array script for *n_chunks* tasks.

        Args:
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the metadata parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            n_chunks: Total number of array tasks.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` configured for array submission.
        """
        command = f"{cls.main} metalist {quote(scan_file)} {quote(params_file)}" \
                   f" --chunk_id ${{FILE_INDEX}} --n_chunks {n_chunks:d}"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')
        return SLURMScript(job_name="metalist_array", command=command, parameters=script_spec)

    @classmethod
    def detect(cls, kind: DetectionKind, scan_file: str, params_file: str, script_file: str,
               n_chunks: int, out_suffix: str | None=None) -> SLURMScript:
        """Build a ``cbclib_cli detect`` array script for *n_chunks* tasks.

        Args:
            kind: ``'streaks'`` or ``'regions'``.
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the detection parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            n_chunks: Total number of array tasks.
            out_suffix: Suffix of the detection files to write.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` configured for array submission.
        """
        command = f"{cls.main} detect {quote(kind)} {quote(scan_file)} {quote(params_file)}" \
                  f" --chunk_id ${{FILE_INDEX}} --n_chunks {n_chunks:d}"
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')
        return SLURMScript(job_name="detect_array", command=command, parameters=script_spec)

    @classmethod
    def refine(cls, scan_file: str, params_file: str, script_file: str,
               hits_dir: HitsDir | None=None, xtal_dir: XtalDir | None=None,
               in_suffix: str | None=None, out_suffix: str | None=None) -> SLURMScript:
        """Build a ``cbclib_cli refine`` array script for *n_chunks* tasks.

        Args:
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the refinement parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            n_chunks: Total number of array tasks.
            hits_dir: Logical directory with detected streaks to read from.
            xtal_dir: Logical crystal orientation directory to read from.
            in_suffix: Suffix of the orientation files to read.
            out_suffix: Suffix of the refinement files to write.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` configured for array submission.
        """
        command = f"{cls.main} refine {quote(scan_file)} {quote(params_file)}" \
                  f" --chunk_id ${{FILE_INDEX}}"
        if hits_dir is not None:
            command += f" --hits-dir {quote(hits_dir)}"
        if xtal_dir is not None:
            command += f" --xtal-dir {quote(xtal_dir)}"
        if in_suffix is not None:
            command += f" --in-suffix {quote(in_suffix)}"
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')
        return SLURMScript(job_name="refine_array", command=command, parameters=script_spec)

    @classmethod
    def post_refine(cls, scan_file: str, params_file: str, script_file: str,
                    in_suffix: str | None=None, out_suffix: str | None=None) -> SLURMScript:
        """Build a ``cbclib_cli post_refine`` array script for *n_chunks* tasks.

        Args:
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the post-refinement parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            in_suffix: Suffix of the refinement files to read.
            out_suffix: Suffix of the post-refinement files to write.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` configured for array submission.
        """
        command = f"{cls.main} post_refine {quote(scan_file)} {quote(params_file)}" \
                  f" --chunk_id ${{FILE_INDEX}}"
        if in_suffix is not None:
            command += f" --in-suffix {quote(in_suffix)}"
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')
        return SLURMScript(job_name="post_refine_array", command=command,
                           parameters=script_spec)

class Scripts:
    """Namespace exposing all ``cbclib_cli`` pipeline scripts and the CLI parser.

    Each attribute is the class implementing the corresponding subcommand.
    :attr:`sbatch` and :attr:`sbatch_array` are factories for building
    :class:`~cbclib_v2.slurm.SLURMScript` objects ready for submission via
    :class:`~cbclib_v2.slurm.SLURMJobManager`.

    Attributes:
        sbatch: :class:`SBatchScripts` — single-job ``sbatch`` script
            factory.
        sbatch_array: :class:`SBatchArrayScripts` — ``sbatch --array``
            script factory.
        compile: :class:`CompileFiles` — merge per-chunk results.
        metadata: :class:`CreateMetadata` — compute background whitefield.
        metalist: :class:`CreateMetaList` — compute PCA metalist.
        detect: :class:`DetectHits` — run streak or region detection.
        index: :class:`IndexingScript` — index detected patterns.
        refine: :class:`RefineScript` — refine indexed orientations.
        post_refine: :class:`PostRefineScript` — post-refine indexed orientations.
    """
    sbatch          : ClassVar[Type[SBatchScripts]] = SBatchScripts
    sbatch_array    : ClassVar[Type[SBatchArrayScripts]] = SBatchArrayScripts
    compile         : ClassVar[Type[CompileFiles]] = CompileFiles
    index           : ClassVar[Type[IndexingScript]] = IndexingScript
    metadata        : ClassVar[Type[CreateMetadata]] = CreateMetadata
    metalist        : ClassVar[Type[CreateMetaList]] = CreateMetaList
    detect          : ClassVar[Type[DetectHits]] = DetectHits
    refine          : ClassVar[Type[RefineScript]] = RefineScript
    post_refine     : ClassVar[Type[PostRefineScript]] = PostRefineScript

    @classmethod
    def parser(cls) -> ArgumentParser:
        """Return the top-level :class:`~argparse.ArgumentParser` for ``cbclib_cli``.

        Registers a subparser for every :class:`BaseScript` subclass found
        on this class.

        Returns:
            Configured :class:`~argparse.ArgumentParser`.
        """
        parser = ArgumentParser(description='Process CBD patterns')
        subparsers = parser.add_subparsers(help='Available subcommands', dest='command')

        for key, Script in cls.__dict__.items():
            if isinstance(Script, type) and issubclass(Script, BaseScript):
                subparser = subparsers.add_parser(key, help=Script.parser_description())
                Script.parser(subparser)
        return parser

def main():
    parser = Scripts.parser()

    args = vars(parser.parse_args())

    print(f"JSON file with the scan parameters: {args['scan']}")

    scan: ScanConfig = ScanConfig.read(args['scan'])
    print(f"Run: {scan.scan_num:d}")
    scan.system.apply()

    if args['command'] == 'compile':
        script = CompileFiles.from_file(args['kind'], args['scan'],
                                          args['in_suffix'], args['out_suffix'])
        script.run()
    elif args['command'] == 'index':
        print(f"JSON file with the indexing parameters: {args['parameters']}")
        script = IndexingScript.from_file(args['scan'], args['parameters'],
                                          args['xtals'], args['hits_dir'],
                                          args['in_suffix'], args['out_suffix'],
                                          args['chunk_id'])
        script.run()
    elif args['command'] == 'metadata':
        print(f"JSON file with the metadata parameters: {args['parameters']}")
        script = CreateMetadata.from_file(args['scan'], args['parameters'])
        script.run()
    elif args['command'] == 'metalist':
        print(f"JSON file with the metadata parameters: {args['parameters']}")
        script = CreateMetaList.from_file(args['scan'], args['parameters'],
                                          args['chunk_id'], args['n_chunks'],
                                          args['n_out'])
        script.run()
    elif args['command'] == 'detect':
        print(f"JSON file with the streak finding parameters: {args['parameters']}")
        script = DetectHits.from_file(args['kind'], args['scan'], args['parameters'],
                                      args['chunk_id'], args['n_chunks'],
                                      args['frames_only'], args['out_suffix'])
        script.run()
    elif args['command'] == 'refine':
        print(f"JSON file with the refinement parameters: {args['parameters']}")
        script = RefineScript.from_file(args['scan'], args['parameters'],
                                        args['hits_dir'], args['xtal_dir'],
                                        args['in_suffix'], args['out_suffix'],
                                        args['chunk_id'])
        script.run()
    elif args['command'] == 'post_refine':
        print(f"JSON file with the post-refinement parameters: {args['parameters']}")
        script = PostRefineScript.from_file(args['scan'], args['parameters'],
                                            args['in_suffix'], args['out_suffix'],
                                            args['chunk_id'])
        script.run()
    else:
        raise ValueError(f"Invalid command: {args['command']}")
