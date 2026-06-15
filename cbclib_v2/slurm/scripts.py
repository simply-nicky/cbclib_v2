from argparse import ArgumentParser
from dataclasses import dataclass
from multiprocessing import cpu_count
import os
import re
from shlex import quote
from typing import ClassVar, Iterator, List, Literal, Type
import h5py
import pandas as pd
from tqdm.auto import tqdm
from ..cuda import Allocator, set_allocator
from .._src.annotations import AnyNamespace, NumPy
from .._src.array_api import default_api, default_rng, Platform
from .._src.config import CPUConfig
from .._src.data_container import compute_index
from .._src.data_processing import CrystMetadata
from .._src.cxi_protocol import H5Handler, TrainIndices, write_hdf
from .._src.run import BaseRun, RunConfig, open_run
from .._src.scripts import (BaseParameters, FinderConfig, IndexingConfig,
                            MetadataParameters, RegionFinderConfig, StreakFinderConfig)
from .._src.scripts import create_metadata, pool_detection, pool_indexing
from .._src.streaks import StackedStreaks, Streaks
from ..indexer import FixedSetup, XtalCell, XtalList, XtalState
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
        set_allocator(self.cuda_allocator, strict=True)

    def cpu_config(self) -> CPUConfig:
        """Return a :class:`~cbclib_v2.CPUConfig` context manager for this thread count."""
        return CPUConfig(num_threads=self.num_threads)

    def array_api(self) -> AnyNamespace:
        """Return the array namespace matching :attr:`platform` (NumPy or CuPy)."""
        return default_api(self.platform)

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
    :doc:`/workflows`).  Used by ``cbclib_cli index`` and the SLURM
    indexing job to locate the detector geometry and crystal unit-cell
    files.

    Attributes:
        setup_file: Path to the detector geometry JSON file (read by
            :class:`~cbclib_v2.indexer.FixedSetup`).
        unit_file: Path to the crystal unit-cell JSON file (read by
            :class:`~cbclib_v2.indexer.XtalCell`).
        xtals_dir: Directory where per-chunk indexing results are written.
    """

    setup_file      : str
    unit_file       : str
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

    def setup(self) -> FixedSetup:
        """Load the fixed detector geometry from :attr:`setup_file`.

        Returns:
            :class:`~cbclib_v2.indexer.FixedSetup` instance.

        Raises:
            ValueError: If :attr:`setup_file` is empty.
        """
        if self.setup_file == str():
            raise ValueError("No setup file provided")
        return FixedSetup.read(self.setup_file)

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
                  dir: str | None=None) -> str:
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
            return get_filename(file_index, suffix, extension)

        if file_index is None:
            filename = get_filename(None, suffix, extension)
            return os.path.join(dir, filename)

        return os.path.join(self.scan_subdir(dir, suffix),
                            get_filename(file_index, str(), extension))

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

@dataclass
class CompileStreaks(BaseScript):
    """Implements ``cbclib_cli compile``.

    Reads all per-chunk streak or region HDF5 files from the scan directory
    and concatenates them into a single merged HDF5 file.

    Attributes:
        kind: ``'streaks'`` or ``'regions'`` — selects which output
            directory to read from.
        scan_file: Path to the scan configuration JSON file.
    """

    kind        : DetectionKind
    scan_file   : str

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('scan', type=str, help='Path to a scan parameters JSON file')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Compile detected streaks into a single table"

    @classmethod
    def from_file(cls, kind: DetectionKind, scan_file: str) -> 'CompileStreaks':
        return cls(kind, scan_file)

    def run(self):
        scan = ScanConfig.read(self.scan_file)
        print(f'Assembling streaks for a scan {scan.scan_num:d}...')

        # Use regex pattern from filename_pattern method instead of glob
        if self.kind == 'streaks':
            hits_dir = scan.detect.streaks_dir
        elif self.kind == 'regions':
            hits_dir = scan.detect.regions_dir
        else:
            raise ValueError(f'Invalid detection kind: {self.kind}')

        scan_dir = scan.scan_subdir(hits_dir)
        scan_files = list(scan.list_files(scan_dir))

        print(f'Found {len(scan_files)} streak files for run {scan.scan_num:d}')
        output_path = os.path.join(hits_dir, scan.scan_file())
        print(f'Writing the detected streaks to the file: {output_path}')

        tables: List[pd.DataFrame | pd.Series] = []
        for scan_file in scan_files:
            path = os.path.join(scan_dir, scan_file)
            tables.append(pd.read_hdf(path, 'data'))
        pd.concat(tables).to_hdf(output_path, key='data')

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
        initial.add_argument('--chunk_id', '-c', type=int, help='ID of the chunk to process')
        initial.add_argument('--n_chunks', '-n', type=int, help='Total number of chunks')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Calculate a list of CBD metadata"

    @classmethod
    def from_file(cls, scan_file: str, params_file: str, chunk_id: int | None, n_chunks: int | None
                  ) -> 'CreateMetaList':
        scan = ScanConfig.read(scan_file)
        params = MetadataParameters.read(params_file)
        return cls(scan, params, chunk_id, n_chunks)

    def run(self):
        print("Configuring the script...")
        xp = self.scan.system.array_api()

        print("Looking for data...")
        run = self.scan.run()
        indices = run.indices()

        if self.chunk_id is not None and self.n_chunks is not None:
            print(f"Processing chunk No. {self.chunk_id}")
            indices = list(indices.split(self.n_chunks))[self.chunk_id]
        else:
            print("Processing the whole scan")
        print(f"Starting to process {len(indices):d} frames...")

        offsets = xp.arange(0, self.scan.metalist.n_frames) - self.scan.metalist.n_frames // 2
        spacing = min(self.scan.metalist.spacing, len(indices) - len(offsets))

        centers = xp.arange(-int(offsets[0]), len(indices) - int(offsets[-1]),
                            spacing)
        frames = centers[:, None] + offsets
        print(f"Creating a metadata list of {frames.shape[0]:d} points...")

        output_path = self.scan.scan_file(self.chunk_id, dir=self.scan.metalist.output_dir)
        dir_path = os.path.dirname(output_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        print(f"The results will be saved to {output_path}")

        handler = H5Handler(CrystMetadata.default_protocol())
        mask, var = xp.ones(1, dtype=bool), xp.zeros(1)

        whitefields = []
        with self.scan.system.cpu_config():
            for index, chunk in tqdm(enumerate(frames), total=frames.shape[0],
                                     desc='Generating the list'):
                images = run.data(indices[chunk], geometry=self.scan.apply_geometry, verbose=False,
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
                                     mask=mask, std=xp.sqrt(var / frames.shape[0]))
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
        initial.add_argument('--chunk_id', '-c', type=int, help='Index of the chunk to process')
        initial.add_argument('--n_chunks', '-n', type=int, help='Number of chunks to process')
        initial.add_argument('--frames-only', action='store_true',
                             help='Only save the list of frames with hits')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Detect streaks in CBD patterns"

    @classmethod
    def from_file(cls, kind: DetectionKind, scan_file: str, params_file: str, chunk_id: int | None,
                  n_chunks: int | None, frames_only: bool) -> 'DetectHits':
        scan = ScanConfig.read(scan_file)
        if kind == 'streaks':
            params = StreakFinderConfig.read(params_file)
        elif kind == 'regions':
            params = RegionFinderConfig.read(params_file)
        else:
            raise ValueError(f"Invalid detection kind: {kind}")
        return cls(scan, params, chunk_id, n_chunks, frames_only)

    @property
    def kind(self) -> DetectionKind:
        """Detection kind derived from the type of :attr:`params`."""
        if isinstance(self.params, StreakFinderConfig):
            return 'streaks'
        if isinstance(self.params, RegionFinderConfig):
            return 'regions'
        raise ValueError(f"Invalid parameters type for detection: {type(self.params)}")

    def run(self):
        xp = self.scan.system.array_api()

        print("Looking for data...")
        run = self.scan.run()
        indices = run.indices()

        if self.chunk_id is not None and self.n_chunks is not None:
            print(f"Processing chunk No. {self.chunk_id}")
            chunk = list(indices.split(self.n_chunks))[self.chunk_id]
        else:
            print("Processing the whole scan")
            chunk = indices

        print(f"Starting to process {len(chunk):d} frames...")

        metadata_path = self.scan.find_metadata(self.chunk_id)
        print(f"Using the metadata saved at {metadata_path}")

        loader = run.worker(self.scan.apply_geometry)
        detector = None if self.scan.apply_geometry else self.scan.data.geometry()
        with self.scan.system.cpu_config():
            streaks = pool_detection(loader, chunk, metadata_path, self.params,
                                     self.scan.system.platform, detector)

        frames, counts = xp.unique_counts(streaks.index)
        hit_frames = frames[counts > self.scan.detect.hit_threshold]
        hits = streaks.take(hit_frames)
        print(f"{hit_frames.size:d} hits were found.")

        if len(hits) > 0:
            if self.frames_only:
                print("Frames only requested, skipping saving the full hits data.")
                output_path = self.scan.scan_file(self.chunk_id, extension='.csv',
                                                  dir=self.scan.detect.streaks_dir)
                pd.DataFrame({'frame': hit_frames}).to_csv(output_path, index=False)
                print(f"The results were saved to {output_path}")
            else:
                print("Preparing the file...")
                df = hits.to_dataframe()
                hit_indices = xp.where(xp.isin(xp.array(list(chunk.index())), hit_frames))[0]
                pulse_ids = run.metadata('pulse_id', chunk[hit_indices])
                pulse_ids = pulse_ids[hits.reset_index().index]
                # If data for each module is saved in a separate file
                # pulse_ids for each module will be stacked along the second axis
                if pulse_ids.ndim > 1:
                    pulse_ids = pulse_ids[:, 0]
                df['pulse_id'] = pulse_ids

                if self.kind == 'streaks':
                    output_dir = self.scan.detect.streaks_dir
                else:
                    output_dir = self.scan.detect.regions_dir

                output_path = self.scan.scan_file(self.chunk_id, dir=output_dir)
                dir_path = os.path.dirname(output_path)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                print(f"The results will be saved to {output_path}")

                df.to_hdf(output_path, key='data')

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
        suffix: Suffix appended to the output directory name.
        chunk_id: Zero-based chunk index (``None`` = whole scan).
        n_chunks: Total number of chunks (``None`` = whole scan).
    """

    scan        : ScanConfig
    params      : IndexingConfig
    xtals       : str
    suffix      : str
    chunk_id    : int | None
    n_chunks    : int | None

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
                             help='Path to an indexing parameters JSON file')
        initial.add_argument('--xtals', '-x', type=str, default=str(),
                             help='Path to a crystal orientations H5 file')
        initial.add_argument('--suffix', '-s', type=str, default=str(),
                             help='Suffix of the folder where the indexing results are saved')
        initial.add_argument('--chunk_id', '-c', type=int, help='ID of the chunk to process')
        initial.add_argument('--n_chunks', '-n', type=int, help='Total number of chunks')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Index detected streaks in CBD patterns"

    @classmethod
    def from_file(cls, scan_file: str, params_file: str, xtals: str, suffix: str,
                  chunk_id: int | None, n_chunks: int | None) -> 'IndexingScript':
        scan = ScanConfig.read(scan_file)
        params = IndexingConfig.read(params_file)
        return cls(scan, params, xtals, suffix, chunk_id, n_chunks)

    def run(self):
        print("Configuring the script...")
        xp = self.scan.system.array_api()

        geometry = self.scan.data.geometry()
        hits_file = self.scan.scan_file(self.chunk_id, dir=self.scan.detect.streaks_dir)
        if not os.path.isfile(hits_file):
            print(f"No streaks file found at {hits_file}")
            return

        print(f"Loading detected streaks from {hits_file}...")
        dataframe = pd.read_hdf(hits_file, 'data')
        if 'module_id' in dataframe.columns:
            num_modules = geometry.num_modules
            streaks = StackedStreaks.import_dataframe(dataframe, num_modules=num_modules, xp=xp)
        else:
            streaks = Streaks.import_dataframe(dataframe, xp=xp)
        assembled = geometry.to_streaks(streaks)
        patterns = geometry.to_patterns(assembled)

        if self.xtals:
            print(f"Loading crystal orientations from {self.xtals}...")
            df = pd.read_hdf(self.xtals, 'data')
            xtals = XtalList.import_dataframe(df, xp=xp).to_xtals()
        else:
            print("No crystal orientations provided, using the unit cell information")
            xtals = self.scan.setup.xtal(xp=xp)
        setup = self.scan.setup.setup()

        print(f"Indexing {len(patterns):d} patterns...")
        with self.scan.system.cpu_config():
            indexed = pool_indexing(patterns, xtals, setup, self.params,
                                    self.scan.system.platform, xp)

        output_path = self.scan.scan_file(self.chunk_id, dir=self.scan.setup.xtals_dir,
                                          suffix=self.suffix)
        dir_path = os.path.dirname(output_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        print(f"Saving the results to {output_path}...")
        df = indexed.to_dataframe()
        df.to_hdf(output_path, key='data')
        with h5py.File(output_path, 'a') as output_file:
            for key in ('files/xtal_file', 'files/hits_file', 'files/setup_file'):
                if key in output_file:
                    del output_file[key]

            if self.xtals:
                output_file['files/xtal_file'] = self.xtals
            else:
                output_file['files/xtal_file'] = self.scan.setup.unit_file
            output_file['files/hits_file'] = hits_file
            output_file['files/setup_file'] = self.scan.setup.setup_file

class SBatchScripts:
    """Factory for single ``sbatch`` job scripts.

    Each classmethod assembles a ``cbclib_cli <subcommand> …`` shell command,
    reads SLURM parameters from *script_file*, and returns a
    :class:`~cbclib_v2.slurm.SLURMScript` ready for submission via
    :meth:`~cbclib_v2.slurm.SLURMJobManager.submit`.
    """

    main        : ClassVar[str] = 'cbclib_cli'

    @classmethod
    def compile(cls, kind: DetectionKind, scan_file: str, script_file: str) -> SLURMScript:
        """Build a ``cbclib_cli compile`` script.

        Args:
            kind: ``'streaks'`` or ``'regions'``.
            scan_file: Path to the scan configuration JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the compile step.
        """
        command = f"{cls.main} compile {quote(kind)} {quote(scan_file)}"
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="compile", command=command, parameters=script_spec)

    @classmethod
    def index(cls, scan_file: str, params_file: str, script_file: str,
              xtals: str | None=None, suffix: str | None=None, chunk_id: int | None=None,
              n_chunks: int | None=None) -> SLURMScript:
        """Build a ``cbclib_cli index`` script.

        Args:
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the indexing parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            xtals: Path to an initial crystal orientations HDF5 file
                (empty string → use unit cell).
            suffix: Output directory suffix.
            chunk_id: Optional chunk index for chunked processing.
            n_chunks: Optional total chunk count.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the indexing step.
        """
        command = f"{cls.main} index {quote(scan_file)} {quote(params_file)} "
        if xtals is not None:
            command += f"--xtals {quote(xtals)} "
        if suffix is not None:
            command += f"--suffix {quote(suffix)} "
        if chunk_id is not None and n_chunks is not None:
            command += f' --chunk_id {chunk_id:d}'
            command += f' --n_chunks {n_chunks:d}'
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
                 chunk_id: int | None=None, n_chunks: int | None=None) -> SLURMScript:
        """Build a ``cbclib_cli metalist`` script.

        Args:
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the metadata parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            chunk_id: Optional chunk index.
            n_chunks: Optional total chunk count.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the metalist step.
        """
        command = f"{cls.main} metalist {quote(scan_file)} {quote(params_file)}"
        if chunk_id is not None and n_chunks is not None:
            command += f' --chunk_id {chunk_id:d}'
            command += f' --n_chunks {n_chunks:d}'
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="metalist", command=command, parameters=script_spec)

    @classmethod
    def detect(cls, kind: DetectionKind, scan_file: str, params_file: str, script_file: str,
               chunk_id: int | None=None, n_chunks: int | None=None, frames_only: bool=False
               ) -> SLURMScript:
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

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the detection step.
        """
        command = f"{cls.main} detect {quote(kind)} {quote(scan_file)} {quote(params_file)}"
        if chunk_id is not None and n_chunks is not None:
            command += f' --chunk_id {chunk_id:d}'
            command += f' --n_chunks {n_chunks:d}'
        if frames_only:
            command += ' --frames-only'
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="detect", command=command, parameters=script_spec)

class SBatchArrayScripts:
    """Factory for ``sbatch --array`` job scripts.

    Like :class:`SBatchScripts` but each script injects
    ``FILE_INDEX=${SLURM_ARRAY_TASK_ID}`` so that the chunk index is taken
    from the SLURM task ID at runtime.  Use with
    :meth:`~cbclib_v2.slurm.SLURMJobManager.submit_array`.
    """

    main        : ClassVar[str] = 'cbclib_cli'

    @classmethod
    def index(cls, scan_file: str, params_file: str, script_file: str, n_chunks: int,
              xtals: str | None=None, suffix: str | None=None) -> SLURMScript:
        """Build a ``cbclib_cli index`` array script for *n_chunks* tasks.

        Args:
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the indexing parameters JSON.
            xtals: Path to an initial crystal orientations HDF5 file.
            suffix: Output directory suffix.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            n_chunks: Total number of array tasks.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` configured for array submission.
        """
        command = f"{cls.main} index {quote(scan_file)} {quote(params_file)} "\
                  f" --chunk_id ${{FILE_INDEX}} --n_chunks {n_chunks:d}"
        if xtals is not None:
            command += f" --xtals {quote(xtals)}"
        if suffix is not None:
            command += f" --suffix {quote(suffix)}"
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
               n_chunks: int) -> SLURMScript:
        """Build a ``cbclib_cli detect`` array script for *n_chunks* tasks.

        Args:
            kind: ``'streaks'`` or ``'regions'``.
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the detection parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            n_chunks: Total number of array tasks.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` configured for array submission.
        """
        command = f"{cls.main} detect {quote(kind)} {quote(scan_file)} {quote(params_file)}" \
                  f" --chunk_id ${{FILE_INDEX}} --n_chunks {n_chunks:d}"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')
        return SLURMScript(job_name="detect_array", command=command, parameters=script_spec)

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
        compile: :class:`CompileStreaks` — merge per-chunk results.
        metadata: :class:`CreateMetadata` — compute background whitefield.
        metalist: :class:`CreateMetaList` — compute PCA metalist.
        detect: :class:`DetectHits` — run streak or region detection.
        index: :class:`IndexingScript` — index detected patterns.
    """

    sbatch          : ClassVar[Type[SBatchScripts]] = SBatchScripts
    sbatch_array    : ClassVar[Type[SBatchArrayScripts]] = SBatchArrayScripts
    compile         : ClassVar[Type[CompileStreaks]] = CompileStreaks
    index           : ClassVar[Type[IndexingScript]] = IndexingScript
    metadata        : ClassVar[Type[CreateMetadata]] = CreateMetadata
    metalist        : ClassVar[Type[CreateMetaList]] = CreateMetaList
    detect          : ClassVar[Type[DetectHits]] = DetectHits

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
        script = CompileStreaks.from_file(args['kind'], args['scan'])
        script.run()
    elif args['command'] == 'index':
        print(f"JSON file with the indexing parameters: {args['parameters']}")
        script = IndexingScript.from_file(args['scan'], args['parameters'],
                                          args['xtals'], args['suffix'],
                                          args['chunk_id'], args['n_chunks'])
        script.run()
    elif args['command'] == 'metadata':
        print(f"JSON file with the metadata parameters: {args['parameters']}")
        script = CreateMetadata.from_file(args['scan'], args['parameters'])
        script.run()
    elif args['command'] == 'metalist':
        print(f"JSON file with the metadata parameters: {args['parameters']}")
        script = CreateMetaList.from_file(args['scan'], args['parameters'],
                                          args['chunk_id'], args['n_chunks'])
        script.run()
    elif args['command'] == 'detect':
        print(f"JSON file with the streak finding parameters: {args['parameters']}")
        script = DetectHits.from_file(args['kind'], args['scan'], args['parameters'],
                                      args['chunk_id'], args['n_chunks'], args['frames_only'])
        script.run()
    else:
        raise ValueError(f"Invalid command: {args['command']}")
