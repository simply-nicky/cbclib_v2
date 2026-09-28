from dataclasses import InitVar, dataclass, field, fields
import json
from multiprocessing import cpu_count
from numbers import Integral
import os
import re
from typing import ClassVar, Iterator, List, Literal, Sequence, overload
import h5py
import pandas as pd
from jax import config as jax_config
from .. import cuda
from ..cuda import Allocator
from .._src.annotations import AnyNamespace, Array, BoolArray, IntArray, JaxNumPy, NumPy, Shape
from .._src.array_api import asnumpy, default_api, Platform
from .._src.config import CPUConfig
from .._src.data_container import ArrayContainer, Container, list_indices
from .._src.cxi_protocol import TrainIndices
from .._src.parser import read_all
from .._src.run import BaseRun, RunConfig, RunList, open_run
from .._src.scripts import BaseParameters
from ..indexer import FixedGeometry, FixedLens, XtalCell, XtalState
from .slurm_manager import SLURMArrayScript, ScriptSpec

ScanNumbers = int | range | List[int]
DetectionKind = Literal['streaks', 'regions']
ImageKind = Literal['full', 'stacked']

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

    def apply(self):
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

    def output_dir(self, kind: DetectionKind) -> str:
        """Return the configured output directory for a detection kind.

        Args:
            kind: Detection representation written by the pipeline.

        Returns:
            Root directory for the requested detection artifacts.
        """
        if kind == 'streaks':
            return self.streaks_dir
        if kind == 'regions':
            return self.regions_dir
        raise ValueError(f"Invalid detection kind: {kind}")

@dataclass
class DetectionAttributes(Container):
    """HDF5 provenance identifying one detection-metadata chunk."""
    scan_num          : int
    detection_kind    : DetectionKind
    hit_threshold     : int
    chunk_id          : int
    n_chunks          : int
    n_frames          : int

    def is_hit(self, n_detections: IntArray) -> BoolArray:
        """Return which detection counts exceed this result's hit threshold."""
        return n_detections > self.hit_threshold

    def hit_rate(self, n_hits: int) -> float:
        """Return the hit fraction for this result's processed-frame count."""
        return n_hits / self.n_frames if self.n_frames else 0.0

    @classmethod
    def read_integer(cls, attributes: h5py.AttributeManager, key: str) -> int:
        """Read one attribute from an HDF5 file and validate its type."""
        if key not in attributes:
            raise ValueError(f"Missing attribute {key}")
        value = attributes[key]
        if not isinstance(value, Integral):
            raise ValueError(f"Invalid attribute {key}: expected int, got {type(value)}")
        return int(value)

    @classmethod
    def read_string(cls, attributes: h5py.AttributeManager, key: str) -> str:
        """Read one attribute from an HDF5 file and validate its type."""
        if key not in attributes:
            raise ValueError(f"Missing attribute {key}")
        value = attributes[key]
        if not isinstance(value, (str, bytes)):
            raise ValueError(f"Invalid attribute {key}: expected str or bytes, got {type(value)}")
        return value.decode('utf-8') if isinstance(value, bytes) else value

    @classmethod
    def read(cls, path: str) -> 'DetectionAttributes':
        """Read and validate detection provenance from one artifact."""
        with h5py.File(path, mode='r') as input_file:
            kind_str = cls.read_string(input_file.attrs, 'detection_kind')
            if kind_str == 'streaks':
                detection_kind = 'streaks'
            elif kind_str == 'regions':
                detection_kind = 'regions'
            else:
                raise ValueError(f"Invalid detection kind: {kind_str}")
            return cls(
                scan_num=cls.read_integer(input_file.attrs, 'scan_num'),
                detection_kind=detection_kind,
                hit_threshold=cls.read_integer(input_file.attrs, 'hit_threshold'),
                chunk_id=cls.read_integer(input_file.attrs, 'chunk_id'),
                n_chunks=cls.read_integer(input_file.attrs, 'n_chunks'),
                n_frames=cls.read_integer(input_file.attrs, 'n_frames'))

    @classmethod
    def concat(cls, chunks: Sequence['DetectionAttributes']) -> 'DetectionAttributes':
        """Validate and aggregate detection artifact attributes."""
        if not chunks:
            raise ValueError("No detection artifacts found")
        expected = (chunks[0].scan_num, chunks[0].detection_kind,
                    chunks[0].hit_threshold, chunks[0].n_chunks)
        n_frames = chunks[0].n_frames
        for chunk in chunks[1:]:
            if (chunk.scan_num, chunk.detection_kind,
                chunk.hit_threshold, chunk.n_chunks) != expected:
                raise ValueError("Detection artifacts describe different scan results")
            n_frames += chunk.n_frames
        return cls(expected[0], expected[1], expected[2], 0, 1, n_frames)

    @classmethod
    def from_scan(cls, scan: 'Scan', kind: DetectionKind, n_frames: int) -> 'DetectionAttributes':
        return cls(scan_num=scan.scan_num, detection_kind=kind,
                   hit_threshold=scan.config.detect.hit_threshold,
                   chunk_id=0, n_chunks=1, n_frames=n_frames)

    @classmethod
    def from_chunk(cls, scan: 'Scan', kind: DetectionKind, chunk_id: int, n_chunks: int,
                   n_frames: int) -> 'DetectionAttributes':
        return cls(scan_num=scan.scan_num, detection_kind=kind,
                   hit_threshold=scan.config.detect.hit_threshold,
                   chunk_id=chunk_id, n_chunks=n_chunks, n_frames=n_frames)

    def write(self, path: str, mode: Literal['a', 'w', 'r+']='a'):
        """Write detection provenance to an HDF5 artifact."""
        with h5py.File(path, mode=mode) as output_file:
            for name, value in self.to_dict().items():
                output_file.attrs[name] = value

@dataclass
class DetectionMetadata(ArrayContainer):
    """Hold hit metadata and derive scan-level detection statistics."""
    index           : IntArray          # (n_hits,)
    filename        : Array             # (n_hits, n_files_per_hit)
    file_index      : IntArray          # (n_hits, n_files_per_hit)
    pulse_id        : IntArray          # (n_hits,)
    n_detections    : IntArray          # (n_hits,)

    @staticmethod
    def serialise_rows(values: Array) -> list[str]:
        """Serialise per-hit source addresses for storage in one table column."""
        return [json.dumps(row.tolist()) for row in values]

    @staticmethod
    def deserialise_rows(values: pd.Series, dtype: type[str] | type[int],
                         xp: AnyNamespace) -> Array:
        """Restore per-hit source addresses from one table column."""
        if values.empty:
            return xp.empty((0, 0), dtype=dtype)
        return xp.asarray([json.loads(value) for value in values], dtype=dtype)

    @classmethod
    def from_run(cls, hits: TrainIndices, pulse_ids: IntArray, n_detections: IntArray,
                 xp: AnyNamespace) -> 'DetectionMetadata':
        """Load facility metadata and bind detection counts to hit frames."""
        if len(hits) == 0:
            return cls(
                index=xp.zeros(0, dtype=int),
                filename=xp.empty((0, 0), dtype=str),
                file_index=xp.empty((0, 0), dtype=int),
                pulse_id=xp.zeros(0, dtype=int),
                n_detections=xp.zeros(0, dtype=int))

        if pulse_ids.ndim > 1:
            pulse_ids = pulse_ids[:, 0]

        index, filenames, file_index = [], [], []
        for record in hits.records():
            index.append(record.index)
            filenames.append(record.filename)
            file_index.append(record.file_index)

        return cls(index=xp.asarray(index), filename=xp.asarray(filenames, dtype=str),
                   file_index=xp.asarray(file_index, dtype=int),
                   pulse_id=xp.asarray(pulse_ids), n_detections=xp.asarray(n_detections))

    @classmethod
    def import_dataframe(cls, dataframe: pd.DataFrame | pd.Series, xp: AnyNamespace
                         ) -> 'DetectionMetadata':
        """Read hit metadata from one detection artifact."""
        filenames = cls.deserialise_rows(dataframe['filename'], str, xp)
        file_index = cls.deserialise_rows(dataframe['file_index'], int, xp)
        return cls(index=xp.asarray(dataframe['index']), filename=filenames,
                   file_index=file_index, pulse_id=xp.asarray(dataframe['pulse_id']),
                   n_detections=xp.asarray(dataframe['n_detections']))

    @property
    def shape(self) -> Shape:
        return self.index.shape

    @property
    def detections_in_hits(self) -> int:
        return int(self.n_detections.sum())

    @property
    def mean_per_hit(self) -> float:
        if self.size == 0:
            return 0.0
        return float(self.n_detections.mean())

    @property
    def median_per_hit(self) -> float:
        if self.size == 0:
            return 0.0
        xp = self.__array_namespace__()
        return float(xp.median(self.n_detections))

    @property
    def maximum_per_hit(self) -> int:
        if self.size == 0:
            return 0
        return int(self.n_detections.max())

    def to_dataframe(self) -> pd.DataFrame:
        """Return the hit metadata in the detection artifact schema."""
        return pd.DataFrame({
            'index': asnumpy(self.index),
            'filename': self.serialise_rows(self.filename),
            'file_index': self.serialise_rows(self.file_index),
            'pulse_id': asnumpy(self.pulse_id),
            'n_detections': asnumpy(self.n_detections)})

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
class ScanArgument:
    """Represent one scan or a compact selection of scans.

    Attributes:
        scan_num: One scan number, a Python range, or a list of scan numbers.
        cli_value: Scalar value or SLURM array expression written to the command.
        task_ids: Scan-indexed SLURM task identifiers, or ``None`` for a scalar job.
    """
    scan_num        : ScanNumbers

    @classmethod
    def read_string(cls, string: str) -> ScanNumbers:
        """Parse an integer, range, or whitespace or underscore-separated list."""
        items = re.split(r'[\s_]+', string.strip())
        if len(items) > 1:
            try:
                return [int(item) for item in items]
            except ValueError as error:
                raise ValueError(f"Invalid scan number list: {string}") from error

        match = re.fullmatch(r'(\d+)-(\d+)(?:-(-?\d+))?', string)
        if match is not None:
            return range(int(match.group(1)), int(match.group(2)),
                         int(match.group(3)) if match.group(3) is not None else 1)
        try:
            return int(string)
        except ValueError as error:
            raise ValueError(f"Invalid scan number selection: {string}") from error

    @classmethod
    def from_string(cls, string: str) -> 'ScanArgument':
        """Parse a scan selection string and return a :class:`ScanArgument`."""
        return cls(cls.read_string(string))

    def enumerate(self) -> List[int]:
        """Validate scan numbers and return them as a non-empty list."""
        numbers = list_indices(self.scan_num)
        if not numbers:
            raise ValueError("scan_num selection must be non-empty")
        if any(scan_num < 0 for scan_num in numbers):
            raise ValueError("scan_num values must be non-negative integers")
        if len(set(numbers)) != len(numbers):
            raise ValueError("scan_num selection must not contain duplicates")
        return numbers

    def slurm_command(self, command: str) -> str:
        return command.format(scan_num=str(self))

    def job_name(self, prefix: str) -> str:
        return f'{prefix}_{str(self)}'

    def slurm_array_script(self, command: str, job_name: str, script_spec: ScriptSpec
                           ) -> SLURMArrayScript:
        command = command.format(scan_num='${SCAN_NUM}')
        script_spec.add_define('SCAN_NUM', '${SLURM_ARRAY_TASK_ID}')

        if isinstance(self.scan_num, int):
            task_ids = [self.scan_num,]
        elif isinstance(self.scan_num, range):
            task_ids = self.scan_num
        elif isinstance(self.scan_num, list):
            task_ids = self.scan_num
        else:
            raise TypeError("scan_num must be an int, range, or list of ints")
        return SLURMArrayScript(job_name=self.job_name(job_name), command=command,
                                parameters=script_spec, task_ids=task_ids)

    def __len__(self) -> int:
        """Return the number of scans represented by this argument."""
        return len(self.enumerate())

    def __iter__(self) -> Iterator['ScanArgument']:
        """Yield one :class:`ScanArgument` per scan number represented."""
        for scan_num in self.enumerate():
            yield ScanArgument(scan_num)

    def __str__(self) -> str:
        """Return the canonical scan selection string."""
        if isinstance(self.scan_num, int):
            return str(self.scan_num)
        if isinstance(self.scan_num, range):
            if self.scan_num.step == 1:
                return f'{self.scan_num.start}-{self.scan_num.stop}'
            return f'{self.scan_num.start}-{self.scan_num.stop}-{self.scan_num.step}'
        return '_'.join(str(num) for num in self.scan_num)

@dataclass
class FileMatch:
    """Hold one scan file matched against the canonical naming pattern.

    Attributes:
        filename: Absolute path of the matched file.
        chunk_id: Zero-based chunk index parsed from an ``_f{index}`` segment,
            or ``None`` if absent.
        suffix: Suffix segment following the chunk index, or ``None`` if absent.
        extension: File extension including the leading dot.
    """
    filename        : str
    chunk_id        : int | None
    suffix          : str | None
    extension       : str

@dataclass
class ScanFiles(BaseParameters):
    """Scan-specific file and directory naming utilities.

    Provides helpers that translate a ``(scan_num, image_kind)`` pair into
    the canonical file and directory names used throughout the pipeline.

    Attributes:
        scan_num: One scan number or a multi-scan selection.
        image_kind: Image layout — ``'full'`` for assembled lab-frame
            images, ``'stacked'`` for per-module stacks.
    """
    scan_num            : InitVar[ScanNumbers]
    image_kind          : InitVar[ImageKind]
    dir_pattern         : ClassVar[str] = 'scan_{scan_num}_{kind}'
    file_pattern        : ClassVar[str] = 'scan_{scan_num}_{kind}'
    scan_name           : str = field(init=False)

    def __post_init__(self, scan_num: ScanNumbers, image_kind: ImageKind):
        self.scan_name = self.file_pattern.format(scan_num=str(ScanArgument(scan_num)),
                                                  kind=image_kind)

    def list_files(self, dir: str) -> Iterator[FileMatch]:
        """Yield :class:`FileMatch` records for scan files found in *dir*.

        Matches files whose name follows the canonical
        ``scan_{num}_{kind}[_f{index}][_{suffix}]{ext}`` pattern.

        Args:
            dir: Directory to search.

        Yields:
            :class:`FileMatch` for each matching file, with the chunk index,
            suffix, and extension parsed out.
        """
        pattern = re.escape(self.scan_name)
        pattern += r'(?:_f(?P<chunk_id>[0-9]{4}))?'
        pattern += r'(?:_(?P<suffix>[^.]+))?'
        pattern += r'(?P<extension>\..*)?$'

        for filename in os.listdir(dir):
            match = re.match(pattern, filename)
            if match is not None:
                chunk_id = match.group('chunk_id')
                yield FileMatch(filename=os.path.join(dir, filename),
                                chunk_id=int(chunk_id) if chunk_id is not None else None,
                                suffix=match.group('suffix'),
                                extension=match.group('extension') or str())

    def list_dirs(self, dir: str) -> Iterator[str]:
        """Yield paths of scan subdirectories found in *dir*.

        Matches directories whose name follows the canonical
        ``scan_{num}_{kind}[_{suffix}]`` pattern.

        Args:
            dir: Parent directory to search.

        Yields:
            Absolute paths of matching subdirectories.
        """
        pattern = self.scan_name
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
        dir = self.scan_name
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

    def scan_file(self, file_index: int | None=None, /, *, suffix: str=str(),
                  extension: str='.h5', dir: str | None=None,
                  make_dirs: bool=False) -> str:
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
            filename = self.scan_name
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
class ScanConfig(BaseParameters):
    """Reusable experiment configuration shared by all pipeline workflows.

    Aggregates configuration that can be reused across multiple scans. Loaded
    from a JSON file via :meth:`~cbclib_v2.scripts.BaseParameters.read`.

    See :doc:`/workflows` for the JSON format and a worked example.

    Attributes:
        image_kind: Image layout — ``'full'`` for assembled lab-frame
            images, ``'stacked'`` for per-module stacks.
        data: Facility data source configuration (HDF5 layout, geometry
            file, module count).
        setup: Crystal geometry and unit-cell file paths.
        detect: Hit-finding thresholds and output directories.
        metadata: Background-whitefield computation parameters.
        metalist: PCA metalist computation parameters.
        system: Compute backend and thread count.
    """
    image_kind      : ImageKind
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

    @overload
    def open_scan(self, scan_num: int) -> 'Scan': ...

    @overload
    def open_scan(self, scan_num: range | List[int]) -> 'ScanList': ...

    def open_scan(self, scan_num: ScanNumbers) -> 'Scan | ScanList':
        """Bind this reusable configuration to one scan or a scan selection.

        Args:
            scan_num: One scan number, a Python range, or a list of scan numbers.

        Returns:
            A :class:`Scan` for an integer or a :class:`ScanList` for a
            multi-scan representation.
        """
        if isinstance(scan_num, int):
            return Scan(scan_num, self)
        return ScanList(scan_num, self)

@dataclass
class Scan:
    """Bind reusable experiment configuration to one concrete scan number.

    Attributes:
        scan_num: Run number identifying the scan.
        config: Experiment configuration shared by one or more scans.
    """
    scan_num        : int
    config          : ScanConfig

    def __post_init__(self):
        if (isinstance(self.scan_num, bool) or not isinstance(self.scan_num, int)
                or self.scan_num < 0):
            raise ValueError("scan_num must be a non-negative integer")

    @property
    def files(self) -> ScanFiles:
        """Return file-naming utilities bound to this scan."""
        return ScanFiles(self.scan_num, self.config.image_kind)

    @property
    def scan_string(self) -> str:
        return str(self.scan_num)

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
            metalist_path = self.files.scan_file(file_index,
                                                 dir=self.config.metalist.output_dir)
            if os.path.isfile(metalist_path):
                return metalist_path

        metalist_path = self.files.scan_file(dir=self.config.metalist.output_dir)
        if os.path.isfile(metalist_path):
            return metalist_path

        metadata_path = os.path.join(self.config.metadata.output_dir,
                                     self.files.scan_file(file_index))
        if os.path.isfile(metadata_path):
            return metadata_path

        err_txt = f"No metadata exists for the scan {self.scan_num}"
        if file_index is not None:
            err_txt += f" and file No. {file_index}"
        raise ValueError(err_txt)

    def run(self) -> BaseRun[int, TrainIndices, RunConfig]:
        """Open the facility run and return a :class:`~cbclib_v2.BaseRun`.

        Dispatches to the appropriate run class based on
        ``data.facility`` (e.g. :class:`~cbclib_v2.XFELRun` for EuXFEL,
        :class:`~cbclib_v2.SwissFELRun` for SwissFEL).

        Returns:
            Opened run object.
        """
        return open_run(self.scan_num, self.config.data)

@dataclass
class ScanList:
    """Bind reusable configuration to an indexable collection of scans.

    Unlike :class:`Scan`, this collection only owns selection and file naming;
    facility runs and metadata lookup remain operations on individual scans.

    Attributes:
        scans: Non-empty list of scans sharing one experiment configuration.
    """
    scan_num        : range | List[int]
    config          : ScanConfig
    files           : ScanFiles = field(init=False)

    def __post_init__(self):
        self.files = ScanFiles(self.scan_num, self.config.image_kind)

    def __len__(self) -> int:
        """Return the number of selected scans."""
        return len(self.scan_num)

    def __iter__(self) -> Iterator[Scan]:
        """Yield each selected scan with the shared configuration."""
        for scan_num in self.scan_num:
            yield Scan(scan_num, self.config)

    @property
    def scan_string(self) -> str:
        """Return the canonical scan selection string."""
        return str(ScanArgument(self.scan_num))

    @overload
    def __getitem__(self, index: int) -> Scan: ...

    @overload
    def __getitem__(self, index: slice) -> 'ScanList': ...

    def __getitem__(self, index: int | slice) -> 'Scan | ScanList':
        """Select one scan or a smaller scan collection by position."""
        if isinstance(index, slice):
            if index.start < 0 or index.stop > len(self.scan_num):
                raise IndexError("ScanList slice out of range")
            return ScanList(self.scan_num[index], self.config)
        return Scan(self.scan_num[index], self.config)

    def run(self) -> RunList:
        return open_run(self.scan_num, self.config.data)
