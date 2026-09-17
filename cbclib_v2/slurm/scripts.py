from argparse import ArgumentParser
from dataclasses import dataclass
import json
import os
from shlex import quote
from typing import (Any, ClassVar, Dict, ItemsView, Iterator, KeysView, List, Literal, ValuesView,
                    overload, Tuple, Type)
import h5py
import pandas as pd
from tqdm.auto import tqdm
from .._src.annotations import AnyNamespace, IntArray
from .._src.array_api import asnumpy, default_rng
from .._src.crystfel import Detector
from .._src.data_container import compute_index
from .._src.data_processing import CrystMetadata
from .._src.cxi_protocol import H5Handler, TrainIndices, write_hdf
from .._src.run import BaseRun, RunConfig
from .._src.scripts import (FinderConfig, IndexingConfig, MetadataParameters, PostRefineConfig,
                            RefineConfig, RegionFinderConfig, StreakFinderConfig)
from .._src.scripts import create_metadata, pool_detection, pool_indexing, scale_background
from .._src.streaks import StackedStreaks, Streaks
from ..indexer import BaseSetup, LinePoints, Miller, Patterns, ResolvedSetup, XtalState
from ..scaler import FullState, ScalerModel, ScalerState
from .config import Scan, ScanArgument, ScanConfig, ScanList, ScanNumbers
from .slurm_manager import SLURMArrayScript, SLURMScript, ScriptSpec

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
        preserve_config: Whether all chunks must carry the same configuration.
    """
    required_tables : Tuple[str, ...]
    optional_tables : Tuple[str, ...] = ()
    preserve_config : bool = False

    @property
    def tables(self) -> Tuple[str, ...]:
        """Return every table allowed by this artifact schema."""
        return self.required_tables + self.optional_tables

@dataclass(frozen=True)
class FileContent:
    """Hold one validated per-chunk artifact and its provenance.

    Attributes:
        chunk_id: Numeric identifier parsed from the source filename.
        path: Source artifact path.
        tables: Validated Pandas tables keyed by their HDF5 names.
        config: Parsed root configuration, when present.
        extra: Source-file provenance stored under the ``extra`` group.
    """
    tables   : Dict[str, pd.DataFrame | pd.Series]
    extra    : Dict[str, str]

@dataclass
class CompiledFiles:
    files    : Dict[int, FileContent]

    def __post_init__(self):
        self.files = dict(sorted(self.files.items()))
        if len(set(self.files)) != len(self.files):
            raise ValueError(f"Duplicate chunk IDs found: {list(self.files.keys())}")

    def __contains__(self, key: str) -> bool:
        """Return whether a table exists in every chunk."""
        return all(key in chunk.tables for chunk in self.files.values())

    def keys(self) -> KeysView[int]:
        return self.files.keys()

    def items(self) -> ItemsView[int, FileContent]:
        return self.files.items()

    def values(self) -> ValuesView[FileContent]:
        return self.files.values()

    @property
    def has_extra(self) -> bool:
        """Return whether any chunk carries provenance metadata."""
        return all(chunk.extra for chunk in self.files.values())

    def compile(self, key: str) -> pd.DataFrame | pd.Series:
        """Concatenate one semantic table across all chunks."""
        tables = []
        for file_id, chunk in self.files.items():
            if key not in chunk.tables:
                raise ValueError(f"No table found for '{key}' in file No. {file_id}")
            table = chunk.tables[key]
            tables.append(table)
        return pd.concat(tables, ignore_index=True)

    def compile_with_offsets(self, key: str, index_key: str, offsets: Dict[int, int]
                             ) -> pd.DataFrame | pd.Series:
        """Concatenate one semantic table with per-file index offsets.

        Args:
            key: Table name to concatenate.
            index_key: Column containing indices local to each input file.
            offsets: Offset for each input file identifier.

        Returns:
            Concatenated table with normalised indices.
        """
        tables = []
        for file_id, chunk in self.files.items():
            if key not in chunk.tables:
                raise ValueError(f"No table found for '{key}' in file No. {file_id}")
            table = chunk.tables[key].copy()
            table[index_key] += offsets[file_id]
            tables.append(table)
        return pd.concat(tables, ignore_index=True)

@dataclass
class CompileFiles(BaseScript):
    """Implements ``cbclib_cli compile``.

    Validates and concatenates per-chunk pipeline artifacts into one HDF5 file.
    Optimisation traces retain their chunk identity, compatible configurations
    are preserved at the root, and source provenance is written to ``extra``.

    Attributes:
        kind: ``'streaks'``, ``'regions'``, ``'xtals'``, ``'solutions'``, or
            ``'reflections'`` — selects which output directory to read from.
        scan: One scan or scan collection compiled as one artifact.
        in_suffix: Suffix selecting the per-chunk input directory.
        out_suffix: Suffix appended to the compiled output filename.
    """
    kind            : FileKind
    scan            : Scan | ScanList
    in_suffix       : str = str()
    out_suffix      : str = str()

    schemas : ClassVar[Dict[FileKind, CompileSchema]] = {
        'streaks': CompileSchema(required_tables=('data', 'metadata')),
        'regions': CompileSchema(required_tables=('data', 'metadata')),
        'xtals': CompileSchema(required_tables=('data',), preserve_config=True),
        'solutions': CompileSchema(required_tables=('data', 'miller', 'stats'),
                                   optional_tables=('candidates',),
                                   preserve_config=True),
        'reflections': CompileSchema(required_tables=('data', 'stats'),
                                     preserve_config=True),
    }

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('kind', type=str, choices=FileKind.__args__,
                             help='Type of chunked files to compile')
        initial.add_argument('scan_num', type=str,
                             help='Scan number, range, or underscore-separated scan list')
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
    def from_file(cls, kind: FileKind, scan_num: ScanNumbers, scan_file: str, in_suffix: str,
                  out_suffix: str) -> 'CompileFiles':
        config = ScanConfig.read(scan_file)
        return cls(kind, config.open_scan(scan_num), in_suffix, out_suffix)

    @property
    def input_dir(self) -> str:
        """Return the directory where per-chunk detection files are read from."""
        if self.kind == 'streaks':
            return self.scan.config.detect.streaks_dir
        if self.kind == 'regions':
            return self.scan.config.detect.regions_dir
        if self.kind == 'xtals':
            return self.scan.config.setup.xtals_dir
        if self.kind == 'solutions':
            return self.scan.config.setup.solutions_dir
        if self.kind == 'reflections':
            return self.scan.config.setup.reflections_dir
        raise ValueError(f"Invalid file kind: {self.kind}")

    @property
    def scan_text(self) -> str:
        """Return a human-readable description of the scan(s) being compiled."""
        if isinstance(self.scan, ScanList):
            return f"scans {self.scan.scan_string}"
        return f"a scan {self.scan.scan_string}"

    @property
    def file_key(self) -> str:
        if isinstance(self.scan, ScanList):
            return 'scan_num'
        return 'chunk_id'

    def load_file(self, path: str, schema: CompileSchema) -> FileContent:
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
        extra: Dict[str, str] = {}
        with h5py.File(path, mode='r') as input_file:
            if 'extra' in input_file:
                group = input_file['extra']
                if not isinstance(group, h5py.Group):
                    raise ValueError(f"Expected 'extra' to be an HDF5 group in file: {path}")

                def read_extra(name: str, item: h5py.Group | h5py.Dataset):
                    if isinstance(item, h5py.Dataset):
                        extra[name] = str(item.asstr()[()])

                group.visititems(read_extra)
        return FileContent(tables, extra)

    def load_config(self, path: str) -> Any:
        """Load the root configuration from one chunk artifact."""
        with h5py.File(path, mode='r') as file:
            if 'config' not in file.attrs:
                raise ValueError(f"Missing 'config' attribute in file: {path}")
            try:
                return json.loads(str(file.attrs['config']))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid configuration in file: {path}") from error

    def list_files(self) -> Iterator[Tuple[int, str]]:
        """Yield the absolute paths of all per-chunk files to compile."""
        if isinstance(self.scan, ScanList):
            for scan in self.scan:
                scan_file = scan.files.scan_file(None, suffix=self.in_suffix,
                                                 dir=self.input_dir)
                yield (scan.scan_num, scan_file)
        else:
            scan_dir = self.scan.files.scan_subdir(self.input_dir, self.in_suffix)
            for match in self.scan.files.list_files(scan_dir):
                if match.chunk_id is not None:
                    yield (match.chunk_id, match.filename)

    def run(self):
        schema = self.schemas[self.kind]

        print(f'Compiling {self.kind} for {self.scan_text}...')

        paths = list(self.list_files())
        print(f'Found {len(paths)} {self.kind} files for {self.scan_text}')
        if not paths:
            raise ValueError(f"No {self.kind} files found for {self.scan_text}")

        files = CompiledFiles({file_id: self.load_file(path, self.schemas[self.kind])
                               for file_id, path in self.list_files()})
        frame_offsets = None
        if isinstance(self.scan, ScanList):
            run_indices = self.scan.run().indices()
            frame_offsets = dict(zip(self.scan.scan_num, run_indices.offsets))

        output_path = self.scan.files.scan_file(suffix=self.out_suffix, dir=self.input_dir)
        print(f'Writing the detected {self.kind} to the file: {output_path}')

        mode = 'w'
        for key in schema.tables:
            if key in schema.optional_tables and not key in files:
                continue
            if key == 'data' and frame_offsets is not None:
                df = files.compile_with_offsets(key, 'index', frame_offsets)
                df.to_hdf(output_path, key=key, mode=mode)
            else:
                files.compile(key).to_hdf(output_path, key=key, mode=mode)
            mode = 'a'

        if files.has_extra:
            with h5py.File(output_path, mode='a') as output_file:
                for file_id, file in files.items():
                    for name, value in file.extra.items():
                        output_file[f'extra/{self.file_key}_{file_id:d}/{name}'] = value

        if schema.preserve_config:
            config = self.load_config(paths[0][1])
            with h5py.File(output_path, mode='a') as output_file:
                output_file.attrs['config'] = json.dumps(config)


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
    scan        : Scan
    params      : MetadataParameters

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('scan_num', type=str,
                             help='Scan number, range, or underscore-separated scan list')
        initial.add_argument('scan', type=str,
                             help='Path to a scan parameters JSON file')
        initial.add_argument('parameters', type=str,
                             help='Path to a metadata parameters JSON file')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Calculate CBD metadata needed for streak detection"

    @classmethod
    def from_file(cls, scan_num: int, scan_file: str, params_file: str) -> 'CreateMetadata':
        scan = ScanConfig.read(scan_file).open_scan(scan_num)
        params = MetadataParameters.read(params_file)
        return cls(scan, params)

    def run(self):
        print("Configuring the script...")
        xp = self.scan.config.system.array_api()
        rng = default_rng(xp=xp)

        run = self.scan.run()

        print(f"Looking for scan {self.scan.scan_string} data...")
        indices = run.indices()
        frames = rng.choice(len(indices), self.scan.config.metadata.n_frames, replace=False)
        indices = indices[frames]

        print(f"Loading {self.scan.config.metadata.n_frames:d} frames...")
        images = run.data(indices, geometry=self.scan.config.apply_geometry, xp=xp)

        print(f"Generating metadata for scan {self.scan.scan_string}...")
        with self.scan.config.system.cpu_config():
            metadata = create_metadata(images, self.params)

        output_file = os.path.join(self.scan.config.metadata.output_dir,
                                   self.scan.files.scan_file())
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
    scan        : Scan
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
        initial.add_argument('scan_num', type=str,
                             help='Scan number, range, or underscore-separated scan list')
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
    def from_file(cls, scan_num: int, scan_file: str, params_file: str,
                  chunk_id: int | None, n_chunks: int | None, n_out: int | None = None
                  ) -> 'CreateMetaList':
        scan = ScanConfig.read(scan_file).open_scan(scan_num)
        params = MetadataParameters.read(params_file)
        return cls(scan, params, chunk_id, n_chunks, n_out)

    def run(self):
        print("Configuring the script...")
        xp = self.scan.config.system.array_api()

        print(f"Looking for scan {self.scan.scan_string} data...")
        run = self.scan.run()
        chunk = self.get_chunk(run.indices(), self.chunk_id, self.n_chunks)
        print(f"Starting to process {len(chunk):d} frames...")

        offsets = (xp.arange(0, self.scan.config.metalist.n_frames)
                   - self.scan.config.metalist.n_frames // 2)
        spacing = min(self.scan.config.metalist.spacing, len(chunk) - len(offsets))

        n_out = self.n_out or (len(chunk) - len(offsets)) // spacing
        centers = xp.linspace(-int(offsets[0]), len(chunk) - int(offsets[-1]) - 1,
                              n_out, dtype=int)
        batches = xp.asarray(centers[:, None] + offsets, dtype=int)
        print(f"Creating a metadata list of {batches.shape[0]:d} points...")

        output_path = self.scan.files.scan_file(
            self.chunk_id, dir=self.scan.config.metalist.output_dir, make_dirs=True)
        print(f"The results will be saved to {output_path}")

        handler = H5Handler(CrystMetadata.default_protocol())
        mask, var = xp.ones(1, dtype=bool), xp.zeros(1)

        whitefields = []
        with self.scan.config.system.cpu_config():
            for index, batch in tqdm(enumerate(batches), total=batches.shape[0],
                                     desc='Generating the list'):
                images = run.data(chunk[batch], geometry=self.scan.config.apply_geometry,
                                  verbose=False, xp=xp)
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
    scan        : Scan
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
        initial.add_argument('scan_num', type=str,
                             help='Scan number, range, or underscore-separated scan list')
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
    def from_file(cls, scan_num: int, kind: DetectionKind, scan_file: str, params_file: str,
                  chunk_id: int | None, n_chunks: int | None, frames_only: bool,
                  out_suffix: str) -> 'DetectHits':
        scan = ScanConfig.read(scan_file).open_scan(scan_num)
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
    def meta_dataframe(run: BaseRun[int, TrainIndices, RunConfig], hits: TrainIndices
                       ) -> pd.DataFrame:
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
        xp = self.scan.config.system.array_api()

        print(f"Looking for scan {self.scan.scan_string} data...")
        run = self.scan.run()
        chunk = self.get_chunk(run.indices(), self.chunk_id, self.n_chunks)
        frames = xp.asarray(list(chunk.index()))

        print(f"Starting to process {len(chunk):d} frames...")

        metadata_path = self.scan.find_metadata(self.chunk_id)
        print(f"Using the metadata saved at {metadata_path}")

        loader = run.worker(self.scan.config.apply_geometry)
        detector = None if self.scan.config.apply_geometry else self.scan.config.data.geometry()
        with self.scan.config.system.cpu_config():
            streaks = pool_detection(loader, chunk, metadata_path, self.params,
                                     self.scan.config.system.platform, detector)

        indices, counts = xp.unique_counts(streaks.index)
        hit_indices = indices[counts > self.scan.config.detect.hit_threshold]
        hits = streaks.take(hit_indices)
        print(f"{hit_indices.size:d} hits were found.")

        if len(hits) > 0:
            if self.kind == 'streaks':
                output_dir = self.scan.config.detect.streaks_dir
            else:
                output_dir = self.scan.config.detect.regions_dir

            if self.frames_only:
                print("Frames only requested, skipping saving the full hits data.")
                output_path = self.scan.files.scan_file(
                    self.chunk_id, extension='.csv', suffix=self.out_suffix,
                    dir=output_dir, make_dirs=True)
                dataframe = pd.DataFrame({'frame': asnumpy(frames[hit_indices])})
                dataframe.to_csv(output_path, index=False)
                print(f"The results were saved to {output_path}")
            else:
                print("Preparing the file...")
                df = hits.to_dataframe(frames)
                metadata = self.meta_dataframe(run, chunk[hit_indices])

                output_path = self.scan.files.scan_file(
                    self.chunk_id, suffix=self.out_suffix, dir=output_dir, make_dirs=True)
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
    scan        : Scan
    params      : IndexingConfig
    xtals       : str
    hits_dir    : HitsDir
    in_suffix   : str
    out_suffix  : str
    chunk_id    : int | None

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('scan_num', type=str,
                             help='Scan number, range, or underscore-separated scan list')
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
    def from_file(cls, scan_num: int, scan_file: str, params_file: str, xtals: str,
                  hits_dir: HitsDir, in_suffix: str, out_suffix: str,
                  chunk_id: int | None) -> 'IndexingScript':
        scan = ScanConfig.read(scan_file).open_scan(scan_num)
        params = IndexingConfig.read(params_file)
        return cls(scan, params, xtals, hits_dir, in_suffix, out_suffix, chunk_id)

    @property
    def hits_directory(self) -> str:
        """Return the configured detection directory selected by :attr:`hits_dir`."""
        if self.hits_dir == 'streaks':
            return self.scan.config.detect.streaks_dir
        if self.hits_dir == 'regions':
            return self.scan.config.detect.regions_dir
        raise ValueError(f"Invalid hits_dir: {self.hits_dir}")

    @property
    def xtal_file(self) -> str:
        """Return the path to the crystal orientations file to read."""
        if self.xtals:
            return self.xtals
        return self.scan.config.setup.unit_file

    def run(self):
        print("Configuring the script...")
        xp = self.scan.config.system.array_api()

        geometry = self.scan.config.data.geometry()
        hits_file = self.scan.files.scan_file(self.chunk_id, suffix=self.in_suffix,
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
            xtals = self.scan.config.setup.xtal(xp=xp)
        geometry = self.scan.config.setup.geometry()

        print(f"Indexing {len(patterns):d} patterns...")
        with self.scan.config.system.cpu_config():
            result = pool_indexing(patterns.to_points(), xtals, geometry, self.params,
                                   self.scan.config.system.platform, xp)

        output_path = self.scan.files.scan_file(
            self.chunk_id, dir=self.scan.config.setup.xtals_dir,
            suffix=self.out_suffix, make_dirs=True)

        print(f"Saving the results to {output_path}...")
        extra = {'xtal_file': self.xtal_file, 'hits_file': hits_file,
                 'setup_file': self.scan.config.setup.setup_file}
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
    scan        : Scan
    params      : RefineConfig
    hits_dir    : HitsDir
    xtal_dir    : XtalDir
    in_suffix   : str
    out_suffix  : str
    chunk_id    : int | None

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('scan_num', type=str,
                             help='Scan number, range, or underscore-separated scan list')
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
    def from_file(cls, scan_num: int, scan_file: str, params_file: str, hits_dir: HitsDir,
                  xtal_dir: XtalDir, in_suffix: str, out_suffix: str,
                  chunk_id: int | None) -> 'RefineScript':
        scan = ScanConfig.read(scan_file).open_scan(scan_num)
        params = RefineConfig.read(params_file)
        return cls(scan, params, hits_dir, xtal_dir, in_suffix, out_suffix, chunk_id)

    @property
    def hits_directory(self) -> str:
        """Return the configured detection directory selected by :attr:`hits_dir`."""
        if self.hits_dir == 'streaks':
            return self.scan.config.detect.streaks_dir
        if self.hits_dir == 'regions':
            return self.scan.config.detect.regions_dir
        raise ValueError(f"Invalid hits_dir: {self.hits_dir}")

    @property
    def setup_file(self) -> str:
        """Return the path to the setup file used for indexing and refinement."""
        if self.xtal_dir == 'xtals':
            return self.scan.config.setup.setup_file
        if self.xtal_dir == 'solutions':
            return self.scan.files.scan_file(
                self.chunk_id, dir=self.scan.config.setup.solutions_dir,
                suffix=self.in_suffix)
        raise ValueError(f"Invalid xtal_dir: {self.xtal_dir}")

    @property
    def xtal_directory(self) -> str:
        """Return the configured orientation directory selected by :attr:`xtal_dir`."""
        if self.xtal_dir == 'xtals':
            return self.scan.config.setup.xtals_dir
        if self.xtal_dir == 'solutions':
            return self.scan.config.setup.solutions_dir
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
        xp = self.scan.config.system.jax_api()

        geometry = self.scan.config.data.geometry()
        in_file = self.scan.files.scan_file(self.chunk_id, dir=self.xtal_directory,
                                            suffix=self.in_suffix)
        if not os.path.isfile(in_file):
            print(f"No indexed crystal orientations file found at {in_file}")
            return

        print(f"Loading indexed crystal orientations from {in_file}...")
        df = pd.read_hdf(in_file, 'data')

        hits_file = self.scan.files.scan_file(self.chunk_id, dir=self.hits_directory)
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

        with self.scan.config.system.cpu_config():
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

        output_path = self.scan.files.scan_file(
            self.chunk_id, dir=self.scan.config.setup.solutions_dir,
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
    scan        : Scan
    params      : PostRefineConfig
    in_suffix   : str
    out_suffix  : str
    chunk_id    : int | None

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('scan_num', type=str,
                             help='Scan number, range, or underscore-separated scan list')
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
    def from_file(cls, scan_num: int, scan_file: str, params_file: str, in_suffix: str,
                  out_suffix: str, chunk_id: int | None) -> 'PostRefineScript':
        scan = ScanConfig.read(scan_file).open_scan(scan_num)
        params = PostRefineConfig.read(params_file)
        return cls(scan, params, in_suffix, out_suffix, chunk_id)

    def run(self):
        print("Configuring the script...")
        xp = self.scan.config.system.jax_api()

        geometry = self.scan.config.data.geometry()
        in_file = self.scan.files.scan_file(
            self.chunk_id, dir=self.scan.config.setup.solutions_dir,
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

            output_path = self.scan.files.scan_file(
                self.chunk_id, dir=self.scan.config.setup.solutions_dir,
                suffix=self.out_suffix, make_dirs=True)

            print(f"Saving the refined crystal orientations to {output_path}...")
            self.params.save(output_path, mode='w', extra=extra)
            post_stats.to_dataframe().to_hdf(output_path, key='stats', mode='a')
            result.to_dataframe().to_hdf(output_path, key='data', mode='a')
            miller.to_dataframe(frames).to_hdf(output_path, key='miller', mode='a')
        else:
            print(f"Skipping post-refinement for {frames.size:d} patterns...")

        reflections = context.to_list(optimised, resolved)

        output_path = self.scan.files.scan_file(
            self.chunk_id, dir=self.scan.config.setup.reflections_dir,
            suffix=self.out_suffix, make_dirs=True)
        print(f"Saving the refined reflections to {output_path}...")

        self.params.save(output_path, mode='w', extra=extra)
        reflections.to_dataframe(frames).to_hdf(output_path, key='data', mode='a')
        stats.to_dataframe().to_hdf(output_path, key='stats', mode='a')

class SBatchScripts:
    """Factory for scalar jobs and scan-indexed ``sbatch`` arrays.

    Each classmethod assembles a ``cbclib_cli <subcommand> …`` shell command,
    reads SLURM parameters from *script_file*, and returns a scalar script for
    one scan or an array script carrying multiple scan numbers.
    """
    main        : ClassVar[str] = 'cbclib_cli'

    @classmethod
    def compile(cls, kind: FileKind, scan_num: ScanNumbers, scan_file: str,
                script_file: str, in_suffix: str | None=None,
                out_suffix: str | None=None) -> SLURMScript:
        """Build a ``cbclib_cli compile`` script.

        Args:
            kind: ``'streaks'``, ``'reflections'``, ``'regions'``, ``'solutions'``,
                or ``'xtals'``.
            scan_num: One scan number or a selection compiled into one artifact.
            scan_file: Path to the scan configuration JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            in_suffix: Suffix of the per-chunk detection files to read.
            out_suffix: Suffix of the merged detection file to write.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the compile step.
        """
        command = (f"{cls.main} compile {quote(kind)} {{scan_num}}"
                   f" {quote(scan_file)}")
        if in_suffix is not None:
            command += f" --in-suffix {quote(in_suffix)}"
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        script_spec = ScriptSpec.read(script_file)

        scan = ScanArgument(scan_num)
        return SLURMScript(scan.slurm_command(command), scan.job_name("compile"), script_spec)

    @overload
    @classmethod
    def index(cls, scan_num: int, scan_file: str, params_file: str, script_file: str,
              xtals: str | None=None, hits_dir: HitsDir='streaks', in_suffix: str | None=None,
              out_suffix: str | None=None, chunk_id: int | None=None) -> SLURMScript: ...

    @overload
    @classmethod
    def index(cls, scan_num: range | List[int], scan_file: str, params_file: str, script_file: str,
              xtals: str | None=None, hits_dir: HitsDir='streaks', in_suffix: str | None=None,
              out_suffix: str | None=None, chunk_id: int | None=None) -> SLURMArrayScript: ...

    @classmethod
    def index(cls, scan_num: ScanNumbers, scan_file: str, params_file: str, script_file: str,
              xtals: str | None=None, hits_dir: HitsDir='streaks', in_suffix: str | None=None,
              out_suffix: str | None=None, chunk_id: int | None=None) -> SLURMScript:
        """Build a ``cbclib_cli index`` script.

        Args:
            scan_num: One scan number or scan numbers used as SLURM array task IDs.
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
        command = (f"{cls.main} index {{scan_num}} {quote(scan_file)} "
                   f"{quote(params_file)} --hits-dir {quote(hits_dir)} ")
        if xtals is not None:
            command += f"--xtals {quote(xtals)} "
        if in_suffix is not None:
            command += f"--in-suffix {quote(in_suffix)} "
        if out_suffix is not None:
            command += f"--out-suffix {quote(out_suffix)} "
        if chunk_id is not None:
            command += f' --chunk_id {chunk_id:d}'
        script_spec = ScriptSpec.read(script_file)

        scan = ScanArgument(scan_num)
        if len(scan) == 1:
            return SLURMScript(scan.slurm_command(command), scan.job_name("index"),
                               script_spec)
        return scan.slurm_array_script(command, "index", script_spec)

    @overload
    @classmethod
    def metadata(cls, scan_num: int, scan_file: str, params_file: str, script_file: str,
                 frames: List[int] | None=None) -> SLURMScript: ...

    @overload
    @classmethod
    def metadata(cls, scan_num: range | List[int], scan_file: str, params_file: str,
                 script_file: str, frames: List[int] | None=None) -> SLURMArrayScript: ...

    @classmethod
    def metadata(cls, scan_num: ScanNumbers, scan_file: str, params_file: str,
                 script_file: str, frames: List[int] | None=None) -> SLURMScript:
        """Build a ``cbclib_cli metadata`` script.

        Args:
            scan_num: One scan number or scan numbers used as SLURM array task IDs.
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the metadata parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            frames: Optional explicit list of frame indices to use.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the metadata step.
        """
        command = (f"{cls.main} metadata {{scan_num}} {quote(scan_file)}"
                   f" {quote(params_file)}")
        if frames is not None:
            command += f' --frames {frames}'
        script_spec = ScriptSpec.read(script_file)

        scan = ScanArgument(scan_num)
        if len(scan) == 1:
            return SLURMScript(scan.slurm_command(command), scan.job_name("metadata"),
                               script_spec)
        return scan.slurm_array_script(command, "metadata", script_spec)

    @overload
    @classmethod
    def metalist(cls, scan_num: int, scan_file: str, params_file: str, script_file: str,
                 chunk_id: int | None=None, n_chunks: int | None=None,
                 n_out: int | None=None) -> SLURMScript: ...

    @overload
    @classmethod
    def metalist(cls, scan_num: range | List[int], scan_file: str, params_file: str,
                 script_file: str, chunk_id: int | None=None, n_chunks: int | None=None,
                 n_out: int | None=None) -> SLURMArrayScript: ...

    @classmethod
    def metalist(cls, scan_num: ScanNumbers, scan_file: str, params_file: str, script_file: str,
                 chunk_id: int | None=None, n_chunks: int | None=None,
                 n_out: int | None=None) -> SLURMScript:
        """Build a ``cbclib_cli metalist`` script.

        Args:
            scan_num: One scan number or scan numbers used as SLURM array task IDs.
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the metadata parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            chunk_id: Optional chunk index.
            n_chunks: Optional total chunk count.
            n_out: Optional number of background estimates to compute.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the metalist step.
        """
        command = (f"{cls.main} metalist {{scan_num}} {quote(scan_file)}"
                   f" {quote(params_file)}")
        if chunk_id is not None and n_chunks is not None:
            command += f' --chunk_id {chunk_id:d}'
            command += f' --n_chunks {n_chunks:d}'
        if n_out is not None:
            command += f' --n_out {n_out:d}'
        script_spec = ScriptSpec.read(script_file)

        scan = ScanArgument(scan_num)
        if len(scan) == 1:
            return SLURMScript(scan.slurm_command(command),scan.job_name("metalist"),
                               script_spec)
        return scan.slurm_array_script(command, "metalist", script_spec)

    @overload
    @classmethod
    def detect(cls, scan_num: int, kind: DetectionKind, scan_file: str,
               params_file: str, script_file: str, chunk_id: int | None=None,
               n_chunks: int | None=None, frames_only: bool=False,
               out_suffix: str | None=None) -> SLURMScript: ...

    @overload
    @classmethod
    def detect(cls, scan_num: range | List[int], kind: DetectionKind, scan_file: str,
               params_file: str, script_file: str, chunk_id: int | None=None,
               n_chunks: int | None=None, frames_only: bool=False,
               out_suffix: str | None=None) -> SLURMArrayScript: ...

    @classmethod
    def detect(cls, scan_num: ScanNumbers, kind: DetectionKind, scan_file: str,
               params_file: str, script_file: str, chunk_id: int | None=None,
               n_chunks: int | None=None, frames_only: bool=False,
               out_suffix: str | None=None) -> SLURMScript:
        """Build a ``cbclib_cli detect`` script.

        Args:
            scan_num: One scan number or scan numbers used as SLURM array task IDs.
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
        command = (f"{cls.main} detect {{scan_num}} {quote(kind)} "
                   f"{quote(scan_file)} {quote(params_file)}")
        if chunk_id is not None and n_chunks is not None:
            command += f' --chunk_id {chunk_id:d}'
            command += f' --n_chunks {n_chunks:d}'
        if frames_only:
            command += ' --frames-only'
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        script_spec = ScriptSpec.read(script_file)
        job_name = f"detect_{kind}"

        scan = ScanArgument(scan_num)
        if len(scan) == 1:
            return SLURMScript(scan.slurm_command(command), scan.job_name(job_name),
                               script_spec)
        return scan.slurm_array_script(command, job_name, script_spec)

    @overload
    @classmethod
    def refine(cls, scan_num: int, scan_file: str, params_file: str, script_file: str,
               hits_dir: HitsDir | None=None, xtal_dir: XtalDir | None=None,
               in_suffix: str | None=None, out_suffix: str | None=None,
               chunk_id: int | None=None) -> SLURMScript: ...

    @overload
    @classmethod
    def refine(cls, scan_num: range | List[int], scan_file: str, params_file: str,
               script_file: str, hits_dir: HitsDir | None=None, xtal_dir: XtalDir | None=None,
               in_suffix: str | None=None, out_suffix: str | None=None,
               chunk_id: int | None=None) -> SLURMArrayScript: ...

    @classmethod
    def refine(cls, scan_num: ScanNumbers, scan_file: str, params_file: str, script_file: str,
               hits_dir: HitsDir | None=None, xtal_dir: XtalDir | None=None,
               in_suffix: str | None=None, out_suffix: str | None=None,
               chunk_id: int | None=None) -> SLURMScript:
        """Build a ``cbclib_cli refine`` script.

        Args:
            scan_num: One scan number or scan numbers used as SLURM array task IDs.
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
        command = (f"{cls.main} refine {{scan_num}} {quote(scan_file)}"
                   f" {quote(params_file)}")
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

        scan = ScanArgument(scan_num)
        if len(scan) == 1:
            return SLURMScript(scan.slurm_command(command), scan.job_name("refine"),
                               script_spec)
        return scan.slurm_array_script(command, "refine", script_spec)

    @overload
    @classmethod
    def post_refine(cls, scan_num: int, scan_file: str, params_file: str, script_file: str,
                    in_suffix: str | None=None, out_suffix: str | None=None,
                    chunk_id: int | None=None) -> SLURMScript: ...

    @overload
    @classmethod
    def post_refine(cls, scan_num: range | List[int], scan_file: str, params_file: str,
                    script_file: str, in_suffix: str | None=None, out_suffix: str | None=None,
                    chunk_id: int | None=None) -> SLURMArrayScript: ...

    @classmethod
    def post_refine(cls, scan_num: ScanNumbers, scan_file: str, params_file: str,
                    script_file: str, in_suffix: str | None=None, out_suffix: str | None=None,
                    chunk_id: int | None=None) -> SLURMScript:
        """Build a ``cbclib_cli post_refine`` script.

        Args:
            scan_num: One scan number or scan numbers used as SLURM array task IDs.
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the post-refinement parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            in_suffix: Suffix of the refinement files to read.
            out_suffix: Suffix of the post-refinement files to write.
            chunk_id: Optional chunk index.

        Returns:
            :class:`~cbclib_v2.slurm.SLURMScript` for the post-refinement step.
        """
        command = (f"{cls.main} post_refine {{scan_num}} {quote(scan_file)}"
                   f" {quote(params_file)}")
        if in_suffix is not None:
            command += f" --in-suffix {quote(in_suffix)}"
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        if chunk_id is not None:
            command += f' --chunk_id {chunk_id:d}'
        script_spec = ScriptSpec.read(script_file)

        scan = ScanArgument(scan_num)
        if len(scan) == 1:
            return SLURMScript(scan.slurm_command(command), scan.job_name("post_refine"),
                               script_spec)
        return scan.slurm_array_script(command, "post_refine", script_spec)

class SBatchArrayScripts:
    """Factory for ``sbatch --array`` job scripts.

    Array tasks may represent scan numbers for independent compilation or
    chunk indices for parallel processing. Use with
    :meth:`~cbclib_v2.slurm.SLURMJobManager.submit_array`.
    """
    main        : ClassVar[str] = 'cbclib_cli'

    @classmethod
    def compile(cls, kind: FileKind, scan_num: range | List[int], scan_file: str,
                script_file: str, in_suffix: str | None=None,
                out_suffix: str | None=None) -> SLURMArrayScript:
        """Build a scan-indexed array that compiles each scan independently.

        Args:
            kind: Type of chunked files to compile.
            scan_num: Scan numbers used as SLURM array task IDs.
            scan_file: Path to the scan configuration JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            in_suffix: Suffix of the per-chunk files to read.
            out_suffix: Suffix of each compiled scan file to write.

        Returns:
            A scan-indexed array script with one compilation task per scan.
        """
        command = (f"{cls.main} compile {quote(kind)} {{scan_num}}"
                   f" {quote(scan_file)}")
        if in_suffix is not None:
            command += f" --in-suffix {quote(in_suffix)}"
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        script_spec = ScriptSpec.read(script_file)

        scans = ScanArgument(scan_num)
        return scans.slurm_array_script(command, "compile_array", script_spec)

    @overload
    @classmethod
    def index(cls, scan_num: int, n_tasks: int, scan_file: str, params_file: str,
              script_file: str, xtals: str | None=None, hits_dir: HitsDir='streaks',
              in_suffix: str | None=None, out_suffix: str | None=None
              ) -> SLURMArrayScript: ...

    @overload
    @classmethod
    def index(cls, scan_num: range | List[int], n_tasks: int, scan_file: str, params_file: str,
              script_file: str, xtals: str | None=None, hits_dir: HitsDir='streaks',
              in_suffix: str | None=None, out_suffix: str | None=None
              ) -> List[SLURMArrayScript]: ...

    @classmethod
    def index(cls, scan_num: ScanNumbers, n_tasks: int, scan_file: str, params_file: str,
              script_file: str, xtals: str | None=None, hits_dir: HitsDir='streaks',
              in_suffix: str | None=None, out_suffix: str | None=None
              ) -> SLURMArrayScript | List[SLURMArrayScript]:
        """Build a ``cbclib_cli index`` array script for *n_tasks* tasks.

        Args:
            scan_num: One scan number, or scan numbers used to build separate chunk arrays.
            n_tasks: Total number of array tasks.
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the indexing parameters JSON.
            xtals: Path to an initial crystal orientations HDF5 file.
            hits_dir: Logical directory with detected streaks to read from:
                ``'streaks'`` or ``'regions'``.
            in_suffix: Suffix of the detection files to read.
            out_suffix: Suffix of the indexing files to write.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.

        Returns:
            One chunk-array script for a scalar scan number, otherwise one script per scan.
        """
        command = (f"{cls.main} index {{scan_num}} {quote(scan_file)} {quote(params_file)}"
                   f" --hits-dir {quote(hits_dir)}"
                   " --chunk_id ${{FILE_INDEX}}")
        if xtals is not None:
            command += f" --xtals {quote(xtals)}"
        if in_suffix is not None:
            command += f" --in-suffix {quote(in_suffix)}"
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        job_name = "index_array"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')

        scans = ScanArgument(scan_num)
        if len(scans) == 1:
            return SLURMArrayScript(scans.slurm_command(command), scans.job_name(job_name),
                                    script_spec, range(n_tasks))

        return [SLURMArrayScript(scan.slurm_command(command), scan.job_name(job_name),
                                 script_spec, range(n_tasks)) for scan in scans]

    @overload
    @classmethod
    def metalist(cls, scan_num: int, n_tasks: int, scan_file: str, params_file: str,
                 script_file: str) -> SLURMArrayScript: ...

    @overload
    @classmethod
    def metalist(cls, scan_num: range | List[int], n_tasks: int, scan_file: str, params_file: str,
                 script_file: str) -> List[SLURMArrayScript]: ...

    @classmethod
    def metalist(cls, scan_num: ScanNumbers, n_tasks: int, scan_file: str, params_file: str,
                 script_file: str) -> SLURMArrayScript | List[SLURMArrayScript]:
        """Build a ``cbclib_cli metalist`` array script for *n_chunks* tasks.

        Args:
            scan_num: One scan number, or scan numbers used to build separate chunk arrays.
            n_tasks: Total number of array tasks.
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the metadata parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.

        Returns:
            One chunk-array script for a scalar scan number, otherwise one script per scan.
        """
        command = (f"{cls.main} metalist {{scan_num}} {quote(scan_file)} {quote(params_file)}"
                   f" --n_chunks {n_tasks:d}"
                   " --chunk_id ${{FILE_INDEX}}")
        job_name = "metalist_array"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')

        scans = ScanArgument(scan_num)
        if len(scans) == 1:
            return SLURMArrayScript(scans.slurm_command(command), scans.job_name(job_name),
                                    script_spec, range(n_tasks))

        return [SLURMArrayScript(scan.slurm_command(command), scan.job_name(job_name),
                                 script_spec, range(n_tasks)) for scan in scans]

    @overload
    @classmethod
    def detect(cls, scan_num: int, n_tasks: int, kind: DetectionKind, scan_file: str,
               params_file: str, script_file: str, out_suffix: str | None=None
               ) -> SLURMArrayScript: ...

    @overload
    @classmethod
    def detect(cls, scan_num: range | List[int], n_tasks: int, kind: DetectionKind, scan_file: str,
               params_file: str, script_file: str, out_suffix: str | None=None
               ) -> List[SLURMArrayScript]: ...

    @classmethod
    def detect(cls, scan_num: ScanNumbers, n_tasks: int, kind: DetectionKind, scan_file: str,
               params_file: str, script_file: str, out_suffix: str | None=None
               ) -> SLURMArrayScript | List[SLURMArrayScript]:
        """Build a ``cbclib_cli detect`` array script for *n_tasks* tasks.

        Args:
            scan_num: One scan number, or scan numbers used to build separate chunk arrays.
            n_tasks: Total number of array tasks.
            kind: ``'streaks'`` or ``'regions'``.
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the detection parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            out_suffix: Suffix of the detection files to write.

        Returns:
            One chunk-array script for a scalar scan number, otherwise one script per scan.
        """
        command = (f"{cls.main} detect {{scan_num}} {quote(kind)} {quote(scan_file)}"
                   f" {quote(params_file)} --n_chunks {n_tasks:d}"
                   " --chunk_id ${{FILE_INDEX}}")
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        job_name = f"detect_{kind}_array"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')

        scans = ScanArgument(scan_num)
        if len(scans) == 1:
            return SLURMArrayScript(scans.slurm_command(command), scans.job_name(job_name),
                                    script_spec, range(n_tasks))
        return [SLURMArrayScript(scan.slurm_command(command), scan.job_name(job_name),
                                 script_spec, range(n_tasks)) for scan in scans]

    @overload
    @classmethod
    def refine(cls, scan_num: int, n_tasks: int, scan_file: str, params_file: str, script_file: str,
               hits_dir: HitsDir | None=None, xtal_dir: XtalDir | None=None,
               in_suffix: str | None=None, out_suffix: str | None=None
               ) -> SLURMArrayScript: ...

    @overload
    @classmethod
    def refine(cls, scan_num: range | List[int], n_tasks: int, scan_file: str, params_file: str,
               script_file: str, hits_dir: HitsDir | None=None, xtal_dir: XtalDir | None=None,
               in_suffix: str | None=None, out_suffix: str | None=None
               ) -> List[SLURMArrayScript]: ...

    @classmethod
    def refine(cls, scan_num: ScanNumbers, n_tasks: int, scan_file: str, params_file: str,
               script_file: str, hits_dir: HitsDir | None=None, xtal_dir: XtalDir | None=None,
               in_suffix: str | None=None, out_suffix: str | None=None
               ) -> SLURMArrayScript | List[SLURMArrayScript]:
        """Build a ``cbclib_cli refine`` array script for *n_tasks* tasks.

        Args:
            scan_num: One scan number, or scan numbers used to build separate chunk arrays.
            n_tasks: Total number of array tasks.
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the refinement parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            hits_dir: Logical directory with detected streaks to read from.
            xtal_dir: Logical crystal orientation directory to read from.
            in_suffix: Suffix of the orientation files to read.
            out_suffix: Suffix of the refinement files to write.

        Returns:
            One chunk-array script for a scalar scan number, otherwise one script per scan.
        """
        command = (f"{cls.main} refine {{scan_num}} {quote(scan_file)} {quote(params_file)}"
                   " --chunk_id ${{FILE_INDEX}}")
        if hits_dir is not None:
            command += f" --hits-dir {quote(hits_dir)}"
        if xtal_dir is not None:
            command += f" --xtal-dir {quote(xtal_dir)}"
        if in_suffix is not None:
            command += f" --in-suffix {quote(in_suffix)}"
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        job_name = "refine_array"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')

        scans = ScanArgument(scan_num)
        if len(scans) == 1:
            return SLURMArrayScript(scans.slurm_command(command), scans.job_name(job_name),
                                    script_spec, range(n_tasks))
        return [SLURMArrayScript(scan.slurm_command(command), scan.job_name(job_name),
                                 script_spec, range(n_tasks)) for scan in scans]

    @overload
    @classmethod
    def post_refine(cls, scan_num: int, n_tasks: int, scan_file: str, params_file: str,
                    script_file: str, in_suffix: str | None=None, out_suffix: str | None=None
                    ) -> SLURMArrayScript: ...

    @overload
    @classmethod
    def post_refine(cls, scan_num: range | List[int], n_tasks: int, scan_file: str,
                    params_file: str, script_file: str, in_suffix: str | None=None,
                    out_suffix: str | None=None) -> List[SLURMArrayScript]: ...

    @classmethod
    def post_refine(cls, scan_num: ScanNumbers, n_tasks: int, scan_file: str, params_file: str,
                    script_file: str, in_suffix: str | None=None, out_suffix: str | None=None
                    ) -> SLURMArrayScript | List[SLURMArrayScript]:
        """Build a ``cbclib_cli post_refine`` array script for *n_tasks* tasks.

        Args:
            scan_num: One scan number, or scan numbers used to build separate chunk arrays.
            n_tasks: Total number of array tasks.
            scan_file: Path to the scan configuration JSON.
            params_file: Path to the post-refinement parameters JSON.
            script_file: Path to the :class:`~cbclib_v2.slurm.ScriptSpec` JSON.
            in_suffix: Suffix of the refinement files to read.
            out_suffix: Suffix of the post-refinement files to write.

        Returns:
            One chunk-array script for a scalar scan number, otherwise one script per scan.
        """
        command = (f"{cls.main} post_refine {{scan_num}} {quote(scan_file)} {quote(params_file)}"
                   " --chunk_id ${{FILE_INDEX}}")
        if in_suffix is not None:
            command += f" --in-suffix {quote(in_suffix)}"
        if out_suffix is not None:
            command += f" --out-suffix {quote(out_suffix)}"
        job_name = "post_refine_array"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')

        scans = ScanArgument(scan_num)
        if len(scans) == 1:
            return SLURMArrayScript(scans.slurm_command(command), scans.job_name(job_name),
                                    script_spec, range(n_tasks))
        return [SLURMArrayScript(scan.slurm_command(command), scan.job_name(job_name),
                                 script_spec, range(n_tasks)) for scan in scans]

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

    config = ScanConfig.read(args['scan'])
    config.system.apply()

    try:
        scan_nums = ScanArgument.read_string(args['scan_num'])
    except ValueError as error:
        raise SystemExit(f"Invalid scan number: {error}") from error

    if args['command'] == 'compile':
        script = CompileFiles.from_file(
            args['kind'], scan_nums, args['scan'], args['in_suffix'], args['out_suffix'])
        script.run()
        return

    for scan_num in ScanArgument(scan_nums).enumerate():
        print(f"Run: {scan_num:d}")
        if args['command'] == 'index':
            print(f"JSON file with the indexing parameters: {args['parameters']}")
            script = IndexingScript.from_file(
                scan_num, args['scan'], args['parameters'], args['xtals'], args['hits_dir'],
                args['in_suffix'], args['out_suffix'], args['chunk_id'])
        elif args['command'] == 'metadata':
            print(f"JSON file with the metadata parameters: {args['parameters']}")
            script = CreateMetadata.from_file(scan_num, args['scan'], args['parameters'])
        elif args['command'] == 'metalist':
            print(f"JSON file with the metadata parameters: {args['parameters']}")
            script = CreateMetaList.from_file(
                scan_num, args['scan'], args['parameters'], args['chunk_id'],
                args['n_chunks'], args['n_out'])
        elif args['command'] == 'detect':
            print(f"JSON file with the streak finding parameters: {args['parameters']}")
            script = DetectHits.from_file(
                scan_num, args['kind'], args['scan'], args['parameters'], args['chunk_id'],
                args['n_chunks'], args['frames_only'], args['out_suffix'])
        elif args['command'] == 'refine':
            print(f"JSON file with the refinement parameters: {args['parameters']}")
            script = RefineScript.from_file(
                scan_num, args['scan'], args['parameters'], args['hits_dir'],
                args['xtal_dir'], args['in_suffix'], args['out_suffix'], args['chunk_id'])
        elif args['command'] == 'post_refine':
            print(f"JSON file with the post-refinement parameters: {args['parameters']}")
            script = PostRefineScript.from_file(
                scan_num, args['scan'], args['parameters'], args['in_suffix'],
                args['out_suffix'], args['chunk_id'])
        else:
            raise ValueError(f"Invalid command: {args['command']}")
        script.run()
