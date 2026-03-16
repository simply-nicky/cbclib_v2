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
    platform        : Platform
    num_threads     : int

    def __post_init__(self):
        if self.num_threads <= 0:
            self.num_threads = cpu_count()
        if self.platform not in ['cpu', 'gpu']:
            raise ValueError(f"Invalid platform: {self.platform}")

    def cpu_config(self) -> CPUConfig:
        return CPUConfig(num_threads=self.num_threads)

    def array_api(self) -> AnyNamespace:
        return default_api(self.platform)

@dataclass
class DetectConfig(BaseParameters):
    hit_threshold   : int
    streaks_dir     : str
    regions_dir     : str

@dataclass
class MetadataConfig(BaseParameters):
    n_frames        : int
    output_dir      : str

@dataclass
class MetaListConfig(BaseParameters):
    n_frames        : int
    spacing         : int
    output_dir      : str

@dataclass
class SetupConfig(BaseParameters):
    setup_file      : str
    unit_file       : str
    xtals_dir       : str

    def unit_cell(self, xp: AnyNamespace=NumPy) -> XtalCell:
        if self.unit_file == str():
            raise ValueError("No crystal file provided")
        return XtalCell.read(self.unit_file, xp)

    def xtal(self, xp: AnyNamespace=NumPy) -> XtalState:
        return self.unit_cell(xp).to_basis()

    def setup(self) -> FixedSetup:
        if self.setup_file == str():
            raise ValueError("No setup file provided")
        return FixedSetup.read(self.setup_file)

@dataclass
class ScanFiles(BaseParameters):
    scan_num            : int
    image_kind          : Literal['full', 'stacked']
    dir_pattern         : ClassVar[str] = 'scan_{scan_num:d}_{kind}'
    file_pattern        : ClassVar[str] = 'scan_{scan_num:d}_{kind}'

    def list_files(self, dir: str) -> Iterator[str]:
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
        pattern = self.dir_pattern.format(scan_num=self.scan_num, kind=self.image_kind)
        pattern += r'(_([^.]+))?'

        for filename in os.listdir(dir):
            path = os.path.join(dir, filename)
            if os.path.isdir(path) and re.match(pattern, filename):
                yield path

    def scan_dir(self, suffix: str=str()) -> str:
        dir = self.dir_pattern.format(scan_num=self.scan_num, kind=self.image_kind)
        if suffix:
            dir += f'_{suffix}'
        return dir

    def scan_subdir(self, dir: str, suffix: str=str()) -> str:
        return os.path.join(dir, self.scan_dir(suffix))

    def scan_file(self, file_index: int | None=None, /, *, suffix: str=str(), extension: str='.h5',
                  dir: str | None=None) -> str:
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
    data            : RunConfig
    setup           : SetupConfig
    detect          : DetectConfig
    metadata        : MetadataConfig
    metalist        : MetaListConfig
    system          : SystemConfig

    @property
    def apply_geometry(self) -> bool:
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
        return open_run(self.scan_num, self.data)

@dataclass
class BaseScript:
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
                pulse_ids = run.metadata('pulse_id', chunk[hits.index_array.unique()])
                df['pulse_id'] = pulse_ids[hits.index_array.reset()]

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
            raise ValueError(f"No streaks file found at {hits_file}")

        print(f"Loading detected streaks from {hits_file}...")
        dataframe = pd.read_hdf(hits_file, 'data')
        if 'module_id' in dataframe.columns:
            num_modules = geometry.num_modules
            streaks = StackedStreaks.import_dataframe(dataframe, num_modules=num_modules, xp=xp)
        else:
            streaks = Streaks.import_dataframe(dataframe, xp=xp)
        patterns = geometry.to_patterns(streaks)

        if self.xtals:
            print(f"Loading crystal orientations from {self.xtals}...")
            df = pd.read_hdf(self.xtals, 'data')
            xtals = XtalList.import_dataframe(df, xp=xp).to_xtals()
        else:
            print("No crystal orientations provided, using the unit cell information")
            xtals = self.scan.setup.xtal(xp=xp)
        setup = self.scan.setup.setup()

        print(f"Indexing {len(patterns):d} patterns...")
        indexed = pool_indexing(patterns, xtals, setup, self.params, self.scan.system.platform, xp)

        output_path = self.scan.scan_file(self.chunk_id, dir=self.scan.setup.xtals_dir,
                                          suffix=self.suffix)
        dir_path = os.path.dirname(output_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        print(f"Saving the results to {output_path}...")
        df = indexed.to_dataframe()
        df.to_hdf(output_path, key='data')
        with h5py.File(output_path, 'a') as output_file:
            if self.xtals:
                output_file['files/xtal_file'] = self.xtals
            else:
                output_file['files/xtal_file'] = self.scan.setup.unit_file
            output_file['files/hits_file'] = hits_file
            output_file['files/setup_file'] = self.scan.setup.setup_file

@dataclass
class DetectStreaks(BaseScript):
    scan        : ScanConfig
    params      : FinderConfig
    file_index  : int | None
    module_id   : int | None
    n_frames    : int | None

    def __post_init__(self):
        if self.file_index is not None:
            self.file_index = compute_index(self.file_index, self.scan.num_files())
        if self.module_id is not None:
            self.module_id = compute_index(self.module_id, self.scan.detector.num_modules)
        if self.n_frames is not None and self.n_frames < 0:
            raise ValueError("n_frames must be a non-negative integer")

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('kind', type=str, choices=['streaks', 'regions'],
                             help='Kind of detection to perform')
        initial.add_argument('scan', type=str,
                             help='Path to a scan parameters JSON file')
        initial.add_argument('parameters', type=str,
                             help='Path to a streak finder parameters JSON file')
        initial.add_argument('--file_index', '-f', type=int, help='Index of the file to process')
        initial.add_argument('--module_index', '-m', type=int,
                             help='Index of the detector module to process')
        initial.add_argument('--n_frames', '-n', type=int, help='Number of frames to process')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Detect streaks in CBD patterns"

    @classmethod
    def from_file(cls, kind: DetectionKind, scan_file: str, params_file: str,
                  file_index: int | None, module_id: int | None, n_frames: int | None
                  ) -> 'DetectStreaks':
        scan = ScanConfig.read(scan_file)
        if kind == 'streaks':
            params = StreakFinderConfig.read(params_file)
        elif kind == 'regions':
            params = RegionFinderConfig.read(params_file)
        else:
            raise ValueError(f"Invalid detection kind: {kind}")
        return cls(scan, params, file_index, module_id, n_frames)

    @property
    def kind(self) -> DetectionKind:
        if isinstance(self.params, StreakFinderConfig):
            return 'streaks'
        if isinstance(self.params, RegionFinderConfig):
            return 'regions'
        raise ValueError(f"Invalid parameters type for detection: {type(self.params)}")

    def run(self):
        if self.file_index is not None:
            print(f"Processing a file No. {self.file_index}")
        if self.params.num_threads == 0:
            self.params.num_threads = cpu_count()

        print("Configuring the script...")
        scan_file = self.scan.scan_file()
        detector = self.scan.detector.geometry()

        if self.module_id is None:
            print("Processing the whole frame")
            ss_idxs, fs_idxs = slice(None), slice(None)
        else:
            print(f"Processing the detector module {self.module_id:d}")
            region = detector.panel(self.module_id).region
            ss_idxs, fs_idxs = region.ss_slice, region.fs_slice

        if 'data' not in scan_file.attributes():
            raise ValueError("No data found in the files")

        print("Looking for data...")
        if self.file_index is not None:
            indices = scan_file.indices('data')
            indices = self.scan.file_indices(indices, self.file_index)
        elif self.n_frames is not None:
            indices = scan_file.indices('data')[:self.n_frames]
        else:
            indices = scan_file.indices('data')
        print(f"Starting to process {len(indices):d} frames...")

        metadata_path = self.scan.find_metadata(self.file_index)
        print(f"Using the metadata saved at {metadata_path}")

        if self.scan.image_kind == 'full':
            print("Using full-frame images for detection")
            pre_processor = self.scan.pre_processor()
        elif self.scan.image_kind == 'stacked':
            print("Using stacked images for detection")
            pre_processor = None
        else:
            raise ValueError(f"Invalid image_kind keyword: {self.scan.image_kind}")

        streaks = pool_detection(scan_file, metadata_path, self.params, self.kind,
                                 (indices, ss_idxs, fs_idxs), detector, pre_processor)

        frames, counts = np.unique_counts(streaks.index)
        hit_frames = frames[counts > self.scan.detect.hit_threshold]
        hits = streaks.take(hit_frames)
        print(f"{hit_frames.size:d} hits were found.")

        if len(hits) > 0:
            print("Preparing the file...")
            df = hits.to_dataframe()
            pid_indices = scan_file.indices('pulse_id')[hits.index_array.unique()]
            pid_indices = self.scan.file_indices(pid_indices,
                                                 module_id=self.scan.detector.starts_at)
            pulse_ids = scan_file.load('pulse_id', idxs=pid_indices)
            df['pulse_id'] = pulse_ids[hits.index_array.reset()]

            if self.kind == 'streaks':
                output_dir = self.scan.detect.streaks_dir
            else:
                output_dir = self.scan.detect.regions_dir

            output_path = self.scan.new_file(output_dir, self.file_index, self.module_id)
            dir_path = os.path.dirname(output_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            print(f"The results will be saved to {output_path}")

            df.to_hdf(output_path, key='data')

@dataclass
class IndexingScript(BaseScript):
    scan        : ScanConfig
    params      : IndexingConfig
    xtals       : str
    suffix      : str
    file_index  : int | None
    module_id   : int | None

    def __post_init__(self):
        if self.file_index is not None:
            self.file_index = compute_index(self.file_index, self.scan.num_files())
        if self.module_id is not None:
            self.module_id = compute_index(self.module_id, self.scan.detector.num_modules)

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('scan', type=str,
                             help='Path to a scan parameters JSON file')
        initial.add_argument('parameters', type=str,
                             help='Path to an indexing parameters JSON file')
        initial.add_argument('--xtals', '-x', type=str, default=str(),
                             help='Path to a crystal orientations HDF5 file')
        initial.add_argument('--suffix', '-s', type=str, default=str(),
                             help='Suffix of the folder where the indexing results are saved')
        initial.add_argument('--file_index', '-f', type=int, help='Index of the file to process')
        initial.add_argument('--module_index', '-m', type=int,
                             help='Index of the detector module to process')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Index detected streaks in CBD patterns"

    @classmethod
    def from_file(cls, scan_file: str, params_file: str, xtals: str, suffix: str,
                  file_index: int | None, module_id: int | None) -> 'IndexingScript':
        scan = ScanConfig.read(scan_file)
        params = IndexingConfig.read(params_file)
        return cls(scan, params, xtals, suffix, file_index, module_id)

    def run(self):
        print("Configuring the script...")
        if self.params.num_threads == 0:
            self.params.num_threads = cpu_count()

        detector = self.scan.detector.geometry()
        hits_file = self.scan.new_file(self.scan.detect.streaks_dir,
                                       self.file_index, self.module_id)
        if not os.path.isfile(hits_file):
            raise ValueError(f"No streaks file found at {hits_file}")

        print(f"Loading detected streaks from {hits_file}...")
        dataframe = pd.read_hdf(hits_file, 'data')
        if 'module_id' in dataframe.columns:
            num_modules = self.scan.detector.num_modules
            streaks = StackedStreaks.import_dataframe(dataframe, num_modules=num_modules)
        else:
            streaks = Streaks.import_dataframe(dataframe)
        patterns = detector.to_patterns(streaks)

        if self.xtals:
            print(f"Loading crystal orientations from {self.xtals}...")
            df = pd.read_hdf(self.xtals, 'data')
            xtals = XtalList.import_dataframe(df).to_xtals()
        else:
            print("No crystal orientations provided, using the unit cell information")
            xtals = self.scan.setup.xtal()
        setup = self.scan.setup.setup()

        print(f"Indexing {len(patterns):d} patterns...")
        indexed = pool_indexing(patterns, xtals, setup, self.params)

        output_path = self.scan.new_file(self.scan.setup.xtals_dir, self.file_index,
                                         self.module_id, self.suffix)
        dir_path = os.path.dirname(output_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        print(f"Saving the results to {output_path}...")
        df = indexed.to_dataframe()
        df.to_hdf(output_path, key='data')
        with h5py.File(output_path, 'a') as output_file:
            if self.xtals:
                output_file['files/xtal_file'] = self.xtals
            else:
                output_file['files/xtal_file'] = self.scan.setup.unit_file
            output_file['files/hits_file'] = hits_file
            output_file['files/setup_file'] = self.scan.setup.setup_file

class SBatchScripts:
    main        : ClassVar[str] = 'cbclib_cli'

    @classmethod
    def compile(cls, kind: DetectionKind, scan_file: str, script_file: str) -> SLURMScript:
        command = f"{cls.main} compile {quote(kind)} {quote(scan_file)}"
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="compile", command=command, parameters=script_spec)

    @classmethod
    def index(cls, scan_file: str, params_file: str, xtals: str, suffix: str,
              script_file: str, chunk_id: int | None=None, n_chunks: int | None=None
              ) -> SLURMScript:
        command = f"{cls.main} index {quote(scan_file)} {quote(params_file)} " \
                  f"{quote(xtals)} {quote(suffix)}"
        if chunk_id is not None and n_chunks is not None:
            command += f' --chunk_id {chunk_id:d}'
            command += f' --n_chunks {n_chunks:d}'
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="index", command=command, parameters=script_spec)

    @classmethod
    def metadata(cls, scan_file: str, params_file: str, script_file: str,
                 frames: List[int] | None=None) -> SLURMScript:
        command = f"{cls.main} metadata {quote(scan_file)} {quote(params_file)}"
        if frames is not None:
            command += f' --frames {frames}'
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="metadata", command=command, parameters=script_spec)

    @classmethod
    def metalist(cls, scan_file: str, params_file: str, script_file: str,
                 chunk_id: int | None=None, n_chunks: int | None=None) -> SLURMScript:
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
        command = f"{cls.main} detect {quote(kind)} {quote(scan_file)} {quote(params_file)}"
        if chunk_id is not None and n_chunks is not None:
            command += f' --chunk_id {chunk_id:d}'
            command += f' --n_chunks {n_chunks:d}'
        if frames_only:
            command += ' --frames-only'
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="detect", command=command, parameters=script_spec)

class SBatchArrayScripts:
    main        : ClassVar[str] = 'cbclib_cli'

    @classmethod
    def index(cls, scan_file: str, params_file: str, xtals: str, suffix: str,
              script_file: str, n_chunks: int) -> SLURMScript:
        command = f"{cls.main} index {quote(scan_file)} {quote(params_file)} {quote(xtals)} "\
                  f"{quote(suffix)} --chunk_id ${{FILE_INDEX}} --n_chunks {n_chunks:d}"
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="detect", command=command, parameters=script_spec)

class SBatchArrayScripts:
    main        : ClassVar[str] = 'cbclib_cli'

    @classmethod
    def index(cls, scan_file: str, params_file: str, xtals: str, suffix: str,
              script_file: str, module_id: int | None=None) -> SLURMScript:
        command = f"{cls.main} index {quote(scan_file)} {quote(params_file)} "\
                  f"{quote(xtals)} {quote(suffix)} --file_index ${{FILE_INDEX}}"
        if module_id is not None:
            command += f' --module_index {module_id:d}'
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')
        return SLURMScript(job_name="index_array", command=command, parameters=script_spec)

    @classmethod
    def metalist(cls, scan_file: str, params_file: str, script_file: str, n_chunks: int
                 ) -> SLURMScript:
        command = f"{cls.main} metalist {quote(scan_file)} {quote(params_file)}" \
                   f" --chunk_id ${{FILE_INDEX}} --n_chunks {n_chunks:d}"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')
        return SLURMScript(job_name="metalist_array", command=command, parameters=script_spec)

    @classmethod
    def detect(cls, kind: DetectionKind, scan_file: str, params_file: str, script_file: str,
               n_chunks: int) -> SLURMScript:
        command = f"{cls.main} detect {quote(kind)} {quote(scan_file)} {quote(params_file)}" \
                  f" --chunk_id ${{FILE_INDEX}} --n_chunks {n_chunks:d}"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')
        return SLURMScript(job_name="detect_array", command=command, parameters=script_spec)

class Scripts:
    sbatch          : ClassVar[Type[SBatchScripts]] = SBatchScripts
    sbatch_array    : ClassVar[Type[SBatchArrayScripts]] = SBatchArrayScripts
    compile         : ClassVar[Type[CompileStreaks]] = CompileStreaks
    index           : ClassVar[Type[IndexingScript]] = IndexingScript
    metadata        : ClassVar[Type[CreateMetadata]] = CreateMetadata
    metalist        : ClassVar[Type[CreateMetaList]] = CreateMetaList
    detect          : ClassVar[Type[DetectHits]] = DetectHits

    @classmethod
    def parser(cls) -> ArgumentParser:
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
