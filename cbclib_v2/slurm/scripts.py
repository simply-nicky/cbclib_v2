from argparse import ArgumentParser
from dataclasses import InitVar, dataclass
from multiprocessing import cpu_count
import os
import re
from shlex import quote
from typing import ClassVar, Iterator, List, Literal, Type, overload
import h5py
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from .._src.annotations import Array, ArrayNamespace, NumPy
from .._src.data_container import DataContainer
from .._src.data_processing import CrystMetadata
from .._src.crystfel import Detector, read_crystfel
from .._src.cxi_protocol import CXIIndices, CXIProtocol, CXIStore, write_hdf
from .._src.scripts import (BaseParameters, DetectionKind, FinderConfig, IndexingConfig,
                            MetadataParameters, RegionFinderConfig, StreakFinderConfig)
from .._src.scripts import create_metadata, pool_detection, pool_indexing
from .._src.streaks import StackedStreaks, Streaks
from ..indexer import FixedSetup, XtalCell, XtalList, XtalState
from ..test_util import compute_index
from .slurm_manager import SLURMScript, ScriptSpec

@dataclass
class CFPreProcessor(DataContainer):
    params  : InitVar[Detector]

    def __post_init__(self, detector: Detector):
        super().__post_init__()
        self.indices = detector.indices()

    def __call__(self, frames: Array) -> Array:
        return self.indices(frames)

@dataclass
class DetectorConfig(BaseParameters):
    file_pattern    : str
    geometry_file   : str
    gain_file       : str = str()
    num_modules     : int = 1
    starts_at       : int = 0
    pedestal_file   : str = str()

    def geometry(self) -> Detector:
        return read_crystfel(self.geometry_file)

    @overload
    def data_files(self, data_dir: str, scan_num: int | None=None, file_index: None=None,
                   module_id: int | None=None) -> List[List[str]]: ...

    @overload
    def data_files(self, data_dir: str, scan_num: int | None, file_index: int,
                   module_id: int | None) -> List[str]: ...

    def data_files(self, data_dir: str, scan_num: int | None=None,
                   file_index: int | None=None, module_id: int | None=None
                   ) -> List[str] | List[List[str]]:
        files = []

        if self.num_modules == 1:
            pattern = self.file_pattern.format(scan_num=scan_num)
            for path in os.listdir(data_dir):
                if re.match(pattern, path):
                    files.append(os.path.join(data_dir, path))
            files = [sorted(files),]
        elif module_id is not None:
            if module_id < self.starts_at or module_id >= self.starts_at + self.num_modules:
                raise ValueError(f'module_id {module_id:d} is out of range '
                                 f'[{self.starts_at:d}, {self.starts_at + self.num_modules - 1:d}]')
            pattern = self.file_pattern.format(scan_num=scan_num, module_id=module_id)
            for path in os.listdir(data_dir):
                if re.match(pattern, path):
                    files.append(os.path.join(data_dir, path))
            files = [sorted(files),]
        else:
            for index in range(self.starts_at, self.starts_at + self.num_modules):
                module_files = []
                pattern = self.file_pattern.format(scan_num=scan_num, module_id=index)
                for path in os.listdir(data_dir):
                    if re.match(pattern, path):
                        module_files.append(os.path.join(data_dir, path))
                files.append(sorted(module_files))

        if file_index is None:
            return files
        return [module_files[file_index] for module_files in files
                if len(module_files) > file_index]

    def module_indices(self, indices: CXIIndices, module_id: int) -> CXIIndices:
        if module_id < self.starts_at or module_id >= self.starts_at + self.num_modules:
            raise ValueError(f'module_id {module_id:d} is out of range '
                             f'[{self.starts_at:d}, {self.starts_at + self.num_modules - 1:d}]')
        return indices[:, module_id - self.starts_at]

    def pre_processor(self, kind: Literal['full', 'stacked']) -> CFPreProcessor | None:
        if kind == 'full':
            return CFPreProcessor(self.geometry())
        if kind == 'stacked':
            return None
        raise ValueError(f'Invalid pre_processor keyword: {kind}')

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

    def unit_cell(self, xp: ArrayNamespace=NumPy) -> XtalCell:
        if self.unit_file == str():
            raise ValueError("No crystal file provided")
        return XtalCell.read(self.unit_file, xp)

    def xtal(self, xp: ArrayNamespace=NumPy) -> XtalState:
        return self.unit_cell(xp).to_basis()

    def setup(self) -> FixedSetup:
        if self.setup_file == str():
            raise ValueError("No setup file provided")
        return FixedSetup.read(self.setup_file)

@dataclass
class ScanFileStructure(BaseParameters):
    folder_pattern      : str
    file_pattern        : str

    def folder(self, scan_num: int, kind: str, suffix: str=str()) -> str:
        folder = self.folder_pattern.format(scan_num=scan_num, kind=kind)
        if suffix:
            folder += f'_{suffix}'
        return folder

    def filename(self, scan_num: int, kind: str, file_index: int | None=None,
                 module_id: int | None=None, suffix: str=str()) -> str:
        path = self.file_pattern.format(scan_num=scan_num, kind=kind)
        filename, extension = os.path.splitext(path)
        if file_index is not None:
            filename += f'_f{file_index:04d}'
        if module_id is not None:
            filename += f'_m{module_id:02d}'
        if suffix:
            filename += f'_{suffix}'
        return filename + extension

    def find_files(self, folder: str, scan_num: int, kind: str) -> Iterator[str]:
        path = self.file_pattern.format(scan_num=scan_num, kind=kind)
        filename, extension = os.path.splitext(path)
        pattern = filename
        pattern += r'(_f[0-9]{4})?'
        pattern += r'(_m[0-9]{2})?'
        pattern += r'(_([^.]+))?'
        pattern += re.escape(extension)

        for filename in os.listdir(folder):
            if re.match(pattern, filename):
                yield os.path.join(folder, filename)

    def find_folders(self, folder: str, scan_num: int, kind: str) -> Iterator[str]:
        pattern = self.folder_pattern.format(scan_num=scan_num, kind=kind)
        pattern += r'(_([^.]+))?'

        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if os.path.isdir(path) and re.match(pattern, filename):
                yield path

FILE_STRUCTURE = ScanFileStructure('scan_{scan_num:d}_{kind}', 'scan_{scan_num:d}_{kind}.h5')

@dataclass
class ScanConfig(BaseParameters):
    detector        : DetectorConfig
    setup           : SetupConfig
    detect          : DetectConfig
    metadata        : MetadataConfig
    metalist        : MetaListConfig
    data_dir        : str
    image_kind      : Literal['full', 'stacked']
    protocol_file   : str
    scan_num        : int

    structure       : ClassVar[ScanFileStructure] = FILE_STRUCTURE

    def __post_init__(self):
        if self.detect.streaks_dir == self.metadata.output_dir:
            raise ValueError("Streaks and metadata must be saved to different folders")
        if self.detect.regions_dir == self.metadata.output_dir:
            raise ValueError("Regions and metadata must be saved to different folders")
        if self.metadata.output_dir == self.metalist.output_dir:
            raise ValueError("Metadata and metalist must be saved to different folders")

    @overload
    def data_files(self, file_index: int, module_id: int | None=None) -> List[str]: ...

    @overload
    def data_files(self, file_index: None=None, module_id: int | None=None
                   ) -> List[List[str]]: ...

    def data_files(self, file_index: int | None=None, module_id: int | None=None
                   ) -> List[str] | List[List[str]]:
        data_dir = self.data_dir.format(scan_num=self.scan_num)
        return self.detector.data_files(data_dir, self.scan_num, file_index, module_id)

    def file_indices(self, indices: CXIIndices, file_index: int | None=None,
                     module_id: int | None=None) -> CXIIndices:
        if file_index is not None:
            data_files = np.asarray(self.data_files(file_index)).ravel()
            mask = np.asarray(np.any(indices.files[..., 0, None] == data_files, axis=-1))

            idxs = np.where(mask)
            file_indices = indices[idxs].reshape((-1, self.detector.num_modules))
        else:
            file_indices = indices

        if module_id is not None:
            return self.detector.module_indices(file_indices, module_id)
        return file_indices

    def find_files(self, folder: str) -> List[str]:
        return sorted(self.structure.find_files(folder, self.scan_num, self.image_kind))

    def find_folders(self, folder: str) -> List[str]:
        return sorted(self.structure.find_folders(folder, self.scan_num, self.image_kind))

    def find_metadata(self, file_index: int | None=None) -> str:
        metalist_path = self.new_file(self.metalist.output_dir, file_index)
        if os.path.isfile(metalist_path):
            return metalist_path

        metadata_path = os.path.join(self.metadata.output_dir, self.new_filename(file_index))
        if os.path.isfile(metadata_path):
            return metadata_path

        err_txt = f"No metadata exists for the scan {self.scan_num}"
        if file_index is not None:
            err_txt += f" and file No. {file_index}"
        raise ValueError(err_txt)

    def num_files(self) -> int:
        data_files = self.data_files()
        for files in data_files:
            if isinstance(files, list):
                return len(files)

            return len(data_files)
        return 0

    def scan_file(self) -> CXIStore:
        data_files = self.data_files()
        if len(data_files) == 0:
            raise ValueError(f'No files found in {self.data_dir} matching '
                             f'the pattern {self.detector.file_pattern}')

        return CXIStore(data_files, protocol=self.protocol())

    def new_filename(self, file_index: int | None=None, module_id: int | None=None,
                     suffix: str=str()) -> str:
        return self.structure.filename(self.scan_num, self.image_kind, file_index, module_id,
                                       suffix)

    def new_file(self, folder: str, file_index: int | None=None, module_id: int | None=None,
                 suffix: str=str()) -> str:
        if file_index is None:
            filename = self.new_filename(None, module_id, suffix)
            return os.path.join(folder, filename)

        return os.path.join(self.new_folder(folder, suffix),
                            self.new_filename(file_index, module_id))

    def new_folder(self, folder: str, suffix: str=str()) -> str:
        return os.path.join(folder, self.structure.folder(self.scan_num, self.image_kind, suffix))

    def pre_processor(self) -> CFPreProcessor | None:
        return self.detector.pre_processor(self.image_kind)

    def protocol(self) -> CXIProtocol:
        return CXIProtocol.read(self.protocol_file)

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

        scan_dir = os.path.join(hits_dir,
                                scan.structure.folder(scan.scan_num, scan.image_kind))
        scan_files = scan.find_files(scan_dir)

        print(f'Found {len(scan_files)} streak files for run {scan.scan_num:d}')
        output_path = os.path.join(hits_dir, scan.new_filename())
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
    frames      : List[int] | None

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('scan', type=str,
                             help='Path to a scan parameters JSON file')
        initial.add_argument('parameters', type=str,
                             help='Path to a metadata parameters JSON file')
        initial.add_argument('--frames', '-f', type=int, nargs='*',
                             help='A list of frames used to generate CBD metadata')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Calculate CBD metadata needed for streak detection"

    @classmethod
    def from_file(cls, scan_file: str, params_file: str,
                  frames: List[int] | None) -> 'CreateMetadata':
        scan = ScanConfig.read(scan_file)
        params = MetadataParameters.read(params_file)
        return cls(scan, params, frames)

    def run(self):
        print("Configuring the script...")
        pre_processor = self.scan.pre_processor()
        scan_file = self.scan.scan_file()

        if self.params.num_threads == 0:
            self.params.num_threads = cpu_count()

        print("Looking for data...")
        indices = scan_file.indices('data')
        if self.frames is None:
            frames = np.random.choice(len(indices), self.scan.metadata.n_frames,
                                      replace=False)
            indices = indices[frames]
        else:
            indices = indices[self.frames]

        print(f"Loading {self.scan.metadata.n_frames:d} frames...")
        images = scan_file.load('data', idxs=indices)
        if pre_processor is not None:
            images = pre_processor(images)

        print("Generating metadata...")
        metadata = create_metadata(images, self.params)

        output_file = os.path.join(self.scan.metadata.output_dir, self.scan.new_filename())
        print(f"Saving to {output_file}")
        write_hdf(metadata, CXIStore(output_file, metadata.protocol))

@dataclass
class CreateMetaList(BaseScript):
    scan        : ScanConfig
    params      : MetadataParameters
    file_index  : int | None

    def __post_init__(self):
        if self.file_index is not None:
            self.file_index = compute_index(self.file_index, self.scan.num_files())

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('scan', type=str,
                             help='Path to a scan parameters JSON file')
        initial.add_argument('parameters', type=str,
                             help='Path to a metadata parameters JSON file')
        initial.add_argument('--file_index', '-f', type=int, help='Index of the file to process')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Calculate a list of CBD metadata"

    @classmethod
    def from_file(cls, scan_file: str, params_file: str, file_index: int | None
                  ) -> 'CreateMetaList':
        scan = ScanConfig.read(scan_file)
        params = MetadataParameters.read(params_file)
        return cls(scan, params, file_index)

    def run(self):
        print("Configuring the script...")
        if self.params.num_threads == 0:
            self.params.num_threads = cpu_count()

        pre_processor = self.scan.pre_processor()
        scan_file = self.scan.scan_file()

        if self.file_index is not None:
            print(f"Processing a file No. {self.file_index}")
            indices = scan_file.indices('data')
            indices = self.scan.file_indices(indices, self.file_index)
        else:
            print("Processing the whole scan")
            indices = scan_file.indices('data')
        print(f"Starting to process {len(indices):d} frames...")

        offsets = np.arange(0, self.scan.metalist.n_frames) - self.scan.metalist.n_frames // 2
        spacing = min(self.scan.metalist.spacing, len(indices) - len(offsets))
        frames = np.arange(-int(offsets[0]), len(indices) - int(offsets[-1]),
                           spacing)[:, None] + offsets
        print(f"Creating a metadata list of {frames.shape[0]:d} points...")

        output_path = self.scan.new_file(self.scan.metalist.output_dir, self.file_index)
        dir_path = os.path.dirname(output_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        print(f"The results will be saved to {output_path}")

        output_file = CXIStore(output_path, CrystMetadata.default_protocol())
        mask, var = np.ones(1, dtype=bool), np.zeros(1)

        whitefields = []
        for index, fidxs in tqdm(enumerate(frames), total=frames.shape[0],
                                 desc='Generating the list'):
            images = scan_file.load('data', indices[fidxs], verbose=False)
            if pre_processor is not None:
                images = pre_processor(images)
            metadata = create_metadata(images, self.params)
            mask = mask & metadata.mask
            var = var + metadata.std**2
            whitefields.append(metadata.whitefield)
            with output_file.file('a') as cxi_file:
                output_file.save('data_frames', fidxs, cxi_file, mode='insert', idxs=index)
                output_file.save('whitefield', metadata.whitefield, cxi_file, mode='insert',
                                 idxs=index)

        print("Performing PC analysis...")
        metadata = CrystMetadata(data_frames=frames, whitefield=np.stack(whitefields, axis=0),
                                 mask=mask, std=np.sqrt(var / frames.shape[0]))
        metadata = metadata.pca()

        print("Saving the rest...")
        write_hdf(metadata, output_file, 'eigen_field', 'eigen_value', 'flatfield', 'mask',
                  'std', file_mode='a')

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
              script_file: str, file_index: int | None=None, module_id: int | None=None
              ) -> SLURMScript:
        command = f"{cls.main} index {quote(scan_file)} {quote(params_file)} " \
                  f"{quote(xtals)} {quote(suffix)}"
        if file_index is not None:
            command += f' --file_index {file_index:d}'
        if module_id is not None:
            command += f' --module_index {module_id:d}'
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
                 file_index: int | None=None) -> SLURMScript:
        command = f"{cls.main} metalist {quote(scan_file)} {quote(params_file)}"
        if file_index is not None:
            command += f' --file_index {file_index:d}'
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="metalist", command=command, parameters=script_spec)

    @classmethod
    def detect(cls, kind: DetectionKind, scan_file: str, params_file: str, script_file: str,
               file_index: int | None=None, module_id: int | None=None,
               n_frames: int | None=None) -> SLURMScript:
        command = f"{cls.main} detect {quote(kind)} {quote(scan_file)} {quote(params_file)}"
        if file_index is not None:
            command += f' --file_index {file_index:d}'
        if module_id is not None:
            command += f' --module_index {module_id:d}'
        if n_frames is not None:
            command += f' --n_frames {n_frames:d}'
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
    def metalist(cls, scan_file: str, params_file: str, script_file: str) -> SLURMScript:
        command = f"{cls.main} metalist {quote(scan_file)} {quote(params_file)}" \
                   " --file_index ${FILE_INDEX}"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')
        return SLURMScript(job_name="metalist_array", command=command, parameters=script_spec)

    @classmethod
    def detect(cls, kind: DetectionKind, scan_file: str, params_file: str, script_file: str,
               module_id: int | None=None, n_frames: int | None=None) -> SLURMScript:
        command = f"{cls.main} detect {quote(kind)} {quote(scan_file)} {quote(params_file)}" \
                   " --file_index ${FILE_INDEX}"
        if module_id is not None:
            command += f' --module_index {module_id:d}'
        if n_frames is not None:
            command += f' --n_frames {n_frames:d}'
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
    detect          : ClassVar[Type[DetectStreaks]] = DetectStreaks

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

    scan = ScanConfig.read(args['scan'])
    print(f"Run: {scan.scan_num:d}")

    if args['command'] == 'compile':
        script = CompileStreaks.from_file(args['kind'], args['scan'])
        script.run()
    elif args['command'] == 'index':
        print(f"JSON file with the indexing parameters: {args['parameters']}")
        script = IndexingScript.from_file(args['scan'], args['parameters'],
                                          args['xtals'], args['suffix'],
                                          args['file_index'], args['module_index'])
        script.run()
    elif args['command'] == 'metadata':
        print(f"JSON file with the metadata parameters: {args['parameters']}")
        script = CreateMetadata.from_file(args['scan'], args['parameters'],
                                          args['frames'])
        script.run()
    elif args['command'] == 'metalist':
        print(f"JSON file with the metadata parameters: {args['parameters']}")
        script = CreateMetaList.from_file(args['scan'], args['parameters'],
                                          args['file_index'])
        script.run()
    elif args['command'] == 'detect':
        print(f"JSON file with the streak finding parameters: {args['parameters']}")
        script = DetectStreaks.from_file(args['kind'], args['scan'], args['parameters'],
                                         args['file_index'], args['module_index'],
                                         args['n_frames'])
        script.run()
    else:
        raise ValueError(f"Invalid command: {args['command']}")
