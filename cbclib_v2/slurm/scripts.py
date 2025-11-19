from argparse import ArgumentParser
from dataclasses import InitVar, dataclass
from multiprocessing import cpu_count
import os
import re
from shlex import quote
from typing import ClassVar, List, Literal, Type, overload
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from .._src.data_processing import CrystMetadata
from .._src.data_container import DataContainer
from .._src.cxi_protocol import CXIIndices, CXIProtocol, CXIStore, write_hdf
from .._src.crystfel import Detector, read_crystfel
from .._src.scripts import BaseParameters, MetadataParameters, StreakFinderConfig
from .._src.scripts import create_metadata, pool_detection, pool_detection_stacked
from .._src.annotations import Array
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
    n_modules       : int = 1
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

        if self.n_modules == 1:
            pattern = self.file_pattern.format(scan_num=scan_num)
            for path in os.listdir(data_dir):
                if re.match(pattern, path):
                    files.append(os.path.join(data_dir, path))
            files = [sorted(files),]
        elif module_id is not None:
            if module_id < self.starts_at or module_id >= self.starts_at + self.n_modules:
                raise ValueError(f'module_id {module_id:d} is out of range '
                                 f'[{self.starts_at:d}, {self.starts_at + self.n_modules - 1:d}]')
            pattern = self.file_pattern.format(scan_num=scan_num, module_id=module_id)
            for path in os.listdir(data_dir):
                if re.match(pattern, path):
                    files.append(os.path.join(data_dir, path))
            files = [sorted(files),]
        else:
            for index in range(self.starts_at, self.starts_at + self.n_modules):
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
        if module_id < self.starts_at or module_id >= self.starts_at + self.n_modules:
            raise ValueError(f'module_id {module_id:d} is out of range '
                             f'[{self.starts_at:d}, {self.starts_at + self.n_modules - 1:d}]')
        return indices[:, module_id - self.starts_at]

    def pre_processor(self, kind: Literal['full', 'stacked']) -> CFPreProcessor | None:
        if kind == 'full':
            return CFPreProcessor(self.geometry())
        if kind == 'stacked':
            return None
        raise ValueError(f'Invalid pre_processor keyword: {kind}')

@dataclass
class StreakConfig(BaseParameters):
    hit_threshold   : int
    output_dir      : str

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
class ScanFileStructure(BaseParameters):
    folder_pattern      : str
    file_pattern        : str

    def folder(self, scan_num: int, kind: str) -> str:
        return self.folder_pattern.format(scan_num=scan_num, kind=kind)

    def filename(self, scan_num: int, kind: str, file_index: int | None=None,
                 module_id: int | None=None, prefix: str=str()) -> str:
        path = self.file_pattern.format(scan_num=scan_num, kind=kind)
        filename, extension = os.path.splitext(path)
        if file_index is not None:
            filename += f'_f{file_index:04d}'
        if module_id is not None:
            filename += f'_m{module_id:02d}'
        if prefix:
            filename += f'_{prefix}'
        return filename + extension

    def filename_pattern(self, scan_num: int, kind: str) -> str:
        path = self.file_pattern.format(scan_num=scan_num, kind=kind)
        filename, extension = os.path.splitext(path)
        pattern = filename
        pattern += r'(_f[0-9]{4})?'
        pattern += r'(_m[0-9]{2})?'
        pattern += re.escape(extension)
        return pattern

FILE_STRUCTURE = ScanFileStructure('scan_{scan_num:d}_{kind}', 'scan_{scan_num:d}_{kind}.h5')

@dataclass
class ScanConfig(BaseParameters):
    detector        : DetectorConfig
    streaks         : StreakConfig
    metadata        : MetadataConfig
    metalist        : MetaListConfig
    data_dir        : str
    image_kind      : Literal['full', 'stacked']
    protocol_file   : str
    scan_num        : int

    structure       : ClassVar[ScanFileStructure] = FILE_STRUCTURE

    def __post_init__(self):
        if self.streaks.output_dir == self.metadata.output_dir:
            raise ValueError("Streaks and metadata must be saved to different folders")
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
            file_indices = indices[idxs].reshape((-1, self.detector.n_modules))
        else:
            file_indices = indices

        if module_id is not None:
            return self.detector.module_indices(file_indices, module_id)
        return file_indices

    def find_metadata(self, file_index: int | None=None) -> str:
        metalist_path = self.metalist_path(file_index)
        if os.path.isfile(metalist_path):
            return metalist_path

        metadata_path = self.metadata_path()
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

    def metadata_path(self) -> str:
        filename = self.structure.filename(self.scan_num, self.image_kind)
        return os.path.join(self.metadata.output_dir, filename)

    def metalist_path(self, file_index: int | None=None) -> str:
        if file_index is None:
            filename = self.structure.filename(self.scan_num, self.image_kind)
            return os.path.join(self.metalist.output_dir, filename)

        folder = os.path.join(self.metalist.output_dir,
                              self.structure.folder(self.scan_num, self.image_kind))
        filename = self.structure.filename(self.scan_num, self.image_kind, file_index)
        return os.path.join(folder, filename)

    def streaks_path(self, file_index: int | None=None, module_id: int | None=None
                     ) -> str:
        if file_index is None:
            filename = self.structure.filename(self.scan_num, self.image_kind, file_index,
                                               module_id)
            return os.path.join(self.streaks.output_dir, filename)

        folder = os.path.join(self.streaks.output_dir,
                              self.structure.folder(self.scan_num, self.image_kind))
        filename = self.structure.filename(self.scan_num, self.image_kind, file_index,
                                           module_id)
        return os.path.join(folder, filename)

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
    scan_file   : str

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
        initial.add_argument('scan', type=str, help='Path to a scan parameters JSON file')
        return initial

    @classmethod
    def parser_description(cls) -> str:
        return "Compile detected streaks into a single table"

    @classmethod
    def from_file(cls, scan_file: str) -> 'CompileStreaks':
        return cls(scan_file)

    def run(self):
        params = ScanConfig.read(self.scan_file)
        print(f'Assembling streaks for a scan {params.scan_num:d}...')

        # Use regex pattern from filename_pattern method instead of glob
        pattern = params.structure.filename_pattern(params.scan_num, params.image_kind)
        streak_dir = os.path.join(params.streaks.output_dir,
                              params.structure.folder(params.scan_num, params.image_kind))
        streak_files = [filename for filename in os.listdir(streak_dir)
                        if re.match(pattern, filename)]
        streak_files = sorted(streak_files)

        print(f'Found {len(streak_files)} streak files for run {params.scan_num:d}')
        filename = params.structure.filename(params.scan_num, params.image_kind)
        output_file = os.path.join(params.streaks.output_dir, filename)
        print(f'Writing the detected streaks to the file: {output_file}')

        tables = []
        for streak_file in streak_files:
            path = os.path.join(streak_dir, streak_file)
            tables.append(pd.read_hdf(path, 'data'))
        pd.concat(tables).to_hdf(output_file, key='data')

@dataclass
class DetectStreaks(BaseScript):
    scan        : ScanConfig
    params      : StreakFinderConfig
    file_index  : int | None
    module_id   : int | None
    n_frames    : int | None

    def __post_init__(self):
        if self.file_index is not None:
            self.file_index = compute_index(self.file_index, self.scan.num_files())
        if self.module_id is not None:
            self.module_id = compute_index(self.module_id, self.scan.detector.n_modules)
        if self.n_frames is not None and self.n_frames < 0:
            raise ValueError("n_frames must be a non-negative integer")

    @classmethod
    def parser(cls, initial: ArgumentParser=ArgumentParser()) -> ArgumentParser:
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
    def from_file(cls, scan_file: str, params_file: str, file_index: int | None,
                  module_id: int | None, n_frames: int | None) -> 'DetectStreaks':
        scan = ScanConfig.read(scan_file)
        params = StreakFinderConfig.read(params_file)
        return cls(scan, params, file_index, module_id, n_frames)

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
            streaks = pool_detection(scan_file, metadata_path, self.params,
                                    (indices, ss_idxs, fs_idxs), self.scan.pre_processor())
        elif self.scan.image_kind == 'stacked':
            streaks = pool_detection_stacked(scan_file, metadata_path, self.params,
                                            (indices, ss_idxs, fs_idxs), detector)
        else:
            raise ValueError(f"Invalid image_kind keyword: {self.scan.image_kind}")

        frames, counts = np.unique_counts(streaks.index)
        hit_frames = frames[counts > self.scan.streaks.hit_threshold]
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

            output_path = self.scan.streaks_path(self.file_index, self.module_id)
            dir_path = os.path.dirname(output_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            print(f"The results will be saved to {output_path}")

            df.to_hdf(output_path, key='data')

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

        output_file = self.scan.metadata_path()
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
    def from_file(cls, scan_file: str, params_file: str,
                  file_index: int | None) -> 'CreateMetaList':
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

        output_path = self.scan.metalist_path(self.file_index)
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

class SBatchScripts:
    main        : ClassVar[str] = 'cbclib_cli'

    @classmethod
    def compile(cls, scan_file: str, script_file: str) -> SLURMScript:
        command = f"{cls.main} compile {quote(scan_file)}"
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="compile", command=command, parameters=script_spec)

    @classmethod
    def streaks(cls,  scan_file: str, params_file: str, script_file: str,
                file_index: int | None=None, module_id: int | None=None,
                n_frames: int | None=None) -> SLURMScript:
        command = f"{cls.main} streaks {quote(scan_file)} {quote(params_file)}"
        if file_index is not None:
            command += f' --file_index {file_index:d}'
        if module_id is not None:
            command += f' --module_index {module_id:d}'
        if n_frames is not None:
            command += f' --n_frames {n_frames:d}'
        script_spec = ScriptSpec.read(script_file)
        return SLURMScript(job_name="streaks", command=command, parameters=script_spec)

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

class SBatchArrayScripts:
    main        : ClassVar[str] = 'cbclib_cli'

    @classmethod
    def streaks(cls, scan_file: str, params_file: str, script_file: str,
                module_id: int | None=None, n_frames: int | None=None) -> SLURMScript:
        command = f"{cls.main} streaks {quote(scan_file)} {quote(params_file)}" \
                   " --file_index ${FILE_INDEX}"
        if module_id is not None:
            command += f' --module_index {module_id:d}'
        if n_frames is not None:
            command += f' --n_frames {n_frames:d}'
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')
        return SLURMScript(job_name="streaks_array", command=command, parameters=script_spec)

    @classmethod
    def metalist(cls, scan_file: str, params_file: str, script_file: str) -> SLURMScript:
        command = f"{cls.main} metalist {quote(scan_file)} {quote(params_file)}" \
                   " --file_index ${FILE_INDEX}"
        script_spec = ScriptSpec.read(script_file)
        script_spec.add_define('FILE_INDEX', '${SLURM_ARRAY_TASK_ID}')
        return SLURMScript(job_name="metalist_array", command=command, parameters=script_spec)

class Scripts:
    sbatch          : ClassVar[Type[SBatchScripts]] = SBatchScripts
    sbatch_array    : ClassVar[Type[SBatchArrayScripts]] = SBatchArrayScripts
    compile         : ClassVar[Type[CompileStreaks]] = CompileStreaks
    streaks         : ClassVar[Type[DetectStreaks]] = DetectStreaks
    metadata        : ClassVar[Type[CreateMetadata]] = CreateMetadata
    metalist        : ClassVar[Type[CreateMetaList]] = CreateMetaList

    @classmethod
    def parser(cls) -> ArgumentParser:
        parser = ArgumentParser(description='Process CBD patterns')
        subparsers = parser.add_subparsers(help='Available subcommands', dest='command')

        for key, attr in cls.__dict__.items():
            if isinstance(attr, type) and issubclass(attr, BaseScript):
                subparser = subparsers.add_parser(key, help=attr.parser_description())
                attr.parser(subparser)
        return parser

def main():
    parser = Scripts.parser()

    args = vars(parser.parse_args())

    print(f"JSON file with the scan parameters: {args['scan']}")

    scan = ScanConfig.read(args['scan'])
    print(f"Run: {scan.scan_num:d}")

    if args['command'] == 'streaks':
        print(f"JSON file with the streak finding parameters: {args['parameters']}")
        script = DetectStreaks.from_file(args['scan'], args['parameters'],
                                         args.get('file_index'), args.get('module_index'),
                                         args.get('n_frames'))
        script.run()
    elif args['command'] == 'metadata':
        print(f"JSON file with the metadata parameters: {args['parameters']}")
        script = CreateMetadata.from_file(args['scan'], args['parameters'],
                                          args.get('frames'))
        script.run()
    elif args['command'] == 'metalist':
        print(f"JSON file with the metadata parameters: {args['parameters']}")
        script = CreateMetaList.from_file(args['scan'], args['parameters'],
                                         args.get('file_index'))
        script.run()
    elif args['command'] == 'compile':
        script = CompileStreaks.from_file(args['scan'])
        script.run()
    else:
        raise ValueError(f"Invalid command: {args['command']}")
