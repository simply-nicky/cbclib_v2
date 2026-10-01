from argparse import Namespace
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
from typing import Any, Literal, cast
import h5py
import pandas as pd
import pytest
from cbclib_v2 import RunConfig, Streaks, cuda, slurm
from cbclib_v2.annotations import NumPy, NumPyNamespace, RealArray
from cbclib_v2._src.cxi_protocol import StackIndex, StackIndices
from cbclib_v2.slurm.scripts import DetectHits
from cbclib_v2.scripts import (DetectConfig, IndexingConfig, LossParameters, MetaListConfig,
                               MetadataConfig, OptimiseParameters, RefineConfig,
                               RefineDataParameters, Scan, ScanConfig, ScheduleParameters,
                               SetupConfig, StreakFinderConfig, StructureParameters,
                               SystemConfig)

Event = str | tuple[object, ...]
HitsDir = Literal['streaks', 'regions']
XtalDir = Literal['xtals', 'solutions']

@dataclass
class DetectionRun:
    pulse_ids : RealArray

    def indices(self) -> StackIndices:
        return StackIndices([StackIndex('input.h5', 3)])

    def worker(self, geometry: bool) -> object:
        return object()

    def metadata(self, attr: str, keys: Any) -> RealArray:
        assert attr == 'pulse_id'
        assert len(keys) == self.pulse_ids.size
        return self.pulse_ids

class ScanFixtures:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def run_config(self) -> RunConfig:
        return RunConfig(facility='XFEL')

    @pytest.fixture
    def setup_config(self, tmp_path: Path) -> SetupConfig:
        return SetupConfig(setup_file=str(tmp_path / 'setup.json'),
                           unit_file=str(tmp_path / 'unit.json'),
                           reflections_dir=str(tmp_path / 'reflections'),
                           solutions_dir=str(tmp_path / 'solutions'),
                           xtals_dir=str(tmp_path / 'xtals'))

    @pytest.fixture
    def detect_config(self, tmp_path: Path) -> DetectConfig:
        return DetectConfig(hit_threshold=1, streaks_dir=str(tmp_path / 'streaks'),
                            regions_dir=str(tmp_path / 'regions'))

    @pytest.fixture
    def metadata_config(self, tmp_path: Path) -> MetadataConfig:
        return MetadataConfig(n_frames=2, output_dir=str(tmp_path / 'metadata'))

    @pytest.fixture
    def metalist_config(self, tmp_path: Path) -> MetaListConfig:
        return MetaListConfig(n_frames=2, spacing=1, output_dir=str(tmp_path / 'metalist'))

    @pytest.fixture
    def system_config(self) -> SystemConfig:
        return SystemConfig(platform='cpu', num_threads=1)

    @pytest.fixture
    def scan(self, run_config: RunConfig, setup_config: SetupConfig,
             detect_config: DetectConfig, metadata_config: MetadataConfig,
             metalist_config: MetaListConfig, system_config: SystemConfig) -> Scan:
        config = ScanConfig(image_kind='full', data=run_config, setup=setup_config,
                            detect=detect_config, metadata=metadata_config,
                            metalist=metalist_config, system=system_config)
        return Scan(7, config)

class TestSystemConfig:
    @pytest.fixture
    def allocator_calls(self) -> list[tuple[str, bool]]:
        return []

    @pytest.fixture
    def patched_allocator(self, monkeypatch: pytest.MonkeyPatch,
                          allocator_calls: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
        def fake_set_allocator(allocator: str, *, strict: bool):
            allocator_calls.append((allocator, strict))

        monkeypatch.setattr(cuda, 'set_allocator', fake_set_allocator)
        return allocator_calls

    @pytest.fixture(params=[
        Namespace(platform='cpu', allocator='default', expected=[]),
        Namespace(platform='gpu', allocator='cuda_malloc_async',
                  expected=[('cuda_malloc_async', True)]),
    ])
    def allocator_case(self, request: pytest.FixtureRequest) -> Namespace:
        return request.param

    @pytest.fixture
    def allocator_config(self, allocator_case: Namespace) -> slurm.SystemConfig:
        return slurm.SystemConfig(platform=allocator_case.platform,
                                  cuda_allocator=allocator_case.allocator)

    def test_invalid_allocator(self):
        # Allocator names are validated when system configuration enters the domain model.
        with pytest.raises(ValueError, match='Invalid CUDA allocator'):
            slurm.SystemConfig.from_dict(
                platform='gpu', cuda_allocator='invalid', num_threads=1)

    def test_allocator_boundary(self, allocator_case: Namespace,
                                allocator_config: slurm.SystemConfig,
                                patched_allocator: list[tuple[str, bool]]):
        allocator_config.apply()

        # Only GPU execution initialises its selected allocator, always in strict mode.
        assert patched_allocator == allocator_case.expected

class TestMain(ScanFixtures):
    @pytest.fixture
    def events(self) -> list[Event]:
        return []

    @pytest.fixture
    def metadata_args(self, scan: Scan) -> Namespace:
        scan_nums = [scan.scan_num, scan.scan_num + 1]
        return Namespace(command='metadata', scan='scan.json',
                         scan_num=' '.join(str(scan_num) for scan_num in scan_nums),
                         parameters='params.json')

    @pytest.fixture
    def compile_args(self, scan: Scan) -> Namespace:
        scan_nums = [scan.scan_num, scan.scan_num + 1]
        return Namespace(command='compile', scan='scan.json',
                         scan_num='_'.join(str(scan_num) for scan_num in scan_nums),
                         kind='streaks', in_suffix='in', out_suffix='out')

    @pytest.fixture
    def log_args(self, scan: Scan) -> Namespace:
        scan_nums = [scan.scan_num, scan.scan_num + 1]
        return Namespace(command='log', scan='scan.json',
                         scan_num='_'.join(str(scan_num) for scan_num in scan_nums),
                         kind='streaks', google='sheets.json', in_suffix='online',
                         sample='lysozyme', notes='alignment check')

    def patch_parser(self, monkeypatch: pytest.MonkeyPatch, args: Namespace):
        class FakeParser:
            def parse_args(self) -> Namespace:
                return args

        monkeypatch.setattr(slurm.Scripts, 'parser',
                            classmethod(lambda cls: FakeParser()))

    def patch_scan(self, monkeypatch: pytest.MonkeyPatch, scan: Scan,
                   events: list[Event]):
        def apply():
            events.append('apply')

        monkeypatch.setattr(scan.config.system, 'apply', apply)
        monkeypatch.setattr(slurm.ScanConfig, 'read',
                            classmethod(lambda cls, _: scan.config))

    def patch_metadata(self, monkeypatch: pytest.MonkeyPatch,
                       events: list[Event]):
        class FakeScript:
            def run(self):
                events.append('run')

        def fake_from_file(cls: type[slurm.CreateMetadata], scan_num: int,
                           scan_file: str, params_file: str) -> FakeScript:
            events.append(('from_file', scan_num, scan_file, params_file))
            return FakeScript()

        monkeypatch.setattr(slurm.CreateMetadata, 'from_file',
                            classmethod(fake_from_file))

    def patch_compile(self, monkeypatch: pytest.MonkeyPatch,
                      events: list[Event]):
        class FakeScript:
            def run(self):
                events.append('run')

        def fake_from_file(cls: type[slurm.CompileFiles], kind: str,
                           scan_num: slurm.ScanNumbers, scan_file: str,
                           in_suffix: str, out_suffix: str) -> FakeScript:
            events.append(
                ('from_file', kind, scan_num, scan_file, in_suffix, out_suffix))
            return FakeScript()

        monkeypatch.setattr(slurm.CompileFiles, 'from_file',
                            classmethod(fake_from_file))

    def patch_log(self, monkeypatch: pytest.MonkeyPatch,
                  events: list[Event]):
        class FakeScript:
            def run(self):
                events.append('run')

        def fake_from_file(cls: type[slurm.LogDetections], scan_num: int, kind: str,
                           scan_file: str,
                           google_file: str, in_suffix: str, sample: str,
                           notes: str) -> FakeScript:
            events.append(('from_file', scan_num, kind, scan_file,
                           google_file, in_suffix, sample, notes))
            return FakeScript()

        monkeypatch.setattr(slurm.LogDetections, 'from_file',
                            classmethod(fake_from_file))

    def test_metadata_multiple_scans(self, scan: Scan, metadata_args: Namespace,
                                     events: list[Event],
                                     monkeypatch: pytest.MonkeyPatch):
        scan_nums = [scan.scan_num, scan.scan_num + 1]
        self.patch_parser(monkeypatch, metadata_args)
        self.patch_scan(monkeypatch, scan, events)
        self.patch_metadata(monkeypatch, events)

        slurm.main()

        # Local CLI execution constructs one worker per scan after configuring the system once.
        assert events == [
            'apply',
            ('from_file', scan_nums[0], metadata_args.scan, metadata_args.parameters),
            'run',
            ('from_file', scan_nums[1], metadata_args.scan, metadata_args.parameters),
            'run',
        ]

    def test_compile_multiple_scans_once(self, scan: Scan, compile_args: Namespace,
                                         events: list[Event],
                                         monkeypatch: pytest.MonkeyPatch):
        scan_nums = [scan.scan_num, scan.scan_num + 1]
        self.patch_parser(monkeypatch, compile_args)
        self.patch_scan(monkeypatch, scan, events)
        self.patch_compile(monkeypatch, events)

        slurm.main()

        # Compilation preserves the selection and dispatches one combined operation.
        assert events == [
            'apply',
            ('from_file', compile_args.kind, scan_nums, compile_args.scan,
             compile_args.in_suffix, compile_args.out_suffix),
            'run',
        ]

    def test_log_multiple_scans(self, scan: Scan, log_args: Namespace,
                                events: list[Event],
                                monkeypatch: pytest.MonkeyPatch):
        scan_nums = [scan.scan_num, scan.scan_num + 1]
        self.patch_parser(monkeypatch, log_args)
        self.patch_scan(monkeypatch, scan, events)
        self.patch_log(monkeypatch, events)

        slurm.main()

        # One CLI process logs scans sequentially to avoid concurrent Sheet writes.
        assert events == [
            'apply',
            ('from_file', scan_nums[0], log_args.kind, log_args.scan,
             log_args.google, log_args.in_suffix, log_args.sample, log_args.notes),
            'run',
            ('from_file', scan_nums[1], log_args.kind, log_args.scan,
             log_args.google, log_args.in_suffix, log_args.sample, log_args.notes),
            'run',
        ]

class TestLogDetections(ScanFixtures):
    @pytest.fixture
    def logger(self, scan: Scan) -> slurm.LogDetections:
        sheets = slurm.GoogleSheetsConfig('spreadsheet-id', 'CBC log')
        return slurm.LogDetections(scan, 'streaks', sheets, in_suffix='online')

    @pytest.fixture
    def chunk_paths(self, scan: Scan) -> list[Path]:
        output_dir = scan.config.detect.streaks_dir
        chunk_dir = Path(scan.files.scan_subdir(output_dir, 'online'))
        chunk_dir.mkdir(parents=True)
        paths = [Path(scan.files.scan_file(
            chunk_id, suffix='online', dir=output_dir)) for chunk_id in (1, 0)]
        for path in paths:
            path.touch()
        return paths

    @pytest.fixture
    def compiled_path(self, scan: Scan, chunk_paths: list[Path]) -> Path:
        compiled = Path(scan.files.scan_file(
            suffix='online', dir=scan.config.detect.streaks_dir))
        compiled.touch()
        return compiled

    def test_prefers_compiled_scan(self, logger: slurm.LogDetections,
                                   compiled_path: Path):
        # A compiled scan takes precedence over its individual chunk artifacts.
        assert logger.result_files() == [str(compiled_path)]

    def test_falls_back_to_ordered_chunks(self, chunk_paths: list[Path],
                                          logger: slurm.LogDetections):
        # Chunk fallback follows numeric chunk order regardless of discovery order.
        assert logger.result_files() == [str(path) for path in reversed(chunk_paths)]

class TestCompileFiles(ScanFixtures):
    @pytest.fixture
    def offsets(self) -> tuple[int, ...]:
        return (1, 2)

    def write_chunk(self, path: Path, offset: int):
        pd.DataFrame({'index': [offset], 'signal': [offset + 10]}).to_hdf(path, key='data')
        pd.DataFrame({'index': [offset], 'pulse_id': [offset + 20],
                      'n_detections': [offset + 2]}).to_hdf(path, key='metadata')

    def write_chunks(self, scan: Scan, offsets: tuple[int, ...]) -> tuple[Path, ...]:
        scan_dir = Path(scan.files.scan_subdir(scan.config.detect.streaks_dir))
        scan_dir.mkdir(parents=True)
        paths = tuple(Path(scan.files.scan_file(
            index, dir=scan.config.detect.streaks_dir))
                      for index in range(len(offsets)))
        for chunk_id, (path, offset) in enumerate(zip(paths, offsets)):
            self.write_chunk(path, offset)
            slurm.DetectionAttributes.from_chunk(
                scan, 'streaks', chunk_id, len(offsets), 1).write(str(path))
        return paths

    @pytest.fixture
    def chunk_paths(self, scan: Scan, offsets: tuple[int, ...]) -> tuple[Path, ...]:
        return self.write_chunks(scan, offsets)

    @pytest.fixture
    def loaded_path(self, tmp_path: Path) -> Path:
        path = tmp_path / 'compiled.h5'
        self.write_chunk(path, 1)
        with h5py.File(path, mode='a') as output_file:
            output_file['extra/chunk_id_0/input_file'] = 'input-0.h5'
            output_file['extra/chunk_id_1/input_file'] = 'input-1.h5'
        return path

    @pytest.fixture
    def scans(self, scan: Scan) -> tuple[Scan, ...]:
        return (scan, Scan(scan.scan_num + 1, scan.config))

    @pytest.fixture
    def frame_offsets(self) -> tuple[int, ...]:
        return (0, 5, 9)

    @pytest.fixture
    def reflection_config(self) -> dict[str, object]:
        return {'method': 'test', 'steps': 2}

    @pytest.fixture
    def reflection_data(self, offsets: tuple[int, ...]) -> pd.DataFrame:
        return pd.DataFrame({
            'index': offsets,
            'h': [1, 2],
            'k': [0, 0],
            'l': [0, 0],
            'I_hkl': [10.0, 20.0],
            'sigma_hkl': [1.0, 2.0],
        })

    @pytest.fixture
    def reflection_stats(self) -> pd.DataFrame:
        return pd.DataFrame({'step': [0, 1], 'loss': [2.0, 1.0]})

    @pytest.fixture
    def reflection_paths(self, scan: Scan, scans: tuple[Scan, ...],
                         reflection_data: pd.DataFrame, reflection_stats: pd.DataFrame,
                         reflection_config: dict[str, object]) -> tuple[Path, ...]:
        paths = []
        for item in scans:
            path = Path(item.files.scan_file(dir=scan.config.setup.reflections_dir))
            path.parent.mkdir(parents=True, exist_ok=True)
            reflection_data.to_hdf(path, key='data')
            reflection_stats.to_hdf(path, key='stats', mode='a')
            with h5py.File(path, mode='a') as output_file:
                output_file.attrs['config'] = json.dumps(reflection_config)
                output_file['extra/input_file'] = f'input-{item.scan_num}.h5'
            paths.append(path)
        return tuple(paths)

    @pytest.fixture
    def scan_list(self, scans: tuple[Scan, ...], reflection_paths: tuple[Path, ...],
                  frame_offsets: tuple[int, ...],
                  monkeypatch: pytest.MonkeyPatch) -> slurm.ScanList:
        scan_list = scans[0].config.open_scan([item.scan_num for item in scans])
        monkeypatch.setattr(slurm.ScanList, 'run', lambda self: Namespace(
            indices=lambda: Namespace(offsets=frame_offsets)))
        return scan_list

    @pytest.fixture
    def solution_config(self) -> dict[str, object]:
        return {'method': 'test', 'steps': 2}

    @pytest.fixture
    def solution_paths(self, scan: Scan,
                       solution_config: dict[str, object]) -> tuple[Path, ...]:
        scan_dir = Path(scan.files.scan_subdir(scan.config.setup.solutions_dir))
        scan_dir.mkdir(parents=True)
        paths = tuple(Path(scan.files.scan_file(
            index, dir=scan.config.setup.solutions_dir)) for index in range(2))
        for chunk_id, path in enumerate(paths):
            pd.DataFrame({'index': [10 + chunk_id]}).to_hdf(path, key='data')
            pd.DataFrame({'index': [0], 'h': [1], 'k': [0], 'l': [0]}).to_hdf(
                path, key='miller', mode='a')
            pd.DataFrame({'step': [0], 'loss': [2.0 - chunk_id]}).to_hdf(
                path, key='stats', mode='a')
            if chunk_id == 0:
                pd.DataFrame({'index': [10], 'loss': [2.0]}).to_hdf(
                    path, key='candidates', mode='a')
            with h5py.File(path, mode='a') as output_file:
                output_file.attrs['config'] = json.dumps(solution_config)
                output_file['extra/input_file'] = f'input-{chunk_id}.h5'
        return paths

    @pytest.fixture
    def incomplete_path(self, scan: Scan) -> Path:
        scan_dir = Path(scan.files.scan_subdir(scan.config.detect.streaks_dir))
        scan_dir.mkdir(parents=True)
        path = Path(scan.files.scan_file(0, dir=scan.config.detect.streaks_dir))
        pd.DataFrame({'index': [0]}).to_hdf(path, key='data')
        return path

    def test_load_nested_provenance(self, scan: Scan, loaded_path: Path):
        compiler = slurm.CompileFiles('streaks', scan)
        content = compiler.load_file(str(loaded_path), compiler.schemas['streaks'])

        # Nested provenance keeps its path relative to the HDF5 ``extra`` group.
        assert content.extra == {
            'chunk_id_0/input_file': 'input-0.h5',
            'chunk_id_1/input_file': 'input-1.h5',
        }

    def test_compile_streaks(self, scan: Scan, chunk_paths: tuple[Path, ...]):
        slurm.CompileFiles('streaks', scan).run()

        output_path = scan.files.scan_file(dir=scan.config.detect.streaks_dir)
        # Detection compilation preserves table order and omits absent provenance.
        for key in ('data', 'metadata'):
            expected = pd.concat([pd.read_hdf(path, key) for path in chunk_paths],
                                 ignore_index=True)
            actual = pd.read_hdf(output_path, key)
            pd.testing.assert_frame_equal(actual, expected)

        attributes = slurm.DetectionAttributes.read(output_path)
        assert attributes == slurm.DetectionAttributes.from_scan(
            scan, 'streaks', len(chunk_paths))

        with h5py.File(output_path, mode='r') as output_file:
            assert 'extra' not in output_file

    def test_compile_multiple_scans(self, scan: Scan, scans: tuple[Scan, ...],
                                    scan_list: slurm.ScanList,
                                    reflection_data: pd.DataFrame,
                                    reflection_stats: pd.DataFrame,
                                    frame_offsets: tuple[int, ...]):
        slurm.CompileFiles('reflections', scan_list).run()
        name = f'scan_{scan.scan_num:d}_{scan.scan_num + 1:d}_full'
        output_path = Path(scan.config.setup.reflections_dir) / f'{name}.h5'
        data = pd.read_hdf(output_path, 'data')
        stats = pd.read_hdf(output_path, 'stats')

        # Frame indices become global while optimisation traces retain their source scan.
        expected_data = pd.concat([
            reflection_data.assign(index=reflection_data['index'] + frame_offset)
            for frame_offset in frame_offsets[:len(scans)]], ignore_index=True)
        expected_stats = pd.concat([
            reflection_stats.assign(scan_num=item.scan_num) for item in scans],
            ignore_index=True)
        pd.testing.assert_frame_equal(data, expected_data)
        pd.testing.assert_frame_equal(stats, expected_stats)
        with h5py.File(output_path, mode='r') as output_file:
            for item in scans:
                path = f'extra/scan_num_{item.scan_num}/input_file'
                assert output_file[path].asstr()[()] == f'input-{item.scan_num}.h5'

    def test_compile_solutions(self, scan: Scan, solution_paths: tuple[Path, ...],
                               solution_config: dict[str, object]):
        slurm.CompileFiles('solutions', scan).run()

        output_path = scan.files.scan_file(dir=scan.config.setup.solutions_dir)
        stats = pd.read_hdf(output_path, 'stats')
        # Required tables concatenate, partially present optional tables are omitted, and
        # configuration and provenance survive compilation.
        assert stats['step'].tolist() == [0, 0]

        with pd.HDFStore(output_path, mode='r') as store:
            assert '/candidates' not in store.keys()
        with h5py.File(output_path, mode='r') as output_file:
            assert json.loads(output_file.attrs['config']) == solution_config
            assert output_file['extra/chunk_id_0/input_file'].asstr()[()] == 'input-0.h5'
            assert output_file['extra/chunk_id_1/input_file'].asstr()[()] == 'input-1.h5'

    def test_missing_required_table(self, scan: Scan, incomplete_path: Path):
        # Artifact schemas reject incomplete input at the compilation boundary.
        with pytest.raises(
                ValueError,
                match=r"Missing required tables \['metadata'\]"):
            slurm.CompileFiles('streaks', scan).run()

class TestDetectHits(ScanFixtures):
    @pytest.fixture
    def detect_scan(self, scan: Scan) -> Scan:
        return Scan(scan.scan_num, scan.config.replace(image_kind='stacked'))

    @pytest.fixture
    def params(self) -> StreakFinderConfig:
        return StreakFinderConfig.__new__(StreakFinderConfig)

    @pytest.fixture
    def hit_run(self, xp: NumPyNamespace) -> DetectionRun:
        return DetectionRun(xp.asarray([10, 12]))

    @pytest.fixture
    def empty_run(self, xp: NumPyNamespace) -> DetectionRun:
        return DetectionRun(xp.zeros(0, dtype=int))

    @pytest.fixture
    def detected_streaks(self, xp: NumPyNamespace) -> Streaks:
        return Streaks(xp.asarray([0, 0, 2, 2, 2]), xp.zeros((5, 4)))

    @pytest.fixture
    def empty_streaks(self, xp: NumPyNamespace) -> Streaks:
        return Streaks(xp.zeros(0, dtype=int), xp.zeros((0, 4)))

    @pytest.fixture
    def hit_script(self, detect_scan: Scan, hit_run: DetectionRun,
                   detected_streaks: Streaks, params: StreakFinderConfig,
                   monkeypatch: pytest.MonkeyPatch) -> DetectHits:
        monkeypatch.setattr(slurm.Scan, 'run', lambda self: hit_run)
        monkeypatch.setattr(slurm.Scan, 'find_metadata',
                            lambda self, file_index=None: 'metadata.h5')
        monkeypatch.setattr('cbclib_v2.slurm.scripts.pool_detection',
                            lambda *args, **kwargs: detected_streaks)
        return slurm.Scripts.detect(detect_scan, params, 0, 1, False, 'online')

    @pytest.fixture
    def empty_script(self, detect_scan: Scan, empty_run: DetectionRun,
                     empty_streaks: Streaks, params: StreakFinderConfig,
                     monkeypatch: pytest.MonkeyPatch) -> DetectHits:
        monkeypatch.setattr(slurm.Scan, 'run', lambda self: empty_run)
        monkeypatch.setattr(slurm.Scan, 'find_metadata',
                            lambda self, file_index=None: 'metadata.h5')
        monkeypatch.setattr('cbclib_v2.slurm.scripts.pool_detection',
                            lambda *args, **kwargs: empty_streaks)
        return slurm.Scripts.detect(detect_scan, params, 0, 1, False, 'online')

    def test_metadata_contains_hit_counts_only(self, detect_scan: Scan,
                                               hit_script: DetectHits):
        hit_script.run()

        path = detect_scan.files.scan_file(
            0, suffix='online', dir=detect_scan.config.detect.streaks_dir)
        metadata = pd.read_hdf(path, 'metadata')
        # Only frames above the strict threshold are retained with their counts.
        assert metadata['index'].tolist() == [0, 2]
        assert metadata['n_detections'].tolist() == [2, 3]
        with h5py.File(path, mode='r') as output_file:
            assert output_file.attrs['n_frames'] == 3

    def test_zero_hits_writes_complete_artifact(self, detect_scan: Scan,
                                                empty_script: DetectHits):
        empty_script.run()

        path = detect_scan.files.scan_file(
            0, suffix='online', dir=detect_scan.config.detect.streaks_dir)
        data = pd.read_hdf(path, 'data')
        metadata = pd.read_hdf(path, 'metadata')
        # Completion is represented even when the detector finds nothing.
        assert data.empty
        assert metadata.empty
        with pd.HDFStore(path, mode='r') as store:
            assert '/stats' not in store.keys()
        with h5py.File(path, mode='r') as output_file:
            assert output_file.attrs['scan_num'] == detect_scan.scan_num
            assert output_file.attrs['detection_kind'] == 'streaks'
            assert output_file.attrs['hit_threshold'] == detect_scan.config.detect.hit_threshold
            assert output_file.attrs['n_frames'] == 3

class TestRoutingConvention(ScanFixtures):
    @pytest.fixture
    def structure(self) -> StructureParameters:
        return StructureParameters(radius=1, connectivity=1)

    @pytest.fixture
    def indexing_config(self, structure: StructureParameters) -> IndexingConfig:
        return IndexingConfig(shape=(3, 3, 3), q_max=0.3, width=1.0, threshold=0.5,
                              n_max=2, vicinity=structure, connectivity=structure)

    @pytest.fixture
    def schedule(self) -> ScheduleParameters:
        return ScheduleParameters(kind='constant', learning_rate=1.0e-3,
                                  min_lr=1.0e-4, num_steps=2)

    @pytest.fixture
    def optimise(self, schedule: ScheduleParameters) -> OptimiseParameters:
        return OptimiseParameters(schedule=schedule, method='adam')

    @pytest.fixture
    def refine_setup(self) -> dict[str, object]:
        return {'focus': 'dynamic', 'sample': 'out-of-focus', 'mode': 'shared',
                'geometry': 'fixed-aperture'}

    @pytest.fixture
    def refine_config(self, optimise: OptimiseParameters,
                      refine_setup: dict[str, object]) -> RefineConfig:
        data = RefineDataParameters(keep='all', quantile=0.5, q_abs=1.0, threshold=0.1)
        loss = LossParameters(kind='l1', projector='line')
        return RefineConfig.from_dict(data=data.to_dict(), loss=loss.to_dict(),
                                      optimise=optimise.to_dict(), setup=refine_setup,
                                      indexed_thr=0.0)

    @pytest.fixture(params=['streaks', 'regions'])
    def hits_dir(self, request: pytest.FixtureRequest) -> HitsDir:
        return cast(HitsDir, request.param)

    @pytest.fixture(params=['xtals', 'solutions'])
    def xtal_dir(self, request: pytest.FixtureRequest) -> XtalDir:
        return cast(XtalDir, request.param)

    @pytest.fixture
    def indexing_script(self, scan: Scan, indexing_config: IndexingConfig,
                        hits_dir: HitsDir) -> slurm.IndexingScript:
        return slurm.IndexingScript(scan=scan, params=indexing_config, xtals=str(),
                                    hits_dir=hits_dir, in_suffix='seg', out_suffix='idx',
                                    chunk_id=3)

    @pytest.fixture
    def refine_script(self, scan: Scan, refine_config: RefineConfig,
                      hits_dir: HitsDir, xtal_dir: XtalDir) -> slurm.RefineScript:
        return slurm.RefineScript(scan=scan, params=refine_config, hits_dir=hits_dir,
                                  xtal_dir=xtal_dir, in_suffix='prev', out_suffix='next',
                                  chunk_id=4)

    def hits_directory(self, scan: Scan, hits_dir: HitsDir) -> str:
        if hits_dir == 'streaks':
            return scan.config.detect.streaks_dir
        return scan.config.detect.regions_dir

    def xtal_directory(self, scan: Scan, xtal_dir: XtalDir) -> str:
        if xtal_dir == 'xtals':
            return scan.config.setup.xtals_dir
        return scan.config.setup.solutions_dir

    def test_indexing_script(self, scan: Scan,
                             indexing_script: slurm.IndexingScript):
        expected_dir = self.hits_directory(scan, indexing_script.hits_dir)
        expected_path = (Path(expected_dir) / scan.files.scan_dir(indexing_script.in_suffix)
                         / scan.files.scan_file(indexing_script.chunk_id))
        actual_path = scan.files.scan_file(indexing_script.chunk_id,
                                           suffix=indexing_script.in_suffix,
                                           dir=indexing_script.hits_directory)

        # A logical hit source selects its configured root and canonical chunk path.
        assert indexing_script.hits_directory == expected_dir
        assert actual_path == str(expected_path)

    def test_refine_script(self, scan: Scan,
                           refine_script: slurm.RefineScript):
        expected_hits = self.hits_directory(scan, refine_script.hits_dir)
        expected_xtals = self.xtal_directory(scan, refine_script.xtal_dir)
        if refine_script.xtal_dir == 'xtals':
            expected_setup = scan.config.setup.setup_file
        else:
            expected_setup = scan.files.scan_file(
                refine_script.chunk_id, dir=scan.config.setup.solutions_dir,
                suffix=refine_script.in_suffix)

        # Refinement routes hit, orientation, and setup inputs independently by logical kind.
        assert refine_script.hits_directory == expected_hits
        assert refine_script.xtal_directory == expected_xtals
        assert refine_script.setup_file == expected_setup

class TestSBatchRouting:
    @pytest.fixture(autouse=True)
    def patch_script_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm.ScriptSpec, 'read',
                            classmethod(lambda cls, _: slurm.ScriptSpec()))

    @pytest.fixture(params=['single', 'array'])
    def indexing_args(self, request: pytest.FixtureRequest) -> Namespace:
        return Namespace(kind=request.param, scan_num=7, n_tasks=4,
                         scan_file='scan.json', params_file='index.json',
                         script_file='slurm.json', hits_dir='regions',
                         in_suffix='seg', out_suffix='idx')

    @pytest.fixture
    def indexing_script(self, indexing_args: Namespace) -> slurm.SLURMScript:
        if indexing_args.kind == 'single':
            return slurm.SBatchScripts.index(
                indexing_args.scan_num, indexing_args.scan_file, indexing_args.params_file,
                indexing_args.script_file, hits_dir=indexing_args.hits_dir,
                in_suffix=indexing_args.in_suffix, out_suffix=indexing_args.out_suffix)
        return slurm.SBatchArrayScripts.index(
            indexing_args.scan_num, indexing_args.n_tasks, indexing_args.scan_file,
            indexing_args.params_file, indexing_args.script_file,
            hits_dir=indexing_args.hits_dir, in_suffix=indexing_args.in_suffix,
            out_suffix=indexing_args.out_suffix)

    @pytest.fixture
    def refine_args(self) -> Namespace:
        return Namespace(scan_num=7, n_tasks=4, scan_file='scan.json',
                         params_file='refine.json', script_file='slurm.json',
                         hits_dir='regions', xtal_dir='solutions',
                         in_suffix='idx', out_suffix='rf')

    @pytest.fixture
    def refine_script(self, refine_args: Namespace) -> slurm.SLURMArrayScript:
        return slurm.SBatchArrayScripts.refine(
            refine_args.scan_num, refine_args.n_tasks, refine_args.scan_file,
            refine_args.params_file, refine_args.script_file,
            hits_dir=refine_args.hits_dir, xtal_dir=refine_args.xtal_dir,
            in_suffix=refine_args.in_suffix, out_suffix=refine_args.out_suffix)

    @pytest.fixture
    def metalist_args(self) -> Namespace:
        return Namespace(scan_num=[373, 374, 380], scan_file='scan.json',
                         params_file='metadata.json', script_file='slurm.json')

    @pytest.fixture
    def metalist_script(self, metalist_args: Namespace) -> slurm.SLURMScript:
        return slurm.SBatchScripts.metalist(
            metalist_args.scan_num, metalist_args.scan_file, metalist_args.params_file,
            metalist_args.script_file)

    @pytest.fixture
    def compile_args(self) -> Namespace:
        return Namespace(kind='streaks', scan_num=range(373, 380, 2),
                         scan_file='scan.json', script_file='slurm.json',
                         in_suffix='chunks', out_suffix='compiled')

    @pytest.fixture
    def compile_script(self, compile_args: Namespace) -> slurm.SLURMArrayScript:
        return slurm.SBatchArrayScripts.compile(
            compile_args.kind, compile_args.scan_num, compile_args.scan_file,
            compile_args.script_file, in_suffix=compile_args.in_suffix,
            out_suffix=compile_args.out_suffix)

    @pytest.fixture
    def log_args(self) -> Namespace:
        return Namespace(scan_num=[373, 374], kind='regions', scan_file='scan.json',
                         google_file='sheets.json', script_file='slurm.json',
                         in_suffix='online', sample='lysozyme', notes='alignment check')

    @pytest.fixture
    def log_script(self, log_args: Namespace) -> slurm.SLURMScript:
        return slurm.SBatchScripts.log(
            log_args.scan_num, log_args.kind, log_args.scan_file, log_args.google_file,
            log_args.script_file, in_suffix=log_args.in_suffix, sample=log_args.sample,
            notes=log_args.notes)

    def option_value(self, command: list[str], option: str) -> str:
        return command[command.index(option) + 1]

    def test_indexing_script(self, indexing_args: Namespace,
                             indexing_script: slurm.SLURMScript) -> None:
        command = shlex.split(indexing_script.command)
        expected = {'--hits-dir': indexing_args.hits_dir,
                    '--in-suffix': indexing_args.in_suffix,
                    '--out-suffix': indexing_args.out_suffix}
        job_prefix = 'index' if indexing_args.kind == 'single' else 'index_array'

        # Single and array jobs preserve the same indexing CLI option contract.
        assert command[2] == str(indexing_args.scan_num)
        assert indexing_script.job_name == f'{job_prefix}_{indexing_args.scan_num}'
        for option, value in expected.items():
            assert self.option_value(command, option) == value
        assert '--input-dir' not in command
        assert '--suffix' not in command

    def test_refine_script(self, refine_args: Namespace,
                           refine_script: slurm.SLURMArrayScript) -> None:
        expected = {'--hits-dir': refine_args.hits_dir,
                    '--xtal-dir': refine_args.xtal_dir,
                    '--in-suffix': refine_args.in_suffix,
                    '--out-suffix': refine_args.out_suffix}
        command = shlex.split(refine_script.command)

        # Refinement array jobs forward both independent input-directory selectors.
        for option, value in expected.items():
            assert self.option_value(command, option) == value
        assert '--input-dir' not in command
        assert '--suffix' not in command

    def test_metalist_scan_list(self, metalist_args: Namespace,
                                metalist_script: slurm.SLURMScript) -> None:
        # A scan selection maps directly to scan-indexed tasks and one shared command.
        assert isinstance(metalist_script, slurm.SLURMArrayScript)
        assert metalist_script.task_ids == metalist_args.scan_num
        assert metalist_script.job_name == 'metalist_' + '_'.join(
            str(scan_num) for scan_num in metalist_args.scan_num)
        command = shlex.split(metalist_script.command)
        assert command[2] == '${SCAN_NUM}'
        assert metalist_script.parameters.define_macros['SCAN_NUM'] == '${SLURM_ARRAY_TASK_ID}'

    def test_compile_scan_range(self, compile_args: Namespace,
                                compile_script: slurm.SLURMArrayScript) -> None:
        command = shlex.split(compile_script.command)
        scan_num = compile_args.scan_num
        # Each task receives one scan number and compiles only that scan's chunks.
        assert compile_script.task_ids == scan_num
        assert compile_script.job_name == (
            f'compile_array_{scan_num.start}-{scan_num.stop}-{scan_num.step}')
        assert command[3] == '${SCAN_NUM}'
        assert compile_script.parameters.define_macros['SCAN_NUM'] == '${SLURM_ARRAY_TASK_ID}'
        assert self.option_value(command, '--in-suffix') == compile_args.in_suffix
        assert self.option_value(command, '--out-suffix') == compile_args.out_suffix

    def test_log_scan_list_is_single_writer(self, log_args: Namespace,
                                            log_script: slurm.SLURMScript) -> None:
        command = shlex.split(log_script.command)

        # The scan selection remains one command handled sequentially by the logger.
        assert command[2] == '_'.join(str(scan_num) for scan_num in log_args.scan_num)
        assert command[3] == log_args.kind
        assert self.option_value(command, '--in-suffix') == log_args.in_suffix
        assert self.option_value(command, '--sample') == log_args.sample
        assert self.option_value(command, '--notes') == log_args.notes
