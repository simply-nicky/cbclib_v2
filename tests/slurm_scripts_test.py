from argparse import Namespace
import json
from multiprocessing import cpu_count
from pathlib import Path
import shlex
from typing import Literal, cast
import h5py
from jax import config as jax_config
import pandas as pd
import pytest
from cbclib_v2 import RunConfig, cuda, slurm
from cbclib_v2.annotations import IntArray, JaxNumPy, NumPy, NumPyNamespace, RealArray
from cbclib_v2.indexer import ResolvedGeometry, RefineResult, ResolvedSetup, XtalState
from cbclib_v2.scripts import (DetectConfig, IndexingConfig, LossParameters, MetaListConfig,
                               MetadataConfig, OptimiseParameters, RefineConfig,
                               RefineDataParameters, Scan, ScanConfig, ScheduleParameters,
                               SetupConfig, StructureParameters, SystemConfig)
from cbclib_v2.test_util import check_close

Event = str | tuple[object, ...]
Platform = Literal['cpu', 'gpu']
SetupMode = Literal['shared', 'per-pattern']
HitsDir = Literal['streaks', 'regions']
XtalDir = Literal['xtals', 'solutions']

class ScanFixtures:
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

    def patch_allocator(self, monkeypatch: pytest.MonkeyPatch,
                        allocator_calls: list[tuple[str, bool]]):
        def fake_set_allocator(allocator: str, *, strict: bool):
            allocator_calls.append((allocator, strict))

        monkeypatch.setattr(cuda, 'set_allocator', fake_set_allocator)

    def test_defaults(self):
        config = slurm.SystemConfig(platform='gpu')

        # The default uses the conservative allocator and every available CPU thread.
        assert config.cuda_allocator == 'default'
        assert config.num_threads == cpu_count()

    def test_invalid_allocator(self):
        # Allocator names are validated when system configuration enters the domain model.
        with pytest.raises(ValueError, match='Invalid CUDA allocator'):
            slurm.SystemConfig(platform='gpu', cuda_allocator='invalid')

    def test_cpu_apply(self, allocator_calls: list[tuple[str, bool]],
                       monkeypatch: pytest.MonkeyPatch):
        self.patch_allocator(monkeypatch, allocator_calls)

        slurm.SystemConfig(platform='cpu').apply()

        # CPU configuration never initializes a CUDA allocator.
        assert allocator_calls == []

    def test_gpu_apply(self, allocator_calls: list[tuple[str, bool]],
                       monkeypatch: pytest.MonkeyPatch):
        self.patch_allocator(monkeypatch, allocator_calls)

        config = slurm.SystemConfig(platform='gpu', cuda_allocator='cuda_malloc_async')
        config.apply()

        # GPU configuration applies its selected allocator in strict mode.
        assert allocator_calls == [(config.cuda_allocator, True)]

    @pytest.mark.parametrize('platform', ['cpu', 'gpu'])
    def test_jax_api(self, platform: Platform, monkeypatch: pytest.MonkeyPatch):
        calls: list[tuple[str, str]] = []

        def fake_update(name: str, value: str):
            calls.append((name, value))

        monkeypatch.setattr(jax_config, 'update', fake_update)

        system = slurm.SystemConfig(platform=platform)
        xp = system.jax_api()

        # The configured system platform is forwarded before returning the JAX namespace.
        assert calls == [('jax_platform_name', system.platform)]
        assert xp is JaxNumPy

class TestMain(ScanFixtures):
    @pytest.fixture
    def events(self) -> list[Event]:
        return []

    def patch_parser(self, monkeypatch: pytest.MonkeyPatch, args: Namespace):
        class FakeParser():
            def parse_args(self) -> Namespace:
                return args

        monkeypatch.setattr(slurm.Scripts, 'parser',
                            classmethod(lambda cls: FakeParser()))

    def patch_scan(self, monkeypatch: pytest.MonkeyPatch, scan: Scan,
                   events: list[Event], error: RuntimeError | None=None):
        def apply():
            if error is not None:
                raise error
            events.append('apply')

        monkeypatch.setattr(scan.config.system, 'apply', apply)
        monkeypatch.setattr(slurm.ScanConfig, 'read',
                            classmethod(lambda cls, _: scan.config))

    def patch_metadata(self, monkeypatch: pytest.MonkeyPatch,
                       events: list[Event]):
        class FakeScript():
            def run(self):
                events.append('run')

        def fake_from_file(scan_num: int, scan_file: str, params_file: str) -> FakeScript:
            events.append(('from_file', scan_num, scan_file, params_file))
            return FakeScript()

        monkeypatch.setattr(slurm.CreateMetadata, 'from_file',
                            classmethod(lambda cls, scan_num, scan_file, params_file:
                                        fake_from_file(scan_num, scan_file, params_file)))

    def patch_refine(self, monkeypatch: pytest.MonkeyPatch,
                     events: list[Event]):
        class FakeScript():
            def run(self):
                events.append('run')

        def fake_from_file(scan_num: int, scan_file: str, params_file: str, hits_dir: HitsDir,
                           xtal_dir: XtalDir, in_suffix: str, out_suffix: str,
                           chunk_id: int | None) -> FakeScript:
            events.append(('from_file', scan_num, scan_file, params_file, hits_dir, xtal_dir,
                           in_suffix, out_suffix, chunk_id))
            return FakeScript()

        monkeypatch.setattr(slurm.RefineScript, 'from_file',
                            classmethod(lambda cls, *args: fake_from_file(*args)))

    def patch_compile(self, monkeypatch: pytest.MonkeyPatch,
                      events: list[Event]):
        class FakeScript():
            def run(self):
                events.append('run')

        def fake_from_file(scan_num: slurm.ScanNumbers, kind: str, scan_file: str,
                           in_suffix: str, out_suffix: str) -> FakeScript:
            events.append(
                ('from_file', scan_num, kind, scan_file, in_suffix, out_suffix))
            return FakeScript()

        monkeypatch.setattr(slurm.CompileFiles, 'from_file',
                            classmethod(lambda cls, *args: fake_from_file(*args)))

    def test_parser_accepts_multiple_scan_numbers(self):
        args = slurm.Scripts.parser().parse_args(
            ['metadata', '7 11', 'scan.json', 'params.json'])

        # A complete scan selection remains one positional CLI value for domain parsing.
        assert args.scan_num == '7 11'

    def test_metadata_multiple_scans(self, scan: Scan, events: list[Event],
                                     monkeypatch: pytest.MonkeyPatch):
        scan_nums = [scan.scan_num, scan.scan_num + 1]
        args = Namespace(command='metadata', scan='scan.json',
                         scan_num=' '.join(str(scan_num) for scan_num in scan_nums),
                         parameters='params.json')
        self.patch_parser(monkeypatch, args)
        self.patch_scan(monkeypatch, scan, events)
        self.patch_metadata(monkeypatch, events)

        slurm.main()

        # Local CLI execution constructs one worker per scan after configuring the system once.
        assert events == [
            'apply',
            ('from_file', scan_nums[0], args.scan, args.parameters),
            'run',
            ('from_file', scan_nums[1], args.scan, args.parameters),
            'run',
        ]

    def test_compile_multiple_scans_once(self, scan: Scan, events: list[Event],
                                         monkeypatch: pytest.MonkeyPatch):
        scan_nums = [scan.scan_num, scan.scan_num + 1]
        args = Namespace(command='compile', scan='scan.json',
                         scan_num='_'.join(str(scan_num) for scan_num in scan_nums),
                         kind='streaks', in_suffix='in', out_suffix='out')
        self.patch_parser(monkeypatch, args)
        self.patch_scan(monkeypatch, scan, events)
        self.patch_compile(monkeypatch, events)

        slurm.main()

        # Compilation preserves the selection and dispatches one combined operation.
        assert events == [
            'apply',
            ('from_file', args.kind, scan_nums, args.scan, args.in_suffix, args.out_suffix),
            'run',
        ]

    def test_refine(self, scan: Scan, events: list[Event],
                    monkeypatch: pytest.MonkeyPatch):
        args = Namespace(command='refine', scan='scan.json', parameters='refine.json',
                         scan_num=str(scan.scan_num),
                         hits_dir='regions', xtal_dir='solutions', in_suffix='gd',
                         out_suffix='rf', chunk_id=2)
        self.patch_parser(monkeypatch, args)
        self.patch_scan(monkeypatch, scan, events)
        self.patch_refine(monkeypatch, events)

        slurm.main()

        expected_call = ('from_file', scan.scan_num, args.scan, args.parameters, args.hits_dir,
                         args.xtal_dir, args.in_suffix, args.out_suffix, args.chunk_id)
        # Refine dispatch forwards every routing argument without renaming or reordering it.
        assert events == ['apply', expected_call, 'run']

    def test_apply_error(self, scan: Scan, events: list[Event],
                         monkeypatch: pytest.MonkeyPatch):
        args = Namespace(command='metadata', scan='scan.json',
                         scan_num=str(scan.scan_num),
                         parameters='params.json')
        error = RuntimeError('allocator setup failed')
        self.patch_parser(monkeypatch, args)
        self.patch_scan(monkeypatch, scan, events, error)
        self.patch_metadata(monkeypatch, events)

        with pytest.raises(RuntimeError, match='allocator setup failed'):
            slurm.main()

        # Failed system configuration stops dispatch before construction or execution.
        assert events == []

class TestCompileFiles(ScanFixtures):
    @pytest.fixture
    def offsets(self) -> tuple[int, ...]:
        return (1, 2)

    def write_chunk(self, path: Path, offset: int):
        pd.DataFrame({'index': [offset], 'signal': [offset + 10]}).to_hdf(path, key='data')
        pd.DataFrame({'index': [offset], 'pulse_id': [offset + 20]}).to_hdf(path,
                                                                            key='metadata')

    def patch_scan(self, monkeypatch: pytest.MonkeyPatch, scan: Scan):
        monkeypatch.setattr(slurm.ScanConfig, 'read',
                            classmethod(lambda cls, _: scan.config))

    def write_chunks(self, scan: Scan, offsets: tuple[int, ...],
                     suffix: str=str()) -> tuple[Path, ...]:
        scan_dir = Path(scan.files.scan_subdir(scan.config.detect.streaks_dir, suffix))
        scan_dir.mkdir(parents=True)
        paths = tuple(Path(scan.files.scan_file(
            index, dir=scan.config.detect.streaks_dir, suffix=suffix))
                      for index in range(len(offsets)))
        for path, offset in zip(paths, offsets):
            self.write_chunk(path, offset)
        return paths

    @pytest.fixture
    def chunk_paths(self, scan: Scan, offsets: tuple[int, ...]) -> tuple[Path, ...]:
        return self.write_chunks(scan, offsets)

    @pytest.fixture
    def suffixed_chunk_paths(self, scan: Scan,
                             offsets: tuple[int, ...]) -> tuple[Path, ...]:
        return self.write_chunks(scan, offsets, suffix='in')

    def test_compile_loading(self, scan: Scan, tmp_path: Path):
        path = tmp_path / 'compiled.h5'
        self.write_chunk(path, 1)
        with h5py.File(path, mode='a') as output_file:
            output_file['extra/chunk_id_0/input_file'] = 'input-0.h5'
            output_file['extra/chunk_id_1/input_file'] = 'input-1.h5'

        compiler = slurm.CompileFiles('streaks', scan)
        content = compiler.load_file(path, compiler.schemas['streaks'])

        # Nested provenance keeps its path relative to the HDF5 ``extra`` group.
        assert content.extra == {
            'chunk_id_0/input_file': 'input-0.h5',
            'chunk_id_1/input_file': 'input-1.h5',
        }

    def test_compile_steaks(self, scan: Scan, chunk_paths: tuple[Path, ...],
                            monkeypatch: pytest.MonkeyPatch):
        self.patch_scan(monkeypatch, scan)

        slurm.CompileFiles('streaks', scan).run()

        output_path = scan.files.scan_file(dir=scan.config.detect.streaks_dir)
        # Detection compilation preserves table order and omits absent provenance.
        for key in ('data', 'metadata'):
            expected = pd.concat([pd.read_hdf(path, key) for path in chunk_paths],
                                 ignore_index=True)
            actual = pd.read_hdf(output_path, key)
            pd.testing.assert_frame_equal(actual, expected)

        with h5py.File(output_path, mode='r') as output_file:
            assert 'extra' not in output_file

    def test_compile_multiple_scans(self, scan: Scan,
                                    offsets: tuple[int, ...],
                                    monkeypatch: pytest.MonkeyPatch):
        self.patch_scan(monkeypatch, scan)
        scans = [scan, Scan(scan.scan_num + 1, scan.config)]
        for item in scans:
            path = Path(item.files.scan_file(dir=scan.config.detect.streaks_dir))
            path.parent.mkdir(parents=True, exist_ok=True)
            data = pd.DataFrame({'index': offsets, 'signal': [offset + 10
                                                              for offset in offsets]})
            metadata = pd.DataFrame({'index': offsets, 'pulse_id': [offset + 20
                                                                    for offset in offsets]})
            data.to_hdf(path, key='data')
            metadata.to_hdf(path, key='metadata', mode='a')
            with h5py.File(path, mode='a') as output_file:
                for chunk_id in range(len(offsets)):
                    output_file[f'extra/chunk_id_{chunk_id}/input_file'] = (
                        f'input-{item.scan_num}-{chunk_id}.h5')
        scan_list = scan.config.open_scan([item.scan_num for item in scans])
        monkeypatch.setattr(slurm.ScanList, 'run', lambda self: Namespace(
            indices=lambda: Namespace(offsets=[0, 5, 9])))
        compiler = slurm.CompileFiles('streaks', scan_list)
        compiler.run()

        name = f'scan_{scan.scan_num:d}_{scan.scan_num + 1:d}_full'
        output_path = Path(scan.config.detect.streaks_dir) / f'{name}.h5'
        data = pd.read_hdf(output_path, 'data')
        metadata = pd.read_hdf(output_path, 'metadata')

        # Run-list offsets place scan-local data indices in the flattened frame space.
        expected_data = pd.DataFrame({'index': [1, 2, 6, 7],
                                      'signal': [11, 12, 11, 12]})
        expected_metadata = pd.concat(
            [pd.DataFrame({'index': offsets,
                           'pulse_id': [offset + 20 for offset in offsets]})
             for _ in scans], ignore_index=True)
        pd.testing.assert_frame_equal(data, expected_data)
        pd.testing.assert_frame_equal(metadata, expected_metadata)
        assert 'scan_num' not in data.columns
        with h5py.File(output_path, mode='r') as output_file:
            for item in scans:
                for chunk_id in range(len(offsets)):
                    path = f'extra/scan_num_{item.scan_num}/chunk_id_{chunk_id}/input_file'
                    assert output_file[path].asstr()[()] == (
                        f'input-{item.scan_num}-{chunk_id}.h5')

    def test_compile_suffixes(self, scan: Scan,
                              suffixed_chunk_paths: tuple[Path, ...],
                              monkeypatch: pytest.MonkeyPatch):
        self.patch_scan(monkeypatch, scan)

        slurm.CompileFiles('streaks', scan, 'in', 'out').run()

        output_path = scan.files.scan_file(dir=scan.config.detect.streaks_dir, suffix='out')
        expected = pd.concat([pd.read_hdf(path, 'data') for path in suffixed_chunk_paths],
                             ignore_index=True)
        # Input and output suffixes change routing without changing table contents.
        dataframe = pd.read_hdf(output_path, 'data')
        assert isinstance(dataframe, pd.DataFrame)
        pd.testing.assert_frame_equal(dataframe, expected)

    def test_compile_solutions(self, scan: Scan, monkeypatch: pytest.MonkeyPatch):
        self.patch_scan(monkeypatch, scan)
        scan_dir = Path(scan.files.scan_subdir(scan.config.setup.solutions_dir))
        scan_dir.mkdir(parents=True)
        config = {'method': 'test', 'steps': 2}
        paths = tuple(Path(scan.files.scan_file(index, dir=scan.config.setup.solutions_dir))
                      for index in range(2))

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
                output_file.attrs['config'] = json.dumps(config)
                output_file[f'extra/input_file'] = f'input-{chunk_id}.h5'

        slurm.CompileFiles('solutions', scan).run()

        output_path = scan.files.scan_file(dir=scan.config.setup.solutions_dir)
        stats = pd.read_hdf(output_path, 'stats')
        # Required tables concatenate, partially present optional tables are omitted, and
        # configuration and provenance survive compilation.
        assert stats['step'].tolist() == [0, 0]

        with pd.HDFStore(output_path, mode='r') as store:
            assert '/candidates' not in store.keys()
        with h5py.File(output_path, mode='r') as output_file:
            assert json.loads(output_file.attrs['config']) == config
            assert output_file['extra/chunk_id_0/input_file'].asstr()[()] == 'input-0.h5'
            assert output_file['extra/chunk_id_1/input_file'].asstr()[()] == 'input-1.h5'

    def test_compile(self, scan: Scan, monkeypatch: pytest.MonkeyPatch):
        self.patch_scan(monkeypatch, scan)
        scan_dir = Path(scan.files.scan_subdir(scan.config.detect.streaks_dir))
        scan_dir.mkdir(parents=True)
        path = scan.files.scan_file(0, dir=scan.config.detect.streaks_dir)
        pd.DataFrame({'index': [0]}).to_hdf(path, key='data')

        # Artifact schemas reject incomplete input at the compilation boundary.
        with pytest.raises(ValueError, match="Missing required tables \['metadata'\]"):
            slurm.CompileFiles('streaks', scan).run()

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
    def patch_script_spec(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(slurm.ScriptSpec, 'read',
                            classmethod(lambda cls, _: slurm.ScriptSpec()))

    @pytest.fixture(params=['single', 'array'])
    def indexing_script(self, request: pytest.FixtureRequest) -> slurm.SLURMScript:
        kwargs = {'hits_dir': 'regions', 'in_suffix': 'seg', 'out_suffix': 'idx'}
        if request.param == 'single':
            return slurm.SBatchScripts.index(
                7, 'scan.json', 'index.json', 'slurm.json', **kwargs)
        return slurm.SBatchArrayScripts.index(
            7, 4, 'scan.json', 'index.json', 'slurm.json', **kwargs)

    def option_value(self, command: list[str], option: str) -> str:
        return command[command.index(option) + 1]

    def test_indexing_script(self, indexing_script: slurm.SLURMScript):
        command = shlex.split(indexing_script.command)
        expected = {'--hits-dir': 'regions', '--in-suffix': 'seg', '--out-suffix': 'idx'}

        # Single and array jobs preserve the same indexing CLI option contract.
        assert command[2] == '7'
        assert indexing_script.job_name in ('index_7', 'index_array_7')
        for option, value in expected.items():
            assert self.option_value(command, option) == value
        assert '--input-dir' not in command
        assert '--suffix' not in command

    def test_refine_script(self):
        expected = {'--hits-dir': 'regions', '--xtal-dir': 'solutions',
                    '--in-suffix': 'idx', '--out-suffix': 'rf'}
        script = slurm.SBatchArrayScripts.refine(
            7, 4, 'scan.json', 'refine.json', 'slurm.json', hits_dir='regions',
            xtal_dir='solutions', in_suffix='idx', out_suffix='rf')
        command = shlex.split(script.command)

        # Refinement array jobs forward both independent input-directory selectors.
        for option, value in expected.items():
            assert self.option_value(command, option) == value
        assert '--input-dir' not in command
        assert '--suffix' not in command

    def test_metalist_scan_list(self):
        script = slurm.SBatchScripts.metalist(
            [373, 374, 380], 'scan.json', 'metadata.json', 'slurm.json')

        # A scan selection maps directly to scan-indexed tasks and one shared command.
        assert isinstance(script, slurm.SLURMArrayScript)
        assert script.task_ids == [373, 374, 380]
        assert script.job_name == 'metalist_373_374_380'
        command = shlex.split(script.command)
        assert command[2] == '${SCAN_NUM}'
        assert script.parameters.define_macros['SCAN_NUM'] == '${SLURM_ARRAY_TASK_ID}'

    def test_compile_scan_range(self):
        script = slurm.SBatchArrayScripts.compile(
            'streaks', range(373, 380, 2), 'scan.json', 'slurm.json',
            in_suffix='chunks', out_suffix='compiled')

        command = shlex.split(script.command)
        # Each task receives one scan number and compiles only that scan's chunks.
        assert script.task_ids == range(373, 380, 2)
        assert script.job_name == 'compile_array_373-380-2'
        assert command[3] == '${SCAN_NUM}'
        assert script.parameters.define_macros['SCAN_NUM'] == '${SLURM_ARRAY_TASK_ID}'
        assert self.option_value(command, '--in-suffix') == 'chunks'
        assert self.option_value(command, '--out-suffix') == 'compiled'

    @pytest.mark.parametrize('scan_num', [[], [7, 7], [-1]])
    def test_scan_list_invalid_numbers(self, scan_num: list[int]):
        # Task selections must be non-empty, unique, and non-negative.
        with pytest.raises(ValueError, match='scan_num'):
            slurm.SBatchScripts.metalist(
                scan_num, 'scan.json', 'metadata.json', 'slurm.json')

class TestRefineResult:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture(params=['shared', 'per-pattern'])
    def mode(self, request: pytest.FixtureRequest) -> SetupMode:
        return cast(SetupMode, request.param)

    @pytest.fixture
    def frames(self, xp: NumPyNamespace) -> IntArray:
        return xp.asarray([7, 7, 9])

    @pytest.fixture
    def loss(self, xp: NumPyNamespace) -> RealArray:
        return xp.asarray([2.0, 1.0, 3.0])

    @pytest.fixture
    def xtal(self, loss: RealArray, xp: NumPyNamespace) -> XtalState:
        scales = xp.arange(1, loss.size + 1)
        return XtalState(xp.asarray(scales[:, None, None] * xp.eye(3)))

    @pytest.fixture
    def geometry(self, mode: SetupMode, loss: RealArray,
                 xp: NumPyNamespace) -> ResolvedGeometry:
        if mode == 'shared':
            size = 1
        else:
            size = loss.size
        offsets = xp.arange(size)[:, None]
        foc_pos = xp.asarray([[0.10, 0.20, -0.40]]) + 0.01 * offsets
        pupil_roi = xp.asarray([[0.15, 0.17, 0.12, 0.16]]) + 0.01 * offsets
        defocus = 0.01 * xp.arange(1, size + 1)
        return ResolvedGeometry(foc_pos, pupil_roi, defocus)

    @pytest.fixture
    def resolved(self, xtal: XtalState, geometry: ResolvedGeometry) -> ResolvedSetup:
        return ResolvedSetup(xtal, geometry)

    @pytest.fixture
    def result(self, frames: IntArray, resolved: ResolvedSetup,
               loss: RealArray) -> RefineResult:
        return RefineResult(frames=frames, resolved=resolved, loss=loss)

    def test_champions(self, result: RefineResult, xp: NumPyNamespace):
        indices = result.champions_only()

        # A champion is the unique minimum-loss candidate for each frame.
        for frame in xp.unique_values(result.frames):
            candidates = xp.where(result.frames == frame)[0]
            selected = indices[result.frames[indices] == frame]
            assert selected.size == 1
            check_close(result.loss[selected], xp.min(result.loss[candidates]))

    def test_dataframe_round_trip(self, result: RefineResult,
                                  xp: NumPyNamespace):
        dataframe = result.to_dataframe()
        frames, restored = ResolvedSetup.import_dataframe(dataframe, index=True, xp=xp)
        if result.resolved.geometry.size == 1:
            expected_geometry = result.resolved.geometry.broadcast(result.frames.size)
        else:
            expected_geometry = result.resolved.geometry

        # Tabular conversion preserves frame labels, loss, crystal state, and geometry.
        assert isinstance(expected_geometry, ResolvedGeometry)
        assert isinstance(restored.geometry, ResolvedGeometry)
        assert xp.all(frames == result.frames)
        check_close(dataframe['loss'].to_numpy(), result.loss)
        check_close(restored.xtal.basis, result.resolved.xtal.basis)
        check_close(restored.geometry.foc_pos, expected_geometry.foc_pos)
        check_close(restored.geometry.pupil_roi, expected_geometry.pupil_roi)
        check_close(restored.geometry.defocus, expected_geometry.defocus)
