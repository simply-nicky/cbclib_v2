from argparse import Namespace
import json
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
                               RefineDataParameters, ScanConfig, ScheduleParameters, SetupConfig,
                               StructureParameters, SystemConfig)
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
             metalist_config: MetaListConfig, system_config: SystemConfig) -> ScanConfig:
        return ScanConfig(scan_num=7, image_kind='full', data=run_config, setup=setup_config,
                          detect=detect_config, metadata=metadata_config,
                          metalist=metalist_config, system=system_config)

class TestSystemConfig():
    @pytest.fixture
    def allocator_calls(self) -> list[tuple[str, bool]]:
        return []

    def patch_allocator(self, monkeypatch: pytest.MonkeyPatch,
                        allocator_calls: list[tuple[str, bool]]) -> None:
        def fake_set_allocator(allocator: str, *, strict: bool) -> None:
            allocator_calls.append((allocator, strict))

        monkeypatch.setattr(cuda, 'set_allocator', fake_set_allocator)

    def test_defaults(self):
        config = slurm.SystemConfig(platform='gpu')

        assert config.cuda_allocator == 'default'
        assert config.num_threads > 0

    def test_invalid_allocator(self):
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

        def fake_update(name: str, value: str) -> None:
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

    def patch_parser(self, monkeypatch: pytest.MonkeyPatch, args: Namespace) -> None:
        class FakeParser():
            def parse_args(self) -> Namespace:
                return args

        monkeypatch.setattr(slurm.Scripts, 'parser',
                            classmethod(lambda cls: FakeParser()))

    def patch_scan(self, monkeypatch: pytest.MonkeyPatch, scan: ScanConfig,
                   events: list[Event], error: RuntimeError | None=None) -> None:
        def apply() -> None:
            if error is not None:
                raise error
            events.append('apply')

        monkeypatch.setattr(scan.system, 'apply', apply)
        monkeypatch.setattr(slurm.ScanConfig, 'read',
                            classmethod(lambda cls, _: scan))

    def patch_metadata(self, monkeypatch: pytest.MonkeyPatch,
                       events: list[Event]) -> None:
        class FakeScript():
            def run(self) -> None:
                events.append('run')

        def fake_from_file(scan_file: str, params_file: str) -> FakeScript:
            events.append(('from_file', scan_file, params_file))
            return FakeScript()

        monkeypatch.setattr(slurm.CreateMetadata, 'from_file',
                            classmethod(lambda cls, scan_file, params_file:
                                        fake_from_file(scan_file, params_file)))

    def patch_refine(self, monkeypatch: pytest.MonkeyPatch,
                     events: list[Event]) -> None:
        class FakeScript():
            def run(self) -> None:
                events.append('run')

        def fake_from_file(scan_file: str, params_file: str, hits_dir: HitsDir,
                           xtal_dir: XtalDir, in_suffix: str, out_suffix: str,
                           chunk_id: int | None) -> FakeScript:
            events.append(('from_file', scan_file, params_file, hits_dir, xtal_dir,
                           in_suffix, out_suffix, chunk_id))
            return FakeScript()

        monkeypatch.setattr(slurm.RefineScript, 'from_file',
                            classmethod(lambda cls, *args: fake_from_file(*args)))

    def test_metadata(self, scan: ScanConfig, events: list[Event],
                      monkeypatch: pytest.MonkeyPatch):
        args = Namespace(command='metadata', scan='scan.json', parameters='params.json')
        self.patch_parser(monkeypatch, args)
        self.patch_scan(monkeypatch, scan, events)
        self.patch_metadata(monkeypatch, events)

        slurm.main()

        # Main applies system configuration before constructing and running the command.
        assert events == ['apply', ('from_file', args.scan, args.parameters), 'run']

    def test_refine(self, scan: ScanConfig, events: list[Event],
                    monkeypatch: pytest.MonkeyPatch):
        args = Namespace(command='refine', scan='scan.json', parameters='refine.json',
                         hits_dir='regions', xtal_dir='solutions', in_suffix='gd',
                         out_suffix='rf', chunk_id=2)
        self.patch_parser(monkeypatch, args)
        self.patch_scan(monkeypatch, scan, events)
        self.patch_refine(monkeypatch, events)

        slurm.main()

        expected_call = ('from_file', args.scan, args.parameters, args.hits_dir,
                         args.xtal_dir, args.in_suffix, args.out_suffix, args.chunk_id)
        # Refine dispatch forwards every routing argument without renaming or reordering it.
        assert events == ['apply', expected_call, 'run']

    def test_apply_error(self, scan: ScanConfig, events: list[Event],
                         monkeypatch: pytest.MonkeyPatch):
        args = Namespace(command='metadata', scan='scan.json', parameters='params.json')
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

    def write_chunk(self, path: Path, offset: int) -> None:
        pd.DataFrame({'index': [offset], 'signal': [offset + 10]}).to_hdf(path, key='data')
        pd.DataFrame({'index': [offset], 'pulse_id': [offset + 20]}).to_hdf(path,
                                                                            key='metadata')

    def patch_scan(self, monkeypatch: pytest.MonkeyPatch, scan: ScanConfig) -> None:
        monkeypatch.setattr(slurm.ScanConfig, 'read',
                            classmethod(lambda cls, _: scan))

    def write_chunks(self, scan: ScanConfig, offsets: tuple[int, ...],
                     suffix: str=str()) -> tuple[Path, ...]:
        scan_dir = Path(scan.scan_subdir(scan.detect.streaks_dir, suffix))
        scan_dir.mkdir(parents=True)
        paths = tuple(Path(scan.scan_file(index, dir=scan.detect.streaks_dir, suffix=suffix))
                      for index in range(len(offsets)))
        for path, offset in zip(paths, offsets):
            self.write_chunk(path, offset)
        return paths

    @pytest.fixture
    def chunk_paths(self, scan: ScanConfig, offsets: tuple[int, ...]) -> tuple[Path, ...]:
        return self.write_chunks(scan, offsets)

    @pytest.fixture
    def suffixed_chunk_paths(self, scan: ScanConfig,
                             offsets: tuple[int, ...]) -> tuple[Path, ...]:
        return self.write_chunks(scan, offsets, suffix='in')

    def test_compile_detection_tables(self, scan: ScanConfig,
                                      chunk_paths: tuple[Path, ...],
                                      monkeypatch: pytest.MonkeyPatch) -> None:
        self.patch_scan(monkeypatch, scan)

        slurm.CompileFiles('streaks', 'scan.json').run()

        output_path = scan.scan_file(dir=scan.detect.streaks_dir)
        # Detection compilation preserves ordered rows and records every source chunk.
        for key in ('data', 'metadata'):
            expected = pd.concat([pd.read_hdf(path, key) for path in chunk_paths],
                                 ignore_index=True)
            actual = pd.read_hdf(output_path, key)
            pd.testing.assert_frame_equal(actual, expected)

        manifest = pd.read_hdf(output_path, 'extra')
        assert manifest['chunk_id'].tolist() == [0, 1]
        assert manifest['source_file'].tolist() == [str(path.resolve()) for path in chunk_paths]

    def test_compile_suffixes(self, scan: ScanConfig,
                              suffixed_chunk_paths: tuple[Path, ...],
                              monkeypatch: pytest.MonkeyPatch) -> None:
        self.patch_scan(monkeypatch, scan)

        slurm.CompileFiles('streaks', 'scan.json', 'in', 'out').run()

        output_path = scan.scan_file(dir=scan.detect.streaks_dir, suffix='out')
        expected = pd.concat([pd.read_hdf(path, 'data') for path in suffixed_chunk_paths],
                             ignore_index=True)
        # Input and output suffixes change routing without changing table contents.
        dataframe = pd.read_hdf(output_path, 'data')
        if isinstance(dataframe, pd.Series):
            raise AssertionError(f"Expected a DataFrame, but got a {type(expected).__name__}")
        pd.testing.assert_frame_equal(dataframe, expected)

    def test_compile_refinement_provenance(self, scan: ScanConfig,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
        self.patch_scan(monkeypatch, scan)
        scan_dir = Path(scan.scan_subdir(scan.setup.solutions_dir))
        scan_dir.mkdir(parents=True)
        config = {'method': 'test', 'steps': 2}
        paths = tuple(Path(scan.scan_file(index, dir=scan.setup.solutions_dir))
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

        slurm.CompileFiles('solutions', 'scan.json').run()

        output_path = scan.scan_file(dir=scan.setup.solutions_dir)
        stats = pd.read_hdf(output_path, 'stats')
        assert stats['step'].tolist() == [0, 0]
        assert stats['chunk_id'].tolist() == [0, 1]
        assert pd.read_hdf(output_path, 'candidates')['index'].tolist() == [10]

        manifest = pd.read_hdf(output_path, 'extra')
        assert manifest['input_file'].tolist() == ['input-0.h5', 'input-1.h5']
        with h5py.File(output_path, mode='r') as output_file:
            assert json.loads(output_file.attrs['config']) == config

    def test_compile_rejects_inconsistent_config(self, scan: ScanConfig,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
        self.patch_scan(monkeypatch, scan)
        scan_dir = Path(scan.scan_subdir(scan.setup.xtals_dir))
        scan_dir.mkdir(parents=True)
        for chunk_id in range(2):
            path = scan.scan_file(chunk_id, dir=scan.setup.xtals_dir)
            pd.DataFrame({'index': [chunk_id]}).to_hdf(path, key='data')
            with h5py.File(path, mode='a') as output_file:
                output_file.attrs['config'] = json.dumps({'chunk': chunk_id})

        with pytest.raises(ValueError, match='does not match'):
            slurm.CompileFiles('xtals', 'scan.json').run()

        assert not Path(scan.scan_file(dir=scan.setup.xtals_dir)).exists()

    def test_compile_rejects_missing_required_table(self, scan: ScanConfig,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
        self.patch_scan(monkeypatch, scan)
        scan_dir = Path(scan.scan_subdir(scan.detect.streaks_dir))
        scan_dir.mkdir(parents=True)
        path = scan.scan_file(0, dir=scan.detect.streaks_dir)
        pd.DataFrame({'index': [0]}).to_hdf(path, key='data')

        with pytest.raises(ValueError, match="Missing required tables \['metadata'\]"):
            slurm.CompileFiles('streaks', 'scan.json').run()

    def test_compile_failure_preserves_output(self, scan: ScanConfig,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
        self.patch_scan(monkeypatch, scan)
        scan_dir = Path(scan.scan_subdir(scan.setup.xtals_dir))
        scan_dir.mkdir(parents=True)
        config = json.dumps({'method': 'test'})
        for chunk_id in range(2):
            path = scan.scan_file(chunk_id, dir=scan.setup.xtals_dir)
            if chunk_id == 0:
                pd.DataFrame({'index': [chunk_id]}).to_hdf(path, key='data')
            else:
                pd.Series([chunk_id], name='index').to_hdf(path, key='data')
            with h5py.File(path, mode='a') as output_file:
                output_file.attrs['config'] = config

        output_path = scan.scan_file(dir=scan.setup.xtals_dir)
        expected = pd.DataFrame({'sentinel': [1]})
        expected.to_hdf(output_path, key='previous')

        with pytest.raises(ValueError, match="Table 'data' has inconsistent types"):
            slurm.CompileFiles('xtals', 'scan.json').run()

        pd.testing.assert_frame_equal(pd.read_hdf(output_path, 'previous'), expected)
        assert list(Path(scan.setup.xtals_dir).glob('.compile-*')) == []

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
    def indexing_script(self, scan: ScanConfig, indexing_config: IndexingConfig,
                        hits_dir: HitsDir) -> slurm.IndexingScript:
        return slurm.IndexingScript(scan=scan, params=indexing_config, xtals=str(),
                                    hits_dir=hits_dir, in_suffix='seg', out_suffix='idx',
                                    chunk_id=3)

    @pytest.fixture
    def refine_script(self, scan: ScanConfig, refine_config: RefineConfig,
                      hits_dir: HitsDir, xtal_dir: XtalDir) -> slurm.RefineScript:
        return slurm.RefineScript(scan=scan, params=refine_config, hits_dir=hits_dir,
                                  xtal_dir=xtal_dir, in_suffix='prev', out_suffix='next',
                                  chunk_id=4)

    def hits_directory(self, scan: ScanConfig, hits_dir: HitsDir) -> str:
        if hits_dir == 'streaks':
            return scan.detect.streaks_dir
        return scan.detect.regions_dir

    def xtal_directory(self, scan: ScanConfig, xtal_dir: XtalDir) -> str:
        if xtal_dir == 'xtals':
            return scan.setup.xtals_dir
        return scan.setup.solutions_dir

    def test_indexing_routes_hits(self, scan: ScanConfig,
                                  indexing_script: slurm.IndexingScript) -> None:
        expected_dir = self.hits_directory(scan, indexing_script.hits_dir)
        expected_path = (Path(expected_dir) / scan.scan_dir(indexing_script.in_suffix)
                         / scan.scan_file(indexing_script.chunk_id))
        actual_path = scan.scan_file(indexing_script.chunk_id,
                                     suffix=indexing_script.in_suffix,
                                     dir=indexing_script.hits_directory)

        # A logical hit source selects its configured root and canonical chunk path.
        assert indexing_script.hits_directory == expected_dir
        assert actual_path == str(expected_path)

    def test_refinement_routes_inputs(self, scan: ScanConfig,
                                      refine_script: slurm.RefineScript) -> None:
        expected_hits = self.hits_directory(scan, refine_script.hits_dir)
        expected_xtals = self.xtal_directory(scan, refine_script.xtal_dir)
        if refine_script.xtal_dir == 'xtals':
            expected_setup = scan.setup.setup_file
        else:
            expected_setup = scan.scan_file(refine_script.chunk_id,
                                            dir=scan.setup.solutions_dir,
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
    def indexing_script(self, request: pytest.FixtureRequest) -> slurm.SLURMScript:
        kwargs = {'hits_dir': 'regions', 'in_suffix': 'seg', 'out_suffix': 'idx'}
        if request.param == 'single':
            return slurm.SBatchScripts.index('scan.json', 'index.json', 'slurm.json', **kwargs)
        return slurm.SBatchArrayScripts.index('scan.json', 'index.json', 'slurm.json', **kwargs)

    def option_value(self, command: list[str], option: str) -> str:
        return command[command.index(option) + 1]

    def test_index_command_uses_common_names(self, indexing_script: slurm.SLURMScript) -> None:
        command = shlex.split(indexing_script.command)
        expected = {'--hits-dir': 'regions', '--in-suffix': 'seg', '--out-suffix': 'idx'}

        # Single and array jobs preserve the same indexing CLI option contract.
        for option, value in expected.items():
            assert self.option_value(command, option) == value
        assert '--input-dir' not in command
        assert '--suffix' not in command

    def test_refine_array_command_uses_common_names(self) -> None:
        expected = {'--hits-dir': 'regions', '--xtal-dir': 'solutions',
                    '--in-suffix': 'idx', '--out-suffix': 'rf'}
        script = slurm.SBatchArrayScripts.refine(
            'scan.json', 'refine.json', 'slurm.json', hits_dir='regions',
            xtal_dir='solutions', in_suffix='idx', out_suffix='rf')
        command = shlex.split(script.command)

        # Refinement array jobs forward both independent input-directory selectors.
        for option, value in expected.items():
            assert self.option_value(command, option) == value
        assert '--input-dir' not in command
        assert '--suffix' not in command

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

    def test_champions(self, result: RefineResult, xp: NumPyNamespace) -> None:
        indices = result.champions_only()

        # A champion is the unique minimum-loss candidate for each frame.
        for frame in xp.unique_values(result.frames):
            candidates = xp.where(result.frames == frame)[0]
            selected = indices[result.frames[indices] == frame]
            assert selected.size == 1
            check_close(result.loss[selected], xp.min(result.loss[candidates]))

    def test_dataframe_round_trip(self, result: RefineResult,
                                  xp: NumPyNamespace) -> None:
        dataframe = result.to_dataframe()
        frames, restored = ResolvedSetup.import_dataframe(dataframe, index=True, xp=xp)
        if result.resolved.geometry.size == 1:
            expected_geometry = result.resolved.geometry.broadcast(result.frames.size)
        else:
            expected_geometry = result.resolved.geometry

        # Tabular conversion preserves frame labels, loss, crystal state, and geometry.
        assert xp.all(frames == result.frames)
        check_close(dataframe['loss'].to_numpy(), result.loss)
        check_close(restored.xtal.basis, result.resolved.xtal.basis)
        check_close(restored.geometry.foc_pos, expected_geometry.foc_pos)
        check_close(restored.geometry.pupil_roi, expected_geometry.pupil_roi)
        if isinstance(expected_geometry, ResolvedGeometry) and \
           isinstance(restored.geometry, ResolvedGeometry):
            check_close(restored.geometry.defocus, expected_geometry.defocus)
