from argparse import Namespace
import json
from pathlib import Path
from typing import Any, Iterator, Literal, cast
import h5py
from jax import config as jax_config
import pandas as pd
import pytest
import cbclib_v2.cuda as cuda
import cbclib_v2.slurm as slurm
from cbclib_v2.annotations import JaxNumPy, NumPy, NumPyNamespace
from cbclib_v2.indexer import ResolvedLens, ResolvedGeometry, ResolvedSetup, XtalState
from cbclib_v2.scripts import (LossParameters, ModelDataParameters, OptimiseParameters,
                               RefineResult, RefinementConfig, RefinementStats,
                               ScheduleParameters)
from cbclib_v2.test_util import check_close

Event = str | tuple[str, ...]
RefinementMode = Literal['shared', 'per-pattern']

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

        assert allocator_calls == []

    def test_gpu_apply(self, allocator_calls: list[tuple[str, bool]],
                       monkeypatch: pytest.MonkeyPatch):
        self.patch_allocator(monkeypatch, allocator_calls)

        slurm.SystemConfig(platform='gpu', cuda_allocator='cuda_malloc_async').apply()

        assert allocator_calls == [('cuda_malloc_async', True)]

    @pytest.mark.parametrize('platform', ['cpu', 'gpu'])
    def test_jax_api(self, platform: str, monkeypatch: pytest.MonkeyPatch):
        calls: list[tuple[str, str]] = []

        def fake_update(name: str, value: str) -> None:
            calls.append((name, value))

        monkeypatch.setattr(jax_config, 'update', fake_update)

        xp = slurm.SystemConfig(platform=platform).jax_api()

        assert calls == [('jax_platform_name', platform)]
        assert xp is JaxNumPy

class TestMain():
    @pytest.fixture
    def events(self) -> list[Event]:
        return []

    def patch_parser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeParser():
            def parse_args(self) -> Namespace:
                return Namespace(command='metadata',
                                 scan='scan.json',
                                 parameters='params.json')

        monkeypatch.setattr(slurm.Scripts, 'parser',
                            classmethod(lambda cls: FakeParser()))

    def patch_scan(self, monkeypatch: pytest.MonkeyPatch, system: Any) -> None:
        class FakeScan():
            scan_num = 373

        FakeScan.system = system

        monkeypatch.setattr(slurm.ScanConfig, 'read',
                            classmethod(lambda cls, _: FakeScan()))

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

    def patch_refine_parser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeParser():
            def parse_args(self) -> Namespace:
                return Namespace(command='refine',
                                 scan='scan.json',
                                 parameters='refine.json',
                                 input_dir='solutions',
                                 in_suffix='gd',
                                 out_suffix='rf',
                                 chunk_id=2,
                                 n_chunks=8)

        monkeypatch.setattr(slurm.Scripts, 'parser',
                            classmethod(lambda cls: FakeParser()))

    def patch_refine(self, monkeypatch: pytest.MonkeyPatch,
                     events: list[Event]) -> None:
        class FakeScript():
            def run(self) -> None:
                events.append('run')

        def fake_from_file(scan_file: str, params_file: str, input_dir: str,
                           in_suffix: str, out_suffix: str,
                           chunk_id: int | None, n_chunks: int | None) -> FakeScript:
            events.append(('from_file', scan_file, params_file, input_dir,
                           in_suffix, out_suffix, str(chunk_id), str(n_chunks)))
            return FakeScript()

        monkeypatch.setattr(slurm.RefinementScript, 'from_file',
                            classmethod(lambda cls, scan_file, params_file, input_dir,
                                        in_suffix, out_suffix, chunk_id, n_chunks:
                                        fake_from_file(scan_file, params_file, input_dir,
                                                       in_suffix, out_suffix, chunk_id,
                                                       n_chunks)))

    def patch_main(self, monkeypatch: pytest.MonkeyPatch, events: list[Event],
                   system: Any) -> None:
        self.patch_parser(monkeypatch)
        self.patch_scan(monkeypatch, system)
        self.patch_metadata(monkeypatch, events)

    def test_metadata(self, events: list[Event], monkeypatch: pytest.MonkeyPatch):
        class FakeSystem():
            def apply(self) -> None:
                events.append('apply')

        self.patch_main(monkeypatch, events, FakeSystem())

        slurm.main()

        assert events == ['apply', ('from_file', 'scan.json', 'params.json'), 'run']

    def test_refine(self, events: list[Event], monkeypatch: pytest.MonkeyPatch):
        class FakeSystem():
            def apply(self) -> None:
                events.append('apply')

        self.patch_refine_parser(monkeypatch)
        self.patch_scan(monkeypatch, FakeSystem())
        self.patch_refine(monkeypatch, events)

        slurm.main()

        assert events == ['apply',
                          ('from_file', 'scan.json', 'refine.json', 'solutions', 'gd', 'rf',
                           '2', '8'),
                          'run']

    def test_apply_error(self, events: list[Event], monkeypatch: pytest.MonkeyPatch):
        class FakeSystem():
            def apply(self) -> None:
                raise RuntimeError('allocator setup failed')

        self.patch_main(monkeypatch, events, FakeSystem())

        with pytest.raises(RuntimeError, match='allocator setup failed'):
            slurm.main()

        assert events == []

class TestCompileStreaks:
    def write_chunk(self, path: Path, offset: int) -> None:
        pd.DataFrame({'index': [offset], 'signal': [offset + 10]}).to_hdf(path, key='data')
        pd.DataFrame({'index': [offset], 'pulse_id': [offset + 20]}).to_hdf(path,
                                                                            key='metadata')
        pd.Series([offset + 30], name='score').to_hdf(path, key='scores')

    def patch_scan(self, monkeypatch: pytest.MonkeyPatch, scan_dir: Path,
                   hits_dir: Path) -> None:
        class FakeDetect:
            streaks_dir = str(hits_dir)
            regions_dir = str(hits_dir)

        class FakeScan:
            scan_num = 7
            detect = FakeDetect()

            def scan_subdir(self, dir: str, suffix: str=str()) -> str:
                return str(scan_dir)

            def list_files(self, dir: str) -> Iterator[str]:
                return iter(sorted(str(path) for path in Path(dir).glob('*.h5')))

            def scan_file(self, suffix: str=str(), dir: str | None=None) -> str:
                name = 'scan_7_full'
                if suffix:
                    name += f'_{suffix}'
                name += '.h5'
                if dir is None:
                    return name
                return str(Path(dir) / name)

        monkeypatch.setattr(slurm.ScanConfig, 'read',
                            classmethod(lambda cls, _: FakeScan()))

    def test_compile_all_pandas_tables(self, tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
        hits_dir = tmp_path / 'hits'
        scan_dir = hits_dir / 'scan_7_full'
        scan_dir.mkdir(parents=True)

        self.write_chunk(scan_dir / 'scan_7_full_f0000.h5', 1)
        self.write_chunk(scan_dir / 'scan_7_full_f0001.h5', 2)
        self.patch_scan(monkeypatch, scan_dir, hits_dir)

        slurm.CompileStreaks('streaks', 'scan.json').run()

        output_path = hits_dir / 'scan_7_full.h5'
        assert pd.read_hdf(output_path, 'data')['index'].tolist() == [1, 2]
        assert pd.read_hdf(output_path, 'metadata')['pulse_id'].tolist() == [21, 22]
        assert pd.read_hdf(output_path, 'scores').tolist() == [31, 32]

    def test_compile_suffixes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hits_dir = tmp_path / 'hits'
        scan_dir = hits_dir / 'scan_7_full_in'
        scan_dir.mkdir(parents=True)

        self.write_chunk(scan_dir / 'scan_7_full_f0000.h5', 1)
        self.write_chunk(scan_dir / 'scan_7_full_f0001.h5', 2)
        self.patch_scan(monkeypatch, scan_dir, hits_dir)

        slurm.CompileStreaks('streaks', 'scan.json', 'in', 'out').run()

        output_path = hits_dir / 'scan_7_full_out.h5'
        assert pd.read_hdf(output_path, 'data')['index'].tolist() == [1, 2]

class TestRoutingConvention:
    @pytest.fixture
    def scan(self) -> Any:
        class FakeDetect:
            streaks_dir = '/data/streaks'
            regions_dir = '/data/regions'

        class FakeSetup:
            xtals_dir = '/data/xtals'
            solutions_dir = '/data/solutions'

        class FakeScan:
            detect = FakeDetect()
            setup = FakeSetup()

            def scan_dir(self, suffix: str=str()) -> str:
                name = 'scan_7_full'
                if suffix:
                    name += f'_{suffix}'
                return name

            def scan_subdir(self, dir: str, suffix: str=str()) -> str:
                return str(Path(dir) / self.scan_dir(suffix))

            def scan_file(self, file_index: int | None=None, /, *, suffix: str=str(),
                          extension: str='.h5', dir: str | None=None) -> str:
                name = 'scan_7_full'
                if file_index is not None:
                    name += f'_f{file_index:04d}'
                if suffix:
                    name += f'_{suffix}'
                name += extension
                if dir is None:
                    return name
                if file_index is None:
                    return str(Path(dir) / name)
                filename = f'scan_7_full_f{file_index:04d}{extension}'
                return str(Path(dir) / self.scan_dir(suffix) / filename)

        return FakeScan()

    def test_indexing_routes_regions_input(self, scan: Any) -> None:
        script = slurm.IndexingScript(scan, None, '', 'regions', 'seg', 'idx', 3, None)

        path = scan.scan_file(script.chunk_id, suffix=script.in_suffix,
                              dir=script.hits_dir)

        assert script.hits_dir == '/data/regions'
        assert path == '/data/regions/scan_7_full_seg/scan_7_full_f0003.h5'

    def test_indexing_routes_streaks_input(self, scan: Any) -> None:
        script = slurm.IndexingScript(scan, None, '', 'streaks', 'str', 'idx', 3, None)

        assert script.hits_dir == '/data/streaks'

    def test_refinement_routes_solutions_input(self, scan: Any) -> None:
        script = slurm.RefinementScript(scan, None, 'solutions', 'prev', 'next', 4, None)

        path = scan.scan_file(script.chunk_id, suffix=script.in_suffix,
                              dir=script.xtals_dir)

        assert script.xtals_dir == '/data/solutions'
        assert path == '/data/solutions/scan_7_full_prev/scan_7_full_f0004.h5'

    def test_refinement_routes_xtals_input(self, scan: Any) -> None:
        script = slurm.RefinementScript(scan, None, 'xtals', 'idx', 'rf', 4, None)

        assert script.xtals_dir == '/data/xtals'

class TestSBatchRouting:
    @pytest.fixture(autouse=True)
    def patch_script_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(slurm.ScriptSpec, 'read',
                            classmethod(lambda cls, _: slurm.ScriptSpec()))

    def test_index_command_uses_common_names(self) -> None:
        script = slurm.SBatchScripts.index('scan.json', 'index.json', 'slurm.json',
                                           input_dir='regions', in_suffix='seg',
                                           out_suffix='idx')

        assert '--input-dir regions' in script.command
        assert '--in-suffix seg' in script.command
        assert '--out-suffix idx' in script.command
        assert '--suffix' not in script.command

    def test_refine_array_command_uses_common_names(self) -> None:
        script = slurm.SBatchArrayScripts.refine('scan.json', 'refine.json', 'slurm.json', 8,
                                                 input_dir='solutions', in_suffix='idx',
                                                 out_suffix='rf')

        assert '--input-dir solutions' in script.command
        assert '--in-suffix idx' in script.command
        assert '--out-suffix rf' in script.command
        assert '--suffix' not in script.command

class TestRefineResult:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture(params=['shared', 'per-pattern'])
    def mode(self, request: pytest.FixtureRequest) -> RefinementMode:
        return cast(RefinementMode, request.param)

    @pytest.fixture
    def config(self, mode: RefinementMode) -> RefinementConfig:
        schedule = ScheduleParameters('constant', 1.0e-3, 1.0e-4, 2)
        optimise = OptimiseParameters(schedule, 'adam')
        data = ModelDataParameters('all', 0.5, 1.0, 0.1, [0.25, 0.75])
        return RefinementConfig(data, LossParameters('l1', 'line'), optimise,
                                'fixed-aperture', mode, 0.2)

    @pytest.fixture
    def result(self, mode: RefinementMode, xp: NumPyNamespace) -> RefineResult:
        basis = xp.stack((xp.eye(3), 2.0 * xp.eye(3), 3.0 * xp.eye(3)))
        if mode == 'shared':
            foc_pos = xp.array([[0.10, 0.20, -0.40]])
            pupil_roi = xp.array([[0.15, 0.17, 0.12, 0.16]])
            z = xp.array([-0.39])
        else:
            foc_pos = xp.array([[0.10, 0.20, -0.40],
                                [0.11, 0.21, -0.40],
                                [0.12, 0.22, -0.40]])
            pupil_roi = xp.array([[0.15, 0.17, 0.12, 0.16],
                                  [0.16, 0.18, 0.13, 0.17],
                                  [0.17, 0.19, 0.14, 0.18]])
            z = xp.array([-0.39, -0.38, -0.37])

        geometry = ResolvedGeometry(ResolvedLens(foc_pos, pupil_roi), z)
        setup = ResolvedSetup(XtalState(basis), geometry)
        stats = RefinementStats([0], [1.0], [1.0e-3], [0.5], [0.1])
        return RefineResult(xp.array([7, 7, 9]), setup, xp.array([2.0, 1.0, 3.0]),
                            xp.array([0.5, 0.8, 0.6]), stats)

    def test_save_restart(self, tmp_path: Path, config: RefinementConfig,
                          result: RefineResult, xp: NumPyNamespace) -> None:
        output = tmp_path / 'refinement.h5'

        result.save(str(output), config)

        candidates = pd.read_hdf(output, 'candidates')
        champions = pd.read_hdf(output, 'data')
        restored = config.import_resolved(ResolvedSetup.import_dataframe(candidates, xp=xp))
        if config.mode == 'per-pattern':
            expected = result.setup.geometry
        else:
            expected = result.setup.geometry.collapse()

        assert champions['index'].tolist() == [7, 9]
        check_close(champions['foc_x'].to_numpy(), candidates['foc_x'].to_numpy()[[1, 2]])
        check_close(restored.resolve(xp).geometry.lens.foc_pos, expected.lens.foc_pos)
        check_close(restored.resolve(xp).geometry.lens.pupil_roi, expected.lens.pupil_roi)
        check_close(restored.resolve(xp).geometry.z, expected.z)

        with h5py.File(output, 'r') as h5_file:
            values = json.loads(h5_file.attrs['refinement_config'])
        assert RefinementConfig.from_dict(**values) == config
