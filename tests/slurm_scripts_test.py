from argparse import Namespace
from pathlib import Path
from typing import Any, Iterator
import pandas as pd
import pytest
import cbclib_v2.slurm.scripts as slurm_scripts

Event = str | tuple[str, ...]

class TestSystemConfig():
    @pytest.fixture
    def allocator_calls(self) -> list[tuple[str, bool]]:
        return []

    def patch_allocator(self, monkeypatch: pytest.MonkeyPatch,
                        allocator_calls: list[tuple[str, bool]]) -> None:
        def fake_set_allocator(allocator: str, *, strict: bool) -> None:
            allocator_calls.append((allocator, strict))

        monkeypatch.setattr(slurm_scripts, 'set_allocator', fake_set_allocator)

    def test_defaults(self):
        config = slurm_scripts.SystemConfig(platform='gpu')

        assert config.cuda_allocator == 'default'
        assert config.num_threads > 0

    def test_invalid_allocator(self):
        with pytest.raises(ValueError, match='Invalid CUDA allocator'):
            slurm_scripts.SystemConfig(platform='gpu', cuda_allocator='invalid')

    def test_cpu_apply(self, allocator_calls: list[tuple[str, bool]],
                       monkeypatch: pytest.MonkeyPatch):
        self.patch_allocator(monkeypatch, allocator_calls)

        slurm_scripts.SystemConfig(platform='cpu').apply()

        assert allocator_calls == []

    def test_gpu_apply(self, allocator_calls: list[tuple[str, bool]],
                       monkeypatch: pytest.MonkeyPatch):
        self.patch_allocator(monkeypatch, allocator_calls)

        slurm_scripts.SystemConfig(platform='gpu',
                                   cuda_allocator='cuda_malloc_async').apply()

        assert allocator_calls == [('cuda_malloc_async', True)]

    @pytest.mark.parametrize('platform', ['cpu', 'gpu'])
    def test_jax_api(self, platform: str, monkeypatch: pytest.MonkeyPatch):
        calls: list[tuple[str, str]] = []

        def fake_update(name: str, value: str) -> None:
            calls.append((name, value))

        monkeypatch.setattr(slurm_scripts.jax_config, 'update', fake_update)

        xp = slurm_scripts.SystemConfig(platform=platform).jax_api()

        assert calls == [('jax_platform_name', platform)]
        assert xp is slurm_scripts.JaxNumPy

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

        monkeypatch.setattr(slurm_scripts.Scripts, 'parser',
                            classmethod(lambda cls: FakeParser()))

    def patch_scan(self, monkeypatch: pytest.MonkeyPatch, system: Any) -> None:
        class FakeScan():
            scan_num = 373

        FakeScan.system = system

        monkeypatch.setattr(slurm_scripts.ScanConfig, 'read',
                            classmethod(lambda cls, _: FakeScan()))

    def patch_metadata(self, monkeypatch: pytest.MonkeyPatch,
                       events: list[Event]) -> None:
        class FakeScript():
            def run(self) -> None:
                events.append('run')

        def fake_from_file(scan_file: str, params_file: str) -> FakeScript:
            events.append(('from_file', scan_file, params_file))
            return FakeScript()

        monkeypatch.setattr(slurm_scripts.CreateMetadata, 'from_file',
                            classmethod(lambda cls, scan_file, params_file:
                                        fake_from_file(scan_file, params_file)))

    def patch_refine_parser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeParser():
            def parse_args(self) -> Namespace:
                return Namespace(command='refine',
                                 scan='scan.json',
                                 parameters='refine.json',
                                 suffix='gd',
                                 chunk_id=2,
                                 n_chunks=8)

        monkeypatch.setattr(slurm_scripts.Scripts, 'parser',
                            classmethod(lambda cls: FakeParser()))

    def patch_refine(self, monkeypatch: pytest.MonkeyPatch,
                     events: list[Event]) -> None:
        class FakeScript():
            def run(self) -> None:
                events.append('run')

        def fake_from_file(scan_file: str, params_file: str, suffix: str,
                           chunk_id: int | None, n_chunks: int | None) -> FakeScript:
            events.append(('from_file', scan_file, params_file, suffix,
                           str(chunk_id), str(n_chunks)))
            return FakeScript()

        monkeypatch.setattr(slurm_scripts.RefinementScript, 'from_file',
                            classmethod(lambda cls, scan_file, params_file, suffix,
                                        chunk_id, n_chunks:
                                        fake_from_file(scan_file, params_file, suffix,
                                                       chunk_id, n_chunks)))

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

        slurm_scripts.main()

        assert events == ['apply', ('from_file', 'scan.json', 'params.json'), 'run']

    def test_refine(self, events: list[Event], monkeypatch: pytest.MonkeyPatch):
        class FakeSystem():
            def apply(self) -> None:
                events.append('apply')

        self.patch_refine_parser(monkeypatch)
        self.patch_scan(monkeypatch, FakeSystem())
        self.patch_refine(monkeypatch, events)

        slurm_scripts.main()

        assert events == ['apply', ('from_file', 'scan.json', 'refine.json', 'gd', '2', '8'),
                          'run']

    def test_apply_error(self, events: list[Event], monkeypatch: pytest.MonkeyPatch):
        class FakeSystem():
            def apply(self) -> None:
                raise RuntimeError('allocator setup failed')

        self.patch_main(monkeypatch, events, FakeSystem())

        with pytest.raises(RuntimeError, match='allocator setup failed'):
            slurm_scripts.main()

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

            def scan_subdir(self, dir: str) -> str:
                return str(scan_dir)

            def list_files(self, dir: str) -> Iterator[str]:
                return iter(sorted(str(path) for path in Path(dir).glob('*.h5')))

            def scan_file(self) -> str:
                return 'scan_7_full.h5'

        monkeypatch.setattr(slurm_scripts.ScanConfig, 'read',
                            classmethod(lambda cls, _: FakeScan()))

    def test_compile_all_pandas_tables(self, tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
        hits_dir = tmp_path / 'hits'
        scan_dir = hits_dir / 'scan_7_full'
        scan_dir.mkdir(parents=True)

        self.write_chunk(scan_dir / 'scan_7_full_f0000.h5', 1)
        self.write_chunk(scan_dir / 'scan_7_full_f0001.h5', 2)
        self.patch_scan(monkeypatch, scan_dir, hits_dir)

        slurm_scripts.CompileStreaks('streaks', 'scan.json').run()

        output_path = hits_dir / 'scan_7_full.h5'
        assert pd.read_hdf(output_path, 'data')['index'].tolist() == [1, 2]
        assert pd.read_hdf(output_path, 'metadata')['pulse_id'].tolist() == [21, 22]
        assert pd.read_hdf(output_path, 'scores').tolist() == [31, 32]
