from argparse import Namespace
from typing import Any
import pytest
import cbclib_v2.slurm.scripts as slurm_scripts

Event = str | tuple[str, str, str]

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

    def test_apply_error(self, events: list[Event], monkeypatch: pytest.MonkeyPatch):
        class FakeSystem():
            def apply(self) -> None:
                raise RuntimeError('allocator setup failed')

        self.patch_main(monkeypatch, events, FakeSystem())

        with pytest.raises(RuntimeError, match='allocator setup failed'):
            slurm_scripts.main()

        assert events == []
