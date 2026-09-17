from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import pytest
from cbclib_v2._src import run as run_module
from cbclib_v2._src.annotations import NDArray
from cbclib_v2._src.cxi_protocol import LoadWorker, StackIndex, StackIndices
from cbclib_v2._src.run import BaseRun, RunConfig, RunList, RunListIndices

@dataclass
class ValueWorker(LoadWorker[NDArray]):
    run_id      : int
    offset      : int = 0

    def __call__(self, index: Any) -> NDArray:
        _, frame_index = index
        return np.asarray(self.offset + 10 * self.run_id + frame_index)

@dataclass
class StubRun:
    run_id      : int
    config      : RunConfig
    n_frames    : int

    def attributes(self) -> List[str]:
        return ['pulse_id']

    def indices(self) -> StackIndices:
        return StackIndices([StackIndex(f'run_{self.run_id}.h5', self.n_frames)])

    def meta_worker(self, attr: str) -> ValueWorker:
        assert attr == 'pulse_id'
        return ValueWorker(self.run_id, offset=100)

    def worker(self, geometry: bool = False) -> ValueWorker:
        return ValueWorker(self.run_id)

class TestRunList:
    @pytest.fixture
    def config(self) -> RunConfig:
        return RunConfig(facility='SwissFEL')

    @pytest.fixture
    def child_runs(self, config: RunConfig) -> Dict[int, StubRun]:
        return {
            2: StubRun(2, config, 2),
            3: StubRun(3, config, 3),
        }

    @pytest.fixture
    def run_list(self, monkeypatch: pytest.MonkeyPatch, config: RunConfig,
                 child_runs: Dict[int, StubRun]) -> RunList:
        original_open_run = run_module.open_run

        def open_stub(run_id: int | range | List[int], config: RunConfig,
                      *, variant: str | None = None) -> Any:
            if isinstance(run_id, int):
                return child_runs[run_id]
            return original_open_run(run_id, config, variant=variant)

        monkeypatch.setattr(run_module, 'open_run', open_stub)
        runs = open_stub(range(2, 4), config)
        assert isinstance(runs, RunList)
        return runs

    def test_indices_flatten_runs(self, run_list: RunList):
        indices = run_list.indices()

        assert isinstance(indices, RunListIndices)
        assert list(indices.index()) == list(range(5))
        assert list(indices) == [
            (0, ('run_2.h5', 0)),
            (0, ('run_2.h5', 1)),
            (1, ('run_3.h5', 0)),
            (1, ('run_3.h5', 1)),
            (1, ('run_3.h5', 2)),
        ]

    def test_indices_slice_across_run_boundary(self, run_list: RunList):
        indices = run_list.indices()[1:4]

        # Global positions remain stable while native indices restart for each run.
        assert list(indices.index()) == [1, 2, 3]
        assert list(indices) == [
            (0, ('run_2.h5', 1)),
            (1, ('run_3.h5', 0)),
            (1, ('run_3.h5', 1)),
        ]
        assert [record.index for record in indices.records()] == [1, 2, 3]

    def test_indices_preserve_reordered_selection(self, run_list: RunList):
        indices = run_list.indices()[[4, 0, 3, 1]]

        assert indices.offsets == [0, 2, 5]
        assert list(indices) == [
            (1, ('run_3.h5', 2)),
            (0, ('run_2.h5', 0)),
            (1, ('run_3.h5', 1)),
            (0, ('run_2.h5', 1)),
        ]
        assert [record.index for record in indices.records()] == [4, 0, 3, 1]

    def test_loaders_dispatch_to_child_runs(self, run_list: RunList):
        indices = run_list.indices()[1:4]

        assert np.array_equal(run_list.data(indices, verbose=False), np.asarray([21, 30, 31]))
        assert np.array_equal(run_list.metadata('pulse_id', indices),
                              np.asarray([121, 130, 131]))
        assert [frame.item() for frame in run_list.visit_frames(indices)] == [21, 30, 31]

    def test_parallel_loader_dispatches_to_child_runs(self, run_list: RunList):
        data = run_list.data(run_list.indices()[1:4], n_processes=2, verbose=False)

        assert np.array_equal(data, np.asarray([21, 30, 31]))

    def test_exposes_base_run_interface(self, run_list: RunList):
        assert isinstance(run_list, BaseRun)
        assert run_list.run_id == [2, 3]
        assert run_list.attributes() == ['pulse_id']
        assert [run.run_id for run in run_list.runs] == [2, 3]

    def test_rejects_empty_collection(self, config: RunConfig):
        with pytest.raises(ValueError, match="requires at least one run ID"):
            RunList([], config)
