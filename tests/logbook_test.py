from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
import pytest
from cbclib_v2.annotations import NumPy, NumPyNamespace
from cbclib_v2.slurm import (DetectionAttributes, DetectionLogEntry, DetectionMetadata,
                             GoogleSheetsConfig, GoogleSheetsLog)

class FakeRequest:
    def __init__(self, response: dict[str, Any] | None=None):
        self.response = response or {}

    def execute(self) -> dict[str, Any]:
        return self.response

class FakeValues:
    def __init__(self, rows: list[list[Any]]):
        self.rows = rows
        self.gets: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.batch_updates: list[dict[str, Any]] = []
        self.appends: list[dict[str, Any]] = []

    def get(self, **kwargs: Any) -> FakeRequest:
        self.gets.append(kwargs)
        return FakeRequest({'values': self.rows})

    def update(self, **kwargs: Any) -> FakeRequest:
        self.updates.append(kwargs)
        return FakeRequest()

    def batchUpdate(self, **kwargs: Any) -> FakeRequest:
        self.batch_updates.append(kwargs)
        return FakeRequest()

    def append(self, **kwargs: Any) -> FakeRequest:
        self.appends.append(kwargs)
        return FakeRequest()

class FakeService:
    def __init__(self, values: FakeValues):
        self._values = values
        self.batch_updates: list[dict[str, Any]] = []

    def spreadsheets(self) -> 'FakeService':
        return self

    def values(self) -> FakeValues:
        return self._values

    def get(self, **kwargs: Any) -> FakeRequest:
        return FakeRequest({'sheets': [{'properties': {
            'sheetId': 17, 'title': 'CBC log'}}]})

    def batchUpdate(self, **kwargs: Any) -> FakeRequest:
        self.batch_updates.append(kwargs)
        return FakeRequest()

class TestDetectionLogEntry:
    def write_result(self, path: Path, chunk_id: int, counts: list[int],
                     xp: NumPyNamespace) -> None:
        frames = xp.arange(10 * chunk_id, 10 * chunk_id + len(counts))
        hit_mask = xp.asarray(counts) > 2
        hit_frames = frames[hit_mask]
        hit_counts = xp.asarray(counts)[hit_mask]
        attributes = DetectionAttributes(7, 'streaks', 2, chunk_id, 2, len(counts))
        metadata = DetectionMetadata(
            index=hit_frames,
            filename=xp.full((hit_frames.size, 1), 'input.h5'),
            file_index=hit_frames[:, None], pulse_id=hit_frames,
            n_detections=hit_counts)
        metadata.to_dataframe().to_hdf(path, key='metadata', mode='w')
        streak_data = [
            {'index': frame, 'x_0': 0.0, 'y_0': 0.0, 'x_1': float(count), 'y_1': 0.0}
            for frame, count in zip(hit_frames, hit_counts)
            for _ in range(count)
        ]
        pd.DataFrame(
            streak_data, columns=['index', 'x_0', 'y_0', 'x_1', 'y_1']
        ).to_hdf(path, key='data', mode='a')
        attributes.write(str(path))

    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def chunk_paths(self, tmp_path: Path, xp: NumPyNamespace) -> list[Path]:
        paths = [tmp_path / 'chunk_0.h5', tmp_path / 'chunk_1.h5']
        self.write_result(paths[0], 0, [0, 3, 5], xp)
        self.write_result(paths[1], 1, [2, 4], xp)
        return paths

    @pytest.fixture
    def entry(self, chunk_paths: list[Path]) -> DetectionLogEntry:
        return DetectionLogEntry.from_files(
            [str(path) for path in chunk_paths], 'online', 'lysozyme', 'alignment check',
            '2026-09-24T12:00:00+00:00')

    @pytest.fixture
    def zero_hit_paths(self, tmp_path: Path, xp: NumPyNamespace) -> list[Path]:
        paths = [tmp_path / 'chunk_0.h5', tmp_path / 'chunk_1.h5']
        self.write_result(paths[0], 0, [0, 1], xp)
        self.write_result(paths[1], 1, [2, 0], xp)
        return paths

    @pytest.fixture
    def zero_hit_entry(self, zero_hit_paths: list[Path]) -> DetectionLogEntry:
        return DetectionLogEntry.from_files([str(path) for path in zero_hit_paths])

    @pytest.fixture
    def empty_metadata(self, xp: NumPyNamespace) -> DetectionMetadata:
        return DetectionMetadata(
            index=xp.zeros(0, dtype=int), filename=xp.empty((0, 0), dtype=str),
            file_index=xp.empty((0, 0), dtype=int), pulse_id=xp.zeros(0, dtype=int),
            n_detections=xp.zeros(0, dtype=int))

    @pytest.fixture
    def frames_only_path(self, tmp_path: Path,
                         empty_metadata: DetectionMetadata) -> Path:
        path = tmp_path / 'frames_only.h5'
        empty_metadata.to_dataframe().to_hdf(path, key='metadata', mode='w')
        DetectionAttributes(7, 'streaks', 2, 0, 1, 4).write(str(path))
        return path

    def test_aggregates_complete_chunks(self, chunk_paths: list[Path],
                                        entry: DetectionLogEntry, xp: NumPyNamespace):
        attributes = DetectionAttributes.read(str(chunk_paths[0]))

        # Chunk attributes retain the frame count and identity written for that chunk.
        assert attributes == DetectionAttributes(7, 'streaks', 2, 0, 2, 3)
        dataframe = pd.read_hdf(chunk_paths[0], key='metadata')
        metadata = DetectionMetadata.import_dataframe(dataframe, xp)
        selected = metadata[metadata.n_detections > 3]
        # Source addresses remain aligned when the inherited container indexing is used.
        assert selected.shape == (1,)
        assert selected.filename.tolist() == [['input.h5']]
        assert selected.file_index.tolist() == [[2]]

        # The threshold is strict, so counts of two are not hits.
        assert entry.key == (7, 'streaks', 'online')
        assert entry.processed_frames == 5
        assert entry.hits == 3
        assert entry.hit_rate == pytest.approx(3 / 5)
        assert entry.detections_in_hits == 12
        assert entry.mean_per_hit == 4.0
        assert entry.median_per_hit == 4.0
        # Each streak contributes its hit's count as its length.
        assert entry.mean_length == pytest.approx((3**2 + 5**2 + 4**2) / 12)
        assert entry.sample == 'lysozyme'
        assert entry.notes == 'alignment check'

    def test_zero_hits(self, zero_hit_entry: DetectionLogEntry):
        # A completed zero-hit scan remains distinguishable from missing output.
        assert zero_hit_entry.processed_frames == 4
        assert zero_hit_entry.hits == 0
        assert zero_hit_entry.hit_rate == 0.0
        assert zero_hit_entry.mean_per_hit == 0.0
        assert zero_hit_entry.mean_length == 0.0

    def test_requires_streak_geometry(self, frames_only_path: Path):
        # A scan summary requires saved geometry from which streak lengths can be derived.
        with pytest.raises(ValueError, match='does not contain streak geometry'):
            DetectionLogEntry.from_files([str(frames_only_path)])

class TestGoogleSheetsLog:
    @pytest.fixture
    def google_dependencies(self) -> None:
        pytest.importorskip('google.auth')
        pytest.importorskip('googleapiclient.discovery')

    def sheet_row(self, entry: DetectionLogEntry) -> list[str | int | float]:
        row = entry.to_row()
        timestamp = datetime.fromisoformat(entry.updated_at).replace(tzinfo=None)
        epoch = datetime(1899, 12, 30)
        row[-1] = (timestamp - epoch).total_seconds() / 86400.0
        return row

    @pytest.fixture
    def entry(self) -> DetectionLogEntry:
        return DetectionLogEntry(7, 'streaks', 'online', 10, 2, 0.2, 9,
                                 4.5, 4.5, 12.5, 2, 'lysozyme', 'alignment check',
                                 '2026-09-24T12:00:00+00:00')

    @pytest.fixture
    def config(self) -> GoogleSheetsConfig:
        return GoogleSheetsConfig('spreadsheet-id', "CBC log")

    @pytest.fixture
    def empty_values(self) -> FakeValues:
        return FakeValues([])

    @pytest.fixture
    def empty_service(self, empty_values: FakeValues) -> FakeService:
        return FakeService(empty_values)

    @pytest.fixture
    def logger(self, config: GoogleSheetsConfig,
               empty_service: FakeService) -> GoogleSheetsLog:
        return GoogleSheetsLog(config, empty_service)

    @pytest.fixture
    def whitespace_values(self) -> FakeValues:
        return FakeValues([[' ']])

    @pytest.fixture
    def whitespace_logger(self, config: GoogleSheetsConfig,
                          whitespace_values: FakeValues) -> GoogleSheetsLog:
        return GoogleSheetsLog(config, FakeService(whitespace_values))

    @pytest.fixture
    def existing_values(self, entry: DetectionLogEntry) -> FakeValues:
        return FakeValues([list(DetectionLogEntry.columns), entry.to_row()])

    @pytest.fixture
    def existing_service(self, existing_values: FakeValues) -> FakeService:
        return FakeService(existing_values)

    @pytest.fixture
    def existing_logger(self, config: GoogleSheetsConfig,
                        existing_service: FakeService) -> GoogleSheetsLog:
        return GoogleSheetsLog(config, existing_service)

    @pytest.fixture
    def later_entry(self) -> DetectionLogEntry:
        return DetectionLogEntry(105, 'streaks', 'online', 10, 2, 0.2, 9,
                                 4.5, 4.5, 12.5, 2, 'sample', 'notes',
                                 '2026-09-24T12:00:00+00:00')

    @pytest.fixture
    def earlier_entry(self) -> DetectionLogEntry:
        return DetectionLogEntry(103, 'streaks', 'online', 10, 1, 0.1, 4,
                                 4.0, 4.0, 8.0, 2, 'sample', 'notes',
                                 '2026-09-24T12:00:00+00:00')

    @pytest.fixture
    def ordered_values(self, later_entry: DetectionLogEntry) -> FakeValues:
        return FakeValues([list(DetectionLogEntry.columns), later_entry.to_row()])

    @pytest.fixture
    def ordered_logger(self, config: GoogleSheetsConfig,
                       ordered_values: FakeValues) -> GoogleSheetsLog:
        return GoogleSheetsLog(config, FakeService(ordered_values))

    @pytest.fixture
    def unsorted_logger(self, ordered_values: FakeValues) -> GoogleSheetsLog:
        config = GoogleSheetsConfig('spreadsheet-id', 'CBC log', sort_rows=False)
        return GoogleSheetsLog(config, FakeService(ordered_values))

    def test_builds_service_from_configured_credentials(
            self, monkeypatch: pytest.MonkeyPatch, google_dependencies: None):
        credentials = object()
        service = object()
        calls: list[tuple[Any, ...]] = []

        def load_credentials(path: str, scopes: list[str]) -> tuple[object, None]:
            calls.append(('load', path, scopes))
            return credentials, None

        def build(api: str, version: str, **kwargs: Any) -> object:
            calls.append(('build', api, version, kwargs))
            return service

        monkeypatch.setattr('google.auth.load_credentials_from_file', load_credentials)
        monkeypatch.setattr('googleapiclient.discovery.build', build)
        config = GoogleSheetsConfig(
            'spreadsheet-id', 'CBC log', credentials_file='/secure/credentials.json')

        # An explicit credential file takes precedence and is passed to the Sheets client.
        assert GoogleSheetsLog(config).build_service() is service
        assert calls == [
            ('load', '/secure/credentials.json', [GoogleSheetsLog.scope]),
            ('build', 'sheets', 'v4', {
                'credentials': credentials, 'cache_discovery': False})]

    def test_builds_service_from_adc_by_default(self, config: GoogleSheetsConfig,
                                                monkeypatch: pytest.MonkeyPatch,
                                                google_dependencies: None):
        credentials = object()
        service = object()
        calls: list[tuple[Any, ...]] = []

        def default(scopes: list[str]) -> tuple[object, None]:
            calls.append(('default', scopes))
            return credentials, None

        def build(api: str, version: str, **kwargs: Any) -> object:
            calls.append(('build', api, version, kwargs))
            return service

        monkeypatch.setattr('google.auth.default', default)
        monkeypatch.setattr('googleapiclient.discovery.build', build)

        # Without a credential file the Sheets client is built from application defaults.
        assert GoogleSheetsLog(config).build_service() is service
        assert calls == [
            ('default', [GoogleSheetsLog.scope]),
            ('build', 'sheets', 'v4', {
                'credentials': credentials, 'cache_discovery': False})]

    def test_initialises_and_writes(self, entry: DetectionLogEntry,
                                   empty_values: FakeValues,
                                   empty_service: FakeService,
                                   logger: GoogleSheetsLog):
        # An empty worksheet receives its schema, formatting, and first data row.
        assert logger.upsert([entry]) == 1

        assert empty_values.updates[0]['body']['values'] == [list(logger.header)]
        assert empty_values.updates[1]['body']['values'] == [self.sheet_row(entry)]
        assert empty_values.appends == []
        assert empty_values.batch_updates == []
        assert empty_values.gets[0]['valueRenderOption'] == 'UNFORMATTED_VALUE'
        assert empty_values.gets[0]['dateTimeRenderOption'] == 'SERIAL_NUMBER'
        assert empty_values.gets[0]['range'] == "'CBC log'!A:N"
        requests = empty_service.batch_updates[0]['body']['requests']
        assert requests[0]['repeatCell']['cell']['userEnteredFormat']['numberFormat'] == {
            'type': 'DATE_TIME', 'pattern': 'yyyy-MM-dd hh:mm:ss'}
        assert len(requests) == 1

    def test_initialises_whitespace_only_sheet(
            self, entry: DetectionLogEntry, whitespace_values: FakeValues,
            whitespace_logger: GoogleSheetsLog):
        # Whitespace-only cells are treated as an empty worksheet.
        assert whitespace_logger.upsert([entry]) == 1

        assert whitespace_values.updates[0]['body']['values'] == [
            list(whitespace_logger.header)]
        assert whitespace_values.updates[1]['body']['values'] == [self.sheet_row(entry)]

    def test_replaces_existing_key(self, entry: DetectionLogEntry,
                                   existing_values: FakeValues,
                                   existing_service: FakeService,
                                   existing_logger: GoogleSheetsLog):
        existing_logger.upsert([entry])

        # A matching stable key is rewritten in place without append or reformat requests.
        assert existing_values.appends == []
        assert existing_values.batch_updates == []
        assert existing_service.batch_updates == []
        assert existing_values.updates[0]['range'] == "'CBC log'!A2:N2"
        assert existing_values.updates[0]['body']['values'] == [self.sheet_row(entry)]

    def test_inserts_by_scan_number(self, earlier_entry: DetectionLogEntry,
                                    later_entry: DetectionLogEntry,
                                    ordered_values: FakeValues,
                                    ordered_logger: GoogleSheetsLog):
        ordered_logger.upsert([earlier_entry])

        # The numeric scan key is primary; kind and suffix break ties.
        assert ordered_values.updates[0]['body']['values'] == [
            self.sheet_row(earlier_entry), self.sheet_row(later_entry)]

    def test_preserves_order_when_sorting_disabled(
            self, earlier_entry: DetectionLogEntry, later_entry: DetectionLogEntry,
            ordered_values: FakeValues, unsorted_logger: GoogleSheetsLog):
        unsorted_logger.upsert([earlier_entry])

        # Disabling sorting preserves existing rows before newly inserted rows.
        assert ordered_values.updates[0]['body']['values'] == [
            self.sheet_row(later_entry), self.sheet_row(earlier_entry)]
