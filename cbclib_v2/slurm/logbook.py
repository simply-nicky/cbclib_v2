"""Detection summaries and Google Sheets experiment logging."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar, Sequence, Tuple, cast

import pandas as pd
from .._src.annotations import NumPy
from .._src.scripts import BaseParameters
from .._src.streaks import Streaks
from .config import DetectionAttributes, DetectionKind, DetectionMetadata, HDFKey

@dataclass(frozen=True)
class DetectionLogEntry:
    """Represent one scan-level detection summary.

    The summary combines the detection count stored with each hit's metadata
    with the lengths derived from its saved streak geometry. A frame is a hit
    when its count is strictly greater than the threshold recorded by the
    detection job.
    """
    scan_num             : int
    kind                 : DetectionKind
    result_suffix        : str
    processed_frames     : int
    hits                 : int
    hit_rate             : float
    detections_in_hits   : int
    mean_per_hit         : float
    median_per_hit       : float
    mean_length          : float
    hit_threshold        : int
    sample               : str
    notes                : str
    updated_at           : str

    columns : ClassVar[Tuple[str, ...]] = (
        'Scan number', 'Detection kind', 'Scan suffix', 'No. of frames', 'No. of hits',
        'Hit rate', 'No. of streaks', 'Average No. of streaks',
        'Median No. of streaks', 'Average streak length', 'Hit threshold', 'Sample',
        'Notes', 'Processed at')

    @property
    def key(self) -> Tuple[int, DetectionKind, str]:
        """Return the stable spreadsheet key for this result."""
        return self.scan_num, self.kind, self.result_suffix

    @classmethod
    def from_files(cls, paths: Sequence[str], result_suffix: str=str(),
                   sample: str=str(), notes: str=str(),
                   updated_at: str | None=None) -> 'DetectionLogEntry':
        """Derive one summary from detection artifacts.

        Args:
            paths: Whole-scan artifact or all chunk artifacts for one scan.
            result_suffix: Detection result suffix used to distinguish runs.
            sample: Optional sample description recorded with the summary.
            notes: Optional free-form notes recorded with the summary.
            updated_at: Optional ISO timestamp, primarily for deterministic callers.

        Returns:
            Aggregated scan-level summary.

        Raises:
            ValueError: If artifacts are absent, inconsistent, or contain no streak geometry.
        """
        xp = NumPy

        attributes, metadata = [], []
        total_length = 0.0
        n_streaks = 0
        for path in paths:
            metadata_df = cast(pd.DataFrame, pd.read_hdf(path, key=HDFKey.metadata, mode='r'))
            try:
                streaks_df = cast(pd.DataFrame, pd.read_hdf(path, key=HDFKey.data, mode='r'))
            except KeyError as error:
                raise ValueError(
                    f"Detection artifact does not contain streak geometry: {path}"
                ) from error
            streaks = Streaks.import_dataframe(streaks_df, NumPy)
            attributes.append(DetectionAttributes.read(path))
            metadata.append(DetectionMetadata.import_dataframe(metadata_df, xp))
            total_length += float(streaks.length.sum())
            n_streaks += streaks.size
        mean_length = total_length / n_streaks if n_streaks else 0.0
        return cls.from_detection(
            DetectionAttributes.concat(attributes),
            DetectionMetadata.concat(metadata), mean_length, result_suffix,
            sample, notes, updated_at)

    @classmethod
    def from_detection(cls, attributes: DetectionAttributes,
                       metadata: DetectionMetadata, mean_length: float,
                       result_suffix: str=str(), sample: str=str(), notes: str=str(),
                       updated_at: str | None=None) -> 'DetectionLogEntry':
        """Create a spreadsheet row from detection attributes and hit metadata."""
        timestamp = updated_at or datetime.now(timezone.utc).isoformat(timespec='seconds')
        return cls(
            scan_num=attributes.scan_num, kind=attributes.detection_kind,
            result_suffix=result_suffix, processed_frames=attributes.n_frames,
            hits=metadata.size, hit_rate=attributes.hit_rate(metadata.size),
            detections_in_hits=metadata.detections_in_hits,
            mean_per_hit=metadata.mean_per_hit, median_per_hit=metadata.median_per_hit,
            mean_length=mean_length, hit_threshold=attributes.hit_threshold,
            sample=sample, notes=notes, updated_at=timestamp)

    @classmethod
    def key_from_row(cls, row: Sequence[Any], row_number: int) -> Tuple[int, str, str]:
        """Parse the stable key from an existing worksheet row."""
        if len(row) < 3:
            raise ValueError(f"Incomplete detection log key in row {row_number}")
        return int(row[0]), str(row[1]), str(row[2])

    def to_row(self) -> list[str | int | float]:
        """Return values in :attr:`GoogleSheetsLog.header` order."""
        return [self.scan_num, self.kind, self.result_suffix, self.processed_frames,
                self.hits, self.hit_rate, self.detections_in_hits, self.mean_per_hit,
                self.median_per_hit, self.mean_length, self.hit_threshold,
                self.sample, self.notes, self.updated_at]

@dataclass(frozen=True)
class GoogleSheetsConfig(BaseParameters):
    """Identify the Google Sheet tab and optional credential file."""
    spreadsheet_id     : str
    worksheet          : str
    sort_rows          : bool = True
    credentials_file   : str | None = None

@dataclass
class GoogleSheetsLog:
    """Upsert detection summaries into a dedicated Google Sheet tab."""
    config       : GoogleSheetsConfig
    service      : Any | None = field(default=None, repr=False)

    scope : ClassVar[str] = 'https://www.googleapis.com/auth/spreadsheets'
    datetime_column : ClassVar[int] = DetectionLogEntry.columns.index('Processed at')
    sheets_epoch : ClassVar[datetime] = datetime(1899, 12, 30)

    @property
    def header(self) -> Tuple[str, ...]:
        """Return the worksheet schema owned by a detection log entry."""
        return DetectionLogEntry.columns

    def build_service(self) -> Any:
        """Build a Sheets API client from configured credentials or ADC."""
        try:
            import google.auth  # type: ignore[reportMissingImports]
            from googleapiclient.discovery import build  # type: ignore[reportMissingImports]
        except ImportError as error:
            raise RuntimeError(
                "Google Sheets support requires google-auth and google-api-python-client"
            ) from error

        if self.config.credentials_file is None:
            credentials, _ = google.auth.default(scopes=[self.scope])
        else:
            credentials, _ = google.auth.load_credentials_from_file(
                self.config.credentials_file, scopes=[self.scope])
        return build('sheets', 'v4', credentials=credentials, cache_discovery=False)

    @property
    def spreadsheets(self) -> Any:
        """Return the authenticated spreadsheets resource."""
        service = self.service
        if service is None:
            service = self.build_service()
            self.service = service
        return service.spreadsheets()

    @property
    def values(self) -> Any:
        """Return the authenticated spreadsheet-values resource."""
        return self.spreadsheets.values()

    def range(self, cells: str) -> str:
        """Return an A1 range quoted for the configured worksheet."""
        worksheet = self.config.worksheet.replace("'", "''")
        return f"'{worksheet}'!{cells}"

    def read_rows(self) -> list[list[Any]]:
        """Read all values in the owned worksheet columns."""
        response = self.values.get(
            spreadsheetId=self.config.spreadsheet_id,
            range=self.range('A:N'), valueRenderOption='UNFORMATTED_VALUE',
            dateTimeRenderOption='SERIAL_NUMBER').execute()
        return response.get('values', [])

    @staticmethod
    def is_blank(rows: Sequence[Sequence[Any]]) -> bool:
        """Return whether the worksheet contains only empty or whitespace cells."""
        return all(isinstance(value, str) and not value.strip()
                   for row in rows for value in row)

    def initialise(self, rows: list[list[Any]]) -> list[list[Any]]:
        """Create or validate the worksheet header."""
        if self.is_blank(rows):
            self.values.update(
                spreadsheetId=self.config.spreadsheet_id,
                range=self.range('A1:N1'), valueInputOption='RAW',
                body={'values': [list(self.header)]}).execute()
            self.format_worksheet()
            return [list(self.header)]
        if tuple(rows[0]) != self.header:
            raise ValueError("Google Sheet header does not match the detection log schema")
        return rows

    def existing_rows(self, rows: Sequence[Sequence[Any]]
                      ) -> dict[Tuple[int, str, str], list[Any]]:
        """Index existing data rows by their stable detection key."""
        existing: dict[Tuple[int, str, str], list[Any]] = {}
        for row_number, row in enumerate(rows[1:], start=2):
            if not row:
                continue
            key = DetectionLogEntry.key_from_row(row, row_number)
            if key in existing:
                raise ValueError(f"Duplicate detection log key in row {row_number}: {key}")
            existing[key] = list(row)
        return existing

    def merge_rows(self, entries: Sequence[DetectionLogEntry],
                   existing: dict[Tuple[int, str, str], list[Any]]
                   ) -> list[list[Any]]:
        """Upsert entries and return rows sorted by scan, kind, and suffix."""
        merged = dict(existing)
        for entry in entries:
            merged[entry.key] = entry.to_row()
        if self.config.sort_rows:
            return [merged[key] for key in sorted(merged)]
        return list(merged.values())

    def write_rows(self, rows: Sequence[Sequence[Any]]):
        """Replace the machine-owned worksheet body with sorted rows."""
        if rows:
            end = len(rows) + 1
            self.values.update(
                spreadsheetId=self.config.spreadsheet_id,
                range=self.range(f'A2:N{end}'), valueInputOption='RAW',
                body={'values': self.to_sheet_rows(rows)}).execute()

    @classmethod
    def datetime_serial(cls, value: str | int | float) -> int | float:
        """Convert an ISO timestamp to the numeric date-time representation used by Sheets."""
        if not isinstance(value, str):
            return value
        timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        return (timestamp - cls.sheets_epoch).total_seconds() / 86400.0

    @classmethod
    def to_sheet_rows(cls, rows: Sequence[Sequence[Any]]) -> list[list[Any]]:
        """Return rows with timestamps encoded as Google Sheets date-time values."""
        result = [list(row) for row in rows]
        for row in result:
            if len(row) > cls.datetime_column:
                row[cls.datetime_column] = cls.datetime_serial(row[cls.datetime_column])
        return result

    def worksheet_id(self) -> int:
        """Return the numeric ID of the configured worksheet tab."""
        response = self.spreadsheets.get(
            spreadsheetId=self.config.spreadsheet_id,
            fields='sheets(properties(sheetId,title))').execute()
        for sheet in response.get('sheets', []):
            properties = sheet.get('properties', {})
            if properties.get('title') == self.config.worksheet:
                return int(properties['sheetId'])
        raise ValueError(f"Google Sheet worksheet not found: {self.config.worksheet}")

    def format_worksheet(self):
        """Format the timestamp column as date-time values."""
        sheet_id = self.worksheet_id()
        self.spreadsheets.batchUpdate(
            spreadsheetId=self.config.spreadsheet_id,
            body={'requests': [
                {'repeatCell': {
                    'range': {
                        'sheetId': sheet_id,
                        'startRowIndex': 1,
                        'startColumnIndex': self.datetime_column,
                        'endColumnIndex': self.datetime_column + 1},
                    'cell': {'userEnteredFormat': {'numberFormat': {
                        'type': 'DATE_TIME', 'pattern': 'yyyy-MM-dd hh:mm:ss'}}},
                    'fields': 'userEnteredFormat.numberFormat'}}
            ]}).execute()

    def upsert(self, entries: Sequence[DetectionLogEntry]) -> int:
        """Insert new scan summaries and replace rows with matching keys.

        Args:
            entries: Scan summaries to publish.

        Returns:
            Number of rows inserted or updated.

        Raises:
            ValueError: If the worksheet header or existing keys are ambiguous.
        """
        if not entries:
            return 0
        if len({entry.key for entry in entries}) != len(entries):
            raise ValueError("Detection log entries contain duplicate keys")
        rows = self.initialise(self.read_rows())
        merged = self.merge_rows(entries, self.existing_rows(rows))
        self.write_rows(merged)
        return len(entries)
