"""
Unit tests for GoogleSheetsRepo and GoogleDriveRepo using mocks.

The central assertion is the repository boundary (Brief 5.1): only the required
columns are fetched, PII columns are never requested, and raw rows never reach
the logs.
"""

from unittest.mock import MagicMock, patch
import pytest

from enums import AchievementLevel, RagStatus
from ingestion.repos.google import GoogleSheetsRepo, GoogleDriveRepo


def column_index(letter: str) -> int:
    index = 0
    for char in letter:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index


class FakeWorksheet:
    """Minimal worksheet double that records exactly which ranges are requested."""

    def __init__(self, title: str, header, rows):
        self.title = title
        self._header = header
        self._rows = rows
        self.requested_ranges = []
        self.updated_cells = []
        self.find_calls = []

    def row_values(self, row_number):
        assert row_number == 1, "only the header row may be read as a full row"
        return list(self._header)

    def col_values(self, column_number):
        return [self._header[column_number - 1]] + [
            r[column_number - 1] for r in self._rows
        ]

    def batch_get(self, ranges, value_render_option="FORMATTED_VALUE"):
        self.requested_ranges = list(ranges)
        values = []
        for rng in ranges:
            start = rng.split(":")[0]
            column = column_index("".join(ch for ch in start if ch.isalpha()))
            values.append([r[column - 1] for r in self._rows])
        return values

    def get_all_records(self, *args, **kwargs):
        raise AssertionError(
            "get_all_records() must never be used: it fetches every column, including PII"
        )

    def find(self, query, in_column=None):
        self.find_calls.append((query, in_column))
        for offset, row in enumerate(self._rows):
            if row[0] == query:
                cell = MagicMock()
                cell.row = offset + 2
                return cell
        return None

    def update_cell(self, row, column, value):
        self.updated_cells.append((row, column, value))


COMPETITION_HEADER = [
    "report_id",
    "reporter_nim",
    "reporter_name",
    "reporter_phone",
    "mentor_name",
    "team_biodata_file_id",
    "participant_photo_file_id",
    "competition_name",
    "competition_year",
    "achievement_level",
    "competition_type",
    "budget_requested",
    "budget_request_id",
    "lpj_file_id",
    "consent_status",
    "submitted_at",
]

COMPETITION_ROWS = [
    [
        "RPT0001",
        "13520001",
        "Budi Rahasia",
        "081234567890",
        "Dosen X",
        "BIO_1",
        "FOTO_1",
        "Lomba Cloud",
        2024,
        AchievementLevel.JUARA_1.value,
        "Business Plan",
        "No",
        "",
        "",
        "Setuju",
        "2024-11-15T10:00:00Z",
    ]
]


@pytest.fixture
def mock_creds():
    return {
        "type": "service_account",
        "project_id": "mock-proj",
        "private_key_id": "key123",
        "private_key": "-----BEGIN PRIVATE KEY-----\nmock\n-----END PRIVATE KEY-----\n",
        "client_email": "test@mock-proj.iam.gserviceaccount.com",
    }


def _build_repo(mock_creds, sheets):
    repo = GoogleSheetsRepo("sheet123", mock_creds)
    spreadsheet = MagicMock()
    spreadsheet.worksheet.side_effect = lambda name: sheets[name]
    repo._spreadsheet = spreadsheet
    return repo


def test_google_sheets_repo_fetches_only_allowed_columns(mock_creds, caplog):
    reports_ws = FakeWorksheet("competition_reports", COMPETITION_HEADER, COMPETITION_ROWS)
    docs_ws = FakeWorksheet(
        "report_documents",
        ["report_id", "drive_file_id", "rag_status"],
        [["RPT0001", "DRV_RPT0001.pdf", RagStatus.BELUM_DIPROSES.value]],
    )

    with patch("gspread.authorize"), patch(
        "google.oauth2.service_account.Credentials.from_service_account_info"
    ):
        repo = _build_repo(
            mock_creds,
            {"competition_reports": reports_ws, "report_documents": docs_ws},
        )

        with caplog.at_level("DEBUG"):
            reports = repo.get_competition_reports()
            docs = repo.get_report_documents()

    # Only the five allowed columns of competition_reports were requested.
    assert reports_ws.requested_ranges == ["A2:A2", "H2:H2", "I2:I2", "J2:J2", "K2:K2"]
    # Only the three allowed columns of report_documents were requested.
    assert docs_ws.requested_ranges == ["A2:A2", "B2:B2", "C2:C2"]

    # PII column letters (B-G) are never requested.
    pii_letters = {"B", "C", "D", "E", "F", "G"}
    requested_letters = {
        "".join(ch for ch in rng.split(":")[0] if ch.isalpha())
        for rng in reports_ws.requested_ranges
    }
    assert requested_letters.isdisjoint(pii_letters)

    # Model boundary: only allowed fields reach the pipeline.
    assert len(reports) == 1
    r = reports[0]
    assert r.report_id == "RPT0001"
    assert str(r.competition_year) == "2024"
    assert r.achievement_level == AchievementLevel.JUARA_1.value
    assert not hasattr(r, "reporter_nim")
    assert not hasattr(r, "reporter_name")
    assert not hasattr(r, "reporter_phone")
    assert not hasattr(r, "mentor_name")

    assert len(docs) == 1
    assert docs[0].report_id == "RPT0001"
    assert docs[0].rag_status == RagStatus.BELUM_DIPROSES

    # No PII value appears in any log record.
    logged = "\n".join(record.getMessage() for record in caplog.records)
    for pii in ("13520001", "Budi Rahasia", "081234567890", "Dosen X", "Setuju"):
        assert pii not in logged


def test_google_sheets_repo_updates_status_column(mock_creds):
    docs_ws = FakeWorksheet(
        "report_documents",
        ["report_id", "drive_file_id", "rag_status"],
        [["RPT0001", "DRV_RPT0001.pdf", RagStatus.BELUM_DIPROSES.value]],
    )

    with patch("gspread.authorize"), patch(
        "google.oauth2.service_account.Credentials.from_service_account_info"
    ):
        repo = _build_repo(mock_creds, {"report_documents": docs_ws})
        ok = repo.update_rag_status("RPT0001", RagStatus.SELESAI)

    assert ok is True
    assert docs_ws.find_calls == [("RPT0001", 1)]
    assert docs_ws.updated_cells == [(2, 3, RagStatus.SELESAI.value)]


def test_google_sheets_repo_rejects_missing_required_column(mock_creds):
    broken = FakeWorksheet(
        "report_documents", ["report_id", "drive_file_id"], [["RPT0001", "DRV"]]
    )

    with patch("gspread.authorize"), patch(
        "google.oauth2.service_account.Credentials.from_service_account_info"
    ):
        repo = _build_repo(mock_creds, {"report_documents": broken})
        with pytest.raises(ValueError, match="missing required columns"):
            repo.get_report_documents()


def test_google_drive_repo_mocked_calls(mock_creds, tmp_path):
    with patch("googleapiclient.discovery.build") as mock_build, patch(
        "google.oauth2.service_account.Credentials.from_service_account_info"
    ):
        mock_service = MagicMock()
        mock_build.return_value = mock_service

        drive_repo = GoogleDriveRepo("folder123", mock_creds)

        mock_file = tmp_path / "test_artifact.zip"
        mock_file.write_text("dummy artifact content")

        mock_service.files().create().execute.return_value = {"id": "uploaded_id_999"}
        file_id = drive_repo.upload_artifact(str(mock_file), "test_artifact.zip")
        assert file_id == "uploaded_id_999"
