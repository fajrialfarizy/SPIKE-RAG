"""
Unit tests for GoogleSheetsRepo and GoogleDriveRepo using mocks.
Verifies that only allowed fields pass repo boundaries and raw rows/personal data are not leaked.
"""

from unittest.mock import MagicMock, patch
import pytest
from enums import CompetitionType, AchievementLevel, RagStatus
from ingestion.repos.google import GoogleSheetsRepo, GoogleDriveRepo


@pytest.fixture
def mock_creds():
    return {
        "type": "service_account",
        "project_id": "mock-proj",
        "private_key_id": "key123",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQD...\n-----END PRIVATE KEY-----\n",
        "client_email": "test@mock-proj.iam.gserviceaccount.com",
    }


def test_google_sheets_repo_privacy_and_columns(mock_creds):
    with patch("gspread.authorize") as mock_auth, patch(
        "google.oauth2.service_account.Credentials.from_service_account_info"
    ):
        mock_client = MagicMock()
        mock_auth.return_value = mock_client
        mock_spreadsheet = MagicMock()
        mock_client.open_by_key.return_value = mock_spreadsheet

        # Mock competition_reports worksheet with personal data included
        ws_reports = MagicMock()
        ws_reports.get_all_records.return_value = [
            {
                "report_id": "RPT0001",
                "reporter_nim": "13520001",  # Personal
                "reporter_name": "Budi Rahasia",  # Personal
                "reporter_phone": "081234567890",  # Personal
                "mentor_name": "Dosen X",  # Personal
                "competition_name": "Lomba Cloud",
                "competition_year": "2024",
                "achievement_level": AchievementLevel.JUARA_1.value,
                "competition_type": CompetitionType.BUSINESS_PLAN.value,
            }
        ]

        # Mock report_documents worksheet (strictly 3 columns)
        ws_docs = MagicMock()
        ws_docs.get_all_records.return_value = [
            {
                "report_id": "RPT0001",
                "drive_file_id": "DRV_RPT0001",
                "rag_status": RagStatus.BELUM_DIPROSES.value,
            }
        ]

        def get_worksheet(name):
            if name == "competition_reports":
                return ws_reports
            elif name == "report_documents":
                return ws_docs
            raise ValueError(f"Unknown worksheet {name}")

        mock_spreadsheet.worksheet.side_effect = get_worksheet

        repo = GoogleSheetsRepo("sheet123", mock_creds)

        # Test competition reports
        reports = repo.get_competition_reports()
        assert len(reports) == 1
        r = reports[0]
        assert r.report_id == "RPT0001"
        assert r.competition_year == 2024
        assert not hasattr(r, "reporter_nim")
        assert not hasattr(r, "reporter_name")
        assert not hasattr(r, "reporter_phone")

        # Test report documents
        docs = repo.get_report_documents()
        assert len(docs) == 1
        assert docs[0].report_id == "RPT0001"
        assert docs[0].rag_status == RagStatus.BELUM_DIPROSES

        # Test update rag status
        mock_cell = MagicMock()
        mock_cell.row = 2
        ws_docs.find.return_value = mock_cell
        ok = repo.update_rag_status("RPT0001", RagStatus.SELESAI)
        assert ok is True
        ws_docs.update_cell.assert_called_with(2, 3, "Selesai")


def test_google_drive_repo_mocked_calls(mock_creds, tmp_path):
    with patch("googleapiclient.discovery.build") as mock_build, patch(
        "google.oauth2.service_account.Credentials.from_service_account_info"
    ):
        mock_service = MagicMock()
        mock_build.return_value = mock_service

        drive_repo = GoogleDriveRepo("folder123", mock_creds)

        # Test upload artifact
        mock_file = tmp_path / "test_artifact.zip"
        mock_file.write_text("dummy artifact content")

        mock_service.files().create().execute.return_value = {"id": "uploaded_id_999"}
        file_id = drive_repo.upload_artifact(str(mock_file), "test_artifact.zip")
        assert file_id == "uploaded_id_999"
