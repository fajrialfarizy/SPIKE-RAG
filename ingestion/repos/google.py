"""
Google Cloud implementation of SheetsRepo and DriveRepo for Production.
Connects via gspread and google-api-python-client with minimal read/write scopes.

Column selection happens at the repository boundary (Brief 5.1): only the five
RAG-allowed columns of competition_reports and the three columns of
report_documents are ever fetched from Google Sheets. Personal/administrative
columns are never downloaded into the process and raw rows are never logged.
"""

import io
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence
import logging

import config
from enums import CompetitionType, AchievementLevel, RagStatus
from ingestion.repos.base import (
    SheetsRepo,
    DriveRepo,
    CompetitionReportMetadata,
    ReportDocumentRow,
)

logger = logging.getLogger(__name__)


class DriveArtifactError(RuntimeError):
    """Raised when the artifact cannot be resolved to exactly one Drive file."""


def column_letter(index_1based: int) -> str:
    """Convert a 1-based column index to its spreadsheet letter (1 -> A, 27 -> AA)."""
    if index_1based < 1:
        raise ValueError(f"Column index must be >= 1, got {index_1based}")
    letters = ""
    index = index_1based
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


class GoogleSheetsRepo(SheetsRepo):
    """
    Production implementation of SheetsRepo using gspread.

    `worksheet.batch_get` is used with one range per required column so that
    PII columns are never requested. `get_all_records()` is deliberately unused
    because it would fetch every column of the worksheet.
    """

    def __init__(self, spreadsheet_id: str, credentials_dict: Dict[str, Any]):
        self.spreadsheet_id = spreadsheet_id
        self.credentials_dict = credentials_dict
        self._client = None
        self._spreadsheet = None

    def _get_spreadsheet(self):
        if self._spreadsheet is None:
            import gspread
            from google.oauth2.service_account import Credentials

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
            ]
            creds = Credentials.from_service_account_info(
                self.credentials_dict, scopes=scopes
            )
            self._client = gspread.authorize(creds)
            self._spreadsheet = self._client.open_by_key(self.spreadsheet_id)
        return self._spreadsheet

    @staticmethod
    def _read_selected_columns(
        worksheet, required_columns: Sequence[str]
    ) -> List[Dict[str, Any]]:
        """
        Fetch ONLY the required columns of a worksheet.

        Returns one dict per data row, keyed by the required column names.
        Raises ValueError when the worksheet header lacks a required column.
        """
        header = [str(cell).strip() for cell in worksheet.row_values(1)]
        missing = [name for name in required_columns if name not in header]
        if missing:
            raise ValueError(
                f"Worksheet '{getattr(worksheet, 'title', '?')}' is missing required "
                f"columns: {missing}"
            )

        id_column = header.index(config.SHEET_KEY_COLUMN) + 1
        # Column A/ID values are only used to determine how many data rows exist.
        id_values = worksheet.col_values(id_column)
        row_count = len(id_values)
        if row_count < 2:
            return []

        ranges = []
        for name in required_columns:
            letter = column_letter(header.index(name) + 1)
            ranges.append(f"{letter}2:{letter}{row_count}")

        columns = worksheet.batch_get(ranges, value_render_option="UNFORMATTED_VALUE")

        rows: List[Dict[str, Any]] = []
        for offset in range(row_count - 1):
            row: Dict[str, Any] = {}
            for name, column in zip(required_columns, columns):
                value = column[offset] if offset < len(column) else ""
                row[name] = value
            rows.append(row)

        logger.info(
            "Fetched %d rows with %d selected columns from worksheet '%s'",
            len(rows),
            len(required_columns),
            getattr(worksheet, "title", "?"),
        )
        return rows

    def get_report_documents(self) -> List[ReportDocumentRow]:
        """Fetch all rows from report_documents worksheet (strictly 3 columns)."""
        ss = self._get_spreadsheet()
        worksheet = ss.worksheet(config.WORKSHEET_REPORT_DOCUMENTS)
        records = self._read_selected_columns(
            worksheet, config.SHEET_COLUMNS_REPORT_DOCUMENTS
        )

        rows: List[ReportDocumentRow] = []
        for r in records:
            report_id = str(r.get("report_id", "")).strip()
            drive_file_id = str(r.get("drive_file_id", "")).strip()
            status_val = str(r.get("rag_status", RagStatus.BELUM_DIPROSES.value)).strip()
            if not report_id:
                continue

            try:
                status = RagStatus(status_val)
            except ValueError:
                status = RagStatus.BELUM_DIPROSES

            rows.append(
                ReportDocumentRow(
                    report_id=report_id,
                    drive_file_id=drive_file_id,
                    rag_status=status,
                )
            )
        logger.info("Retrieved %d document rows from report_documents", len(rows))
        return rows

    def get_competition_reports(self) -> List[CompetitionReportMetadata]:
        """
        Fetch competition reports restricted to the five RAG-allowed columns.
        Metadata enum/year validity is decided later by the ingestion validator.
        """
        ss = self._get_spreadsheet()
        worksheet = ss.worksheet(config.WORKSHEET_COMPETITION_REPORTS)
        records = self._read_selected_columns(
            worksheet, config.SHEET_COLUMNS_COMPETITION_REPORTS
        )

        reports: List[CompetitionReportMetadata] = []
        for r in records:
            report_id = str(r.get("report_id", "")).strip()
            if not report_id:
                continue

            reports.append(
                CompetitionReportMetadata(
                    report_id=report_id,
                    competition_name=str(r.get("competition_name", "")).strip(),
                    competition_year=str(r.get("competition_year", "")).strip(),
                    achievement_level=str(r.get("achievement_level", "")).strip(),
                    competition_type=str(r.get("competition_type", "")).strip(),
                )
            )
        logger.info(
            "Retrieved %d competition report metadata rows", len(reports)
        )
        return reports

    def update_rag_status(self, report_id: str, new_status: RagStatus) -> bool:
        """Update rag_status for a given report_id in report_documents."""
        ss = self._get_spreadsheet()
        worksheet = ss.worksheet(config.WORKSHEET_REPORT_DOCUMENTS)

        header = [str(cell).strip() for cell in worksheet.row_values(1)]
        if config.SHEET_COLUMN_RAG_STATUS not in header:
            raise ValueError(
                f"Worksheet '{config.WORKSHEET_REPORT_DOCUMENTS}' has no "
                f"'{config.SHEET_COLUMN_RAG_STATUS}' column"
            )

        cell = worksheet.find(report_id, in_column=1)
        if not cell:
            logger.error("Could not find report_id %s in report_documents", report_id)
            return False

        status_column = header.index(config.SHEET_COLUMN_RAG_STATUS) + 1
        worksheet.update_cell(cell.row, status_column, new_status.value)
        logger.info("Updated report %s rag_status to %s", report_id, new_status.value)
        return True


class GoogleDriveRepo(DriveRepo):
    """
    Production implementation of DriveRepo using google-api-python-client.
    Downloads PDF files and handles ChromaDB sync artifacts.
    """

    def __init__(self, folder_id: str, credentials_dict: Dict[str, Any]):
        self.folder_id = folder_id
        self.credentials_dict = credentials_dict
        self._service = None

    def _get_service(self):
        if self._service is None:
            from googleapiclient.discovery import build
            from google.oauth2.service_account import Credentials

            scopes = [
                "https://www.googleapis.com/auth/drive.file",
                "https://www.googleapis.com/auth/drive.readonly",
            ]
            creds = Credentials.from_service_account_info(
                self.credentials_dict, scopes=scopes
            )
            self._service = build("drive", "v3", credentials=creds)
        return self._service

    def download_pdf(self, file_id: str, destination_path: str) -> bool:
        """Download file by exact Drive file_id."""
        from googleapiclient.http import MediaIoBaseDownload

        service = self._get_service()
        dest = Path(destination_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            request = service.files().get_media(fileId=file_id)
            with io.FileIO(str(dest), "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            logger.info("Successfully downloaded Drive file %s to %s", file_id, dest)
            return True
        except Exception as e:
            logger.error("Failed to download Drive file %s: %s", file_id, str(e))
            return False

    def _find_files(self, remote_filename: str) -> List[Dict[str, Any]]:
        """
        Resolve an artifact by exact name inside the configured folder.

        Zero or several matches is an error: guessing would either create a
        duplicate or publish the wrong zip (fix pack P2).
        """
        query = (
            f"name = '{remote_filename}' and '{self.folder_id}' in parents "
            f"and trashed = false"
        )
        response = (
            self._get_service()
            .files()
            .list(
                q=query,
                spaces="drive",
                fields="files(id, name, modifiedTime, size)",
                pageSize=10,
            )
            .execute()
        )
        files = list(response.get("files") or [])
        if len(files) > 1:
            raise DriveArtifactError(
                f"'{remote_filename}' appears {len(files)} times in Drive folder "
                f"'{self.folder_id}'. Keep exactly one copy before re-running."
            )
        return files

    def upload_artifact(self, local_path: str, remote_filename: str) -> str:
        """
        Upload or replace an artifact, never duplicating it.

        An existing file is updated in place by its Drive file id; only a missing
        file is created. The caller uploads the zip before the manifest, so a
        failed manifest upload leaves the old manifest pointing at the old zip and
        the app refuses the mismatch instead of serving half an artifact.
        """
        from googleapiclient.http import MediaFileUpload

        service = self._get_service()
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Local artifact not found: {local_path}")

        media = MediaFileUpload(str(src), resumable=True)
        existing = self._find_files(remote_filename)
        if existing:
            file_id = str(existing[0]["id"])
            service.files().update(
                fileId=file_id, media_body=media, fields="id"
            ).execute()
            logger.info(
                "Replaced artifact %s in Drive (id %s)", remote_filename, file_id
            )
            return file_id

        file_metadata = {
            "name": remote_filename,
            "parents": [self.folder_id],
        }
        file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        file_id = str(file.get("id", ""))
        if not file_id:
            raise DriveArtifactError(
                f"Drive did not return an id for the uploaded '{remote_filename}'."
            )
        logger.info("Uploaded artifact %s to Drive with id %s", remote_filename, file_id)
        return file_id

    def download_artifact(self, remote_filename: str, destination_path: str) -> bool:
        """Download the one artifact with this exact name, or fail loudly."""
        items = self._find_files(remote_filename)
        if not items:
            raise DriveArtifactError(
                f"'{remote_filename}' was not found in Drive folder "
                f"'{self.folder_id}'."
            )

        return self.download_pdf(str(items[0]["id"]), destination_path)
