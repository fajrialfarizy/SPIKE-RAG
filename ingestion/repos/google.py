"""
Google Cloud implementation of SheetsRepo and DriveRepo for Production.
Connects via gspread and google-api-python-client with minimal read/write scopes.
Strictly excludes personal data from logs and pipeline boundaries.
"""

import io
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from enums import CompetitionType, AchievementLevel, RagStatus
from ingestion.repos.base import (
    SheetsRepo,
    DriveRepo,
    CompetitionReportMetadata,
    ReportDocumentRow,
)

logger = logging.getLogger(__name__)


class GoogleSheetsRepo(SheetsRepo):
    """
    Production implementation of SheetsRepo using gspread.
    Enforces privacy by reading only the required 5 columns from competition_reports
    and strictly 3 columns from report_documents.
    Raw row contents with personal data are never logged.
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

    def get_report_documents(self) -> List[ReportDocumentRow]:
        """Fetch all rows from report_documents worksheet (strictly 3 columns)."""
        ss = self._get_spreadsheet()
        worksheet = ss.worksheet("report_documents")
        records = worksheet.get_all_records()

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
        Fetch competition reports filtered strictly to RAG-allowed fields.
        All administrative / personal fields (NIM, phone, name, mentor) are ignored.
        """
        ss = self._get_spreadsheet()
        worksheet = ss.worksheet("competition_reports")
        records = worksheet.get_all_records()

        reports: List[CompetitionReportMetadata] = []
        for r in records:
            report_id = str(r.get("report_id", "")).strip()
            if not report_id:
                continue

            try:
                c_type = CompetitionType(str(r.get("competition_type", "")).strip())
                a_level = AchievementLevel(str(r.get("achievement_level", "")).strip())
                year = int(str(r.get("competition_year", "0")).strip())
            except ValueError as e:
                logger.warning(
                    "Skipping report %s with invalid metadata enum or year: %s",
                    report_id,
                    str(e),
                )
                continue

            reports.append(
                CompetitionReportMetadata(
                    report_id=report_id,
                    competition_name=str(r.get("competition_name", "")).strip(),
                    competition_year=year,
                    achievement_level=a_level,
                    competition_type=c_type,
                )
            )
        logger.info(
            "Retrieved %d valid competition report metadata entries", len(reports)
        )
        return reports

    def update_rag_status(self, report_id: str, new_status: RagStatus) -> bool:
        """Update rag_status for a given report_id in report_documents."""
        ss = self._get_spreadsheet()
        worksheet = ss.worksheet("report_documents")
        cell = worksheet.find(report_id)
        if not cell:
            logger.error("Could not find report_id %s in report_documents", report_id)
            return False

        # rag_status is in column 3
        worksheet.update_cell(cell.row, 3, new_status.value)
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

            scopes = ["https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive.readonly"]
            creds = Credentials.from_service_account_info(
                self.credentials_dict, scopes=scopes
            )
            self._service = build("drive", "v3", credentials=creds)
        return self._service

    def download_pdf(self, file_id: str, destination_path: str) -> bool:
        """Download file by exact Drive file_id."""
        from googleapiclient.http import MediaIoBaseDownload
        import googleapiclient.errors

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

    def upload_artifact(self, local_path: str, remote_filename: str) -> str:
        """Upload an artifact to the designated Drive folder."""
        from googleapiclient.http import MediaFileUpload

        service = self._get_service()
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Local artifact not found: {local_path}")

        file_metadata = {
            "name": remote_filename,
            "parents": [self.folder_id],
        }
        media = MediaFileUpload(str(src), resumable=True)
        file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        file_id = file.get("id", "")
        logger.info("Uploaded artifact %s to Drive with id %s", remote_filename, file_id)
        return file_id

    def download_artifact(self, remote_filename: str, destination_path: str) -> bool:
        """Search and download an artifact by filename from the designated Drive folder."""
        service = self._get_service()
        dest = Path(destination_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        query = f"name = '{remote_filename}' and '{self.folder_id}' in parents and trashed = false"
        results = (
            service.files()
            .list(q=query, spaces="drive", fields="files(id, name)")
            .execute()
        )
        items = results.get("files", [])
        if not items:
            logger.warning("Remote artifact %s not found in folder %s", remote_filename, self.folder_id)
            return False

        remote_file_id = items[0]["id"]
        return self.download_pdf(remote_file_id, destination_path)
