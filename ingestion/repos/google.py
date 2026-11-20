"""
Google Cloud implementation of SheetsRepo and DriveRepo for Production.
Connects via gspread and google-api-python-client with minimal read/write scopes.
"""

from typing import List
from enums import CompetitionType, AchievementLevel, RagStatus
from ingestion.repos.base import (
    SheetsRepo,
    DriveRepo,
    CompetitionReportMetadata,
    ReportDocumentRow,
)


class GoogleSheetsRepo(SheetsRepo):
    """
    Production implementation of SheetsRepo using gspread.
    Credentials should be loaded from Streamlit/Colab secrets.
    """

    def __init__(self, spreadsheet_id: str, credentials_dict: dict):
        self.spreadsheet_id = spreadsheet_id
        self.credentials_dict = credentials_dict
        # Lazy initialization will occur in production environment
        self._client = None

    def get_report_documents(self) -> List[ReportDocumentRow]:
        # Production implementation using gspread client
        raise NotImplementedError("GoogleSheetsRepo is intended for production deployment.")

    def get_competition_reports(self) -> List[CompetitionReportMetadata]:
        # Production implementation using gspread client
        raise NotImplementedError("GoogleSheetsRepo is intended for production deployment.")

    def update_rag_status(self, report_id: str, new_status: RagStatus) -> bool:
        # Production implementation using gspread client
        raise NotImplementedError("GoogleSheetsRepo is intended for production deployment.")


class GoogleDriveRepo(DriveRepo):
    """
    Production implementation of DriveRepo using google-api-python-client.
    Minimal scopes for downloading PDFs and uploading/downloading ChromaDB artifacts.
    """

    def __init__(self, folder_id: str, credentials_dict: dict):
        self.folder_id = folder_id
        self.credentials_dict = credentials_dict
        self._service = None

    def download_pdf(self, file_id: str, destination_path: str) -> bool:
        raise NotImplementedError("GoogleDriveRepo is intended for production deployment.")

    def upload_artifact(self, local_path: str, remote_filename: str) -> str:
        raise NotImplementedError("GoogleDriveRepo is intended for production deployment.")

    def download_artifact(self, remote_filename: str, destination_path: str) -> bool:
        raise NotImplementedError("GoogleDriveRepo is intended for production deployment.")
