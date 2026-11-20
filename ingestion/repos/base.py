"""
Repository interfaces and models for Sheets and Drive access.
Enables clean separation between Google Cloud production APIs and local test mocks.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field
from enums import CompetitionType, AchievementLevel, RagStatus


class CompetitionReportMetadata(BaseModel):
    """
    Metadata model containing only the fields allowed for RAG processing (Brief 5.1).
    All personal data (NIM, name, phone, etc.) are strictly excluded.
    """
    report_id: str
    competition_name: str
    competition_year: int
    achievement_level: AchievementLevel
    competition_type: CompetitionType


class ReportDocumentRow(BaseModel):
    """
    Representation of report_documents worksheet row (strictly 3 columns).
    """
    report_id: str
    drive_file_id: str
    rag_status: RagStatus = RagStatus.BELUM_DIPROSES


class SheetsRepo(ABC):
    """Abstract interface for accessing metadata sheets."""

    @abstractmethod
    def get_report_documents(self) -> List[ReportDocumentRow]:
        """Fetch all rows from report_documents."""
        pass

    @abstractmethod
    def get_competition_reports(self) -> List[CompetitionReportMetadata]:
        """Fetch competition reports filtered to RAG-allowed fields."""
        pass

    @abstractmethod
    def update_rag_status(self, report_id: str, new_status: RagStatus) -> bool:
        """Update the rag_status column for a given report_id."""
        pass


class DriveRepo(ABC):
    """Abstract interface for downloading files and managing Chroma artifacts."""

    @abstractmethod
    def download_pdf(self, file_id: str, destination_path: str) -> bool:
        """Download evaluation PDF to the specified local path."""
        pass

    @abstractmethod
    def upload_artifact(self, local_path: str, remote_filename: str) -> str:
        """Upload ChromaDB zip / manifest artifact and return file ID."""
        pass

    @abstractmethod
    def download_artifact(self, remote_filename: str, destination_path: str) -> bool:
        """Download artifact by remote filename to local path."""
        pass
