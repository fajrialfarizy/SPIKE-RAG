"""Repository abstractions for Sheets and Drive."""

from ingestion.repos.base import (
    SheetsRepo,
    DriveRepo,
    CompetitionReportMetadata,
    ReportDocumentRow,
)
from ingestion.repos.local import LocalSheetsRepo, LocalDriveRepo
from ingestion.repos.google import GoogleSheetsRepo, GoogleDriveRepo

__all__ = [
    "SheetsRepo",
    "DriveRepo",
    "CompetitionReportMetadata",
    "ReportDocumentRow",
    "LocalSheetsRepo",
    "LocalDriveRepo",
    "GoogleSheetsRepo",
    "GoogleDriveRepo",
]
