"""
Local filesystem and CSV implementation of SheetsRepo and DriveRepo.
Used for testing, local offline development, and evaluation on dummy data.
"""

import csv
import shutil
from pathlib import Path
from typing import List
from enums import CompetitionType, AchievementLevel, RagStatus
from ingestion.repos.base import (
    SheetsRepo,
    DriveRepo,
    CompetitionReportMetadata,
    ReportDocumentRow,
)


class LocalSheetsRepo(SheetsRepo):
    """Local CSV implementation of SheetsRepo."""

    def __init__(self, competition_reports_csv: Path, report_documents_csv: Path):
        self.competition_reports_csv = Path(competition_reports_csv)
        self.report_documents_csv = Path(report_documents_csv)

    def get_report_documents(self) -> List[ReportDocumentRow]:
        if not self.report_documents_csv.exists():
            return []

        rows: List[ReportDocumentRow] = []
        with open(self.report_documents_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Strip keys and values
                cleaned = {k.strip(): v.strip() for k, v in row.items() if k}
                if not cleaned.get("report_id"):
                    continue
                rows.append(
                    ReportDocumentRow(
                        report_id=cleaned["report_id"],
                        drive_file_id=cleaned["drive_file_id"],
                        rag_status=RagStatus(cleaned.get("rag_status", RagStatus.BELUM_DIPROSES.value)),
                    )
                )
        return rows

    def get_competition_reports(self) -> List[CompetitionReportMetadata]:
        if not self.competition_reports_csv.exists():
            return []

        reports: List[CompetitionReportMetadata] = []
        with open(self.competition_reports_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned = {k.strip(): v.strip() for k, v in row.items() if k}
                report_id = cleaned.get("report_id")
                if not report_id:
                    continue

                try:
                    c_type = CompetitionType(cleaned.get("competition_type", ""))
                    a_level = AchievementLevel(cleaned.get("achievement_level", ""))
                    year = int(cleaned.get("competition_year", "0"))
                except ValueError:
                    # Skip invalid enum or year parsing errors
                    continue

                reports.append(
                    CompetitionReportMetadata(
                        report_id=report_id,
                        competition_name=cleaned.get("competition_name", ""),
                        competition_year=year,
                        achievement_level=a_level,
                        competition_type=c_type,
                    )
                )
        return reports

    def update_rag_status(self, report_id: str, new_status: RagStatus) -> bool:
        if not self.report_documents_csv.exists():
            return False

        rows = []
        updated = False
        with open(self.report_documents_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or ["report_id", "drive_file_id", "rag_status"]
            for row in reader:
                if row.get("report_id", "").strip() == report_id:
                    row["rag_status"] = new_status.value
                    updated = True
                rows.append(row)

        if updated:
            with open(self.report_documents_csv, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return updated


class LocalDriveRepo(DriveRepo):
    """Local filesystem implementation of DriveRepo."""

    def __init__(self, local_pdf_dir: Path, artifacts_dir: Path):
        self.local_pdf_dir = Path(local_pdf_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.local_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def download_pdf(self, file_id: str, destination_path: str) -> bool:
        # Exact file matching per Supervisor feedback (Brief 5.1 & local repo specs)
        # file_id in Google Drive maps to the stored filename or file_id in local storage
        candidate = self.local_pdf_dir / file_id
        if not candidate.exists() and not file_id.endswith(".pdf"):
            candidate = self.local_pdf_dir / f"{file_id}.pdf"

        # Check candidate exact existence
        if candidate.is_file():
            dest = Path(destination_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate, dest)
            return True
        return False

    def upload_artifact(self, local_path: str, remote_filename: str) -> str:
        src = Path(local_path)
        dest = self.artifacts_dir / remote_filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return dest.name

    def download_artifact(self, remote_filename: str, destination_path: str) -> bool:
        src = self.artifacts_dir / remote_filename
        if src.exists():
            dest = Path(destination_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            return True
        return False
