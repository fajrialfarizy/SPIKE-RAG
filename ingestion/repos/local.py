"""
Local filesystem and CSV implementation of SheetsRepo and DriveRepo.
Used for testing, local offline development, and evaluation on dummy data.

CSV reads resolve column indexes from the header first, so only the required
columns are materialised (same select-at-the-boundary rule as the Google repo).
"""

import csv
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import config
from enums import RagStatus
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

    @staticmethod
    def _read_selected_columns(
        path: Path, required_columns: Sequence[str]
    ) -> List[Dict[str, Any]]:
        """Read only the required columns of a CSV, resolving them from the header."""
        if not path.exists():
            return []

        with open(path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = [cell.strip() for cell in next(reader)]
            except StopIteration:
                return []

            missing = [name for name in required_columns if name not in header]
            if missing:
                raise ValueError(
                    f"CSV '{path.name}' is missing required columns: {missing}"
                )

            indexes = [header.index(name) for name in required_columns]
            rows: List[Dict[str, Any]] = []
            for raw in reader:
                if not raw:
                    continue
                row: Dict[str, Any] = {}
                for name, index in zip(required_columns, indexes):
                    row[name] = raw[index].strip() if index < len(raw) else ""
                rows.append(row)
            return rows

    def get_report_documents(self) -> List[ReportDocumentRow]:
        rows: List[ReportDocumentRow] = []
        for cleaned in self._read_selected_columns(
            self.report_documents_csv, config.SHEET_COLUMNS_REPORT_DOCUMENTS
        ):
            if not cleaned.get("report_id"):
                continue
            try:
                status = RagStatus(
                    cleaned.get("rag_status") or RagStatus.BELUM_DIPROSES.value
                )
            except ValueError:
                status = RagStatus.BELUM_DIPROSES

            rows.append(
                ReportDocumentRow(
                    report_id=cleaned["report_id"],
                    drive_file_id=cleaned.get("drive_file_id", ""),
                    rag_status=status,
                )
            )
        return rows

    def get_competition_reports(self) -> List[CompetitionReportMetadata]:
        """Return raw metadata rows; enum/year validation happens in the pipeline."""
        reports: List[CompetitionReportMetadata] = []
        for cleaned in self._read_selected_columns(
            self.competition_reports_csv, config.SHEET_COLUMNS_COMPETITION_REPORTS
        ):
            report_id = cleaned.get("report_id")
            if not report_id:
                continue

            reports.append(
                CompetitionReportMetadata(
                    report_id=report_id,
                    competition_name=cleaned.get("competition_name", ""),
                    competition_year=cleaned.get("competition_year", ""),
                    achievement_level=cleaned.get("achievement_level", ""),
                    competition_type=cleaned.get("competition_type", ""),
                )
            )
        return reports

    def update_rag_status(self, report_id: str, new_status: RagStatus) -> bool:
        if not self.report_documents_csv.exists():
            return False

        rows = []
        updated = False
        with open(self.report_documents_csv, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = (
                reader.fieldnames
                or list(config.SHEET_COLUMNS_REPORT_DOCUMENTS)
            )
            for row in reader:
                if (row.get("report_id") or "").strip() == report_id:
                    row["rag_status"] = new_status.value
                    updated = True
                rows.append(row)

        if updated:
            with open(
                self.report_documents_csv, mode="w", encoding="utf-8", newline=""
            ) as f:
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
        """Copy an evaluation PDF from the local store.

        Exact matching only: `file_id` must equal the stored filename exactly.
        No implicit '.pdf' suffix is appended, so a wrong id fails loudly instead
        of silently resolving to a similarly named file.
        """
        candidate = self.local_pdf_dir / file_id
        if not candidate.is_file():
            return False

        dest = Path(destination_path)
        if candidate.resolve() == dest.resolve():
            return True

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate, dest)
        return True

    def upload_artifact(self, local_path: str, remote_filename: str) -> str:
        src = Path(local_path)
        dest = self.artifacts_dir / remote_filename
        if src.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        return dest.name

    def download_artifact(self, remote_filename: str, destination_path: str) -> bool:
        src = self.artifacts_dir / remote_filename
        if not src.is_file():
            return False

        dest = Path(destination_path)
        if src.resolve() == dest.resolve():
            return True

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return True
