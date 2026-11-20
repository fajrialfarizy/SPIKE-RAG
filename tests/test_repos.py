"""
Unit tests for LocalSheetsRepo and LocalDriveRepo.
Verifies file operations, CSV parsing, data privacy exclusion, and status updates.
"""

import pytest
import csv
from pathlib import Path
from enums import CompetitionType, AchievementLevel, RagStatus
from ingestion.repos.local import LocalSheetsRepo, LocalDriveRepo


@pytest.fixture
def temp_fixtures(tmp_path):
    reports_csv = tmp_path / "competition_reports.csv"
    documents_csv = tmp_path / "report_documents.csv"
    pdf_dir = tmp_path / "pdfs"
    artifacts_dir = tmp_path / "artifacts"
    pdf_dir.mkdir()
    artifacts_dir.mkdir()

    # Create dummy competition reports with personal data included
    with open(reports_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "report_id",
                "reporter_nim",
                "reporter_name",
                "reporter_phone",
                "mentor_name",
                "competition_name",
                "competition_year",
                "achievement_level",
                "competition_type",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "report_id": "RPT0001",
            "reporter_nim": "13520001",
            "reporter_name": "Rahasia",
            "reporter_phone": "08123456789",
            "mentor_name": "Dosen X",
            "competition_name": "Lomba A",
            "competition_year": "2024",
            "achievement_level": AchievementLevel.JUARA_1.value,
            "competition_type": CompetitionType.BUSINESS_PLAN.value,
        })
        # Invalid row
        writer.writerow({
            "report_id": "RPT0002",
            "reporter_nim": "13520002",
            "competition_name": "Lomba B",
            "competition_year": "invalid_year",
            "achievement_level": "Juara Khayalan",
            "competition_type": "Tipe Aneh",
        })

    # Create dummy report documents (strictly 3 columns)
    with open(documents_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["report_id", "drive_file_id", "rag_status"]
        )
        writer.writeheader()
        writer.writerow({
            "report_id": "RPT0001",
            "drive_file_id": "DRV_0001",
            "rag_status": RagStatus.BELUM_DIPROSES.value,
        })

    return reports_csv, documents_csv, pdf_dir, artifacts_dir


def test_local_sheets_repo_privacy_and_filtering(temp_fixtures):
    reports_csv, documents_csv, _, _ = temp_fixtures
    repo = LocalSheetsRepo(reports_csv, documents_csv)

    reports = repo.get_competition_reports()
    # The repo returns raw metadata rows; invalid enums are rejected by the
    # pipeline validator (not silently dropped here), so both rows come back.
    assert len(reports) == 2
    r = reports[0]
    assert r.report_id == "RPT0001"
    assert int(r.competition_year) == 2024
    assert r.achievement_level == AchievementLevel.JUARA_1
    assert r.competition_type == CompetitionType.BUSINESS_PLAN

    invalid = {row.report_id: row for row in reports}["RPT0002"]
    assert invalid.achievement_level == "Juara Khayalan"
    assert invalid.competition_year == "invalid_year"
    # Verify no PII attributes exist on the model
    assert not hasattr(r, "reporter_nim")
    assert not hasattr(r, "reporter_phone")
    assert not hasattr(r, "mentor_name")

    # Verify report_documents fetching
    docs = repo.get_report_documents()
    assert len(docs) == 1
    assert docs[0].report_id == "RPT0001"
    assert docs[0].rag_status == RagStatus.BELUM_DIPROSES

    # Update status to Selesai
    success = repo.update_rag_status("RPT0001", RagStatus.SELESAI)
    assert success is True
    updated_docs = repo.get_report_documents()
    assert updated_docs[0].rag_status == RagStatus.SELESAI


def test_local_drive_repo_operations(temp_fixtures):
    _, _, pdf_dir, artifacts_dir = temp_fixtures
    drive_repo = LocalDriveRepo(pdf_dir, artifacts_dir)

    # Place a dummy file
    sample_pdf = pdf_dir / "DRV_0001.pdf"
    sample_pdf.write_text("Dummy PDF content", encoding="utf-8")

    # Exact matching only: the bare id must NOT resolve to 'DRV_0001.pdf'
    assert drive_repo.download_pdf("DRV_0001", str(artifacts_dir / "nope.pdf")) is False
    assert not (artifacts_dir / "nope.pdf").exists()

    # Download PDF by exact file name
    dest_path = artifacts_dir / "downloaded.pdf"
    ok = drive_repo.download_pdf("DRV_0001.pdf", str(dest_path))
    assert ok is True
    assert dest_path.exists()
    assert dest_path.read_text(encoding="utf-8") == "Dummy PDF content"

    # Uploading into the directory that already holds the source must not raise
    same_name = drive_repo.upload_artifact(str(dest_path), "downloaded.pdf")
    assert same_name == "downloaded.pdf"

    # Upload artifact
    artifact_name = drive_repo.upload_artifact(str(dest_path), "chroma_backup.zip")
    assert artifact_name == "chroma_backup.zip"
    assert (artifacts_dir / "chroma_backup.zip").exists()

    # Download artifact
    dest_artifact = pdf_dir / "retrieved_artifact.zip"
    ok2 = drive_repo.download_artifact("chroma_backup.zip", str(dest_artifact))
    assert ok2 is True
    assert dest_artifact.exists()
