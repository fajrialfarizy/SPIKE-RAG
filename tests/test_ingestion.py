"""
Unit and integration tests for Ingestion Pipeline (Fase 1 Acceptance).
Covers:
- First run: valid reports -> Selesai; defects -> Proses gagal with log reasons; Lainnya skipped.
- Second run without changes: no reprocessing, chunk count identical.
- Updating drive_file_id replaces old chunks cleanly without duplicates.
- Simulated mid-storage failure cleanly deletes partial chunks and sets status Proses gagal.
- Assertion that no chunk exceeds 512 tokens.
- Exactly 8 metadata fields.
- Zero PII leaks remaining in ChromaDB.
- Manifest generation and artifact zip restoration.
- Heading tolerance test with various formatting styles.
"""

import pytest
import shutil
from pathlib import Path

from enums import CompetitionType, AchievementLevel, SectionType, RagStatus
import config
from ingestion.repos.local import LocalSheetsRepo, LocalDriveRepo
from ingestion.chroma_store import ChromaStore
from ingestion.pipeline import IngestionPipeline
from ingestion.pdf_parser import PDFParser
from ingestion.sync import ChromaSync
from scripts.make_dummy_corpus import generate_corpus


@pytest.fixture
def test_env(tmp_path):
    """Sets up a clean temporary workspace for ingestion pipeline tests."""
    generate_corpus()

    # Copy fixtures to temp
    reports_csv = tmp_path / "competition_reports.csv"
    docs_csv = tmp_path / "report_documents.csv"
    shutil.copyfile(config.FIXTURES_DIR / "competition_reports.csv", reports_csv)
    shutil.copyfile(config.FIXTURES_DIR / "report_documents.csv", docs_csv)

    pdf_dir = tmp_path / "dummy_pdfs"
    shutil.copytree(config.DUMMY_PDF_DIR, pdf_dir)

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()

    sheets_repo = LocalSheetsRepo(reports_csv, docs_csv)
    drive_repo = LocalDriveRepo(pdf_dir, artifacts_dir)
    chroma_store = ChromaStore(persist_dir=chroma_dir)

    pipeline = IngestionPipeline(
        sheets_repo=sheets_repo,
        drive_repo=drive_repo,
        chroma_store=chroma_store,
        temp_download_dir=tmp_path / "temp_downloads",
    )

    return pipeline, sheets_repo, drive_repo, chroma_store, tmp_path


def test_first_run_ingestion_and_edge_cases(test_env):
    pipeline, sheets_repo, drive_repo, chroma_store, tmp_path = test_env

    summary = pipeline.run()

    # 1. Valid reports RPT0001 - RPT0010 and valid edge cases RPT0013, RPT0015 must be Selesai
    docs_map = {d.report_id: d for d in sheets_repo.get_report_documents()}

    for i in range(1, 11):
        rid = f"RPT{i:04d}"
        assert docs_map[rid].rag_status == RagStatus.SELESAI

    # RPT0013 (over char limit) is valid and must pass with warning
    assert docs_map["RPT0013"].rag_status == RagStatus.SELESAI
    assert any("RPT0013" in w for w in summary.warnings)

    # RPT0015 (PII case) is valid, PII is scrubbed and status Selesai
    assert docs_map["RPT0015"].rag_status == RagStatus.SELESAI
    assert "RPT0015" in summary.pii_redactions
    assert summary.pii_redactions["RPT0015"]["nim"] > 0

    # 2. Defective reports must be Proses gagal
    # RPT0011 (scanned / no text)
    assert docs_map["RPT0011"].rag_status == RagStatus.PROSES_GAGAL
    # RPT0012 (missing section)
    assert docs_map["RPT0012"].rag_status == RagStatus.PROSES_GAGAL

    # 3. Lainnya (RPT0099) must be skipped
    assert any(s["report_id"] == "RPT0099" for s in summary.skipped_reports)

    # 4. Strict assertions: No chunk > 512 tokens & Exactly 8 metadata fields
    assert summary.max_tokens_recorded <= config.CHUNK_HARD_LIMIT_TOKENS
    all_chunks = chroma_store.collection.get(include=["metadatas", "documents"])
    for m in all_chunks["metadatas"]:
        assert len(m) == 8
        assert "report_id" in m
        assert "chunk_id" in m
        assert "competition_name" in m
        assert "competition_year" in m
        assert "achievement_level" in m
        assert "competition_type" in m
        assert "section_type" in m
        assert "source_file_id" in m


def test_second_run_idempotency_no_reprocessing(test_env):
    pipeline, sheets_repo, drive_repo, chroma_store, tmp_path = test_env

    # First run
    pipeline.run()
    initial_chunk_count = chroma_store.count_chunks()
    assert initial_chunk_count > 0

    # Second run without changes
    summary2 = pipeline.run()
    # All previously finished documents should be skipped
    assert summary2.processed_count == 0
    assert summary2.skipped_count >= 10
    # Chunk count in vector store must be identical
    assert chroma_store.count_chunks() == initial_chunk_count


def test_update_drive_file_id_replaces_old_chunks(test_env):
    pipeline, sheets_repo, drive_repo, chroma_store, tmp_path = test_env

    pipeline.run()
    old_chunks_r1 = chroma_store.collection.get(where={"report_id": "RPT0001"})
    old_count = len(old_chunks_r1["ids"])

    # Simulate admin updating drive_file_id for RPT0001
    new_file_id = "DRV_RPT0001_UPDATED_V2"
    # Create updated PDF file in repo
    updated_pdf = drive_repo.local_pdf_dir / f"{new_file_id}.pdf"
    sample_pdf = drive_repo.local_pdf_dir / "DRV_RPT0001_NBPC.pdf"
    shutil.copyfile(sample_pdf, updated_pdf)

    # Update report_documents CSV
    with open(sheets_repo.report_documents_csv, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("DRV_RPT0001_NBPC", new_file_id)
    with open(sheets_repo.report_documents_csv, "w", encoding="utf-8") as f:
        f.write(content)

    # Re-run pipeline
    summary_update = pipeline.run()
    assert "RPT0001" in summary_update.processed_reports

    new_chunks_r1 = chroma_store.collection.get(where={"report_id": "RPT0001"})
    assert len(new_chunks_r1["ids"]) == old_count
    # All new chunks must reference the updated source_file_id
    for m in new_chunks_r1["metadatas"]:
        assert m["source_file_id"] == new_file_id


def test_simulated_failure_cleans_partial_chunks(test_env):
    pipeline, sheets_repo, drive_repo, chroma_store, tmp_path = test_env

    # Run with simulated failure on RPT0001
    summary = pipeline.run(simulate_failure_on_report="RPT0001")

    # Status must be Proses gagal
    docs_map = {d.report_id: d for d in sheets_repo.get_report_documents()}
    assert docs_map["RPT0001"].rag_status == RagStatus.PROSES_GAGAL

    # Partial chunks must be completely wiped from ChromaDB
    res = chroma_store.collection.get(where={"report_id": "RPT0001"})
    assert len(res["ids"]) == 0


def test_no_pii_remains_in_chromadb(test_env):
    pipeline, sheets_repo, drive_repo, chroma_store, tmp_path = test_env

    pipeline.run()
    all_docs = chroma_store.collection.get(include=["documents"])

    for doc_text in all_docs["documents"]:
        # Check that no simulated PII strings leaked
        assert "13520099" not in doc_text
        assert "081234567890" not in doc_text
        assert "budi.santoso2024@gmail.com" not in doc_text
        assert "hendra@institusi.ac.id" not in doc_text


def test_manifest_and_artifact_roundtrip(test_env):
    pipeline, sheets_repo, drive_repo, chroma_store, tmp_path = test_env

    pipeline.run()

    sync = ChromaSync(chroma_store=chroma_store, drive_repo=drive_repo)
    zip_path = tmp_path / "artifacts" / "chroma_backup.zip"
    manifest_path = tmp_path / "artifacts" / "manifest.json"

    manifest_data = sync.sync_to_drive(zip_path, manifest_path)
    assert manifest_data["report_count"] >= 10
    assert manifest_data["chunk_count"] > 0
    assert manifest_data["embedding_model"] == config.EMBEDDING_MODEL_NAME

    # Test restoring to a new directory
    restore_dir = tmp_path / "restored_chroma"
    sync.restore_from_zip(zip_path, restore_dir)
    assert (restore_dir / "chroma.sqlite3").exists()


def test_heading_detection_tolerance():
    """Verify heading parser is tolerant to variations in casing, numbering, and spacing."""
    parser = PDFParser()

    varied_text = (
        "1. JUDUL IDE: Inovasi Solar Cell\n"
        "Ini adalah penjelasan judul ide yang sangat inovatif.\n\n"
        "2) abstrak :\n"
        "Executive summary mengenai panel surya generasi ketiga dengan efisiensi tinggi.\n\n"
        "III - Evaluasi Tim :\n"
        "Kerjasama tim berjalan sangat baik selama masa persiapan kompetisi.\n\n"
        "4.   Saran Pengembangan  \n"
        "Disarankan menambah kapasitas baterai penyimpanan daya untuk malam hari.\n"
    )

    success, sections, err = parser.parse_sections(varied_text)
    assert success is True
    assert "Inovasi Solar Cell" in sections["judul_ide"]
    assert "Executive summary" in sections["executive_summary"]
    assert "Kerjasama tim" in sections["evaluasi_tim"]
    assert "Disarankan menambah" in sections["saran_pengembangan"]
