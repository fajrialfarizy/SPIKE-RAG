"""
Ingestion pipeline acceptance tests (Phase 1).

Fully offline: the chunker gets an injected word-count tokenizer and the pipeline
an explicit hash embedder, so no model weights are loaded. Embedding-model
identity is still enforced by the ChromaStore guard against config.
"""

import json
import shutil
from pathlib import Path

import pytest

import config
from enums import RagStatus
from ingestion.chroma_store import ChromaStore
from ingestion.chunker import SectionAwareChunker
from ingestion.pipeline import IngestionPipeline
from ingestion.repos.local import LocalDriveRepo, LocalSheetsRepo
from ingestion.sync import ChromaSync
from scripts.make_dummy_corpus import generate_corpus
from tests.stubs import HashEmbedder, WordCountTokenizer

FORBIDDEN_PII = (
    "13520099",
    "081234567890",
    "+6281198765432",
    "budi.santoso2024@gmail.com",
    "hendra@institusi.ac.id",
    "Budi Santoso",
    "Ahmad Fauzi",
    "Hendra",
)


@pytest.fixture(scope="module", autouse=True)
def corpus():
    """Generate the dummy corpus once per module (deterministic, ~2s)."""
    generate_corpus()


@pytest.fixture
def env(tmp_path):
    reports_csv = tmp_path / "competition_reports.csv"
    docs_csv = tmp_path / "report_documents.csv"
    shutil.copyfile(config.FIXTURES_DIR / "competition_reports.csv", reports_csv)
    shutil.copyfile(config.FIXTURES_DIR / "report_documents.csv", docs_csv)

    pdf_dir = tmp_path / "dummy_pdfs"
    shutil.copytree(config.DUMMY_PDF_DIR, pdf_dir)

    sheets = LocalSheetsRepo(reports_csv, docs_csv)
    drive = LocalDriveRepo(pdf_dir, tmp_path / "artifacts")
    store = ChromaStore(persist_dir=tmp_path / "chroma_db")

    pipeline = IngestionPipeline(
        sheets_repo=sheets,
        drive_repo=drive,
        chroma_store=store,
        embedder=HashEmbedder(),
        chunker=SectionAwareChunker(tokenizer=WordCountTokenizer()),
        temp_download_dir=tmp_path / "temp_downloads",
        log_dir=tmp_path / "logs",
    )
    return pipeline, sheets, drive, store, tmp_path


def statuses(sheets):
    return {row.report_id: row.rag_status for row in sheets.get_report_documents()}


def test_first_run_marks_valid_defective_and_skipped(env):
    pipeline, sheets, _, store, _ = env

    summary = pipeline.run()
    state = statuses(sheets)

    for report_id in [f"RPT{i:04d}" for i in range(1, 11)]:
        assert state[report_id] == RagStatus.SELESAI, report_id

    # Char-limit and PII documents are valid; injection document is stored as data.
    assert state["RPT0013"] == RagStatus.SELESAI
    assert state["RPT0015"] == RagStatus.SELESAI
    assert state["RPT0014"] == RagStatus.SELESAI

    assert state["RPT0011"] == RagStatus.PROSES_GAGAL
    assert state["RPT0012"] == RagStatus.PROSES_GAGAL

    # Lainnya row is skipped and its status is left untouched.
    assert state["RPT0099"] == RagStatus.BELUM_DIPROSES

    assert summary.processed_count == 13
    assert summary.skipped_count == 1
    assert summary.failed_count == 2
    assert [r["report_id"] for r in summary.skipped_reports] == ["RPT0099"]
    assert {r["report_id"] for r in summary.failed_reports} == {"RPT0011", "RPT0012"}

    reasons = {r["report_id"]: r["reason"] for r in summary.failed_reports}
    assert "scan" in reasons["RPT0011"].lower()
    assert "saran_pengembangan" in reasons["RPT0012"]

    # A failed document does not stop the batch: later reports are processed.
    assert "RPT0013" in summary.processed_reports
    assert "RPT0015" in summary.processed_reports
    assert store.count_chunks() > 0


def test_chunk_limits_and_metadata_shape(env):
    pipeline, _, _, store, _ = env
    summary = pipeline.run()

    assert summary.max_tokens_recorded <= config.CHUNK_HARD_LIMIT_TOKENS

    stored = store.collection.get(include=["metadatas", "documents"])
    required = {
        "report_id",
        "competition_name",
        "competition_year",
        "achievement_level",
        "competition_type",
        "section_type",
        "source_file_id",
        "chunk_id",
    }
    for metadata in stored["metadatas"]:
        assert len(metadata) == 8
        assert set(metadata) == required
        assert isinstance(metadata["competition_year"], int)
        assert metadata["section_type"] in {
            "executive_summary",
            "evaluasi_tim",
            "saran_pengembangan",
        }

    # No judul_ide chunk, and every chunk carries the context header.
    assert all("judul_ide" != m["section_type"] for m in stored["metadatas"])
    assert all(doc.startswith("Kompetisi: ") for doc in stored["documents"])
    assert all("Judul ide: " in doc.split("\n\n")[0] for doc in stored["documents"])

    # Chunk ids are deterministic and namespaced by report_id.
    for chunk_id, metadata in zip(stored["ids"], stored["metadatas"]):
        assert chunk_id == metadata["chunk_id"]
        assert chunk_id.startswith(f"{metadata['report_id']}:{metadata['section_type']}:")


def test_second_run_does_not_reprocess(env):
    pipeline, _, _, store, _ = env

    first = pipeline.run()
    first_count = store.count_chunks()

    second = pipeline.run()

    assert second.processed_count == 0
    assert second.skipped_count == 14  # 13 Selesai + 1 Lainnya
    assert store.count_chunks() == first_count
    assert second.chunks_per_report == {}
    assert first.chunks_per_report  # non-empty on the first run


def test_changed_drive_file_id_replaces_chunks_without_duplicates(env):
    pipeline, sheets, drive, store, _ = env
    pipeline.run()

    before = store.collection.get(where={"report_id": "RPT0001"})
    before_count = len(before["ids"])
    assert before_count > 0

    new_file_id = "DRV_RPT0001_UPDATED_V2.pdf"
    shutil.copyfile(
        drive.local_pdf_dir / "DRV_RPT0001_NBPC.pdf", drive.local_pdf_dir / new_file_id
    )
    content = sheets.report_documents_csv.read_text(encoding="utf-8")
    sheets.report_documents_csv.write_text(
        content.replace("DRV_RPT0001_NBPC.pdf", new_file_id), encoding="utf-8"
    )

    summary = pipeline.run()

    assert "RPT0001" in summary.processed_reports
    after = store.collection.get(where={"report_id": "RPT0001"})
    assert len(after["ids"]) == before_count
    assert len(set(after["ids"])) == len(after["ids"])
    assert {m["source_file_id"] for m in after["metadatas"]} == {new_file_id}


def test_injected_mid_write_failure_cleans_up_and_retry_succeeds(env, monkeypatch):
    pipeline, sheets, _, store, _ = env

    real_add = store._add

    def failing_add(ids, documents, metadatas, embeddings):
        # Only RPT0001 fails, and it fails AFTER one chunk reached the collection.
        if not str(ids[0]).startswith("RPT0001"):
            return real_add(ids, documents, metadatas, embeddings)
        store.collection.add(
            ids=[ids[0]],
            documents=[documents[0]],
            metadatas=[metadatas[0]],
            embeddings=[embeddings[0]],
        )
        raise RuntimeError("injected mid-write failure")

    monkeypatch.setattr(store, "_add", failing_add)
    summary = pipeline.run()

    state = statuses(sheets)
    assert state["RPT0001"] == RagStatus.PROSES_GAGAL
    assert store.count_chunks_for_report("RPT0001") == 0
    assert any(
        "Gagal menyimpan" in r["reason"]
        for r in summary.failed_reports
        if r["report_id"] == "RPT0001"
    )
    # Other documents still completed.
    assert state["RPT0002"] == RagStatus.SELESAI

    monkeypatch.undo()

    retry = pipeline.run()
    assert "RPT0001" in retry.processed_reports
    assert store.count_chunks_for_report("RPT0001") > 0
    assert statuses(sheets)["RPT0001"] == RagStatus.SELESAI


def test_no_pii_left_in_chroma_and_funding_amount_survives(env):
    pipeline, _, _, store, _ = env
    summary = pipeline.run()

    stored = store.collection.get(include=["documents"])
    joined = "\n".join(stored["documents"])

    for leaked in FORBIDDEN_PII:
        assert leaked not in joined, f"PII leaked into Chroma: {leaked}"

    assert joined.count(config.PII_REDACTION_REPLACEMENT) >= 4
    # The legitimate funding amount is preserved (negative PII case).
    assert "25000000" in joined
    assert "Rp 25.000.000" in joined

    redactions = summary.pii_redactions
    assert redactions["RPT0015"]["nim"] >= 1
    assert redactions["RPT0015"]["phone"] >= 2
    assert redactions["RPT0015"]["email"] >= 2
    assert redactions["RPT0015"]["name"] >= 3
    # The monetary figure must not be counted as a redaction.
    assert redactions.get("RPT0001", {}).get("nim", 0) == 0


def test_run_log_and_json_summary_are_written(env):
    pipeline, _, _, _, tmp_path = env
    summary = pipeline.run()

    log_path = Path(summary.log_path)
    summary_path = Path(summary.summary_path)
    assert log_path.is_file()
    assert summary_path.is_file()
    assert log_path.parent == tmp_path / "logs"

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in (
        "started_at",
        "finished_at",
        "processed",
        "skipped",
        "failed",
        "processed_reports",
        "skipped_reports",
        "failed_reports",
        "warnings",
        "char_limit_warnings",
        "pii_redactions",
        "name_redactions",
        "chunks_per_report",
        "total_chunks",
        "max_tokens_observed",
    ):
        assert key in payload, key

    assert payload["processed"] == 13
    assert payload["failed"] == 2
    assert payload["max_tokens_observed"] <= config.CHUNK_HARD_LIMIT_TOKENS
    assert len(payload["char_limit_warnings"]) == 1
    assert "RPT0013" in payload["char_limit_warnings"][0]
    assert payload["name_redactions"]["RPT0015"] >= 1

    # No document text and no raw personal data in the log file.
    log_text = log_path.read_text(encoding="utf-8")
    for leaked in FORBIDDEN_PII:
        assert leaked not in log_text
    assert "NutriTrack" not in log_text


def test_manifest_and_artifact_roundtrip(env):
    pipeline, _, drive, store, tmp_path = env
    pipeline.run()

    sync = ChromaSync(chroma_store=store, drive_repo=drive)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    zip_path = work_dir / "chroma_backup.zip"
    manifest_path = work_dir / "manifest.json"

    manifest = sync.sync_to_drive(zip_path, manifest_path)
    assert manifest["report_count"] >= 10
    assert manifest["chunk_count"] == store.count_chunks()
    assert manifest["embedding_model"] == config.EMBEDDING_MODEL_NAME
    assert manifest["embedding_dimension"] == config.EMBEDDING_DIMENSION
    assert manifest["schema_version"] == config.MANIFEST_SCHEMA_VERSION
    assert manifest["zip_checksum"] == ChromaSync.calculate_file_hash(zip_path)

    # Reload from the uploaded copies, not from the local working files.
    download_dir = tmp_path / "downloaded"
    download_dir.mkdir()
    assert drive.download_artifact("chroma_backup.zip", str(download_dir / "chroma_backup.zip"))
    assert drive.download_artifact("manifest.json", str(download_dir / "manifest.json"))

    downloaded_manifest = json.loads(
        (download_dir / "manifest.json").read_text(encoding="utf-8")
    )
    ChromaSync.validate_manifest(downloaded_manifest)

    restore_dir = tmp_path / "restored_chroma"
    ChromaSync.restore_from_zip(download_dir / "chroma_backup.zip", restore_dir)
    assert (restore_dir / "chroma.sqlite3").exists()

    restored_store = ChromaStore(persist_dir=restore_dir)
    assert restored_store.count_chunks() == store.count_chunks()
    assert restored_store.get_all_report_ids() == store.get_all_report_ids()


def test_invalid_metadata_fails_with_validator_reason(tmp_path):
    (tmp_path / "c.csv").write_text(
        "report_id,competition_name,competition_year,achievement_level,competition_type\n"
        "RPT9001,Lomba A,2024,Juara Khayalan,Essay\n"
        "RPT9002,Lomba B,bukan-tahun,Juara 1,Essay\n"
        "RPT0001,Lomba C,2024,Juara 1,Essay\n",
        encoding="utf-8",
    )
    (tmp_path / "d.csv").write_text(
        "report_id,drive_file_id,rag_status\n"
        "RPT9001,DRV_RPT0001_NBPC.pdf,Belum diproses\n"
        "RPT9002,DRV_RPT0001_NBPC.pdf,Belum diproses\n"
        "RPT0001,DRV_RPT0001_NBPC.pdf,Belum diproses\n",
        encoding="utf-8",
    )
    shutil.copytree(config.DUMMY_PDF_DIR, tmp_path / "pdfs")

    sheets = LocalSheetsRepo(tmp_path / "c.csv", tmp_path / "d.csv")
    drive = LocalDriveRepo(tmp_path / "pdfs", tmp_path / "artifacts")
    store = ChromaStore(persist_dir=tmp_path / "chroma_db")
    pipeline = IngestionPipeline(
        sheets,
        drive,
        store,
        embedder=HashEmbedder(),
        chunker=SectionAwareChunker(tokenizer=WordCountTokenizer()),
        temp_download_dir=tmp_path / "dl",
        log_dir=tmp_path / "logs",
    )

    summary = pipeline.run()
    reasons = {r["report_id"]: r["reason"] for r in summary.failed_reports}

    assert "achievement_level tidak valid" in reasons["RPT9001"]
    assert "competition_year" in reasons["RPT9002"]
    assert statuses(sheets)["RPT9001"] == RagStatus.PROSES_GAGAL
    assert statuses(sheets)["RPT0001"] == RagStatus.SELESAI
