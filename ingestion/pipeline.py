"""
Main Ingestion Pipeline for Lab Prestasi RAG System.
Orchestrates:
1. Merging report_documents with competition_reports via report_id.
2. Filtering out 'Lainnya' and unchanged documents.
3. Downloading PDF and extracting text with PyMuPDF.
4. Validating readability and section structure.
5. Scrubbing PII (NIM, phone, email) with [DIREDAKSI].
6. Section-aware chunking and intfloat/multilingual-e5-small embedding.
7. Atomic delete-then-insert into ChromaDB.
8. Updating rag_status in Sheets (reasons only in log).
9. Compressing vector store and uploading manifest.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, field

from enums import CompetitionType, AchievementLevel, SectionType, RagStatus
import config
from ingestion.repos.base import SheetsRepo, DriveRepo
from ingestion.pdf_parser import PDFParser
from ingestion.validator import IngestionValidator
from ingestion.pii_scrubber import PIIScrubber
from ingestion.chunker import SectionAwareChunker
from ingestion.embedder import E5Embedder
from ingestion.chroma_store import ChromaStore
from ingestion.sync import ChromaSync

logger = logging.getLogger("ingestion_pipeline")


@dataclass
class IngestionSummary:
    processed_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    processed_reports: List[str] = field(default_factory=list)
    skipped_reports: List[Dict[str, str]] = field(default_factory=list)
    failed_reports: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    char_limit_warnings: List[str] = field(default_factory=list)
    pii_redactions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    chunks_per_report: Dict[str, int] = field(default_factory=dict)
    max_tokens_recorded: int = 0
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    log_path: str = ""
    summary_path: str = ""

    def name_redactions(self) -> Dict[str, int]:
        return {
            report_id: counts.get("name", 0)
            for report_id, counts in self.pii_redactions.items()
            if counts.get("name", 0)
        }


class IngestionPipeline:
    """Executes end-to-end ingestion batch process."""

    def __init__(
        self,
        sheets_repo: SheetsRepo,
        drive_repo: DriveRepo,
        chroma_store: ChromaStore,
        embedder: Optional[E5Embedder] = None,
        temp_download_dir: Optional[Path] = None,
        chunker: Optional[SectionAwareChunker] = None,
        log_dir: Optional[Path] = None,
    ):
        self.sheets_repo = sheets_repo
        self.drive_repo = drive_repo
        self.chroma_store = chroma_store
        self.embedder = embedder or E5Embedder()
        self.pdf_parser = PDFParser()
        self.validator = IngestionValidator()
        self.pii_scrubber = PIIScrubber()
        self.chunker = chunker or SectionAwareChunker()
        self.temp_download_dir = temp_download_dir or (config.DATA_DIR / "temp_downloads")
        self.temp_download_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir) if log_dir else Path(config.LOG_DIR)
        self._file_handler: Optional[logging.Handler] = None

    def _start_run_logging(self) -> Path:
        """Attach a per-run FileHandler so each batch keeps its own log file."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        log_path = self.log_dir / f"{config.LOG_FILE_PREFIX}_{timestamp}.log"

        if self._file_handler is not None:
            logger.removeHandler(self._file_handler)
            self._file_handler.close()

        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(config.LOG_LEVEL)
        self._file_handler = handler
        return log_path

    def _write_run_summary(
        self, summary: IngestionSummary, started: datetime
    ) -> Path:
        finished = datetime.now(timezone.utc)
        summary.started_at = started.isoformat()
        summary.finished_at = finished.isoformat()
        summary.duration_seconds = round((finished - started).total_seconds(), 3)

        summary_path = self.log_dir / f"run_{summary.started_at.replace(':', '').replace('.', '')}.json"
        payload = {
            "started_at": summary.started_at,
            "finished_at": summary.finished_at,
            "duration_seconds": summary.duration_seconds,
            "processed": summary.processed_count,
            "skipped": summary.skipped_count,
            "failed": summary.failed_count,
            "processed_reports": summary.processed_reports,
            "skipped_reports": summary.skipped_reports,
            "failed_reports": summary.failed_reports,
            "warnings": summary.warnings,
            "char_limit_warnings": summary.char_limit_warnings,
            "pii_redactions": summary.pii_redactions,
            "name_redactions": summary.name_redactions(),
            "chunks_per_report": summary.chunks_per_report,
            "total_chunks": sum(summary.chunks_per_report.values()),
            "max_tokens_observed": summary.max_tokens_recorded,
            "log_file": summary.log_path,
        }
        summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary.summary_path = str(summary_path)
        logger.info("Run summary written to %s", summary_path)
        return summary_path

    def run(self) -> IngestionSummary:
        """
        Run the ingestion pipeline across all documents in report_documents.
        """
        started = datetime.now(timezone.utc)
        log_path = self._start_run_logging()

        summary = IngestionSummary()
        summary.log_path = str(log_path)
        logger.info("Starting Lab Prestasi ingestion batch...")

        # 1. Read report_documents and competition_reports
        docs_rows = self.sheets_repo.get_report_documents()
        reports_meta = {r.report_id: r for r in self.sheets_repo.get_competition_reports()}

        for doc_row in docs_rows:
            report_id = doc_row.report_id
            drive_file_id = doc_row.drive_file_id
            current_status = doc_row.rag_status

            logger.info("Inspecting report_id: %s (status: %s)", report_id, current_status.value)

            # Check matching metadata in competition_reports
            if report_id not in reports_meta:
                reason = "Tidak ada baris pasangan di competition_reports"
                logger.error("Report %s failed: %s", report_id, reason)
                self.sheets_repo.update_rag_status(report_id, RagStatus.PROSES_GAGAL)
                summary.failed_count += 1
                summary.failed_reports.append({"report_id": report_id, "reason": reason})
                continue

            meta = reports_meta[report_id]

            # Brief 5.2 #3: If competition_type = Lainnya, skip + warning.
            # The row is kept defensively for production data that should not
            # contain it; rag_status is deliberately left unchanged.
            if str(meta.competition_type).strip() == CompetitionType.LAINNYA.value:
                reason = "competition_type = 'Lainnya' tidak pernah diproses ke RAG"
                logger.warning("Report %s dilewati: %s", report_id, reason)
                summary.skipped_count += 1
                summary.skipped_reports.append({"report_id": report_id, "reason": reason})
                continue

            # Brief 5.2 #3: metadata enum and year validation.
            metadata_ok, metadata_reason = self.validator.validate_metadata(
                str(meta.competition_type),
                str(meta.achievement_level),
                meta.competition_year,
            )
            if not metadata_ok:
                logger.error("Report %s failed: %s", report_id, metadata_reason)
                self.sheets_repo.update_rag_status(report_id, RagStatus.PROSES_GAGAL)
                summary.failed_count += 1
                summary.failed_reports.append(
                    {"report_id": report_id, "reason": metadata_reason}
                )
                continue

            competition_year = int(meta.competition_year)

            # Brief 5.2 #2: Determine if processing is needed
            stored_source_file_id = self.chroma_store.get_source_file_id_for_report(report_id)
            if current_status == RagStatus.SELESAI and stored_source_file_id == drive_file_id:
                reason = "Dokumen Selesai dengan drive_file_id tidak berubah (dilewati)"
                logger.info("Report %s dilewati: %s", report_id, reason)
                summary.skipped_count += 1
                summary.skipped_reports.append({"report_id": report_id, "reason": reason})
                continue

            # Step 4: Download PDF (local temp name keyed by report_id, not drive id)
            local_pdf_path = self.temp_download_dir / f"{report_id}.pdf"
            download_ok = self.drive_repo.download_pdf(drive_file_id, str(local_pdf_path))
            if not download_ok:
                reason = f"Gagal mengunduh file PDF dari Drive (file_id: {drive_file_id})"
                logger.error("Report %s failed: %s", report_id, reason)
                self.sheets_repo.update_rag_status(report_id, RagStatus.PROSES_GAGAL)
                summary.failed_count += 1
                summary.failed_reports.append({"report_id": report_id, "reason": reason})
                continue

            # Extract text & validate readability
            try:
                cleaned_text = self.pdf_parser.extract_and_clean_text(local_pdf_path)
                parse_ok, sections, parse_err = self.pdf_parser.parse_sections(
                    cleaned_text
                )
            except Exception as e:
                reason = f"Gagal membaca PDF: {str(e)}"
                logger.error("Report %s failed during parsing: %s", report_id, reason)
                self.sheets_repo.update_rag_status(report_id, RagStatus.PROSES_GAGAL)
                summary.failed_count += 1
                summary.failed_reports.append({"report_id": report_id, "reason": reason})
                continue

            if not parse_ok:
                logger.error("Report %s parsing failed: %s", report_id, parse_err)
                self.sheets_repo.update_rag_status(report_id, RagStatus.PROSES_GAGAL)
                summary.failed_count += 1
                summary.failed_reports.append({"report_id": report_id, "reason": parse_err})
                continue

            # Check character limits (warning only)
            char_warnings = self.validator.check_character_limits(sections)
            for w in char_warnings:
                logger.warning("[%s] %s", report_id, w)
                summary.warnings.append(f"[{report_id}] {w}")
                summary.char_limit_warnings.append(f"[{report_id}] {w}")

            # Step 8: PII Scrubbing
            scrubbed_sections = {}
            report_redactions = {
                category: 0 for category in self.pii_scrubber.CATEGORIES
            }
            for sec_name, text in sections.items():
                s_text, counts = self.pii_scrubber.scrub(text)
                scrubbed_sections[sec_name] = s_text
                for k, v in counts.items():
                    report_redactions[k] += v

            total_redactions = sum(report_redactions.values())
            if total_redactions > 0:
                logger.info("[%s] Terdeteksi %d PII diredaksi: %s", report_id, total_redactions, report_redactions)
                summary.pii_redactions[report_id] = report_redactions

            # Step 9-10: Chunking and embedding.
            # A failure here (malformed section, model unavailable) must fail only
            # this document and leave the batch running (Brief 5.2 #11).
            try:
                judul_ide = scrubbed_sections.get("judul_ide", "")
                report_chunks = []

                for sec_type in [
                    SectionType.EXECUTIVE_SUMMARY,
                    SectionType.EVALUASI_TIM,
                    SectionType.SARAN_PENGEMBANGAN,
                ]:
                    sec_text = scrubbed_sections.get(sec_type.value, "")
                    if not sec_text:
                        continue

                    chunks = self.chunker.chunk_section(
                        report_id=report_id,
                        competition_name=meta.competition_name,
                        competition_year=competition_year,
                        competition_type=str(meta.competition_type),
                        achievement_level=str(meta.achievement_level),
                        source_file_id=drive_file_id,
                        judul_ide=judul_ide,
                        section_type=sec_type,
                        section_content=sec_text,
                    )
                    report_chunks.extend(chunks)

                if not report_chunks:
                    raise ValueError("Tidak ada chunk yang dihasilkan dari dokumen")

                texts_to_embed = [c.embed_text for c in report_chunks]
                embeddings = self.embedder.embed_passages(texts_to_embed)
            except Exception as e:
                reason = f"Gagal chunking/embedding: {str(e)}"
                logger.error("Report %s failed during chunking: %s", report_id, reason)
                self.sheets_repo.update_rag_status(report_id, RagStatus.PROSES_GAGAL)
                summary.failed_count += 1
                summary.failed_reports.append({"report_id": report_id, "reason": reason})
                continue

            # Record chunk stats only after chunking succeeded
            summary.chunks_per_report[report_id] = len(report_chunks)
            for c in report_chunks:
                if c.token_count > summary.max_tokens_recorded:
                    summary.max_tokens_recorded = c.token_count

            # Step 10: Delete-then-insert into ChromaDB
            try:
                self.chroma_store.insert_chunks(
                    chunks=report_chunks,
                    embeddings=embeddings,
                )
            except Exception as e:
                reason = f"Gagal menyimpan ke ChromaDB: {str(e)}"
                logger.error("Report %s failed during Chroma storage: %s", report_id, reason)
                self.sheets_repo.update_rag_status(report_id, RagStatus.PROSES_GAGAL)
                summary.failed_count += 1
                summary.failed_reports.append({"report_id": report_id, "reason": reason})
                continue

            # Status 'Selesai' ONLY after all chunks safely stored
            self.sheets_repo.update_rag_status(report_id, RagStatus.SELESAI)
            summary.processed_count += 1
            summary.processed_reports.append(report_id)
            logger.info("Successfully ingested report %s (%d chunks)", report_id, len(report_chunks))

        logger.info(
            "Batch finished: %d processed, %d skipped, %d failed",
            summary.processed_count,
            summary.skipped_count,
            summary.failed_count,
        )
        self._write_run_summary(summary, started)
        return summary
