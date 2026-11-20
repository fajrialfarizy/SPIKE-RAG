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
    pii_redactions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    chunks_per_report: Dict[str, int] = field(default_factory=dict)
    max_tokens_recorded: int = 0


class IngestionPipeline:
    """Executes end-to-end ingestion batch process."""

    def __init__(
        self,
        sheets_repo: SheetsRepo,
        drive_repo: DriveRepo,
        chroma_store: ChromaStore,
        embedder: Optional[E5Embedder] = None,
        temp_download_dir: Optional[Path] = None,
    ):
        self.sheets_repo = sheets_repo
        self.drive_repo = drive_repo
        self.chroma_store = chroma_store
        self.embedder = embedder or E5Embedder()
        self.pdf_parser = PDFParser()
        self.validator = IngestionValidator()
        self.pii_scrubber = PIIScrubber()
        self.chunker = SectionAwareChunker()
        self.temp_download_dir = temp_download_dir or (config.DATA_DIR / "temp_downloads")
        self.temp_download_dir.mkdir(parents=True, exist_ok=True)

    def run(self, simulate_failure_on_report: Optional[str] = None) -> IngestionSummary:
        """
        Run the ingestion pipeline across all documents in report_documents.
        """
        summary = IngestionSummary()
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

            # Brief 5.2 #3: If competition_type = Lainnya, skip + warning
            if meta.competition_type == CompetitionType.LAINNYA:
                reason = "competition_type = 'Lainnya' tidak pernah diproses ke RAG"
                logger.warning("Report %s dilewati: %s", report_id, reason)
                summary.skipped_count += 1
                summary.skipped_reports.append({"report_id": report_id, "reason": reason})
                continue

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
            cleaned_text = self.pdf_parser.extract_and_clean_text(local_pdf_path)
            parse_ok, sections, parse_err = self.pdf_parser.parse_sections(cleaned_text)

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

            # Step 8: PII Scrubbing
            scrubbed_sections = {}
            report_redactions = {"nim": 0, "phone": 0, "email": 0}
            for sec_name, text in sections.items():
                s_text, counts = self.pii_scrubber.scrub(text)
                scrubbed_sections[sec_name] = s_text
                for k, v in counts.items():
                    report_redactions[k] += v

            total_redactions = sum(report_redactions.values())
            if total_redactions > 0:
                logger.info("[%s] Terdeteksi %d PII diredaksi: %s", report_id, total_redactions, report_redactions)
                summary.pii_redactions[report_id] = report_redactions

            # Step 9: Chunking
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
                    competition_year=meta.competition_year,
                    competition_type=meta.competition_type.value,
                    achievement_level=meta.achievement_level.value,
                    source_file_id=drive_file_id,
                    judul_ide=judul_ide,
                    section_type=sec_type,
                    section_content=sec_text,
                )
                report_chunks.extend(chunks)

            # Record chunk stats
            summary.chunks_per_report[report_id] = len(report_chunks)
            for c in report_chunks:
                if c.token_count > summary.max_tokens_recorded:
                    summary.max_tokens_recorded = c.token_count

            # Embedding
            texts_to_embed = [c.embed_text for c in report_chunks]
            embeddings = self.embedder.embed_passages(texts_to_embed)

            # Step 10: Delete-then-insert into ChromaDB
            should_fail = (simulate_failure_on_report == report_id)
            try:
                self.chroma_store.insert_chunks(
                    chunks=report_chunks,
                    embeddings=embeddings,
                    simulate_failure=should_fail,
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
        return summary
