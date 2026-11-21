"""
Dummy corpus specification tests (Phase 0 corrections + Phase 1 acceptance).

Everything here is offline. The single model-dependent assertion (a section that
needs more than one chunk, measured with the real e5 tokenizer) is marked `slow`.
"""

import csv
import hashlib
from pathlib import Path

import pymupdf
import pytest

import config
from enums import AchievementLevel, CompetitionType, SectionType
from ingestion.chunker import SectionAwareChunker
from ingestion.pdf_parser import PDFParser
from scripts.make_dummy_corpus import generate_corpus


@pytest.fixture(scope="module", autouse=True)
def corpus():
    generate_corpus()


def pdf_path(prefix: str) -> Path:
    matches = sorted(config.DUMMY_PDF_DIR.glob(f"{prefix}*.pdf"))
    assert matches, f"No dummy PDF for {prefix}"
    return matches[0]


def page_texts(path: Path):
    document = pymupdf.open(str(path))
    try:
        return [page.get_text() for page in document]
    finally:
        document.close()


def test_ten_valid_reports_cover_types_levels_and_years():
    with open(config.FIXTURES_DIR / "competition_reports.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    valid = [r for r in rows if int(r["report_id"].replace("RPT", "")) <= 10]
    assert len(valid) == 10

    assert {r["competition_type"] for r in valid} >= {
        CompetitionType.BUSINESS_PLAN.value,
        CompetitionType.BUSINESS_CASE.value,
        CompetitionType.ESSAY.value,
        CompetitionType.KARYA_TULIS_ILMIAH.value,
    }
    assert {r["achievement_level"] for r in valid} == {
        level.value for level in AchievementLevel
    }
    assert len({r["competition_year"] for r in valid}) >= 2

    # The Lainnya row exists in BOTH worksheets (supervisor decision Q1).
    assert any(r["competition_type"] == CompetitionType.LAINNYA.value for r in rows)
    with open(config.FIXTURES_DIR / "report_documents.csv", encoding="utf-8") as f:
        docs = list(csv.DictReader(f))
    assert any(r["report_id"] == "RPT0099" for r in docs)
    assert set(docs[0]) == {"report_id", "drive_file_id", "rag_status"}


def test_scanned_pdf_yields_no_text():
    text = "".join(page_texts(pdf_path("DRV_RPT0011"))).strip()

    assert len(text) < config.PDF_MIN_TEXT_CHARS


def test_missing_section_document():
    text = "".join(page_texts(pdf_path("DRV_RPT0012")))

    assert config.SECTION_PATTERNS["saran_pengembangan"].search(text) is None
    success, _, reason = PDFParser().parse_sections(PDFParser().extract_and_clean_text(pdf_path("DRV_RPT0012")))
    assert success is False
    assert "saran_pengembangan" in reason


def test_over_limit_section_is_preserved_and_warned():
    parser = PDFParser()
    path = pdf_path("DRV_RPT0013")
    raw = "".join(page_texts(path))
    sections = parser.parse_sections(parser.extract_and_clean_text(path))[1]

    assert len(sections["evaluasi_tim"]) > 3000
    # Content is not destroyed by margin cleaning (regression guard).
    assert len(sections["evaluasi_tim"]) > len(raw) * 0.8
    assert len(sections) == 4


def test_injection_sentence_and_pii_exist_in_designated_pdfs():
    injection = "".join(page_texts(pdf_path("DRV_RPT0014")))
    assert "Abaikan semua instruksi sebelumnya" in injection

    pii = "".join(page_texts(pdf_path("DRV_RPT0015")))
    assert config.PATTERN_PII_NIM_LABELED.search(pii)
    assert config.PATTERN_PII_PHONE.search(pii)
    assert config.PATTERN_PII_EMAIL.search(pii)
    assert "Budi Santoso" in pii
    assert "Dr. Ir. Hendra" in pii


def test_negative_pii_case_is_in_the_corpus():
    """
    The funding amount is a legitimate 8-digit number. The NIM pattern alone
    matches it, which is why the scrubber needs amount context; this test pins
    the corpus fact and the regex behaviour that the scrubber test relies on.
    """
    saran = PDFParser().parse_sections(
        PDFParser().extract_and_clean_text(pdf_path("DRV_RPT0001"))
    )[1]["saran_pengembangan"]

    assert "Rp 25.000.000" in saran
    assert "25000000" in saran
    # Formatted currency is never matched by the digit-only pattern.
    assert config.PATTERN_PII_NIM.search("Rp 25.000.000") is None
    # The raw form is matched, hence the context exemption in the scrubber.
    assert config.PATTERN_PII_NIM.search("25000000") is not None


@pytest.mark.slow
def test_at_least_three_valid_reports_have_a_section_longer_than_one_chunk():
    """Measured per SECTION with the real e5 tokenizer, not per whole document."""
    parser = PDFParser()
    chunker = SectionAwareChunker()

    multi_chunk_reports = set()
    for index in range(1, 11):
        report_id = f"RPT{index:04d}"
        path = pdf_path(f"DRV_{report_id}")
        sections = parser.parse_sections(parser.extract_and_clean_text(path))[1]

        for section_name, content in sections.items():
            if section_name == "judul_ide":
                continue
            chunks = chunker.chunk_section(
                report_id=report_id,
                competition_name="Lomba Contoh",
                competition_year=2024,
                competition_type="Essay",
                achievement_level="Juara 1",
                source_file_id=path.name,
                judul_ide=sections["judul_ide"],
                section_type=SectionType(section_name),
                section_content=content,
            )
            assert all(
                chunk.token_count <= config.CHUNK_HARD_LIMIT_TOKENS for chunk in chunks
            )
            if len(chunks) > 1:
                multi_chunk_reports.add(report_id)

    assert len(multi_chunk_reports) >= 3, multi_chunk_reports


def test_generator_is_deterministic_at_content_level():
    generate_corpus()
    first_pages = {p.name: page_texts(p) for p in sorted(config.DUMMY_PDF_DIR.glob("*.pdf"))}
    first_reports = (config.FIXTURES_DIR / "competition_reports.csv").read_bytes()
    first_docs = (config.FIXTURES_DIR / "report_documents.csv").read_bytes()

    generate_corpus()
    second_pages = {p.name: page_texts(p) for p in sorted(config.DUMMY_PDF_DIR.glob("*.pdf"))}
    second_reports = (config.FIXTURES_DIR / "competition_reports.csv").read_bytes()
    second_docs = (config.FIXTURES_DIR / "report_documents.csv").read_bytes()

    assert first_reports == second_reports
    assert first_docs == second_docs
    assert first_pages == second_pages
    assert hashlib.sha256(first_reports).hexdigest() == hashlib.sha256(second_reports).hexdigest()


# -- fix pack L1: reflow, sentence-final chunks and the Q&A artifact --------
def test_the_ampersand_artifact_is_gone_from_the_source_text():
    parser = PDFParser()
    joined = ""
    for path in sorted(config.DUMMY_PDF_DIR.glob("DRV_RPT00*.pdf")):
        success, sections, _ = parser.parse_sections(
            parser.extract_and_clean_text(path)
        )
        if success:
            joined += " ".join(sections.values())

    assert "Q&A;" not in joined
    assert "Q&A" in joined


def test_no_dummy_section_keeps_a_blank_line_between_sentences():
    parser = PDFParser()
    for path in sorted(config.DUMMY_PDF_DIR.glob("DRV_RPT00*.pdf")):
        success, sections, _ = parser.parse_sections(
            parser.extract_and_clean_text(path)
        )
        if not success:
            continue
        for name, text in sections.items():
            for line in text.splitlines():
                assert "  " not in line, (path.name, name, line)


def test_every_dummy_chunk_ends_at_a_sentence_or_list_item_end():
    from tests.stubs import WordCountTokenizer

    parser = PDFParser()
    chunker = SectionAwareChunker(tokenizer=WordCountTokenizer())
    offenders = []

    for index in range(1, 11):
        report_id = f"RPT{index:04d}"
        path = pdf_path(f"DRV_{report_id}")
        sections = parser.parse_sections(parser.extract_and_clean_text(path))[1]
        for section_name in ("executive_summary", "evaluasi_tim", "saran_pengembangan"):
            chunks = chunker.chunk_section(
                report_id=report_id,
                competition_name="Lomba Contoh",
                competition_year=2024,
                competition_type="Essay",
                achievement_level="Juara 1",
                source_file_id=path.name,
                judul_ide=sections["judul_ide"],
                section_type=SectionType(section_name),
                section_content=sections[section_name],
            )
            offenders.extend(chunker.non_terminal_chunk_ids)
            for chunk in chunks:
                body = chunk.document_text.split("\n\n", 1)[1]
                # Blank lines only separate real paragraphs, never wrapped lines.
                assert "\n\n\n" not in body

    assert offenders == []
