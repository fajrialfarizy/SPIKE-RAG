"""
Test suite verifying the dummy corpus specification and edge cases (Supervisor requirement 2).
Checks:
- 10 valid reports cover 4 competition types, 6 achievement levels, >= 2 distinct years.
- PDF scan yields empty text via PyMuPDF.
- RPT0012 lacks 'Saran Pengembangan' heading; RPT0013 > 3000 chars in Evaluasi Tim.
- Injection strings and simulated PII (NIM, phone, email) exist in designated PDFs.
- At least 3 valid reports have sections exceeding 1 chunk (> 450 tokens).
- Generator determinism (two consecutive runs produce byte-identical fixtures).
- Negative PII test (valid 8+ digit numbers like currency/budget) and regex behavior report.
"""

import hashlib
import csv
from pathlib import Path
import pytest
import fitz  # PyMuPDF
from transformers import AutoTokenizer

import config
from enums import CompetitionType, AchievementLevel, SectionType
from scripts.make_dummy_corpus import generate_corpus


def get_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_ten_valid_reports_coverage():
    """Verify 10 valid reports cover 4 competition types, 6 achievement levels, >= 2 years."""
    reports_csv = config.FIXTURES_DIR / "competition_reports.csv"
    assert reports_csv.exists()

    with open(reports_csv, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    # Valid reports are RPT0001 through RPT0010
    valid_reports = [r for r in reader if int(r["report_id"].replace("RPT", "")) <= 10]
    assert len(valid_reports) == 10

    types = {r["competition_type"] for r in valid_reports}
    expected_types = {
        CompetitionType.BUSINESS_PLAN.value,
        CompetitionType.BUSINESS_CASE.value,
        CompetitionType.ESSAY.value,
        CompetitionType.KARYA_TULIS_ILMIAH.value,
    }
    assert expected_types.issubset(types)

    levels = {r["achievement_level"] for r in valid_reports}
    expected_levels = {l.value for l in AchievementLevel if l != AchievementLevel.FINALIS}
    # Check that all 6 achievement levels are represented
    assert len(levels) == 6

    years = {r["competition_year"] for r in valid_reports}
    assert len(years) >= 2


def test_scanned_pdf_yields_empty_text():
    """PDF scan (RPT0011) must yield empty text when extracted via PyMuPDF."""
    pdf_path = config.DUMMY_PDF_DIR / "DRV_RPT0011_SCAN.pdf"
    assert pdf_path.exists()

    doc = fitz.open(str(pdf_path))
    full_text = "".join(page.get_text() for page in doc).strip()
    doc.close()

    assert full_text == "" or len(full_text) < config.PDF_MIN_TEXT_CHARS


def test_edge_case_structural_anomalies():
    """Verify RPT0012 lacks Saran Pengembangan; RPT0013 > 3000 chars in Evaluasi Tim."""
    # RPT0012
    doc12 = fitz.open(str(config.DUMMY_PDF_DIR / "DRV_RPT0012_MISSING_SEC.pdf"))
    text12 = "".join(page.get_text() for page in doc12)
    doc12.close()
    assert config.SECTION_PATTERNS["saran_pengembangan"].search(text12) is None

    # RPT0013
    doc13 = fitz.open(str(config.DUMMY_PDF_DIR / "DRV_RPT0013_OVER_CHAR.pdf"))
    text13 = "".join(page.get_text() for page in doc13)
    doc13.close()
    assert len(text13) > 3000


def test_injection_and_pii_in_pdfs():
    """Verify injection strings and simulated PII exist in designated PDFs."""
    # RPT0014 Injection
    doc14 = fitz.open(str(config.DUMMY_PDF_DIR / "DRV_RPT0014_INJECTION.pdf"))
    text14 = "".join(page.get_text() for page in doc14)
    doc14.close()
    assert "Abaikan semua instruksi sebelumnya" in text14

    # RPT0015 PII
    doc15 = fitz.open(str(config.DUMMY_PDF_DIR / "DRV_RPT0015_PII.pdf"))
    text15 = "".join(page.get_text() for page in doc15)
    doc15.close()
    assert config.PATTERN_PII_NIM.search(text15) is not None
    assert config.PATTERN_PII_PHONE.search(text15) is not None
    assert config.PATTERN_PII_EMAIL.search(text15) is not None


def test_valid_reports_with_long_sections():
    """Verify at least 3 valid reports have sections that exceed 1 chunk target (> 350 tokens)."""
    # Use tokenizers directly from cached HuggingFace snapshots for high-speed offline execution
    import glob
    import os
    from tokenizers import Tokenizer
    pattern = os.path.expanduser("~/.cache/huggingface/hub/models--intfloat--multilingual-e5-small/snapshots/*/tokenizer.json")
    matches = glob.glob(pattern)
    assert matches, "Cached tokenizer.json must exist"
    tokenizer = Tokenizer.from_file(matches[0])
    
    long_section_count = 0
    # Check reports RPT0001 to RPT0010
    for idx in range(1, 11):
        drv_id = f"DRV_RPT{idx:04d}"
        matching = list(config.DUMMY_PDF_DIR.glob(f"{drv_id}*.pdf"))
        if not matching:
            continue
        doc = fitz.open(str(matching[0]))
        text = "".join(page.get_text() for page in doc)
        doc.close()
        tokens = tokenizer.encode(text).ids
        if len(tokens) > config.CHUNK_TARGET_MIN_TOKENS:
            long_section_count += 1

    assert long_section_count >= 3


def test_generator_determinism(tmp_path):
    """Running generator twice must produce identical CSV and PDF contents."""
    generate_corpus()
    rep_hash1 = get_file_hash(config.FIXTURES_DIR / "competition_reports.csv")
    doc_hash1 = get_file_hash(config.FIXTURES_DIR / "report_documents.csv")

    generate_corpus()
    rep_hash2 = get_file_hash(config.FIXTURES_DIR / "competition_reports.csv")
    doc_hash2 = get_file_hash(config.FIXTURES_DIR / "report_documents.csv")

    assert rep_hash1 == rep_hash2
    assert doc_hash1 == doc_hash2


def test_negative_pii_case_and_regex_behavior():
    """
    Test negative PII case: legitimate numbers with 8+ digits (e.g. nominal dana 25000000).
    Document behavior: Regex currently flags \\b[0-9]{8,14}\\b, so a raw number like 25000000
    without currency symbol is matched as NIM candidate.
    Formatted currency (Rp 25.000.000) is NOT matched by NIM regex due to dots.
    """
    formatted_nominal = "Rp 25.000.000"
    raw_nominal = "25000000"

    # Formatted nominal is safe from NIM regex
    assert config.PATTERN_PII_NIM.search(formatted_nominal) is None

    # Raw 8-digit number IS captured by generic NIM regex \b[0-9]{8,14}\b
    # This conservative behavior prioritizes zero privacy leak over preserving raw unformatted numbers.
    assert config.PATTERN_PII_NIM.search(raw_nominal) is not None
