"""
Unit tests for configuration module.
Verifies all limits, regex patterns, and constants.
"""

import pytest
import config
from enums import SectionType


def test_config_constants():
    assert config.EMBEDDING_MODEL_NAME == "intfloat/multilingual-e5-small"
    assert config.LLM_MODEL_NAME == "openai/gpt-oss-20b"
    assert config.EMBEDDING_PASSAGE_PREFIX == "passage: "
    assert config.EMBEDDING_QUERY_PREFIX == "query: "
    assert config.CHUNK_HARD_LIMIT_TOKENS == 512
    assert config.CHUNK_TARGET_MIN_TOKENS <= config.CHUNK_TARGET_MAX_TOKENS


def test_report_id_regex():
    pattern = config.PATTERN_REPORT_ID
    assert pattern.match("RPT0001")
    assert pattern.match("RPT12345")
    assert not pattern.match("RPT12")  # Less than 4 digits
    assert not pattern.match("REPORT0001")


def test_citation_tag_regex():
    pattern = config.PATTERN_CITATION_TAG
    matches = pattern.findall("Menurut laporan [S1] dan juga [S23], kendala utama...")
    assert matches == ["S1", "S23"]


def test_pii_patterns():
    # Test NIM detection
    nim_text = "NIM mahasiswa: 13520099."
    assert config.PATTERN_PII_NIM.search(nim_text)

    # Test Phone detection
    phone_text = "Kontak: 081234567890 atau +6281198765432"
    phones = config.PATTERN_PII_PHONE.findall(phone_text)
    assert len(phones) == 2

    # Test Email detection
    email_text = "Email pelapor adalah budi.santoso@univ.ac.id"
    emails = config.PATTERN_PII_EMAIL.findall(email_text)
    assert emails == ["budi.santoso@univ.ac.id"]


def test_section_patterns():
    patterns = config.SECTION_PATTERNS

    # Judul Ide
    assert patterns["judul_ide"].search("Judul Ide: Inovasi Pintar")
    assert patterns["judul_ide"].search("1. Judul Ide")

    # Executive Summary / Abstrak
    assert patterns["executive_summary"].search("Executive Summary:")
    assert patterns["executive_summary"].search("2. Abstrak")

    # Evaluasi Tim
    assert patterns["evaluasi_tim"].search("Evaluasi Tim")
    assert patterns["evaluasi_tim"].search("3) Evaluasi Tim :")

    # Saran Pengembangan
    assert patterns["saran_pengembangan"].search("Saran Pengembangan")
    assert patterns["saran_pengembangan"].search("4. Saran Pengembangan")
