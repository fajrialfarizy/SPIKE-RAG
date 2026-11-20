"""
Configuration module for Lab Prestasi RAG MVP.
Centralizes all parameters, thresholds, regex patterns, and constants.
No magic numbers allowed.
"""

from pathlib import Path
import re

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DUMMY_PDF_DIR = DATA_DIR / "dummy_pdfs"
TESTS_DIR = BASE_DIR / "tests"
FIXTURES_DIR = TESTS_DIR / "fixtures"

# Model Configuration
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_PASSAGE_PREFIX = "passage: "
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_DIMENSION = 384

# LLM Configuration (Groq)
LLM_MODEL_NAME = "openai/gpt-oss-20b"
LLM_TEMPERATURE_ANALYZER = 0.0
LLM_TEMPERATURE_CHAT = 0.2
LLM_MAX_OUTPUT_TOKENS = 1024
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF_FACTOR = 2.0
LLM_TIMEOUT_SECONDS = 30.0

# Chunking Parameters
CHUNK_TARGET_MIN_TOKENS = 350
CHUNK_TARGET_MAX_TOKENS = 450
CHUNK_HARD_LIMIT_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 0

# Character Limits per Section (PRD 6.2)
SECTION_CHAR_LIMITS = {
    "judul_ide": 200,
    "executive_summary": 2000,
    "evaluasi_tim": 3000,
    "saran_pengembangan": 3000,
}

# PDF Minimum Text Length for Scanned Document Detection
PDF_MIN_TEXT_CHARS = 100

# Google Sheets schema (Brief 5.1)
# Only these columns may cross the repository boundary. PII and administrative
# columns are never fetched, never modelled and never logged.
WORKSHEET_COMPETITION_REPORTS = "competition_reports"
WORKSHEET_REPORT_DOCUMENTS = "report_documents"
SHEET_KEY_COLUMN = "report_id"
SHEET_COLUMN_RAG_STATUS = "rag_status"
SHEET_COLUMNS_COMPETITION_REPORTS = (
    "report_id",
    "competition_name",
    "competition_year",
    "achievement_level",
    "competition_type",
)
SHEET_COLUMNS_REPORT_DOCUMENTS = (
    "report_id",
    "drive_file_id",
    "rag_status",
)

# ChromaDB Settings
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "lab_prestasi_evaluations"
CHROMA_DISTANCE_METRIC = "cosine"
CHROMA_METADATA_METRIC_KEY = "hnsw:space"
CHROMA_METADATA_EMBEDDING_MODEL_KEY = "embedding_model"
MANIFEST_SCHEMA_VERSION = "1.0"
MANIFEST_FILE_NAME = "manifest.json"
MANIFEST_CHECK_INTERVAL_SECONDS = 300  # 5 minutes TTL

# Chat & Retrieval Parameters
CHAT_MAX_INPUT_CHARS = 1000
MAX_REPORT_IDS_BYPASS = 3

# Specific Query Retrieval Settings
RETRIEVAL_SPECIFIC_K = 6
RETRIEVAL_SPECIFIC_MAX_CHUNKS_PER_REPORT = 3
RETRIEVAL_CANDIDATE_MULTIPLIER = 3

# Cross Report Query Retrieval Settings
RETRIEVAL_CROSS_REPORT_K = 12
RETRIEVAL_CROSS_REPORT_MAX_CHUNKS_PER_REPORT = 2

# Catalog Mode Settings
RETRIEVAL_CATALOG_MAX_REPORTS = 50

# Similarity threshold (to be tuned in Phase 4; initial conservative threshold)
RETRIEVAL_SIMILARITY_THRESHOLD = 0.50

# Regex Patterns
PATTERN_REPORT_ID = re.compile(r"RPT[0-9]{4,}")
PATTERN_CITATION_TAG = re.compile(r"\[(S[0-9]+)\]")
PATTERN_YEAR = re.compile(r"\b(20[1-9][0-9])\b")

# PII Scrubbing Regex Patterns (Indonesian Context)
# NIM in Indonesian universities typically ranges from 8 to 14 digits (e.g., ITB 8 digits, UI 10 digits, UGM 9-11 digits)
PATTERN_PII_NIM = re.compile(r"\b[0-9]{8,14}\b")
PATTERN_PII_PHONE = re.compile(r"(?:\+62|62|08)[0-9]{8,12}\b")
PATTERN_PII_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

PII_REDACTION_REPLACEMENT = "[DIREDAKSI]"

# Section heading detection (Brief 5.2 #6)
# Tolerant to numbering (arabic or roman), ``1.``/``2)``/``III -`` styles, mixed case,
# extra whitespace and an optional trailing colon or dash.
SECTION_HEADING_LABELS = {
    "judul_ide": ("Judul Ide",),
    "executive_summary": (
        "Executive Summary / Abstrak",
        "Executive Summary",
        "Abstrak",
    ),
    "evaluasi_tim": ("Evaluasi Tim",),
    "saran_pengembangan": ("Saran Pengembangan",),
}

# Optional numbering prefix: "1.", "2)", "3 -", "IV.", "III -"
PATTERN_SECTION_NUMBER_PREFIX = (
    r"(?:(?:\d{1,2}|[IVXLC]{1,6})\s*[\.\)\-\u2013\u2014:]\s*)?"
)


def _build_heading_pattern(labels: tuple) -> "re.Pattern[str]":
    alternatives = "|".join(
        r"\s+".join(re.escape(part) for part in label.split()) for label in labels
    )
    return re.compile(
        rf"^[ \t]*{PATTERN_SECTION_NUMBER_PREFIX}(?:{alternatives})[ \t]*[:\-\u2013\u2014]?[ \t]*",
        re.IGNORECASE | re.MULTILINE,
    )


SECTION_PATTERNS = {
    name: _build_heading_pattern(labels)
    for name, labels in SECTION_HEADING_LABELS.items()
}

# Page numbers and running headers/footers (Brief 5.2 #4)
PATTERN_PAGE_NUMBER = re.compile(
    r"^[ \t]*(?:Halaman|Page|Hal\.?)[ \t]*[-\u2013\u2014]?[ \t]*\d{1,3}"
    r"(?:[ \t]*(?:dari|of|/)[ \t]*\d{1,3})?[ \t]*[-\u2013\u2014]?[ \t]*$"
    r"|^[ \t]*[-\u2013\u2014]?[ \t]*\d{1,3}[ \t]*[-\u2013\u2014]?[ \t]*$",
    re.MULTILINE,
)
# Lines in the first/last N lines of a page that repeat on most pages are margins.
REPEATED_MARGIN_LINES = 3
REPEATED_MARGIN_RATIO = 0.6
REPEATED_MARGIN_MIN_PAGES = 2

# Authentication Settings
AUTH_MAX_FAILED_ATTEMPTS = 5
AUTH_LOCKOUT_DURATION_SECONDS = 300  # 5 minutes
