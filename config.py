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
# Catalog reads metadata only; this bounds a single Chroma get() before dedupe.
RETRIEVAL_CATALOG_FETCH_LIMIT = 500

# Similarity threshold (similarity = 1 - cosine distance).
# TODO(Phase 4): this is a conservative placeholder. It must be tuned from the
# evaluation set (Brief 5.5 / PRD 8.1), not guessed; do not treat it as final.
RETRIEVAL_SIMILARITY_THRESHOLD = 0.50

# Regex Patterns
PATTERN_REPORT_ID = re.compile(r"RPT[0-9]{4,}")
PATTERN_CITATION_TAG = re.compile(r"\[(S[0-9]+)\]")
PATTERN_YEAR = re.compile(r"\b(20[1-9][0-9])\b")

# PII Scrubbing (Brief 5.2 #8, PRD 6.3)
# Order matters: email -> phone -> labelled NIM -> unlabelled NIM -> names.
PII_REDACTION_REPLACEMENT = "[DIREDAKSI]"

PATTERN_PII_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PATTERN_PII_PHONE = re.compile(r"(?:\+62|62|08)[0-9]{8,12}\b")

# NIM/NPM/NRP are always redacted when a label is present.
PATTERN_PII_NIM_LABELED = re.compile(
    r"(?i)\b(?:(?:NIM|NPM|NRP)\b|nomor\s+induk)\s*[:.=\-]?\s*(?P<number>[0-9]{8,14})"
)
# Unlabelled 8-14 digit numbers are treated as NIM unless an amount context
# precedes them (see PII_CURRENCY_CONTEXT_WORDS). Numbers written with thousand
# separators (25.000.000) never match this pattern.
PATTERN_PII_NIM = re.compile(r"\b[0-9]{8,14}\b")
PII_CURRENCY_CONTEXT_WORDS = (
    "rp",
    "idr",
    "usd",
    "rupiah",
    "dana",
    "biaya",
    "anggaran",
    "hadiah",
    "nominal",
    "sebesar",
    "senilai",
)
PII_CURRENCY_CONTEXT_WINDOW = 40

# Personal names. No NER dependency: honourifics, role labels, academic
# suffixes and an optional deny-list are used instead (residual risk documented).
PII_NAME_TOKEN = r"[A-Z][A-Za-z'\-]{1,}"
PII_NAME_SEQUENCE = rf"{PII_NAME_TOKEN}(?:\s+{PII_NAME_TOKEN}){{0,3}}"
PII_NAME_TITLES = (
    "Dr",
    "Prof",
    "Ir",
    "Drs",
    "Dra",
    "Bapak",
    "Pak",
    "Ibu",
    "Bu",
    "Mas",
    "Mbak",
    "Kak",
    "Sdr",
    "Saudara",
    "Sdri",
)
PII_NAME_ROLE_LABELS = (
    "kontak pelapor",
    "dosen pembimbing",
    "ketua tim",
    "anggota tim",
    "pembimbing",
    "mentor",
    "narasumber",
    "pelapor",
)
PATTERN_PII_NAME_WITH_TITLE = re.compile(
    rf"\b(?:{'|'.join(PII_NAME_TITLES)})\s*\.?\s*"
    rf"(?:(?:{'|'.join(PII_NAME_TITLES)})\s*\.?\s*)*{PII_NAME_SEQUENCE}"
)
PATTERN_PII_NAME_AFTER_ROLE = re.compile(
    rf"\b(?i:{'|'.join(PII_NAME_ROLE_LABELS)})\s+(?P<name>{PII_NAME_SEQUENCE})"
)
PATTERN_PII_NAME_WITH_SUFFIX = re.compile(
    rf"\b{PII_NAME_SEQUENCE}\s*,\s*(?:S|M|A)\.\s?[A-Za-z]{{1,3}}\.?"
    rf"(?:\s*,?\s*(?:S|M|A)\.\s?[A-Za-z]{{1,3}}\.?)*"
)
# Optional deny-list of dummy (or later: real) names; the real file is gitignored.
PII_NAME_DENYLIST_PATH = BASE_DIR / "pii_denylist.txt"

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
REPEATED_MARGIN_LINES = 2
REPEATED_MARGIN_RATIO = 0.6
REPEATED_MARGIN_MIN_PAGES = 2
# Running headers/footers are short. The cap prevents deleting repeated body text.
REPEATED_MARGIN_MAX_CHARS = 80

# Run logging (Brief 5.2 #11)
# Logs carry operational metadata only: never raw rows, credentials or document text.
LOG_DIR = BASE_DIR / "logs"
LOG_LEVEL = "INFO"
LOG_FILE_PREFIX = "ingestion"

# Authentication Settings
AUTH_MAX_FAILED_ATTEMPTS = 5
AUTH_LOCKOUT_DURATION_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Context header robustness (Brief 5.3)
# Overlong metadata is truncated with an ellipsis inside the chunk header instead
# of aborting the document. Chroma metadata keeps the full, untruncated value, so
# filters and citations still see the real competition name.
CONTEXT_HEADER_COMPETITION_NAME_MAX_CHARS = 120
CONTEXT_HEADER_JUDUL_IDE_MAX_CHARS = 160
CONTEXT_HEADER_ELLIPSIS = "\u2026"


# ---------------------------------------------------------------------------
# Core RAG (Phase 2). No magic numbers: every value lives here.

# Logging for the chat pipeline. Operational metadata only: never question text,
# answers, document text or the API key.
CORE_LOG_FILE_PREFIX = "chat"

# Groq client (https://console.groq.com/docs, checked 2026-11-20)
# openai/gpt-oss-20b supports reasoning_effort low|medium|high (default medium).
# reasoning_format is NOT supported for GPT-OSS models: reasoning is returned in
# message.reasoning, and include_reasoning=False keeps it out of the response.
LLM_API_KEY_ENV_VAR = "GROQ_API_KEY"
LLM_INCLUDE_REASONING = False
LLM_REASONING_EFFORT_ANALYZER = "low"
LLM_REASONING_EFFORT_ANSWER = "medium"
LLM_ANALYZER_MAX_OUTPUT_TOKENS = 400
LLM_JSON_MODE_ENABLED = True
LLM_JSON_RESPONSE_FORMAT = {"type": "json_object"}
LLM_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
LLM_RETRY_BASE_DELAY_SECONDS = 1.0
LLM_RETRY_MAX_DELAY_SECONDS = 8.0

# Analyzer (LLM call #1)
ANALYZER_MAX_SEARCH_QUERY_CHARS = 300
ANALYZER_FALLBACK_MODE = "specific"
ANALYZER_MAX_QUESTION_CHARS = CHAT_MAX_INPUT_CHARS
ANALYZER_YEAR_MIN = 2000
ANALYZER_YEAR_MAX = 2100

# Retrieval limits per mode: k chunks returned, max chunks per report.
RETRIEVAL_LIMITS = {
    "specific": {
        "k": RETRIEVAL_SPECIFIC_K,
        "max_per_report": RETRIEVAL_SPECIFIC_MAX_CHUNKS_PER_REPORT,
    },
    "cross_report": {
        "k": RETRIEVAL_CROSS_REPORT_K,
        "max_per_report": RETRIEVAL_CROSS_REPORT_MAX_CHUNKS_PER_REPORT,
    },
}

# Context builder: one alias per unique report, document text wrapped as data.
CONTEXT_ALIAS_PREFIX = "S"
CONTEXT_DOCUMENT_OPEN_TEMPLATE = (
    '<dokumen id={alias} kompetisi="{competition_name}" jenis="{competition_type}" '
    'tahun="{competition_year}" pencapaian="{achievement_level}" '
    'bagian="{section_type}">'
)
CONTEXT_DOCUMENT_CLOSE_TAG = "</dokumen>"
CONTEXT_DOCUMENT_SEPARATOR = "\n\n"
CONTEXT_CATALOG_OPEN_TAG = "<daftar_laporan>"
CONTEXT_CATALOG_CLOSE_TAG = "</daftar_laporan>"
CONTEXT_CATALOG_LINE_TEMPLATE = (
    '- {alias} | kompetisi="{competition_name}" jenis="{competition_type}" '
    'tahun="{competition_year}" pencapaian="{achievement_level}"'
)

# Citations (Brief 5.6)
CITATION_REPORT_ID_PATTERN = re.compile(r"RPT[0-9]{4,}")
# Report IDs written by the model, with or without brackets, so they can be
# validated against the retrieved set and removed when fabricated.
PATTERN_LLM_REPORT_ID = re.compile(r"\[\s*(RPT[0-9]{4,})\s*\]|(RPT[0-9]{4,})")
SOURCE_BLOCK_TITLE = "**Sumber**"
SOURCE_BLOCK_LINE_TEMPLATE = (
    "- {report_id} — {competition_name} ({competition_type}, {competition_year}, "
    "{achievement_level})"
)
SOURCE_BLOCK_SEPARATOR = "\n\n"

# Output guardrails (Brief 5.7)
# Narrow, targeted detector for requests about personal/administrative data.
PATTERN_REQUEST_PII_ID_NUMBER = re.compile(r"(?i)\b(?:nim|npm|nrp|nomor\s+induk)\b")
PATTERN_REQUEST_PII_CONTACT = re.compile(
    r"(?i)\b(?:nomor\s+(?:telepon|hp|handphone|wa|whatsapp)|no\.?\s*(?:hp|wa)|whatsapp|"
    r"e-?mail|surel|alamat(?:\s+rumah)?)\b"
)
PATTERN_REQUEST_PII_INTENT = re.compile(
    r"(?i)\b(?:siapa|siapa\s+saja|nama|nomor|alamat|kontak|berikan|sebutkan|tampilkan|"
    r"minta|mengungkap|apa)\b"
)
PATTERN_REQUEST_PII_IDENTITY = re.compile(
    r"(?i)\b(?:siapa|nama)\b.{0,40}\b(?:pelapor|ketua\s+tim|anggota\s+tim|mentor|"
    r"pembimbing|mahasiswa)\b"
)
PATTERN_REQUEST_PII_PERSONAL_DATA = re.compile(
    r"(?i)\b(?:data\s+pribadi|identitas\s+(?:pelapor|ketua\s+tim|anggota\s+tim|mentor|"
    r"pembimbing|mahasiswa))\b"
)
PATTERN_REQUEST_PII_ENGLISH = re.compile(
    r"(?i)\b(?:student\s+id|phone\s+number|contact\s+number|email\s+address|"
    r"who\s+is\s+the\s+(?:leader|reporter|mentor|supervisor|member))\b"
)

# System-prompt leak detector: shingle overlap between answer and system prompt.
LEAK_NGRAM_SIZE = 8
LEAK_MIN_MATCHED_SHINGLES = 3
LEAK_OVERLAP_RATIO_THRESHOLD = 0.5
