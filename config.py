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
# Which embedding implementation is serving answers. Recorded in the startup log
# and checked against the manifest once an alternative runtime exists
# (docs/ONNX_PLAN.md). Only "sentence-transformers" is implemented.
EMBEDDING_RUNTIME = "sentence-transformers"
EMBEDDING_PASSAGE_PREFIX = "passage: "
EMBEDDING_QUERY_PREFIX = "query: "
EMBEDDING_DIMENSION = 384
# One dummy query at startup so the first real question does not load the model.
EMBEDDING_WARMUP_QUERY = "pemanasan model embedding"

# LLM Configuration (Groq)
LLM_MODEL_NAME = "openai/gpt-oss-20b"
LLM_TEMPERATURE_ANALYZER = 0.0
LLM_TEMPERATURE_CHAT = 0.2
# Answer completion cap. Data from the supervisor's 2026-11-21 run: answer
# completions were 185-1,024 tokens, 159-525 of them reasoning, and one answer
# hit the old 1,024 cap. With ANSWER_REASONING_EFFORT="low" (~160-250 reasoning
# tokens observed), a ~200-word Indonesian answer is ~300 tokens, so 800 leaves
# room for reasoning plus the whole answer without paying for a long answer.
LLM_MAX_OUTPUT_TOKENS = 800
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF_FACTOR = 2.0
# Per-call provider timeout. Bounded further by the remaining pipeline budget.
LLM_TIMEOUT_SECONDS = 20.0
# Model-and-processing budget for one question (report K2, narrowed in fix pack 7).
# Queue time is excluded from it: a wait for the minute token window is bounded
# separately by QUEUE_MAX_WAIT_SECONDS and refunded to this budget, so a
# legitimate wait can never consume the time the model needs. 60 s covers the
# analyzer (~1-3 s), retrieval (~1-2 s), one answer call (~9-30 s at effort
# "low") and the single citation retry, with a wide margin.
PIPELINE_TOTAL_DEADLINE_SECONDS = 60.0
# Fix pack 7: how long one question may wait for the per-minute token window.
# A wait that ends when the previous question's tokens age out (the observed case
# is about 20-30 s) is still served; anything longer answers RATE_LIMITED at once
# with retry_after_seconds instead of burning the deadline in the queue.
QUEUE_MAX_WAIT_SECONDS = 45.0

# Chunking Parameters
CHUNK_TARGET_MIN_TOKENS = 350
CHUNK_TARGET_MAX_TOKENS = 450
CHUNK_HARD_LIMIT_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 0

# PDF reflow (fix pack L1). PyMuPDF returns one "\n" per visual line and no
# paragraph marker, so the parser rebuilds paragraphs before chunking:
# - a blank line keeps a paragraph break,
# - a list-item marker starts a new paragraph,
# - a line ending in a hyphen followed by a lowercase line is joined without the
#   hyphen ("terburu-" + "buru" -> "terburu-buru"),
# - a detected heading is never joined to the line below it.
PDF_PARAGRAPH_SEPARATOR = "\n\n"
PDF_LINE_JOIN_SEPARATOR = " "
PDF_HYPHEN_BREAK_CHARS = ("-", "\u2010")
PATTERN_PDF_LIST_ITEM = re.compile(r"^(?:\d{1,2}[.)]|[-*\u2022])\s+")
# A chunk may only end at a sentence end, at the end of a list item, or inside a
# single sentence that alone exceeds the token budget (word-window split).
CHUNK_TERMINAL_PUNCTUATION = (".", "?", "!", "\u2026")
PATTERN_CHUNK_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
CHUNK_PARAGRAPH_SEPARATOR = "\n\n"

# Context assembly: blank lines between sentences of the same paragraph are
# collapsed, so a wrapped line never costs tokens or reaches the model as a
# paragraph break.
CONTEXT_PARAGRAPH_SEPARATOR = "\n\n"
PATTERN_CONTEXT_BLANK_RUN = re.compile(r"\n{3,}")

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
# Fix pack H4b: two chunks per report (k stays 6) so a specific answer shows at
# least three distinct reports when the collection offers them. Phase 4 sweeps
# this value against the evaluation set.
RETRIEVAL_SPECIFIC_MAX_CHUNKS_PER_REPORT = 2
RETRIEVAL_CANDIDATE_MULTIPLIER = 3

# Cross Report Query Retrieval Settings
RETRIEVAL_CROSS_REPORT_K = 12
RETRIEVAL_CROSS_REPORT_MAX_CHUNKS_PER_REPORT = 2

# Structured free-text answers (fix pack 9, O1). specific, the exact report-id
# route and cross_report without compare_by answer through the same strict JSON
# schema the comparison path already uses, because free prose ignored the bullet
# and citation caps: the real run returned 12 uncited bullets and the citation
# repair retry then exceeded the queue wait. The schema makes the cap and the
# citation a field of the answer instead of an instruction the model may skip.
# "text" is the old free-prose path; it is selectable only by config and is
# never used as a runtime fallback.
ANSWER_FORMAT = "structured"
ANSWER_FORMAT_VALUES = ("structured", "text")
ANSWER_JSON_SCHEMA_NAME = "answer"
# Per-point caps are validated by the backend, not by the schema (the documented
# keyword subset does not promise maxItems/minItems).
ANSWER_POINT_MAX_WORDS = 40
ANSWER_CITE_MIN = 1
ANSWER_CITE_MAX = 2
# Justification for 700: a structured answer is at most 6 points of 40 words
# (~240 tokens of Indonesian text) plus per-point citation arrays, and effort
# "low" adds the measured 160-250 reasoning tokens; 700 leaves room for the
# worst allowed payload without paying for a long free answer. The previous
# free-text cap was 800.
LLM_MAX_OUTPUT_TOKENS_FREE = 700
# One repair retry when a structured answer has no valid point at all, and when
# a comparison group that has documents ends with no valid point. Invalid JSON
# and a truncated completion are never retried: they cost quota and fail the
# same way.
STRUCTURED_REPAIR_RETRIES = 1
# The repair retry may wait longer for the minute token window than a first
# call, because refusing it turns a recoverable answer into RATE_LIMITED. It is
# still bounded: a student waits at most this long before the final status, and
# a wait beyond it is refused at once with retry_after_seconds.
QUEUE_MAX_WAIT_RETRY_SECONDS = 60.0
ANSWER_BULLET_TEMPLATE = "- {text} {citations}"

# Comparison mode (fix pack H3): each group contributes distinct reports with
# one chunk per report, so every listed team is a separate source.
COMPARE_REPORTS_PER_GROUP = 3
# A group keeps at least this many distinct reports while trimming, as long as
# the group has that many to begin with.
COMPARE_MIN_REPORTS_PER_GROUP = 2
# One chunk per report: prefer the sections that carry learning and advice.
COMPARE_PREFERRED_SECTIONS = ("evaluasi_tim", "saran_pengembangan")
# Candidates fetched per group: enough to find the configured number of distinct
# reports after the per-report cap and the similarity threshold.
RETRIEVAL_COMPARE_CANDIDATE_MULTIPLIER = 6
# Comparison answers need more context than a single-report answer, but not the
# full 12-chunk cross_report budget.
MAX_CONTEXT_TOKENS_COMPARE = 3200

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
# Answer reasoning effort (report F1f). Measured on 2026-11-21: answers spent
# 159-525 reasoning tokens at "medium". "low" keeps that out of the token
# budget; the smoke script can still override it for a comparison run.
ANSWER_REASONING_EFFORT = "low"
ANSWER_REASONING_EFFORT_VALUES = ("low", "medium", "high")
# Where the instruction block goes. "system" (default) keeps a system message;
# "user_prefixed" puts the same text at the start of the user message. Only an
# explicit A/B test should change it.
LLM_INSTRUCTION_ROLE = "system"
LLM_INSTRUCTION_ROLES = ("system", "user_prefixed")
LLM_INSTRUCTION_USER_SEPARATOR = "\n\n"
LLM_ANALYZER_MAX_OUTPUT_TOKENS = 400
LLM_JSON_MODE_ENABLED = True
LLM_JSON_RESPONSE_FORMAT = {"type": "json_object"}
LLM_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
# Fix pack 11 (S1a/S2d): a non-retryable 4xx is a request rejection, not a
# transient failure. The provider's error body is parsed for diagnosis, but only
# the documented fields are kept and the message is dropped when it echoes the
# request. These statuses are reported as error_type=invalid_request.
LLM_NON_RETRYABLE_REQUEST_STATUSES = (400, 401, 403, 404, 413, 422)
LLM_PROVIDER_ERROR_MAX_CHARS = 500
LLM_PROVIDER_ERROR_ECHO_WINDOW_CHARS = 40
LLM_RETRY_BASE_DELAY_SECONDS = 1.0
LLM_RETRY_MAX_DELAY_SECONDS = 8.0
# A rate-limit reset beyond this is treated as a daily window: waiting can never
# help inside one question, so the call fails fast instead of retrying.
LLM_DAILY_LIMIT_MIN_RESET_SECONDS = 60.0
# Process-wide cap on simultaneous provider calls (Streamlit runs in threads).
LLM_MAX_CONCURRENT_CALLS = 2

# Context token budget (report F1d). Chunks are dropped lowest-score first until
# the rendered context fits, keeping one chunk per selected report when possible.
# Reduced from 3,500 after the 2026-11-21 quota data: an 8,000 TPM / 200,000 TPD
# account cannot afford ~2,000-token contexts. 2,500 tokens still fits about six
# 350-450-token chunks.
MAX_CONTEXT_TOKENS = 2500
# Comparison answers are structured JSON (fix pack J1): up to 3 groups x 3 points
# plus 2 paired contrasts (2 sides each), every item with its own citations.
# 900 completion tokens leaves room for ~160-250 reasoning tokens at effort
# "low" and the measured 7-point payloads (~1,100 characters of JSON), i.e. the
# documented caps would have to be hit at every level to approach it.
# Justification: the previous free-text cap was 800; the structured answer adds
# per-point citation arrays but removes the prose headings the backend now writes.
LLM_MAX_OUTPUT_TOKENS_COMPARE = 900
# Comparison mode has its own budget (fix pack H3); specific mode and the exact
# report-id route keep MAX_CONTEXT_TOKENS.
CONTEXT_TOKEN_ESTIMATOR_CHARS_PER_TOKEN = 2.5

# Prompt token estimator for the limiter (fix pack 7). The old rule (one
# chars/2.5 ratio over the whole prompt) over-estimated a real answer prompt by
# 60-68%: it ignored how much better the model tokenizer packs Indonesian prose.
# Now the STATIC parts (system prompt, addenda, structured-output schema) are
# counted exactly with the e5 tokenizer, and only the DYNAMIC parts (context,
# template, hint, question) use this ratio. Fitted from measured runs (prompt
# tokens minus the exact static count):
#   specific: user 6,974 chars -> 1,850 dynamic tokens -> 3.77 chars/token
#   compare : user 8,740 chars -> 2,098 dynamic tokens -> 4.17 chars/token
# 4.3 with the 12% margin stays above both measured prompts without doubling
# them: specific ~3.0k vs 2.9k actual, compare ~4.2k vs 3.7k actual. Erring high
# is the safe direction (the provider refuses a request that is too large).
TOKEN_ESTIMATE_DYNAMIC_CHARS_PER_TOKEN = 4.3
TOKEN_ESTIMATE_MARGIN_RATIO = 0.12
# The counting tokenizer is encode-only: it must count a static prompt of any
# length without the e5 model's 512-token warning, and nothing it counts is ever
# fed to the model. Its model_max_length is raised so transformers never warns.
TOKEN_COUNT_MAX_LENGTH = 100_000_000
# Completion tokens reserved per call (fix pack 7). The provider's ceiling check
# charges the declared completion cap up front, so the cap stays at its real value
# (LLM_MAX_OUTPUT_TOKENS / ..._COMPARE / LLM_ANALYZER_MAX_OUTPUT_TOKENS) and the
# reservation uses the expected completion instead. Measured completions: free
# text 529, compare 538, analyzer about 60-80.
COMPLETION_RESERVATION_TOKENS = 600
COMPLETION_RESERVATION_TOKENS_COMPARE = 700
COMPLETION_RESERVATION_TOKENS_ANALYZER = 120
# Concise-answer target for the addendum (catalog mode excepted).
ANSWER_TARGET_WORDS = 180
ANSWER_MAX_BULLETS = 6
# Fix pack 7: addendum v1.4 also caps the points per report. The backend trims
# any answer above ANSWER_MAX_BULLETS with the same per-report cap (no retry, no
# extra model call).
ANSWER_MAX_BULLETS_PER_REPORT = 2
# Fix pack 10 (R1): "Apa saja saran pengembangan pada RPT0002?" asks for EVERY
# item of one report, and the 2-points-per-report cap hid the third one. On the
# exact report-id route, and whenever the context holds a single report, the
# per-report cap is this value instead; the total cap (ANSWER_MAX_BULLETS) still
# applies, so one report can fill the whole answer.
ANSWER_ID_ROUTE_MAX_POINTS_PER_REPORT = 6

# Analyzer (LLM call #1)
# Fix pack 8 (N4): the answer's post-processing is timed per stage so a slow
# first question can be attributed to one stage instead of "post_ms".
POST_PHASE_STAGES = (
    "context",
    "sanitize",
    "citations",
    "guardrails",
    "trim",
    "metrics",
    "coverage",
    "sources",
    "render",
    "compare_validate",
)
ANALYZER_MAX_SEARCH_QUERY_CHARS = 300
ANALYZER_FALLBACK_MODE = "specific"
ANALYZER_MAX_QUESTION_CHARS = CHAT_MAX_INPUT_CHARS
ANALYZER_YEAR_MIN = 2000
ANALYZER_YEAR_MAX = 2100

# Analyzer normalization (fix pack F2b). Keys are matched case-insensitively and
# ignoring punctuation and repeated whitespace; values are the canonical enums.
ANALYZER_COMPETITION_TYPE_SYNONYMS = {
    "business plan": "Business Plan",
    "businessplan": "Business Plan",
    "rencana bisnis": "Business Plan",
    "bp": "Business Plan",
    "business case": "Business Case",
    "businesscase": "Business Case",
    "studi kasus": "Business Case",
    "case": "Business Case",
    "essay": "Essay",
    "esai": "Essay",
    "esei": "Essay",
    "karya tulis ilmiah": "Karya Tulis Ilmiah",
    "karya tulis": "Karya Tulis Ilmiah",
    "kti": "Karya Tulis Ilmiah",
}
ANALYZER_ACHIEVEMENT_LEVEL_SYNONYMS = {
    "juara 1": "Juara 1",
    "juara i": "Juara 1",
    "juara pertama": "Juara 1",
    "juara ke 1": "Juara 1",
    "peringkat 1": "Juara 1",
    "juara 2": "Juara 2",
    "juara ii": "Juara 2",
    "juara kedua": "Juara 2",
    "juara ke 2": "Juara 2",
    "peringkat 2": "Juara 2",
    "runner up": "Juara 2",
    "juara 3": "Juara 3",
    "juara iii": "Juara 3",
    "juara ketiga": "Juara 3",
    "juara ke 3": "Juara 3",
    "peringkat 3": "Juara 3",
    "juara harapan": "Juara Harapan",
    "harapan": "Juara Harapan",
    "most aspiring": "Most Aspiring dan penghargaan serupa",
    "most aspiring dan penghargaan serupa": "Most Aspiring dan penghargaan serupa",
    "penghargaan serupa": "Most Aspiring dan penghargaan serupa",
    "finalis": "Finalis",
    "finalist": "Finalis",
    "belum juara": "Finalis",
    "non juara": "Finalis",
}

# Deterministic extractor (fix pack F2c): explicit type keywords in the question.
ANALYZER_TYPE_KEYWORDS = {
    "business plan": "Business Plan",
    "rencana bisnis": "Business Plan",
    "bp": "Business Plan",
    "business case": "Business Case",
    "studi kasus": "Business Case",
    "case": "Business Case",
    "essay": "Essay",
    "esai": "Essay",
    "karya tulis ilmiah": "Karya Tulis Ilmiah",
    "karya tulis": "Karya Tulis Ilmiah",
    "kti": "Karya Tulis Ilmiah",
}
# A four-digit year standing on its own, never part of a longer number.
PATTERN_ANALYZER_YEAR = re.compile(r"(?<![\d.])(20[0-9]{2})(?![\d.])")
ANALYZER_YEAR_MONEY_WINDOW = 24
PATTERN_ANALYZER_MONEY_PREFIX = re.compile(
    r"(?i)(?:rp|idr|usd|\$|biaya|dana|anggaran|hadiah|nominal|sebesar|senilai)"
    r"[^0-9]{0,8}$"
)
PATTERN_ANALYZER_MONEY_SUFFIX = re.compile(
    r"(?i)^\s*(?:juta|ribu|miliar|milyar|jt|rb|million)\b"
)

# Comparison grouping (fix pack F3). The analyzer may ask for a balanced
# comparison; retrieval then runs one sub-retrieval per group and merges them.
COMPARE_BY_OPTIONS = ("achievement_group", "competition_type")
COMPARISON_AWARD_LEVELS = (
    "Juara 1",
    "Juara 2",
    "Juara 3",
    "Juara Harapan",
    "Most Aspiring dan penghargaan serupa",
)
# Fix pack L2: "juara" was misleading because the award group also contains
# Juara Harapan and Most Aspiring. The label is the model-facing name (schema
# enum, <kelompok nama=...>, <dokumen kelompok=...>, coverage line); the display
# label is the plain heading the backend renders.
COMPARISON_GROUPS = {
    "achievement_group": (
        {
            "label": "meraih penghargaan",
            "display_label": (
                "Tim yang meraih penghargaan (Juara 1-3, Juara Harapan, Most Aspiring)"
            ),
            "filter": {"achievement_level": COMPARISON_AWARD_LEVELS},
        },
        {
            "label": "belum juara (finalis)",
            "display_label": "Tim belum juara (finalis)",
            "filter": {"achievement_level": ("Finalis",)},
        },
    ),
    "competition_type": (
        {
            "label": "Business Plan",
            "display_label": "Business Plan",
            "filter": {"competition_type": ("Business Plan",)},
        },
        {
            "label": "Business Case",
            "display_label": "Business Case",
            "filter": {"competition_type": ("Business Case",)},
        },
        {
            "label": "Essay",
            "display_label": "Essay",
            "filter": {"competition_type": ("Essay",)},
        },
        {
            "label": "Karya Tulis Ilmiah",
            "display_label": "Karya Tulis Ilmiah",
            "filter": {"competition_type": ("Karya Tulis Ilmiah",)},
        },
    ),
}


def comparison_display_label(compare_by: str, label: str) -> str:
    """Plain heading for one group label (the model-facing name by default)."""
    for group in COMPARISON_GROUPS.get(compare_by, ()):
        if group["label"] == label:
            return group.get("display_label") or label
    return label

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
# Aliases are LAP1, LAP2, ... (fix pack F4a): S1/S2 collide with Indonesian degree
# terms and D1-D4 with diploma levels.
CONTEXT_ALIAS_PREFIX = "LAP"
CITATION_ALIAS_WORD = "LAP"
CONTEXT_DOCUMENT_OPEN_TEMPLATE = (
    '<dokumen id={alias} kompetisi="{competition_name}" jenis="{competition_type}" '
    'tahun="{competition_year}" pencapaian="{achievement_level}" '
    'bagian="{section_type}" bagian_label="{section_label}"{group_attribute}>'
)
CONTEXT_DOCUMENT_CLOSE_TAG = "</dokumen>"
# Grouped comparison mode wraps documents in an explicit group block, so the
# answer model never has to guess which team belongs to which group.
CONTEXT_GROUP_OPEN_TEMPLATE = '<kelompok nama="{label}">'
CONTEXT_GROUP_CLOSE_TAG = "</kelompok>"
CONTEXT_GROUP_ATTRIBUTE_TEMPLATE = ' kelompok="{label}"'
CONTEXT_DOCUMENT_SEPARATOR = "\n\n"
# The coverage line the backend appends to comparison answers (fix pack H3b).
COVERAGE_BLOCK_SEPARATOR = "\n\n"
COVERAGE_PREFIX = "Ringkasan ini didasarkan pada "
COVERAGE_GROUP_TEMPLATE = "{shown} dari {total} laporan {label}"
# Fix pack 9 (O2a): commas between groups, "dan" only before the last one, so
# the sentence stays readable with two and with four groups.
COVERAGE_JOINER = ", "
COVERAGE_FINAL_JOINER = " dan "
COVERAGE_TWO_GROUP_JOINER = " dan "
COVERAGE_SUFFIX = "."
COVERAGE_CROSS_TEMPLATE = (
    "Ringkasan ini didasarkan pada {shown} dari {total} laporan yang sesuai."
)
CONTEXT_CATALOG_OPEN_TAG = "<daftar_laporan>"
CONTEXT_CATALOG_CLOSE_TAG = "</daftar_laporan>"
CONTEXT_CATALOG_LINE_TEMPLATE = (
    '- {alias} | kompetisi="{competition_name}" jenis="{competition_type}" '
    'tahun="{competition_year}" pencapaian="{achievement_level}"'
)

# Citations (Brief 5.6, fix pack F4b). The model writes [LAP1]; the backend
# accepts brackets, parentheses, comma lists, adjacent tokens and a bare token.
CITATION_REPORT_ID_PATTERN = re.compile(r"RPT[0-9]{4,}")
PATTERN_ALIAS_LIST = re.compile(
    r"[\[\(]\s*(LAP[0-9]{1,3}(?:\s*[,;/]\s*LAP[0-9]{1,3})+)\s*[\]\)]"
)
PATTERN_ALIAS_SINGLE = re.compile(
    r"(?P<open>[\[\(])?\s*(?P<alias>LAP[0-9]{1,3})\s*(?P<close>[\]\)])?"
)
# Safety net: after post-processing no alias token may remain in the answer. It
# is case-insensitive on purpose: a variant the substitution does not understand
# must fail loudly instead of reaching the student.
PATTERN_ALIAS_ANY = re.compile(r"\bLAP[0-9]{1,3}\b", re.IGNORECASE)
# Spacing normalization before a resolved citation: "di[RPT0001]" -> "di [RPT0001]".
PATTERN_CITATION_TIGHT_LEAD = re.compile(r"(?<=[^\s(\[*])(?=\[RPT[0-9]{4,}\](?:\s*,\s*RPT[0-9]{4,}\])*)")
# A citation group for the answer-style metrics (mid-sentence and parentheses).
PATTERN_CITATION_GROUP = re.compile(r"\[RPT[0-9]{4,}(?:\s*,\s*RPT[0-9]{4,})*\]")
# A competition name this short is too generic to treat as a citation claim.
CITATION_MENTION_MIN_CHARS = 8
# Report IDs written by the model, with or without brackets, so they can be
# validated against the retrieved set and removed when fabricated.
PATTERN_LLM_REPORT_ID = re.compile(r"\[\s*(RPT[0-9]{4,})\s*\]|(RPT[0-9]{4,})")
SOURCE_BLOCK_TITLE = "**Sumber**"
SOURCE_BLOCK_LINE_TEMPLATE = (
    "- {report_id} — {competition_name} ({competition_type}, {competition_year}, "
    "{achievement_level})"
)
SOURCE_BLOCK_SEPARATOR = "\n\n"

# A model answer that starts with the first sentence of NOT_FOUND_RESPONSE is a
# not-found declaration, not an uncited answer (fix pack H2). The comparison uses
# this many normalized characters, so a missing or slightly reworded second
# sentence still matches. Matching ignores case, whitespace and Markdown markers.
NOT_FOUND_ANSWER_PREFIX_CHARS = 40
# Markdown markers are removed before comparison, so **bold** or a heading hash
# cannot hide a model-declared refusal from the detector.
PATTERN_NOT_FOUND_MARKDOWN = re.compile(r"[*_`#>~]+")

# Real Groq limits for the account used by the supervisor (Groq console limits
# page, read 2026-11-21). Cached prompt tokens do not count toward them
# (https://console.groq.com/docs/rate-limits, checked 2026-11-21). TPD is not
# exposed by any response header, so the limiter tracks it locally.
GROQ_LIMITS = {
    "model": LLM_MODEL_NAME,
    "requests_per_minute": 30,
    "requests_per_day": 1000,
    "tokens_per_minute": 8000,
    "tokens_per_day": 200000,
    "source": "Groq console limits page",
    "date": "2026-11-21",
}
# Rolling windows used by the token-aware limiter and the daily-reset fallback.
TOKEN_LIMITER_MINUTE_WINDOW_SECONDS = 60.0
TOKEN_LIMITER_DAY_WINDOW_SECONDS = 86400.0

# Catalog answers are rendered by the backend (no answer-LLM call).
CATALOG_MAX_ROWS = 30
CATALOG_HEADER_TEMPLATE = "Tersedia {total} laporan evaluasi kompetisi."
CATALOG_TYPE_LINE_TEMPLATE = "- Jenis kompetisi: {value}"
CATALOG_YEAR_LINE_TEMPLATE = "- Tahun: {value}"
CATALOG_ACHIEVEMENT_LINE_TEMPLATE = "- Pencapaian: {value}"
CATALOG_ROW_TEMPLATE = (
    "- {report_id} — {competition_name} ({competition_type}, "
    "{competition_year}, {achievement_level})"
)
CATALOG_TRUNCATED_TEMPLATE = (
    "Menampilkan {shown} dari {total} laporan; persempit dengan jenis kompetisi, "
    "tahun, atau pencapaian."
)
CATALOG_VALUE_SEPARATOR = ", "
CATALOG_COUNT_SEPARATOR = "; "
CATALOG_EMPTY_VALUE = "tidak diketahui"

# Readable labels for the section types used in the answer context.
SECTION_TYPE_LABELS = {
    "executive_summary": "Executive Summary",
    "evaluasi_tim": "Evaluasi Tim",
    "saran_pengembangan": "Saran Pengembangan",
}

# Smoke run pacing (fix pack F1g): Groq's gpt-oss-20b allows ~30 requests per
# minute, and one question spends up to two requests.
SMOKE_DEFAULT_DELAY_SECONDS = 20.0
SMOKE_DEFAULT_TOKEN_BUDGET = 60000
SMOKE_RETRIEVED_PREVIEW = 5

# Backend answer sanitizer (fix pack F4d): HTML is stripped, Markdown kept.
SANITIZER_MAX_BLANK_LINES_PATTERN = "\n\n"
# Invisible and look-alike characters the real model emitted (fix pack 8, N5b):
# non-breaking hyphens in "end-to-end"/"terburu-buruan", and no-break or thin
# spaces in "83 %". They are normalized to their ASCII equivalents; the model's
# wording is otherwise untouched.
SANITIZER_CHARACTER_MAP = {
    "\u2011": "-",  # non-breaking hyphen
    "\u2010": "-",  # hyphen
    "\u00a0": " ",  # no-break space
    "\u202f": " ",  # narrow no-break space
    "\u2009": " ",  # thin space
    "\u200b": "",  # zero-width space
    "\u200c": "",  # zero-width non-joiner
    "\u200d": "",  # zero-width joiner
    "\ufeff": "",  # byte-order mark
}

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

# Input-side instruction-override detector (report K4). Deliberately narrow: a
# reveal/override verb must appear together with a target phrase, so legitimate
# questions that merely contain "instruksi" or "prompt" still reach the model.
PATTERN_SECURITY_PROMPT_EXTRACT = re.compile(
    r"(?i)\b(?:tampilkan|tunjukkan|cetak|berikan|tulis(?:kan)?|ulangi|bocorkan|ungkapkan|"
    r"reveal|show|print|display|repeat|leak|dump)\b[^.?!\n]{0,60}"
    r"\b(?:system\s*prompt|prompt\s*sistem|instruksi\s*(?:sistem|internal|tersembunyi|awal)|"
    r"konfigurasi\s*(?:internal|sistem|chatbot)|hidden\s+(?:instructions?|prompt)|"
    r"system\s+instructions?|aturan\s+internal)\b"
)
PATTERN_SECURITY_INSTRUCTION_OVERRIDE = re.compile(
    r"(?i)\b(?:abaikan|ignore|lupakan|forget|lewati|bypass)\b[^.?!\n]{0,40}"
    r"\b(?:instruksi|instruction|aturan|rule|prompt)s?\b[^.?!\n]{0,30}"
    r"\b(?:sebelumnya|sebelum|di\s*atas|previous(?:ly)?|above|earlier|prior)\b"
)
# Covers both word orders: "abaikan instruksi sebelumnya" and
# "ignore previous instructions".
PATTERN_SECURITY_IGNORE_PREVIOUS = re.compile(
    r"(?i)\b(?:abaikan|ignore)\s+(?:semua\s+|all\s+)?"
    r"(?:(?:previous|sebelumnya)\s+)?(?:instruksi|instructions?|prompts?|aturan|rules?)\b"
)

# System-prompt leak detector: shingle overlap between answer and system prompt.
LEAK_NGRAM_SIZE = 8
LEAK_MIN_MATCHED_SHINGLES = 3
LEAK_OVERLAP_RATIO_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Phase 3: Streamlit application, authentication and knowledge-base sync.

APP_TITLE = "Lab Prestasi"
APP_LAYOUT = "centered"

# Password hashing (OWASP Password Storage Cheat Sheet, checked 2026-11-20:
# PBKDF2-HMAC-SHA256 with 600,000 iterations).
AUTH_HASH_SCHEME = "pbkdf2_sha256"
AUTH_PBKDF2_ALGORITHM = "sha256"
AUTH_PBKDF2_ITERATIONS = 600_000
AUTH_MIN_PBKDF2_ITERATIONS = 100_000
AUTH_PBKDF2_SALT_BYTES = 16
AUTH_MIN_PASSWORD_LENGTH = 12
# A small fixed delay on every failed attempt (config owns the number).
AUTH_FAILURE_DELAY_SECONDS = 0.5

# Per-session quota guards for the shared Groq allowance. These are browser
# session counters only, so they slow one tab down; they do not protect the shared
# account from several sessions at once (reported as a known limitation).
SESSION_COOLDOWN_SECONDS = 5.0
SESSION_HOURLY_MESSAGE_CAP = 10
SESSION_MAX_QUESTIONS_PER_DAY = 15
SESSION_DAY_WINDOW_SECONDS = 86400
# The answer call's own token footprint at the configured caps: completion plus
# the context budget plus the instruction and question blocks. Used to predict a
# queue wait before the question is sent (fix pack J4).
QUEUE_MESSAGE_ESTIMATE_TOKENS = 4800
# Only a wait longer than this is worth a message.
QUEUE_MESSAGE_MIN_WAIT_SECONDS = 3.0
# A RATE_LIMITED answer that knows how long to wait rounds the wait to this step
# (up) before telling the student (fix pack 7).
RETRY_AFTER_ROUNDING_SECONDS = 5.0
CHAT_HISTORY_MAX_TURNS = 50

# Knowledge-base sources. "local" serves an existing chroma_db + manifest;
# "drive" downloads the published artifact from the private Drive folder.
KB_SOURCE_LOCAL = "local"
KB_SOURCE_DRIVE = "drive"
KB_SOURCES = (KB_SOURCE_LOCAL, KB_SOURCE_DRIVE)
KB_DRIVE_SCOPES = ("https://www.googleapis.com/auth/drive.readonly",)
# Manifest written by scripts/run_ingestion.py next to its artifact bundle.
ARTIFACT_DIR = DATA_DIR / "artifacts"
ARTIFACT_MANIFEST_PATH = ARTIFACT_DIR / MANIFEST_FILE_NAME

KB_DRIVE_MANIFEST_NAME = "manifest.json"
KB_DRIVE_ARTIFACT_NAME = "chroma_backup.zip"
KB_MANIFEST_TTL_SECONDS = 300.0
KB_WORK_DIR = DATA_DIR / "kb_work"
KB_VERSION_DIR_PREFIX = "kb_"
KB_ZIP_MAX_MEMBERS = 5000
KB_ZIP_MAX_UNCOMPRESSED_BYTES = 250 * 1024 * 1024
KB_REPORT_COUNT_TOLERANCE = 0

# Secrets/environment keys read by core.settings. Values are never logged.
SETTINGS_ENV_AUTH_USERNAME = "LAB_PRESTASI_USERNAME"
SETTINGS_ENV_AUTH_PASSWORD_HASH = "LAB_PRESTASI_PASSWORD_HASH"
SETTINGS_ENV_KB_SOURCE = "KB_SOURCE"
SETTINGS_ENV_KB_LOCAL_DIR = "KB_LOCAL_DIR"
SETTINGS_ENV_KB_MANIFEST_PATH = "KB_MANIFEST_PATH"
SETTINGS_ENV_KB_DRIVE_FOLDER_ID = "KB_DRIVE_FOLDER_ID"
# Optional exact Drive file ids (fix pack P2). When set, the app downloads those
# files directly instead of resolving names; a stale id fails loudly.
SETTINGS_ENV_KB_DRIVE_MANIFEST_FILE_ID = "KB_DRIVE_MANIFEST_FILE_ID"
SETTINGS_ENV_KB_DRIVE_ARTIFACT_FILE_ID = "KB_DRIVE_ARTIFACT_FILE_ID"
SETTINGS_ENV_KB_WORK_DIR = "KB_WORK_DIR"

# ---------------------------------------------------------------------------
# Structured comparison answers (fix pack J1). The model returns JSON; the
# backend validates it and writes every heading, citation and count itself.
COMPARE_JSON_SCHEMA_NAME = "compare_answer"
COMPARE_POINTS_PER_GROUP_MAX = 3
# Fix pack 8: the free "differences" list is replaced by paired contrasts, one
# side per group on the same dimension. A contrast is the only shape that can be
# validated: a free difference naming one group only was dropped every time.
COMPARE_CONTRASTS_MAX = 2
COMPARE_POINT_MAX_WORDS = 40
COMPARE_DIMENSION_MAX_WORDS = 6
COMPARE_CITE_MIN = 1
COMPARE_CITE_MAX = 2
# Fix pack 11 (S1e): the contrast shape is a configuration choice, never a
# runtime fallback. "pick" is the fix pack 10 shape (a number per group);
# "sides" is the fix pack 9 shape (one {group, text, cite} side per group), which
# is known to have worked on the real model. Both keep the fix pack 10 dimension
# rules and the citation ceilings.
COMPARE_CONTRAST_MODE = "pick"
COMPARE_CONTRAST_MODE_VALUES = ("pick", "sides")
COMPARE_CONTRAST_SIDES = "sides"
COMPARE_CONTRAST_GROUP = "group"
COMPARE_CONTRAST_TEXT = "text"
COMPARE_CONTRAST_CITE = "cite"
# Only the "sides" mode carries its own text, so only it needs these ceilings.
COMPARE_CONTRAST_TEXT_MAX_WORDS = 30
COMPARE_CONTRAST_CITE_MAX = 2
# Fix pack 10 (R2d): the model picks a contrast by NUMBER (see
# COMPARE_CONTRAST_PICK), so the backend renders the validated point text and the
# validated citations. A contrast never carries text of its own, which removes
# the repeated claims the real run produced on the Q3/Q4 contrast sides.
# A dimension is compared against this list as a WHOLE normalized string, never
# word by word: "Kendala Keuangan" is concrete, "Kendala" is not. A generic or
# empty dimension is not dropped; the backend renders the neutral label below
# and counts generic_dimension (non-blocking).
COMPARE_GENERIC_DIMENSIONS = (
    "pembelajaran",
    "kendala",
    "perbedaan",
    "umum",
    "lainnya",
    "hal",
    "aspek",
    "perbandingan",
)
COMPARE_NEUTRAL_DIMENSION = "Perbandingan"
# Model-facing group keys (fix pack 9, O2b): the model returns g1, g2, ... and
# the backend maps them to labels, so it can never invent a group name.
COMPARE_GROUP_KEY_TEMPLATE = "g{index}"
COMPARE_GROUP_KEY_LINE = "- {key} = {label}"
# Fix pack 10 (R2b): a contrast names the 1-based number of one point per group,
# as the model wrote those points. The backend maps the number to the validated
# point and drops the contrast when that point did not survive validation.
COMPARE_CONTRAST_PICK = "pick"
# Fix pack 11 (S1d): the pick values are STRINGS with an enum generated per
# request ("1".."N"), not open integers. String+enum is the one construct the
# real model already accepted in the fix pack 9 schema, and the backend parses
# the number and validates its range, so no undocumented keyword is needed.
COMPARE_PICK_NUMBER_TEMPLATE = "{number}"
# Fix pack 8: the backend writes the group heading in bold, followed by a blank
# line, so a group name reads as a heading instead of a stray prose line.
COMPARE_GROUP_HEADING_TEMPLATE = "**{label}**"
COMPARE_CONTRASTS_HEADING = "**Perbedaan utama**"
COMPARE_CONTRAST_DIMENSION_TEMPLATE = "- {dimension}"
COMPARE_CONTRAST_SIDE_TEMPLATE = "  - {label}: {text} {citations}"
COMPARE_EMPTY_GROUP = (
    "Tidak ada poin yang dapat ditelusuri ke dokumen untuk kelompok ini."
)
# Fix pack 9 (O2e): a group that HAS documents but produced no valid point after
# the repair retry must never claim there is no evidence. This sentence says the
# attempt failed, which is the truth, and asks the student to retry.
COMPARE_GROUP_EMPTY_RETRY = (
    "Poin untuk kelompok ini belum dapat disusun dari dokumen yang diambil pada "
    "percobaan ini. Silakan coba ulangi pertanyaan."
)
COMPARE_BULLET_TEMPLATE = "- {text} {citations}"
# One citation format everywhere (fix pack L6): separate brackets, one space,
# never a comma inside the brackets.
CITATION_GROUP_TEMPLATE = "[{report_id}]"
CITATION_GROUP_JOINER = " "
COMPARE_SECTION_SEPARATOR = "\n\n"
COMPARE_LINE_SEPARATOR = "\n"

# Non-blocking negation metric (fix pack K2). It only measures; it never changes
# an answer. A sentence counts when it contains a negation marker AND one of the
# subject markers, e.g. "Tim belum melakukan simulasi" or "tidak ada anggota yang
# pasif" (the second is legitimate, which is why this is not a guardrail).
NEGATION_MARKERS = (
    "tidak",
    "belum",
    "kurang",
    "tanpa",
    "tak",
    "bukan",
    "mengabaikan",
    "gagal",
)
# Group and heading labels are not claims. These phrases are removed before
# matching, so "Tim belum juara kurang menyiapkan riset" counts once ("kurang")
# while the label line "belum juara" counts zero.
NEGATION_LABEL_PHRASES = (
    "Tim yang meraih penghargaan (Juara 1-3, Juara Harapan, Most Aspiring)",
    "Tim belum juara (finalis)",
    "belum juara (finalis)",
    "belum juara",
    "meraih penghargaan",
    "tim belum juara (finalis)",
    "tim belum juara",
    "tim yang meraih penghargaan",
)
# Backend-written lines are skipped entirely: coverage, source block title/rows.
NEGATION_SKIP_LINE_PREFIXES = ("Ringkasan ini didasarkan pada", "**Sumber**")
# Fix pack 9 (O4): backend sentences the model never wrote. The empty-group
# sentence was counted as a model negation in the real run, so a line that
# starts with any of these phrases is not measured at all.
NEGATION_BACKEND_PHRASES = (
    COVERAGE_PREFIX,
    SOURCE_BLOCK_TITLE,
    COMPARE_EMPTY_GROUP,
    COMPARE_GROUP_EMPTY_RETRY,
)
PATTERN_NEGATION_SOURCE_ROW = re.compile(r"^-\s*RPT[0-9]{4,}\b")
NEGATION_SUBJECT_MARKERS = (
    "tim",
    "laporan",
    "kelompok",
    "finalis",
    "juara",
    "peserta",
    "anggota",
    "mahasiswa",
)
NEGATION_SAMPLE_LIMIT = 5
NEGATION_SENTENCE_SPLIT_PATTERN = r"(?<=[.!?])\s+"

# Non-blocking causal-connector metric (fix pack 8, N2b). It measures how many
# answer sentences join two facts with a link the source may not state. It never
# changes an answer; samples stay in the debug object.
# Fix pack 10 (R3b): the real run joined two separately sourced facts with
# "mengakibatkan", which was not in the list.
CAUSAL_CONNECTORS = (
    "sehingga",
    "karena",
    "akibatnya",
    "membuat",
    "memengaruhi",
    "menyebabkan",
    "mengakibatkan",
    "berakibat",
)
CAUSAL_SAMPLE_LIMIT = 5

# Fix pack 10 (R3b): more non-blocking wording metrics. They cannot detect a
# verb-strength error or an invented causal link (only a human can); they are a
# rough signal for Phase 4 rating. No answer is changed or blocked by them.
# A phrase that states a lesson or an interpretation rather than a document fact.
INTERPRETIVE_PHRASES = (
    "mengajarkan",
    "pentingnya",
    "menunjukkan nilai",
    "menyoroti pentingnya",
    "menegaskan pentingnya",
    "membuktikan",
    "memberi pelajaran",
)
# A verb stronger than the source usually states it ("kurang menggali" written
# as "mengabaikan").
STRONG_VERBS = (
    "mengabaikan",
    "gagal",
    "tidak mampu",
    "tidak berhasil",
    "mengorbankan",
    "menguasai",
)
WORDING_SAMPLE_LIMIT = 5

# Fix pack 10 (R3b): support_overlap is the share of a point's content words
# found in the text of the reports that point cites. Content words are lowercased
# tokens of at least SUPPORT_MIN_WORD_CHARS characters that are not stopwords.
# The threshold only flags a point for human review; nothing is blocked.
SUPPORT_MIN_WORD_CHARS = 4
SUPPORT_OVERLAP_MIN = 0.5
SUPPORT_STOPWORDS = (
    "adalah",
    "agar",
    "akan",
    "antara",
    "atau",
    "bagi",
    "bahwa",
    "banyak",
    "belum",
    "bisa",
    "dalam",
    "dan",
    "dapat",
    "dari",
    "dengan",
    "juga",
    "karena",
    "kepada",
    "kini",
    "kurang",
    "lebih",
    "masih",
    "mereka",
    "namun",
    "oleh",
    "para",
    "pada",
    "saat",
    "sangat",
    "serta",
    "setelah",
    "sehingga",
    "seperti",
    "sudah",
    "telah",
    "tentang",
    "terhadap",
    "tidak",
    "untuk",
)
SUPPORT_SAMPLE_LIMIT = 5
# Fix pack 11 (S3): a point whose cited reports contributed no retrieved text is
# reported separately from a low-overlap point, so a reporting bug (or a point
# citing an unresolvable report) is never mistaken for weak support.
SUPPORT_CITED_TEXT_MISSING = "cited_text_missing"
# A debug sample keeps this many characters of the point it flags.
SUPPORT_SAMPLE_TEXT_CHARS = 160

# Operational logging for the app layer (metadata only).
APP_LOG_FILE_PREFIX = "app"
PYTHON_MIN_SQLITE_VERSION = (3, 35, 0)
PYTHON_MIN_FOR_SQLITE_NOTE = "3.35.0"
