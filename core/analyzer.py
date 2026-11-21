"""
Analyzer: LLM call #1 (Brief 5.5 step 2, fix pack F2).

The model returns JSON; nothing it says is trusted. Path of one analyzer result:

1. the raw payload is kept for the debug object,
2. every filter value is normalized (case, whitespace, punctuation, synonyms,
   year strings) before it is checked against the enum sets in enums.py,
3. invalid values are dropped and the reason is recorded,
4. a deterministic extractor reads obvious type keywords and a four-digit year
   from the question itself and is merged by union with the validated filters.

The deterministic extractor exists because a model can legitimately answer with
empty filters; the question's explicit constraint ("business plan", "tahun 2023")
must not be lost in that case. It is config-driven and conservative: a number
that belongs to a monetary amount is never treated as a year.

A broken response (invalid JSON, provider failure, unusable payload) falls back
to in_scope=true / mode=specific / no filters, and the fallback is logged without
the question text. A deadline or rate-limit failure is re-raised instead: another
call would only fail the same way.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import config
from core import prompts
from core.deadline import Deadline, DeadlineExceeded
from core.llm_client import LLMClient, LLMError, LLMRateLimitError
from enums import AchievementLevel, CompetitionType, QueryMode

logger = logging.getLogger("core_analyzer")

ALLOWED_COMPETITION_TYPES = tuple(
    competition_type.value
    for competition_type in CompetitionType
    if competition_type is not CompetitionType.LAINNYA
)
ALLOWED_ACHIEVEMENT_LEVELS = tuple(level.value for level in AchievementLevel)
FILTER_KEYS = ("competition_type", "competition_year", "achievement_level")


class AnalyzerParseError(ValueError):
    """Raised when the analyzer response contains no parsable JSON object."""


@dataclass(frozen=True)
class AnalyzerResult:
    in_scope: bool
    mode: QueryMode
    search_query: str
    filters: Dict[str, List[Any]] = field(default_factory=dict)
    used_fallback: bool = False
    dropped_invalid_values: int = 0
    latency_ms: float = 0.0
    # Fix pack L5a: the analyzer call reports the same split as every other call,
    # instead of leaving the smoke output at "n/a".
    queue_wait_ms: float = 0.0
    model_latency_ms: float = 0.0
    usage: Dict[str, int] = field(default_factory=dict)
    error_type: Optional[str] = None
    raw_output: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    filters_before_validation: Dict[str, Any] = field(default_factory=dict)
    dropped_reasons: List[str] = field(default_factory=list)
    deterministic_filters: Dict[str, List[Any]] = field(default_factory=dict)
    compare_by: Optional[str] = None


def parse_analyzer_output(raw: str) -> Dict[str, Any]:
    """
    Parse the analyzer payload tolerantly.

    Markdown fences and surrounding prose are tolerated (the model may wrap JSON),
    but there is no guessing: without a JSON object this raises.
    """
    text = (raw or "").strip()
    if not text:
        raise AnalyzerParseError("Analyzer returned an empty response.")

    fenced = re.match(r"^```[a-zA-Z]*\s*(?P<body>.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group("body").strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise AnalyzerParseError("Analyzer response contains no JSON object.")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AnalyzerParseError("Analyzer response is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise AnalyzerParseError("Analyzer response is not a JSON object.")
    return payload


# -- normalization ---------------------------------------------------------
def _normalize_key(value: Any) -> str:
    """Case, punctuation and whitespace tolerant key for the synonym maps."""
    text = str(value or "").casefold()
    text = re.sub(r"[^0-9a-z\u00c0-\u024f]+", " ", text)
    return " ".join(text.split())


COMPETITION_TYPE_SYNONYMS = {
    _normalize_key(key): value
    for key, value in config.ANALYZER_COMPETITION_TYPE_SYNONYMS.items()
}
ACHIEVEMENT_LEVEL_SYNONYMS = {
    _normalize_key(key): value
    for key, value in config.ANALYZER_ACHIEVEMENT_LEVEL_SYNONYMS.items()
}
TYPE_KEYWORDS = {
    _normalize_key(key): value
    for key, value in config.ANALYZER_TYPE_KEYWORDS.items()
}


def normalize_competition_type(value: Any) -> Optional[str]:
    key = _normalize_key(value)
    if key in COMPETITION_TYPE_SYNONYMS:
        return COMPETITION_TYPE_SYNONYMS[key]
    return None


def normalize_achievement_level(value: Any) -> Optional[str]:
    key = _normalize_key(value)
    if key in ACHIEVEMENT_LEVEL_SYNONYMS:
        return ACHIEVEMENT_LEVEL_SYNONYMS[key]
    return None


def _as_list(raw_values: Any) -> Optional[list]:
    if raw_values is None:
        return []
    if isinstance(raw_values, list):
        return raw_values
    return None


def _clean_competition_types(raw_values: Any, dropped: List[str]) -> List[str]:
    values = _as_list(raw_values)
    if values is None:
        dropped.append("competition_type:not_a_list")
        return []

    kept: List[str] = []
    for value in values:
        if isinstance(value, bool):
            dropped.append("competition_type:not_a_string")
            continue
        candidate = normalize_competition_type(value)
        if candidate is None or candidate not in ALLOWED_COMPETITION_TYPES:
            dropped.append(f"competition_type:unknown:{_normalize_key(value)}")
            continue
        if candidate not in kept:
            kept.append(candidate)
    return sorted(kept)


def _clean_achievement_levels(raw_values: Any, dropped: List[str]) -> List[str]:
    values = _as_list(raw_values)
    if values is None:
        dropped.append("achievement_level:not_a_list")
        return []

    kept: List[str] = []
    for value in values:
        if isinstance(value, bool):
            dropped.append("achievement_level:not_a_string")
            continue
        candidate = normalize_achievement_level(value)
        if candidate is None or candidate not in ALLOWED_ACHIEVEMENT_LEVELS:
            dropped.append(f"achievement_level:unknown:{_normalize_key(value)}")
            continue
        if candidate not in kept:
            kept.append(candidate)
    return sorted(kept)


def _clean_years(raw_values: Any, dropped: List[str]) -> List[int]:
    """Keep plausible four-digit competition years only."""
    values = _as_list(raw_values)
    if values is None:
        dropped.append("competition_year:not_a_list")
        return []

    kept: List[int] = []
    for value in values:
        if isinstance(value, bool):
            dropped.append(f"competition_year:not_a_year:{value}")
            continue
        try:
            year = int(str(value).strip())
        except (TypeError, ValueError):
            dropped.append(f"competition_year:not_a_year:{_normalize_key(value)}")
            continue
        if not (config.ANALYZER_YEAR_MIN <= year <= config.ANALYZER_YEAR_MAX):
            dropped.append(f"competition_year:out_of_range:{year}")
            continue
        if year not in kept:
            kept.append(year)
    return sorted(kept)


# -- deterministic extractor ----------------------------------------------
def _is_monetary_context(question: str, match: "re.Match[str]") -> bool:
    before = question[max(0, match.start() - config.ANALYZER_YEAR_MONEY_WINDOW) : match.start()]
    after = question[match.end() : match.end() + config.ANALYZER_YEAR_MONEY_WINDOW]
    if config.PATTERN_ANALYZER_MONEY_PREFIX.search(before):
        return True
    return config.PATTERN_ANALYZER_MONEY_SUFFIX.match(after) is not None


def extract_deterministic_filters(question: str) -> Dict[str, List[Any]]:
    """
    Read explicit constraints straight from the question (fix pack F2c).

    Only competition type keywords and a single four-digit year are extracted.
    An achievement level is deliberately never inferred here: "juara vs belum
    juara" is a comparison, not a filter.
    """
    text = question or ""
    normalized = f" {_normalize_key(text)} "
    filters: Dict[str, List[Any]] = {}

    types: List[str] = []
    for keyword in sorted(TYPE_KEYWORDS, key=len, reverse=True):
        if f" {keyword} " in normalized:
            canonical = TYPE_KEYWORDS[keyword]
            if canonical not in types:
                types.append(canonical)
    if types:
        filters["competition_type"] = sorted(types)

    years: List[int] = []
    for match in config.PATTERN_ANALYZER_YEAR.finditer(text):
        year = int(match.group(1))
        if not (config.ANALYZER_YEAR_MIN <= year <= config.ANALYZER_YEAR_MAX):
            continue
        if _is_monetary_context(text, match):
            continue
        if year not in years:
            years.append(year)
    if years:
        filters["competition_year"] = sorted(years)

    return filters


def _merge_filters(
    llm_filters: Dict[str, List[Any]], deterministic: Dict[str, List[Any]]
) -> Dict[str, List[Any]]:
    """Union per field: neither source may erase what the other found."""
    merged: Dict[str, List[Any]] = {}
    for key in FILTER_KEYS:
        values: List[Any] = []
        for value in list(llm_filters.get(key) or []) + list(deterministic.get(key) or []):
            if value not in values:
                values.append(value)
        if values:
            merged[key] = sorted(values, key=lambda item: str(item))
    return merged


# -- validation ------------------------------------------------------------
def validate_analysis(
    payload: Dict[str, Any],
    question: str,
    *,
    latency_ms: float = 0.0,
    usage: Optional[Dict[str, int]] = None,
    raw_output: str = "",
    queue_wait_ms: float = 0.0,
    model_latency_ms: float = 0.0,
) -> AnalyzerResult:
    """Validate every analyzer field against the enum sets and config limits."""
    dropped: List[str] = []

    in_scope_raw = payload.get("in_scope", True)
    if isinstance(in_scope_raw, bool):
        in_scope = in_scope_raw
    else:
        in_scope = True
        dropped.append("in_scope:not_a_bool")

    mode_raw = payload.get("mode")
    try:
        mode = QueryMode(mode_raw)
    except (ValueError, TypeError):
        mode = QueryMode(config.ANALYZER_FALLBACK_MODE)
        dropped.append(f"mode:unknown:{_normalize_key(mode_raw)}")

    search_query_raw = payload.get("search_query")
    if isinstance(search_query_raw, str):
        search_query = " ".join(search_query_raw.split())
    else:
        search_query = ""
        dropped.append("search_query:not_a_string")
    if not search_query:
        search_query = " ".join(question.split())
    search_query = search_query[: config.ANALYZER_MAX_SEARCH_QUERY_CHARS].strip()

    compare_by_raw = payload.get("compare_by")
    compare_by: Optional[str] = None
    if compare_by_raw not in (None, "", "null", "None"):
        candidate = _normalize_key(compare_by_raw).replace(" ", "_")
        if candidate not in config.COMPARE_BY_OPTIONS:
            dropped.append(f"compare_by:unknown:{candidate}")
        elif mode is not QueryMode.CROSS_REPORT:
            dropped.append(f"compare_by:not_cross_report:{candidate}")
        else:
            compare_by = candidate

    filters_raw = payload.get("filters")
    filters: Dict[str, List[Any]] = {}
    if isinstance(filters_raw, dict):
        types = _clean_competition_types(filters_raw.get("competition_type"), dropped)
        levels = _clean_achievement_levels(filters_raw.get("achievement_level"), dropped)
        years = _clean_years(filters_raw.get("competition_year"), dropped)
        if types:
            filters["competition_type"] = types
        if years:
            filters["competition_year"] = years
        if levels:
            filters["achievement_level"] = levels
    elif filters_raw is not None:
        dropped.append("filters:not_an_object")

    deterministic = extract_deterministic_filters(question)
    merged = _merge_filters(filters, deterministic)

    return AnalyzerResult(
        in_scope=in_scope,
        mode=mode,
        search_query=search_query,
        filters=merged,
        used_fallback=False,
        dropped_invalid_values=len(dropped),
        latency_ms=round(latency_ms, 1),
        queue_wait_ms=round(queue_wait_ms, 1),
        model_latency_ms=round(model_latency_ms, 1),
        usage=dict(usage or {}),
        raw_output=raw_output,
        payload=dict(payload),
        filters_before_validation=dict(filters_raw) if isinstance(filters_raw, dict) else {},
        dropped_reasons=dropped,
        deterministic_filters=deterministic,
        compare_by=compare_by,
    )


def fallback_result(
    question: str,
    *,
    reason: str,
    latency_ms: float = 0.0,
    raw_output: str = "",
    queue_wait_ms: float = 0.0,
    model_latency_ms: float = 0.0,
) -> AnalyzerResult:
    """Fallback mandated by Brief 5.5: in scope, specific, no filters."""
    logger.warning("analyzer_fallback reason=%s", reason)
    return AnalyzerResult(
        in_scope=True,
        mode=QueryMode(config.ANALYZER_FALLBACK_MODE),
        search_query=" ".join(question.split())[
            : config.ANALYZER_MAX_SEARCH_QUERY_CHARS
        ],
        filters={},
        used_fallback=True,
        latency_ms=round(latency_ms, 1),
        queue_wait_ms=round(queue_wait_ms, 1),
        model_latency_ms=round(model_latency_ms, 1),
        error_type=reason,
        raw_output=raw_output,
        dropped_reasons=[f"fallback:{reason}"],
    )


class Analyzer:
    """Runs the analyzer prompt and never lets a bad payload escape."""

    def __init__(self, llm: LLMClient, system_prompt: str = prompts.ANALYZER_SYSTEM_PROMPT):
        self.llm = llm
        self.system_prompt = system_prompt

    def analyze(
        self, question: str, deadline: Optional[Deadline] = None
    ) -> AnalyzerResult:
        try:
            completion = self.llm.complete(
                self.system_prompt,
                prompts.ANALYZER_USER_TEMPLATE.format(
                    question=question[: config.ANALYZER_MAX_QUESTION_CHARS]
                ),
                temperature=config.LLM_TEMPERATURE_ANALYZER,
                max_tokens=config.LLM_ANALYZER_MAX_OUTPUT_TOKENS,
                reasoning_effort=config.LLM_REASONING_EFFORT_ANALYZER,
                json_mode=True,
                deadline=deadline,
            )
        except (DeadlineExceeded, LLMRateLimitError):
            # A budget or allowance problem affects the whole question; a silent
            # fallback would only spend another call that is certain to fail.
            raise
        except LLMError:
            return fallback_result(question, reason="llm_error")
        except Exception:
            # No exception text is logged: it may contain provider payloads.
            return fallback_result(question, reason="unexpected_error")

        latency_ms = completion.latency_ms
        try:
            payload = parse_analyzer_output(completion.content)
        except AnalyzerParseError:
            return fallback_result(
                question,
                reason="invalid_json",
                latency_ms=latency_ms,
                raw_output=completion.content,
                queue_wait_ms=completion.queue_wait_ms,
                model_latency_ms=completion.model_latency_ms,
            )

        return validate_analysis(
            payload,
            question,
            latency_ms=latency_ms,
            usage=completion.usage.to_dict(),
            raw_output=completion.content,
            queue_wait_ms=completion.queue_wait_ms,
            model_latency_ms=completion.model_latency_ms,
        )
