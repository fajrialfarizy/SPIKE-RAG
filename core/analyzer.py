"""
Analyzer: LLM call #1 (Brief 5.5 step 2).

The model returns JSON; nothing it says is trusted. Every field is validated in
code against the enum sets in enums.py, invalid values are dropped, and the search
query is length-capped. A broken response (invalid JSON, provider failure, unusable
payload) falls back to in_scope=true / mode=specific / no filters, and the fallback
is logged without the question text.

Out-of-scope is only set when the model clearly says so; when in doubt the question
stays in scope, as Brief 5.5 requires.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    usage: Dict[str, int] = field(default_factory=dict)
    error_type: Optional[str] = None


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


def _clean_values(
    raw_values: Any, allowed: tuple, *, cast=str
) -> tuple[List[Any], int]:
    """Keep only allowed values, deduplicated and sorted; count what was dropped."""
    if raw_values is None:
        return [], 0
    if not isinstance(raw_values, list):
        return [], 1

    kept: List[Any] = []
    dropped = 0
    for value in raw_values:
        if isinstance(value, bool):
            dropped += 1
            continue
        try:
            candidate = cast(value)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if candidate not in allowed:
            dropped += 1
            continue
        if candidate not in kept:
            kept.append(candidate)

    return sorted(kept, key=lambda item: str(item)), dropped


def validate_analysis(
    payload: Dict[str, Any],
    question: str,
    *,
    latency_ms: float = 0.0,
    usage: Optional[Dict[str, int]] = None,
) -> AnalyzerResult:
    """Validate every analyzer field against the enum sets and config limits."""
    dropped = 0

    in_scope_raw = payload.get("in_scope", True)
    if isinstance(in_scope_raw, bool):
        in_scope = in_scope_raw
    else:
        in_scope = True
        dropped += 1

    mode_raw = payload.get("mode")
    try:
        mode = QueryMode(mode_raw)
    except (ValueError, TypeError):
        mode = QueryMode(config.ANALYZER_FALLBACK_MODE)
        dropped += 1

    search_query_raw = payload.get("search_query")
    if isinstance(search_query_raw, str):
        search_query = " ".join(search_query_raw.split())
    else:
        search_query = ""
        dropped += 1
    if not search_query:
        search_query = " ".join(question.split())
    search_query = search_query[: config.ANALYZER_MAX_SEARCH_QUERY_CHARS].strip()

    filters_raw = payload.get("filters")
    filters: Dict[str, List[Any]] = {}
    if isinstance(filters_raw, dict):
        types, dropped_types = _clean_values(
            filters_raw.get("competition_type"), ALLOWED_COMPETITION_TYPES
        )
        levels, dropped_levels = _clean_values(
            filters_raw.get("achievement_level"), ALLOWED_ACHIEVEMENT_LEVELS
        )
        years, dropped_years = _clean_years(filters_raw.get("competition_year"))

        dropped += dropped_types + dropped_levels + dropped_years
        if types:
            filters["competition_type"] = types
        if years:
            filters["competition_year"] = years
        if levels:
            filters["achievement_level"] = levels
    elif filters_raw is not None:
        dropped += 1

    return AnalyzerResult(
        in_scope=in_scope,
        mode=mode,
        search_query=search_query,
        filters=filters,
        used_fallback=False,
        dropped_invalid_values=dropped,
        latency_ms=round(latency_ms, 1),
        usage=dict(usage or {}),
    )


def _clean_years(raw_values: Any) -> tuple[List[int], int]:
    """Keep plausible four-digit competition years only."""
    if raw_values is None:
        return [], 0
    if not isinstance(raw_values, list):
        return [], 1

    kept: List[int] = []
    dropped = 0
    for value in raw_values:
        if isinstance(value, bool):
            dropped += 1
            continue
        try:
            year = int(value)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not (config.ANALYZER_YEAR_MIN <= year <= config.ANALYZER_YEAR_MAX):
            dropped += 1
            continue
        if year not in kept:
            kept.append(year)

    return sorted(kept), dropped


def fallback_result(
    question: str, *, reason: str, latency_ms: float = 0.0
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
        error_type=reason,
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
                prompts.ANALYZER_USER_TEMPLATE.format(question=question),
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
                question, reason="invalid_json", latency_ms=latency_ms
            )

        return validate_analysis(
            payload,
            question,
            latency_ms=latency_ms,
            usage=completion.usage.to_dict(),
        )
