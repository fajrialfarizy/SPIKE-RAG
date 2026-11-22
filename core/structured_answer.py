"""
Structured free-text answers (fix pack 9, O1).

The free-text modes (specific, the exact report-id route and cross_report without
compare_by) used to ask for prose with a citation and a bullet cap in the
addendum. The real run ignored both: 12 bullets, no [LAPn] at all, and the
citation repair retry then spent 3,800 tokens and still failed the queue wait.
The model now returns the same strict JSON the comparison path already uses:

    {"found": bool, "points": [{"text": str, "cite": [alias]}]}

The backend owns the contract, exactly as for comparisons:

- ``build_schema`` returns a Groq Structured Outputs schema (strict mode: every
  property is required, every object sets ``additionalProperties: false``).
- ``parse_payload`` parses one JSON object with no partial recovery.
- ``validate_payload`` resolves every alias against the retrieved set, drops a
  point without a valid citation, strips alias/report-id tokens from the text,
  caps points per primary report and then caps the total.
- ``render_answer`` writes the Markdown the student sees.

Citations only ever come from the model's own ``cite`` fields. The backend never
infers or attaches one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import config


class StructuredAnswerError(ValueError):
    """Raised when the structured answer payload cannot be used."""


@dataclass(frozen=True)
class AnswerPoint:
    """One validated bullet: backend-safe text plus resolved report ids."""

    text: str
    report_ids: List[str]


@dataclass(frozen=True)
class StructuredAnswerValidation:
    found: bool
    points: List[AnswerPoint] = field(default_factory=list)
    incidents: Dict[str, int] = field(default_factory=dict)
    cited_report_ids: List[str] = field(default_factory=list)
    dropped_points: int = 0


def build_schema() -> Dict[str, Any]:
    """Strict JSON Schema for one free-text answer."""
    return {
        "type": "object",
        "properties": {
            "found": {"type": "boolean"},
            "points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "cite": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text", "cite"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["found", "points"],
        "additionalProperties": False,
    }


def response_format() -> Dict[str, Any]:
    """The ``response_format`` object Groq expects for strict structured output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": config.ANSWER_JSON_SCHEMA_NAME,
            "strict": True,
            "schema": build_schema(),
        },
    }


def parse_payload(raw: str) -> Dict[str, Any]:
    """Parse the completion as one JSON object. No fences, no partial recovery."""
    text = (raw or "").strip()
    if not text:
        raise StructuredAnswerError("The answer was empty.")
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise StructuredAnswerError("The answer is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise StructuredAnswerError("The answer is not a JSON object.")
    return payload


_ALIAS_TOKEN = re.compile(r"[\[\(]?\s*LAP[0-9]{1,3}\s*[\]\)]?", re.IGNORECASE)
_REPORT_ID_TOKEN = re.compile(r"[\[\(]?\s*RPT[0-9]{4,}\s*[\]\)]?", re.IGNORECASE)


def _clean_text(raw: Any, incidents: Dict[str, int]) -> str:
    """Model text without alias/identifier tokens, normalized and word-capped."""
    text = str(raw or "")
    if _ALIAS_TOKEN.search(text):
        incidents["alias_in_text"] += 1
        text = _ALIAS_TOKEN.sub(" ", text)
    if _REPORT_ID_TOKEN.search(text):
        incidents["report_id_in_text"] += 1
        text = _REPORT_ID_TOKEN.sub(" ", text)
    text = " ".join(text.split()).strip(" -")

    words = text.split()
    if len(words) > config.ANSWER_POINT_MAX_WORDS:
        incidents["point_too_long"] += 1
        text = " ".join(words[: config.ANSWER_POINT_MAX_WORDS])
    return text


def _normalize_alias(value: Any) -> str:
    return str(value or "").strip().strip("[]() ").upper()


def _clean_point(
    raw: Any,
    *,
    alias_map: Mapping[str, str],
    retrieved: Sequence[str],
    incidents: Dict[str, int],
) -> Optional[AnswerPoint]:
    """One bullet after validation, or None when it cannot be traced."""
    if not isinstance(raw, dict):
        incidents["malformed_point"] += 1
        return None

    text = _clean_text(raw.get("text"), incidents)
    if not text:
        incidents["empty_point"] += 1
        return None

    raw_cites = raw.get("cite")
    if isinstance(raw_cites, str):
        raw_cites = [raw_cites]
    if not isinstance(raw_cites, list):
        raw_cites = []

    resolved: List[str] = []
    for item in raw_cites:
        alias = _normalize_alias(item)
        report_id = alias_map.get(alias)
        if report_id is None or report_id not in retrieved:
            incidents["unknown_alias"] += 1
            continue
        if report_id not in resolved:
            resolved.append(report_id)

    # Fix pack 9: the cite cap is part of the contract, so a point over it is
    # dropped whole instead of being cut and kept (a cut list attributes the
    # text to the wrong reports).
    if len(resolved) > config.ANSWER_CITE_MAX:
        incidents["cite_over_limit_dropped"] += 1
        return None

    if len(resolved) < config.ANSWER_CITE_MIN:
        incidents["point_uncited"] += 1
        return None

    return AnswerPoint(text=text, report_ids=resolved)


def validate_payload(
    payload: Mapping[str, Any],
    *,
    alias_map: Mapping[str, str],
    retrieved: Sequence[str],
    max_points_per_report: int = config.ANSWER_MAX_BULLETS_PER_REPORT,
) -> StructuredAnswerValidation:
    """
    Validate every point and citation; never render unvalidated text.

    Fix pack 10 (R1): ``max_points_per_report`` is the ordinary breadth cap, and
    the pipeline raises it to ANSWER_ID_ROUTE_MAX_POINTS_PER_REPORT when the
    question names an explicit report (or the context holds one report), because
    such a question asks for every item the report states. The total cap always
    applies.
    """
    incidents = {
        "unknown_alias": 0,
        "point_uncited": 0,
        "alias_in_text": 0,
        "report_id_in_text": 0,
        "point_too_long": 0,
        "cite_over_limit_dropped": 0,
        "malformed_point": 0,
        "empty_point": 0,
        "found_not_boolean": 0,
    }

    found = payload.get("found")
    if not isinstance(found, bool):
        incidents["found_not_boolean"] += 1
        raise StructuredAnswerError("The answer has no boolean 'found' field.")

    raw_points = payload.get("points")
    if raw_points is None:
        raw_points = []
    if not isinstance(raw_points, list):
        raise StructuredAnswerError("The answer has no points array.")

    if not found:
        return StructuredAnswerValidation(found=False)

    points: List[AnswerPoint] = []
    dropped_points = 0
    per_report: Dict[str, int] = {}
    for raw_point in raw_points:
        point = _clean_point(
            raw_point, alias_map=alias_map, retrieved=retrieved, incidents=incidents
        )
        if point is None:
            continue
        primary = point.report_ids[0]
        if per_report.get(primary, 0) >= max_points_per_report:
            dropped_points += 1
            continue
        if len(points) >= config.ANSWER_MAX_BULLETS:
            dropped_points += 1
            continue
        per_report[primary] = per_report.get(primary, 0) + 1
        points.append(point)

    cited: List[str] = []
    for point in points:
        for report_id in point.report_ids:
            if report_id not in cited:
                cited.append(report_id)

    return StructuredAnswerValidation(
        found=True,
        points=points,
        incidents={key: value for key, value in incidents.items() if value},
        cited_report_ids=cited,
        dropped_points=dropped_points,
    )


def _citation(report_ids: Sequence[str]) -> str:
    """One citation format everywhere: [RPT0001] [RPT0002] (fix pack L6)."""
    return config.CITATION_GROUP_JOINER.join(
        config.CITATION_GROUP_TEMPLATE.format(report_id=report_id)
        for report_id in report_ids
    )


def render_answer(validation: StructuredAnswerValidation) -> str:
    """Render the validated points as one bullet each; no intro, no closing."""
    return "\n".join(
        config.ANSWER_BULLET_TEMPLATE.format(
            text=point.text, citations=_citation(point.report_ids)
        )
        for point in validation.points
    )
