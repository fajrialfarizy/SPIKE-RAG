"""
Structured comparison answers (fix pack J1, keyed in fix pack 9, by reference in
fix pack 10).

The comparison path asks the model for JSON only:

    {"groups": {"g1": [{"text": ..., "cite": [...]}], "g2": [...]},
     "contrasts": [{"dimension": ..., "pick": {"g1": 1, "g2": 2}}]}

Fix pack 9 replaced label strings with model-facing keys (g1, g2, ...): the model
never writes a group name, so it cannot invent one, and the backend owns the
mapping to the configured label. All keys are required (strict mode), but only
the groups that were requested and actually have documents are in the schema, so
an empty group is never rendered or counted.

Fix pack 10 (R2) replaced contrast sides with picks: a contrast names the 1-based
number of one point per group, exactly as the model wrote those points. The
model writes no contrast text at all, so the backend renders the VALIDATED point
text and citations. The real run repeated the group points on the contrast sides
and added merged claims there.

The model never writes a heading, a citation, a count or the Sumber block:

- ``build_schema`` returns a Groq Structured Outputs schema (strict mode: every
  property is required and every object sets ``additionalProperties: false``;
  docs checked 2026-11-21). Per-point caps are enforced here, not in the schema,
  because the documented keyword subset does not promise ``minItems``/``maxItems``.
- ``parse_payload`` parses the raw completion with no partial recovery.
- ``validate_payload`` resolves every alias against the retrieved set, enforces
  group membership and the citation ceilings, and counts each rejection.
- ``render_answer`` writes the Markdown the student sees.

A point over its citation ceiling is DROPPED, never cut: the real run cut three
cited aliases to two and attributed a claim to the wrong reports. A contrast
whose pick points at a dropped or out-of-range point is dropped with a reason
code, because it has no validated text to render.

Invalid JSON, a truncated completion or a payload that is not an object is a
technical error and is never retried.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import config


class CompareFormatError(ValueError):
    """Raised when the structured comparison payload cannot be used."""


@dataclass(frozen=True)
class ComparePoint:
    """One validated bullet: backend-safe text plus resolved report ids."""

    text: str
    report_ids: List[str]


@dataclass(frozen=True)
class CompareSection:
    """One comparison group, in the order config declares it."""

    label: str
    points: List[ComparePoint]


@dataclass(frozen=True)
class CompareContrastSide:
    """One group's side of a contrast: validated point text and report ids."""

    label: str
    text: str
    report_ids: List[str]


@dataclass(frozen=True)
class CompareContrast:
    """The same dimension described by every comparison group."""

    dimension: str
    sides: List[CompareContrastSide]


@dataclass(frozen=True)
class CompareValidation:
    sections: List[CompareSection]
    contrasts: List[CompareContrast]
    incidents: Dict[str, int] = field(default_factory=dict)
    cited_report_ids: List[str] = field(default_factory=list)
    reports_cited_per_group: Dict[str, int] = field(default_factory=dict)
    # Fix pack 10 (R6): every group point in the model's own order, with its
    # 1-based index and whether it survived validation, so a smoke run can be
    # checked against the picks.
    points_by_key: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not any(section.points for section in self.sections) and not self.contrasts

    @property
    def group_empty(self) -> List[str]:
        """Groups that produced no valid point; every section has documents."""
        return [section.label for section in self.sections if not section.points]


@dataclass(frozen=True)
class _PointResult:
    """One raw point after validation; ``reason`` is empty when it was kept."""

    text: str
    report_ids: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.reason


def _point_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "cite": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["text", "cite"],
        "additionalProperties": False,
    }


def _pick_schema(group_keys: Sequence[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {key: {"type": "integer"} for key in group_keys},
        "required": list(group_keys),
        "additionalProperties": False,
    }


def _contrast_schema(group_keys: Sequence[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "dimension": {"type": "string"},
            config.COMPARE_CONTRAST_PICK: _pick_schema(group_keys),
        },
        "required": ["dimension", config.COMPARE_CONTRAST_PICK],
        "additionalProperties": False,
    }


def build_schema(group_keys: Sequence[str]) -> Dict[str, Any]:
    """Strict JSON Schema for one comparison answer over the given group keys."""
    keys = list(group_keys)
    return {
        "type": "object",
        "properties": {
            "groups": {
                "type": "object",
                "properties": {key: {"type": "array", "items": _point_schema()} for key in keys},
                "required": keys,
                "additionalProperties": False,
            },
            "contrasts": {
                "type": "array",
                "items": _contrast_schema(keys),
            },
        },
        "required": ["groups", "contrasts"],
        "additionalProperties": False,
    }


def response_format(group_keys: Sequence[str]) -> Dict[str, Any]:
    """The ``response_format`` object Groq expects for strict structured output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": config.COMPARE_JSON_SCHEMA_NAME,
            "strict": True,
            "schema": build_schema(group_keys),
        },
    }


def parse_payload(raw: str) -> Dict[str, Any]:
    """Parse the completion as one JSON object. No fences, no partial recovery."""
    text = (raw or "").strip()
    if not text:
        raise CompareFormatError("The comparison answer was empty.")
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise CompareFormatError("The comparison answer is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise CompareFormatError("The comparison answer is not a JSON object.")
    return payload


_ALIAS_TOKEN = re.compile(r"[\[\(]?\s*LAP[0-9]{1,3}\s*[\]\)]?", re.IGNORECASE)
_REPORT_ID_TOKEN = re.compile(r"[\[\(]?\s*RPT[0-9]{4,}\s*[\]\)]?", re.IGNORECASE)
_DIMENSION_TRIM = " .,;:!?()[]{}\"'"


def _clean_text(
    raw: Any,
    incidents: Dict[str, int],
    *,
    max_words: int = config.COMPARE_POINT_MAX_WORDS,
    limit_incident: str = "point_too_long",
) -> str:
    """Model text without alias/identifier tokens, normalized and word-capped."""
    text = str(raw or "")
    if _ALIAS_TOKEN.search(text):
        incidents["alias_in_text"] += 1
        text = _ALIAS_TOKEN.sub(" ", text)
    if _REPORT_ID_TOKEN.search(text):
        incidents["report_id_in_text"] += 1
        text = _REPORT_ID_TOKEN.sub(" ", text)
    text = " ".join(text.split())
    text = text.strip(" -")

    words = text.split()
    if len(words) > max_words:
        incidents[limit_incident] += 1
        text = " ".join(words[:max_words])
    return text


def _normalize_alias(value: Any) -> str:
    """Brackets, spaces and case never change which alias the model meant."""
    return str(value or "").strip().strip("[]() \t").upper()


def _resolve_cites(
    raw_cites: Any,
    *,
    label: Optional[str],
    allowed_reports: Mapping[str, str],
    alias_map: Mapping[str, str],
    retrieved: Sequence[str],
    incidents: Dict[str, int],
) -> List[str]:
    """Resolved report ids for one point, in the model's own order."""
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
        if label is not None and allowed_reports.get(report_id) != label:
            incidents["cross_group_citation"] += 1
            continue
        if report_id not in resolved:
            resolved.append(report_id)
    return resolved


def _clean_point(
    raw: Any,
    *,
    allowed_reports: Mapping[str, str],
    alias_map: Mapping[str, str],
    retrieved: Sequence[str],
    incidents: Dict[str, int],
    label: Optional[str] = None,
) -> _PointResult:
    """
    One bullet after validation, with the reason it was dropped if it was.

    ``allowed_reports`` maps a report id to the comparison group it belongs to;
    a citation outside that group is removed and counted.
    """
    if not isinstance(raw, dict):
        incidents["malformed_point"] += 1
        return _PointResult(text="", reason="malformed_point")

    text = _clean_text(raw.get("text"), incidents)
    if not text:
        incidents["empty_point"] += 1
        return _PointResult(text="", reason="empty_point")

    resolved = _resolve_cites(
        raw.get("cite"),
        label=label,
        allowed_reports=allowed_reports,
        alias_map=alias_map,
        retrieved=retrieved,
        incidents=incidents,
    )

    if len(resolved) > config.COMPARE_CITE_MAX:
        incidents["cite_over_limit_dropped"] += 1
        return _PointResult(text=text, report_ids=resolved, reason="cite_over_limit_dropped")

    if len(resolved) < config.COMPARE_CITE_MIN:
        incidents["point_uncited"] += 1
        return _PointResult(text=text, reason="point_uncited")

    return _PointResult(text=text, report_ids=resolved)


def _normalize_dimension(dimension: str) -> str:
    """Case, whitespace and edge punctuation removed; the WHOLE string compared."""
    return " ".join(str(dimension or "").split()).casefold().strip(_DIMENSION_TRIM)


_GENERIC_DIMENSIONS = frozenset(
    _normalize_dimension(item) for item in config.COMPARE_GENERIC_DIMENSIONS
)


def _is_generic_dimension(dimension: str) -> bool:
    """
    True when the dimension names no concrete topic (fix pack 10, R2d).

    The full normalized dimension is compared for EXACT equality with the
    configured list, never word by word: "Kendala Keuangan" is a concrete topic
    and must pass, while "Kendala" does not.
    """
    return _normalize_dimension(dimension) in _GENERIC_DIMENSIONS


def _clean_contrast(
    raw: Any,
    *,
    group_keys: Mapping[str, str],
    points_by_key: Mapping[str, List[_PointResult]],
    incidents: Dict[str, int],
) -> Optional[CompareContrast]:
    """
    One contrast resolved from the model's picks, or None with a reason code.

    Every group key must be picked, each pick is a 1-based index into that
    group's points AS THE MODEL WROTE THEM, and the picked point must have
    survived validation. A generic or empty dimension is rendered with the
    neutral label and counted, never dropped.
    """

    def reject(reason: str) -> None:
        incidents["contrast_invalid"] += 1
        key = f"contrast_invalid_{reason}"
        incidents[key] = incidents.get(key, 0) + 1

    if not isinstance(raw, dict):
        reject("malformed")
        return None

    dimension = _clean_text(
        raw.get("dimension"),
        incidents,
        max_words=config.COMPARE_DIMENSION_MAX_WORDS,
        limit_incident="dimension_too_long",
    )
    normalized = _normalize_dimension(dimension)
    if not normalized or normalized in _GENERIC_DIMENSIONS:
        incidents["generic_dimension"] += 1
        dimension = config.COMPARE_NEUTRAL_DIMENSION

    pick = raw.get(config.COMPARE_CONTRAST_PICK)
    if not isinstance(pick, dict):
        reject("pick_malformed")
        return None
    missing = [key for key in group_keys if key not in pick]
    if missing:
        reject("pick_missing")
        return None
    unknown = [key for key in pick if key not in group_keys]
    if unknown:
        reject("pick_unknown")
        return None

    sides: List[CompareContrastSide] = []
    for key, label in group_keys.items():
        value = pick.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            reject("pick_type")
            return None
        results = points_by_key.get(key, [])
        if value < 1 or value > len(results):
            reject("pick_range")
            return None
        result = results[value - 1]
        if not result.ok:
            # The picked point was dropped during validation, so the contrast
            # has no validated text to stand on.
            reject("pick_dropped")
            return None
        sides.append(
            CompareContrastSide(
                label=label,
                text=result.text,
                report_ids=list(result.report_ids),
            )
        )

    return CompareContrast(dimension=dimension, sides=sides)


def validate_payload(
    payload: Mapping[str, Any],
    *,
    group_keys: Mapping[str, str],
    alias_map: Mapping[str, str],
    report_group: Mapping[str, str],
    retrieved: Sequence[str],
) -> CompareValidation:
    """Validate every group, point and citation; never render unvalidated text."""
    incidents = {
        "unknown_alias": 0,
        "cross_group_citation": 0,
        "point_uncited": 0,
        "unknown_label": 0,
        "malformed_point": 0,
        "empty_point": 0,
        "alias_in_text": 0,
        "report_id_in_text": 0,
        "point_too_long": 0,
        "cite_over_limit_dropped": 0,
        "points_over_limit": 0,
        "contrast_invalid": 0,
        "contrast_truncated": 0,
        "dimension_too_long": 0,
        "generic_dimension": 0,
    }

    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, dict):
        raise CompareFormatError("The comparison answer has no groups object.")
    unknown = [key for key in raw_groups if key not in group_keys]
    if unknown:
        # Strict output cannot produce an unknown key, so this is a contract
        # violation and is never rendered (fix pack 9 replaced free labels with
        # keys, and a key the backend does not know has no label to map to).
        raise CompareFormatError("The comparison answer has an unknown group key.")
    missing = [key for key in group_keys if key not in raw_groups]
    if missing:
        raise CompareFormatError("The comparison answer is missing a group key.")

    points_by_key: Dict[str, List[_PointResult]] = {}
    by_label: Dict[str, List[ComparePoint]] = {
        label: [] for label in group_keys.values()
    }
    for key, label in group_keys.items():
        raw_points = raw_groups.get(key)
        results: List[_PointResult] = []
        if not isinstance(raw_points, list):
            incidents["malformed_point"] += 1
            points_by_key[key] = results
            continue
        for raw_point in raw_points:
            result = _clean_point(
                raw_point,
                allowed_reports=report_group,
                alias_map=alias_map,
                retrieved=retrieved,
                incidents=incidents,
                label=label,
            )
            if result.ok and len(by_label[label]) >= config.COMPARE_POINTS_PER_GROUP_MAX:
                incidents["points_over_limit"] += 1
                result = _PointResult(
                    text=result.text,
                    report_ids=result.report_ids,
                    reason="points_over_limit",
                )
            if result.ok:
                by_label[label].append(
                    ComparePoint(text=result.text, report_ids=list(result.report_ids))
                )
            results.append(result)
        points_by_key[key] = results

    contrasts: List[CompareContrast] = []
    raw_contrasts = payload.get("contrasts")
    if isinstance(raw_contrasts, list):
        for raw_contrast in raw_contrasts:
            if len(contrasts) >= config.COMPARE_CONTRASTS_MAX:
                incidents["contrast_truncated"] += 1
                continue
            contrast = _clean_contrast(
                raw_contrast,
                group_keys=group_keys,
                points_by_key=points_by_key,
                incidents=incidents,
            )
            if contrast is not None:
                contrasts.append(contrast)
    elif raw_contrasts is not None:
        incidents["contrast_invalid"] += 1
        incidents["contrast_invalid_malformed"] = (
            incidents.get("contrast_invalid_malformed", 0) + 1
        )

    sections = [
        CompareSection(label=label, points=by_label[label])
        for label in group_keys.values()
    ]

    cited: List[str] = []
    reports_per_group: Dict[str, int] = {}
    for section in sections:
        per_group: List[str] = []
        for point in section.points:
            for report_id in point.report_ids:
                if report_id not in cited:
                    cited.append(report_id)
                if report_id not in per_group:
                    per_group.append(report_id)
        reports_per_group[section.label] = len(per_group)
    for contrast in contrasts:
        for side in contrast.sides:
            for report_id in side.report_ids:
                if report_id not in cited:
                    cited.append(report_id)

    points_debug: Dict[str, List[Dict[str, Any]]] = {}
    for key, results in points_by_key.items():
        points_debug[key] = [
            {
                "index": index,
                "text": result.text,
                "cites": list(result.report_ids),
                "kept": result.ok,
                "reason": result.reason,
            }
            for index, result in enumerate(results, start=1)
        ]

    return CompareValidation(
        sections=sections,
        contrasts=contrasts,
        incidents={key: value for key, value in incidents.items() if value},
        cited_report_ids=cited,
        reports_cited_per_group=reports_per_group,
        points_by_key=points_debug,
    )


def _citation(report_ids: Sequence[str]) -> str:
    """One citation format everywhere: [RPT0001] [RPT0002] (fix pack L6)."""
    return config.CITATION_GROUP_JOINER.join(
        config.CITATION_GROUP_TEMPLATE.format(report_id=report_id)
        for report_id in report_ids
    )


def render_answer(
    validation: CompareValidation,
    compare_by: Optional[str] = None,
    *,
    degraded: bool = False,
) -> str:
    """
    Render validated sections, paired contrasts and their citations.

    ``degraded`` is the fix pack 9 fallback after a failed repair retry: only the
    groups that produced points are rendered, the contrasts are omitted, and a
    group with documents but no point says the attempt failed instead of claiming
    there is no evidence.
    """
    blocks: List[str] = []

    for section in validation.sections:
        if not section.points and not degraded:
            continue
        lines = [
            config.COMPARE_GROUP_HEADING_TEMPLATE.format(
                label=config.comparison_display_label(compare_by or "", section.label)
            ),
            "",
        ]
        if section.points:
            lines.extend(
                config.COMPARE_BULLET_TEMPLATE.format(
                    text=point.text, citations=_citation(point.report_ids)
                )
                for point in section.points
            )
        else:
            lines.append(config.COMPARE_GROUP_EMPTY_RETRY)
        blocks.append(config.COMPARE_LINE_SEPARATOR.join(lines))

    if validation.contrasts and not degraded:
        lines = [config.COMPARE_CONTRASTS_HEADING, ""]
        for index, contrast in enumerate(validation.contrasts):
            if index:
                lines.append("")
            lines.append(
                config.COMPARE_CONTRAST_DIMENSION_TEMPLATE.format(
                    dimension=contrast.dimension
                )
            )
            lines.extend(
                config.COMPARE_CONTRAST_SIDE_TEMPLATE.format(
                    label=config.comparison_display_label(compare_by or "", side.label),
                    text=side.text,
                    citations=_citation(side.report_ids),
                )
                for side in contrast.sides
            )
        blocks.append(config.COMPARE_LINE_SEPARATOR.join(lines))

    return config.COMPARE_SECTION_SEPARATOR.join(blocks)
