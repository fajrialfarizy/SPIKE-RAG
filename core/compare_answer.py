"""
Structured comparison answers (fix pack J1).

The comparison path asks the model for JSON only:

    {"groups": [{"label": ..., "points": [{"text": ..., "cite": [...]}]}],
     "contrasts": [{"dimension": ...,
                    "sides": [{"label": ..., "text": ..., "cite": [...]}]}]}

A contrast is a pair of sides on the same dimension, exactly one per group
(fix pack 8, N1). The free "differences" list could name one group only, which
made every difference unsupported; a paired shape can be validated instead.

The model never writes a heading, a citation, a count or the Sumber block. This
module owns the contract:

- ``build_schema`` returns a Groq Structured Outputs schema (strict mode: every
  property is required and every object sets ``additionalProperties: false``;
  docs checked 2026-11-21). Per-point caps are enforced here, not in the schema,
  because the documented keyword subset does not promise ``minItems``/``maxItems``.
- ``parse_payload`` parses the raw completion with no partial recovery.
- ``validate_payload`` resolves every alias against the retrieved set, enforces
  group membership, drops a point whose citations are all invalid, and counts
  each rejection as an incident.
- ``render_answer`` writes the Markdown the student sees.

Invalid JSON, a truncated completion or a payload that is not an object is a
technical error, never a partially rendered answer, and is never retried: a retry
costs quota and would fail the same way.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
    """One group's side of a contrast: text plus this group's report ids."""

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

    @property
    def is_empty(self) -> bool:
        return not any(section.points for section in self.sections) and not self.contrasts


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


def _side_schema(labels: Sequence[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(labels)},
            "text": {"type": "string"},
            "cite": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["label", "text", "cite"],
        "additionalProperties": False,
    }


def _contrast_schema(labels: Sequence[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "dimension": {"type": "string"},
            "sides": {"type": "array", "items": _side_schema(labels)},
        },
        "required": ["dimension", "sides"],
        "additionalProperties": False,
    }


def build_schema(labels: Sequence[str]) -> Dict[str, Any]:
    """Strict JSON Schema for one comparison answer over the given group labels."""
    return {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": list(labels)},
                        "points": {"type": "array", "items": _point_schema()},
                    },
                    "required": ["label", "points"],
                    "additionalProperties": False,
                },
            },
            "contrasts": {
                "type": "array",
                "items": _contrast_schema(labels),
            },
        },
        "required": ["groups", "contrasts"],
        "additionalProperties": False,
    }


def response_format(labels: Sequence[str]) -> Dict[str, Any]:
    """The ``response_format`` object Groq expects for strict structured output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": config.COMPARE_JSON_SCHEMA_NAME,
            "strict": True,
            "schema": build_schema(labels),
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
    return str(value or "").strip().strip("[]() ").upper()


def _clean_point(
    raw: Any,
    *,
    allowed_reports: Mapping[str, str],
    alias_map: Mapping[str, str],
    retrieved: Sequence[str],
    incidents: Dict[str, int],
    label: Optional[str] = None,
) -> Optional[ComparePoint]:
    """
    One bullet after validation, or None when it cannot be traced to a report.

    ``allowed_reports`` maps a report id to the comparison group it belongs to;
    a citation outside that group is removed and counted.
    """
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
        if label is not None and allowed_reports.get(report_id) != label:
            incidents["cross_group_citation"] += 1
            continue
        if report_id not in resolved:
            resolved.append(report_id)

    if len(resolved) > config.COMPARE_CITE_MAX:
        incidents["cite_over_limit"] += 1
        resolved = resolved[: config.COMPARE_CITE_MAX]

    if len(resolved) < config.COMPARE_CITE_MIN:
        incidents["uncited_point_dropped"] += 1
        return None

    return ComparePoint(text=text, report_ids=resolved)


def _clean_contrast(
    raw: Any,
    *,
    required_labels: Sequence[str],
    allowed_reports: Mapping[str, str],
    alias_map: Mapping[str, str],
    retrieved: Sequence[str],
    incidents: Dict[str, int],
) -> Optional[CompareContrast]:
    """
    One contrast with exactly one supported side per group, or None.

    Missing, duplicated or unknown labels drop the whole contrast, as does a
    side that ends up with no citation after citations outside its own group are
    removed. A contrast beyond the configured limit is counted as truncated, not
    as invalid.
    """
    if not isinstance(raw, dict):
        incidents["contrast_invalid"] += 1
        return None

    dimension = _clean_text(
        raw.get("dimension"),
        incidents,
        max_words=config.COMPARE_DIMENSION_MAX_WORDS,
        limit_incident="dimension_too_long",
    )
    if not dimension:
        incidents["contrast_invalid"] += 1
        return None

    raw_sides = raw.get("sides")
    if not isinstance(raw_sides, list):
        incidents["contrast_invalid"] += 1
        return None

    by_label: Dict[str, CompareContrastSide] = {}
    for raw_side in raw_sides:
        if not isinstance(raw_side, dict):
            incidents["contrast_invalid"] += 1
            return None
        label = str(raw_side.get("label") or "").strip()
        if label not in required_labels or label in by_label:
            incidents["contrast_invalid"] += 1
            return None

        text = _clean_text(
            raw_side.get("text"),
            incidents,
            max_words=config.COMPARE_CONTRAST_TEXT_MAX_WORDS,
            limit_incident="contrast_text_too_long",
        )
        if not text:
            incidents["contrast_invalid"] += 1
            return None

        raw_cites = raw_side.get("cite")
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
            if allowed_reports.get(report_id) != label:
                incidents["cross_group_citation"] += 1
                continue
            if report_id not in resolved:
                resolved.append(report_id)

        if len(resolved) > config.COMPARE_CITE_MAX:
            incidents["cite_over_limit"] += 1
            resolved = resolved[: config.COMPARE_CITE_MAX]

        if not resolved:
            incidents["contrast_invalid"] += 1
            return None

        by_label[label] = CompareContrastSide(
            label=label, text=text, report_ids=resolved
        )

    if set(by_label) != set(required_labels):
        incidents["contrast_invalid"] += 1
        return None

    return CompareContrast(
        dimension=dimension, sides=[by_label[label] for label in required_labels]
    )


def validate_payload(
    payload: Mapping[str, Any],
    *,
    labels: Sequence[str],
    alias_map: Mapping[str, str],
    report_group: Mapping[str, str],
    retrieved: Sequence[str],
) -> CompareValidation:
    """Validate every group, point and citation; never render unvalidated text."""
    incidents = {
        "unknown_alias": 0,
        "cross_group_citation": 0,
        "uncited_point_dropped": 0,
        "unknown_label": 0,
        "malformed_point": 0,
        "empty_point": 0,
        "alias_in_text": 0,
        "report_id_in_text": 0,
        "point_too_long": 0,
        "cite_over_limit": 0,
        "points_over_limit": 0,
        "contrast_invalid": 0,
        "contrast_truncated": 0,
        "dimension_too_long": 0,
        "contrast_text_too_long": 0,
    }

    by_label: Dict[str, List[ComparePoint]] = {label: [] for label in labels}
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list):
        raise CompareFormatError("The comparison answer has no groups array.")

    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            incidents["malformed_point"] += 1
            continue
        label = str(raw_group.get("label") or "").strip()
        if label not in by_label:
            incidents["unknown_label"] += 1
            continue
        raw_points = raw_group.get("points")
        if not isinstance(raw_points, list):
            incidents["malformed_point"] += 1
            continue
        for raw_point in raw_points:
            point = _clean_point(
                raw_point,
                allowed_reports=report_group,
                alias_map=alias_map,
                retrieved=retrieved,
                incidents=incidents,
                label=label,
            )
            if point is None:
                continue
            if len(by_label[label]) >= config.COMPARE_POINTS_PER_GROUP_MAX:
                incidents["points_over_limit"] += 1
                continue
            by_label[label].append(point)

    contrasts: List[CompareContrast] = []
    # Fix pack 8: a contrast must have exactly one side per group that has
    # documents, each supported by that group's own reports. A free difference
    # that named one group only could never be validated, so every such point was
    # dropped. A configured but empty group cannot be one side of a contrast.
    active_labels = [
        label
        for label in labels
        if any(report_group.get(report_id) == label for report_id in retrieved)
    ]
    raw_contrasts = payload.get("contrasts")
    if isinstance(raw_contrasts, list):
        for raw_contrast in raw_contrasts:
            if len(contrasts) >= config.COMPARE_CONTRASTS_MAX:
                incidents["contrast_truncated"] += 1
                continue
            contrast = _clean_contrast(
                raw_contrast,
                required_labels=active_labels,
                allowed_reports=report_group,
                alias_map=alias_map,
                retrieved=retrieved,
                incidents=incidents,
            )
            if contrast is not None:
                contrasts.append(contrast)
    elif raw_contrasts is not None:
        incidents["contrast_invalid"] += 1

    sections = [
        CompareSection(label=label, points=by_label[label]) for label in labels
    ]

    cited: List[str] = []
    for section in sections:
        for point in section.points:
            for report_id in point.report_ids:
                if report_id not in cited:
                    cited.append(report_id)
    for contrast in contrasts:
        for side in contrast.sides:
            for report_id in side.report_ids:
                if report_id not in cited:
                    cited.append(report_id)

    return CompareValidation(
        sections=sections,
        contrasts=contrasts,
        incidents={key: value for key, value in incidents.items() if value},
        cited_report_ids=cited,
    )


def _citation(report_ids: Sequence[str]) -> str:
    """One citation format everywhere: [RPT0001] [RPT0002] (fix pack L6)."""
    return config.CITATION_GROUP_JOINER.join(
        config.CITATION_GROUP_TEMPLATE.format(report_id=report_id)
        for report_id in report_ids
    )


def render_answer(
    validation: CompareValidation, compare_by: Optional[str] = None
) -> str:
    """Render validated sections, paired contrasts and their citations."""
    blocks: List[str] = []

    for section in validation.sections:
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
            lines.append(config.COMPARE_EMPTY_GROUP)
        blocks.append(config.COMPARE_LINE_SEPARATOR.join(lines))

    if validation.contrasts:
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
