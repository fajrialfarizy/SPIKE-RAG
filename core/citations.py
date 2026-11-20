"""
Citation post-processing (Brief 5.6, PRD 8.2; fix pack F4).

The model only ever writes aliases (LAP1, LAP2, ...). The backend:

1. replaces a valid alias with its report_id, e.g. [RPT0003],
2. accepts every form the model may produce: [LAP1], [LAP1, LAP2],
   [LAP1][LAP2], (LAP2) and a bare LAP2 token,
3. deletes an unknown alias and counts an incident,
4. keeps a raw RPT id only when it is in the retrieved set, otherwise deletes it
   and counts an incident (the model may never introduce a source),
5. fails loudly if any alias token survives post-processing,
6. builds the "Sumber" block from cited reports only, using Chroma metadata.

No report metadata is ever invented: an id without stored metadata is skipped.
``find_uncited_mentions`` reports retrieved reports whose competition name appears
in the prose without a citation, so the pipeline can retry and then refuse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import config


class CitationPostconditionError(RuntimeError):
    """Raised when an alias token survives post-processing."""


@dataclass(frozen=True)
class CitationResult:
    text: str
    cited_report_ids: List[str] = field(default_factory=list)
    incidents: Dict[str, int] = field(default_factory=dict)


def has_citation(text: str) -> bool:
    """True when the text contains at least one alias token."""
    return config.PATTERN_ALIAS_ANY.search(text) is not None


def _normalize(text: str) -> str:
    return " ".join((text or "").casefold().split())


def find_uncited_mentions(
    text: str,
    cited_report_ids: Sequence[str],
    report_metadata: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    """
    Retrieved reports named in the prose without any citation.

    A report counts as mentioned when its competition name appears verbatim
    (case-insensitive, whitespace-normalized) and the answer never cites it.
    """
    lowered = _normalize(text)
    cited = set(cited_report_ids)
    missing: List[str] = []

    for report_id, metadata in report_metadata.items():
        if report_id in cited:
            continue
        name = _normalize(str(metadata.get("competition_name") or ""))
        if len(name) < config.CITATION_MENTION_MIN_CHARS:
            continue
        if name in lowered:
            missing.append(report_id)

    return sorted(missing)


def _resolve_alias(
    alias: str,
    alias_map: Mapping[str, str],
    allowed: set,
    incidents: Dict[str, int],
) -> Optional[str]:
    report_id = alias_map.get(alias)
    if report_id is None or report_id not in allowed:
        incidents["unknown_alias"] += 1
        return None
    return f"[{report_id}]"


def process_citations(
    answer: str,
    alias_map: Mapping[str, str],
    retrieved_report_ids: Sequence[str],
) -> CitationResult:
    """Resolve aliases, drop fabricated ids and report what happened."""
    incidents = {"unknown_alias": 0, "fabricated_report_id": 0}
    allowed = set(retrieved_report_ids)
    text = answer or ""

    def replace_list(match: "re.Match[str]") -> str:
        aliases = re.findall(r"LAP[0-9]{1,3}", match.group(0))
        resolved = [
            item
            for item in (
                _resolve_alias(alias, alias_map, allowed, incidents)
                for alias in aliases
            )
            if item
        ]
        return ", ".join(resolved)

    text = config.PATTERN_ALIAS_LIST.sub(replace_list, text)

    def replace_alias(match: "re.Match[str]") -> str:
        resolved = _resolve_alias(match.group("alias"), alias_map, allowed, incidents)
        return resolved or ""

    text = config.PATTERN_ALIAS_SINGLE.sub(replace_alias, text)

    def replace_report_id(match: "re.Match[str]") -> str:
        report_id = match.group(1) or match.group(2)
        if report_id in allowed:
            return f"[{report_id}]"
        incidents["fabricated_report_id"] += 1
        return ""

    text = config.PATTERN_LLM_REPORT_ID.sub(replace_report_id, text)

    # Removing a token can leave empty brackets, doubled spaces or a space in
    # front of punctuation behind.
    text = re.sub(r"[\[\(]\s*[\]\)]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text).strip()
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)

    if config.PATTERN_ALIAS_ANY.search(text):
        raise CitationPostconditionError(
            "An alias token survived citation post-processing."
        )

    cited: List[str] = []
    for report_id in config.CITATION_REPORT_ID_PATTERN.findall(text):
        if report_id in allowed and report_id not in cited:
            cited.append(report_id)

    return CitationResult(text=text, cited_report_ids=cited, incidents=incidents)


def build_source_block(
    cited_report_ids: Sequence[str],
    report_metadata: Mapping[str, Mapping[str, Any]],
) -> str:
    """Build the Sumber block from cited reports that have stored metadata."""
    lines: List[str] = [config.SOURCE_BLOCK_TITLE]

    for report_id in cited_report_ids:
        metadata = report_metadata.get(report_id)
        if not metadata:
            continue
        lines.append(
            config.SOURCE_BLOCK_LINE_TEMPLATE.format(
                report_id=report_id,
                competition_name=metadata.get("competition_name", ""),
                competition_type=metadata.get("competition_type", ""),
                competition_year=metadata.get("competition_year", ""),
                achievement_level=metadata.get("achievement_level", ""),
            )
        )

    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def compose_answer(text: str, source_block: str) -> str:
    """Attach the source block to the answer text."""
    if not source_block:
        return text
    return f"{text}{config.SOURCE_BLOCK_SEPARATOR}{source_block}"
