"""
Citation post-processing (Brief 5.6, PRD 8.2).

The model only ever writes aliases ([S1]). The backend:

1. replaces a valid alias with its report_id, e.g. [RPT0003],
2. deletes an unknown alias and counts an incident,
3. keeps a raw RPT id only when it is in the retrieved set, otherwise deletes it
   and counts an incident (the model may never introduce a source),
4. builds the "Sumber" block from cited reports only, using Chroma metadata.

No report metadata is ever invented: an id without stored metadata is skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

import config


@dataclass(frozen=True)
class CitationResult:
    text: str
    cited_report_ids: List[str] = field(default_factory=list)
    incidents: Dict[str, int] = field(default_factory=dict)


def has_citation(text: str) -> bool:
    """True when the text contains at least one alias token."""
    return config.PATTERN_CITATION_TAG.search(text) is not None


def process_citations(
    answer: str,
    alias_map: Mapping[str, str],
    retrieved_report_ids: Sequence[str],
) -> CitationResult:
    """Resolve aliases, drop fabricated ids and report what happened."""
    incidents = {"unknown_alias": 0, "fabricated_report_id": 0}
    allowed = set(retrieved_report_ids)

    def replace_alias(match: "re.Match[str]") -> str:
        alias = match.group(1)
        report_id = alias_map.get(alias)
        if report_id is None or report_id not in allowed:
            incidents["unknown_alias"] += 1
            return ""
        return f"[{report_id}]"

    text = config.PATTERN_CITATION_TAG.sub(replace_alias, answer)

    def replace_report_id(match: "re.Match[str]") -> str:
        report_id = match.group(1) or match.group(2)
        if report_id in allowed:
            return f"[{report_id}]"
        incidents["fabricated_report_id"] += 1
        return ""

    text = config.PATTERN_LLM_REPORT_ID.sub(replace_report_id, text)

    # Removing a token can leave empty brackets, doubled spaces or a space in
    # front of punctuation behind.
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text).strip()

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
