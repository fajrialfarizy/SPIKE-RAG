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


# -- citation relocation (fix pack J3a) -----------------------------------
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_BULLET_PREFIX = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)")
_TRAILING_PUNCTUATION = ".,;:!?"
_TRAILING_ALIAS_GROUP = re.compile(
    r"[\[\(]\s*LAP[0-9]{1,3}(?:\s*[,;/]\s*LAP[0-9]{1,3})*\s*[\]\)]$"
)
_NOISE = " \t\r\n.,;:!?*_-\u2013\u2014()[]{}\"'"


def _has_prose(text: str) -> bool:
    """True when the fragment still carries words (not just punctuation)."""
    return bool(text.strip(_NOISE))


def _relocate_segment(segment: str) -> str:
    """
    Move mid-sentence alias citations to the end of the sentence.

    A sentence whose citations are followed by prose (``[LAP1] juga ...``) or
    that names aliases in parentheses (``Tim juara (LAP2, LAP3) menekankan``)
    gets every such mention removed and the deduplicated union appended just
    before the closing punctuation. Aliases are never left in the prose.
    """
    matches = list(config.PATTERN_ALIAS_SINGLE.finditer(segment))
    if not matches:
        return segment

    last = matches[-1]
    if _has_prose(segment[last.end():]):
        moved = list(matches)
    else:
        moved = [
            match
            for match in matches[:-1]
            if _has_prose(segment[match.end():last.start()])
        ]
    if not moved:
        return segment

    aliases: List[str] = []
    for match in moved:
        alias = match.group("alias")
        if alias not in aliases:
            aliases.append(alias)

    ordered = sorted(moved, key=lambda item: item.start())
    consecutive = all(
        not segment[left.end():right.start()].strip(" \t,;/()[]{}")
        for left, right in zip(ordered, ordered[1:])
    )

    if consecutive:
        # "(LAP2, LAP3)" or "[LAP1] [LAP2]": drop the whole span, including the
        # parentheses the model wrapped around the aliases.
        result = segment[: ordered[0].start()] + segment[ordered[-1].end():]
    else:
        result = segment
        for match in reversed(ordered):
            result = result[: match.start()] + result[match.end():]

    # Removing a mention can leave "()" or "(, )" behind.
    result = re.sub(r"[\[\(]\s*[,;]?\s*[\]\)]", "", result)
    result = re.sub(r"[ \t]{2,}", " ", result)

    trailing = ""
    stripped = result.rstrip()
    while stripped and stripped[-1] in _TRAILING_PUNCTUATION:
        trailing = stripped[-1] + trailing
        stripped = stripped[:-1].rstrip()

    # A citation group that already ends the sentence joins the moved aliases, in
    # reading order, instead of leaving two adjacent groups behind.
    existing = _TRAILING_ALIAS_GROUP.search(stripped)
    if existing:
        for alias in re.findall(r"LAP[0-9]{1,3}", existing.group(0)):
            if alias not in aliases:
                aliases.append(alias)
        stripped = stripped[: existing.start()].rstrip()

    citation = f"[{', '.join(aliases)}]"
    if not stripped:
        return citation + trailing
    return f"{stripped} {citation}{trailing}"


def relocate_citations(text: str) -> str:
    """Per line and sentence, keep every alias citation at the end of its unit."""
    lines = (text or "").split("\n")
    output: List[str] = []
    for line in lines:
        prefix = ""
        body = line
        bullet = _BULLET_PREFIX.match(line)
        if bullet:
            prefix = bullet.group(1)
            body = line[bullet.end():]
        segments = _SENTENCE_SPLIT.split(body)
        relocated = " ".join(_relocate_segment(segment) for segment in segments)
        output.append(f"{prefix}{relocated}" if prefix else relocated)
    return "\n".join(output)


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
    # Fix pack J3a: relocate before resolving, so an alias never survives inside
    # a sentence or inside parentheses.
    text = relocate_citations(answer or "")

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

    # Placement normalization (fix pack G4a): a citation glued to the previous
    # word ("di[RPT0001]") gets the separating space the style rules expect.
    text = config.PATTERN_CITATION_TIGHT_LEAD.sub(" ", text)

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


def citation_style_metrics(text: str) -> Dict[str, int]:
    """
    Non-blocking style measurements for the debug object (fix pack G4b).

    ``mid_sentence_citations`` counts citation groups that are followed by more
    prose on the same line; ``unbalanced_parentheses`` counts the absolute
    difference between opening and closing parentheses in the answer.
    """
    content = text or ""
    mid_sentence = 0

    for match in config.PATTERN_CITATION_GROUP.finditer(content):
        line_end = content.find("\n", match.end())
        rest = content[match.end() :] if line_end == -1 else content[match.end() : line_end]
        rest = config.PATTERN_CITATION_GROUP.sub("", rest)
        if rest.strip(" \t.,;:!?)-—*"):
            mid_sentence += 1

    return {
        "mid_sentence_citations": mid_sentence,
        "unbalanced_parentheses": abs(content.count("(") - content.count(")")),
    }


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
