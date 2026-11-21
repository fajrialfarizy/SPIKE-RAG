"""
Context builder (Brief 5.5 step 4).

The answer model never sees a report_id. Every unique report gets one alias
(S1, S2, ...) and its chunks are wrapped in a <dokumen> delimiter whose
attributes carry the report metadata. The backend keeps the alias -> report_id
map, which is the only place citations can come from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import config
from core.retriever import RetrievedChunk


class ContextBudgetError(RuntimeError):
    """Raised when even a single chunk cannot fit the context token budget."""

CONTEXT_ATTRIBUTE_FIELDS = (
    "competition_name",
    "competition_type",
    "competition_year",
    "achievement_level",
    "section_type",
)


@dataclass(frozen=True)
class ContextBundle:
    """Rendered context plus the alias bookkeeping the backend keeps private."""

    text: str
    alias_map: Dict[str, str] = field(default_factory=dict)
    report_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    chunk_ids: List[str] = field(default_factory=list)

    @property
    def report_ids(self) -> List[str]:
        return list(dict.fromkeys(self.alias_map.values()))


def _attribute(value: Any) -> str:
    """Keep delimiter attributes well formed even for hostile document text."""
    text = " ".join(str(value if value is not None else "").split())
    return text.replace('"', "'")


def _alias_for(index: int) -> str:
    return f"{config.CONTEXT_ALIAS_PREFIX}{index}"


def _group_label(chunk: RetrievedChunk) -> Any:
    """Comparison group of a chunk, when retrieval was grouped."""
    return getattr(chunk, "group_label", None)


def _section_label(section_type: Any) -> str:
    """Human-readable section name next to the raw section key."""
    key = str(section_type or "")
    return config.SECTION_TYPE_LABELS.get(key, key)


def _neutralize_delimiters(text: str) -> str:
    """Stop document text from closing or reopening the data delimiter."""
    return text.replace("</dokumen>", "(/dokumen)").replace(
        "<dokumen", "(dokumen"
    )


class ContextBuilder:
    """Builds the delimited context for chunks or for the catalog list."""

    def build_chunks(self, chunks: Sequence[RetrievedChunk]) -> ContextBundle:
        alias_map: Dict[str, str] = {}
        alias_by_report: Dict[str, str] = {}
        report_metadata: Dict[str, Dict[str, Any]] = {}
        chunk_ids: List[str] = []
        blocks: List[str] = []
        current_group: Any = None

        for chunk in chunks:
            report_id = chunk.report_id
            if not report_id:
                raise ValueError("Retrieved chunk has no report_id in its metadata.")

            if report_id not in alias_by_report:
                alias = _alias_for(len(alias_map) + 1)
                alias_map[alias] = report_id
                alias_by_report[report_id] = alias
                report_metadata[report_id] = {
                    field_name: chunk.metadata.get(field_name)
                    for field_name in CONTEXT_ATTRIBUTE_FIELDS
                }

            alias = alias_by_report[report_id]
            metadata = chunk.metadata

            group = _group_label(chunk)
            if group != current_group:
                if current_group is not None:
                    blocks.append(config.CONTEXT_GROUP_CLOSE_TAG)
                if group is not None:
                    blocks.append(
                        config.CONTEXT_GROUP_OPEN_TEMPLATE.format(
                            label=_attribute(group)
                        )
                    )
                current_group = group

            opening = config.CONTEXT_DOCUMENT_OPEN_TEMPLATE.format(
                alias=alias,
                competition_name=_attribute(metadata.get("competition_name")),
                competition_type=_attribute(metadata.get("competition_type")),
                competition_year=_attribute(metadata.get("competition_year")),
                achievement_level=_attribute(metadata.get("achievement_level")),
                section_type=_attribute(metadata.get("section_type")),
                section_label=_attribute(_section_label(metadata.get("section_type"))),
                group_attribute=(
                    config.CONTEXT_GROUP_ATTRIBUTE_TEMPLATE.format(
                        label=_attribute(group)
                    )
                    if group is not None
                    else ""
                ),
            )
            blocks.append(
                f"{opening}\n{_neutralize_delimiters(chunk.document)}\n"
                f"{config.CONTEXT_DOCUMENT_CLOSE_TAG}"
            )
            chunk_ids.append(chunk.chunk_id)

        if current_group is not None:
            blocks.append(config.CONTEXT_GROUP_CLOSE_TAG)

        return ContextBundle(
            text=config.CONTEXT_DOCUMENT_SEPARATOR.join(blocks),
            alias_map=alias_map,
            report_metadata=report_metadata,
            chunk_ids=chunk_ids,
        )

    def build_chunks_within_budget(
        self,
        chunks: Sequence[RetrievedChunk],
        token_counter: Any,
        max_tokens: int = config.MAX_CONTEXT_TOKENS,
    ) -> Tuple[ContextBundle, List[str]]:
        """
        Render the context and drop chunks until it fits the token budget.

        The lowest-scoring chunk is dropped first, and a report keeps one chunk
        for as long as any report still has more than one, so a multi-chunk
        report cannot crowd every other report out. When the last remaining chunk
        alone exceeds the budget the build fails loudly instead of truncating
        document text.
        """
        remaining: List[RetrievedChunk] = list(chunks)

        while remaining:
            bundle = self.build_chunks(remaining)
            if token_counter.count_tokens(bundle.text) <= max_tokens:
                kept_ids = {chunk.chunk_id for chunk in remaining}
                dropped = [
                    chunk.chunk_id for chunk in chunks if chunk.chunk_id not in kept_ids
                ]
                return bundle, dropped

            if len(remaining) == 1:
                raise ContextBudgetError(
                    f"A single chunk needs more than the {max_tokens}-token context "
                    "budget; refusing to truncate document text."
                )

            per_report: Dict[str, int] = {}
            per_group: Dict[Any, int] = {}
            for chunk in remaining:
                per_report[chunk.report_id] = per_report.get(chunk.report_id, 0) + 1
                group = _group_label(chunk)
                if group is not None:
                    per_group[group] = per_group.get(group, 0) + 1

            # Grouped comparisons: never remove the last chunk of a group, so the
            # answer always keeps both sides of the comparison.
            pool = remaining
            if per_group:
                pool = [
                    chunk
                    for chunk in remaining
                    if per_group.get(_group_label(chunk), 0) > 1
                ] or remaining

            removable = [
                chunk for chunk in pool if per_report[chunk.report_id] > 1
            ]
            pool = removable or pool
            victim = min(pool, key=lambda chunk: (chunk.score or 0.0, chunk.chunk_id))
            remaining.remove(victim)

        return self.build_chunks([]), [chunk.chunk_id for chunk in chunks]

    def build_catalog(
        self, reports: Sequence[Dict[str, Any]]
    ) -> ContextBundle:
        """Catalog mode: a structured report list, no document text."""
        alias_map: Dict[str, str] = {}
        report_metadata: Dict[str, Dict[str, Any]] = {}
        lines: List[str] = []

        for report in reports:
            report_id = report.get("report_id")
            if not report_id:
                raise ValueError("Catalog report has no report_id.")

            alias = _alias_for(len(alias_map) + 1)
            alias_map[alias] = report_id
            report_metadata[report_id] = {
                field_name: report.get(field_name)
                for field_name in CONTEXT_ATTRIBUTE_FIELDS
                if field_name in report
            }
            lines.append(
                config.CONTEXT_CATALOG_LINE_TEMPLATE.format(
                    alias=alias,
                    competition_name=_attribute(report.get("competition_name")),
                    competition_type=_attribute(report.get("competition_type")),
                    competition_year=_attribute(report.get("competition_year")),
                    achievement_level=_attribute(report.get("achievement_level")),
                )
            )

        text = "\n".join(
            [config.CONTEXT_CATALOG_OPEN_TAG, *lines, config.CONTEXT_CATALOG_CLOSE_TAG]
        )
        return ContextBundle(text=text, alias_map=alias_map, report_metadata=report_metadata)
