"""
Backend-built coverage lines for comparison answers (fix pack H3b).

The model never writes these lines. They state, from Chroma metadata only (no
embeddings, no LLM), how many distinct reports the answer actually rests on and
how many reports the active filters contain. A student then sees that "tim juara
menekankan ..." rests on 3 of 9 award reports, instead of assuming it covers all
nine.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import config
from core.retriever import build_where_filter, merge_filter_dicts


def distinct_report_ids(metadatas: Iterable[Dict[str, Any]]) -> list:
    """Unique report ids in first-seen order."""
    seen = set()
    ordered = []
    for metadata in metadatas:
        report_id = (metadata or {}).get("report_id")
        if not report_id or report_id in seen:
            continue
        seen.add(report_id)
        ordered.append(report_id)
    return ordered


def count_reports(
    knowledge_base: Any,
    filters: Dict[str, Sequence[Any]] | None = None,
) -> int:
    """Distinct reports matching the active filters (metadata read only)."""
    metadatas = knowledge_base.list_report_metadatas(where=build_where_filter(filters))
    return len(distinct_report_ids(metadatas))


def comparison_totals(
    knowledge_base: Any,
    filters: Dict[str, Sequence[Any]] | None,
    compare_by: str,
) -> list:
    """Configured groups in order, each with its total distinct-report count."""
    totals = []
    for group in config.COMPARISON_GROUPS.get(compare_by, ()):
        merged = merge_filter_dicts(filters, group.get("filter"))
        # Fix pack L2: the coverage sentence uses the model-facing group name, so
        # the answer, the headings and the sentence never disagree.
        coverage_label = group["label"]
        if any(not values for values in merged.values()):
            totals.append(
                {
                    "label": group["label"],
                    "coverage_label": coverage_label,
                    "total": 0,
                }
            )
            continue
        totals.append(
            {
                "label": group["label"],
                "coverage_label": coverage_label,
                "total": count_reports(knowledge_base, merged),
            }
        )
    return totals


def build_coverage_line(
    totals: Sequence[Dict[str, Any]],
    shown_counts: Dict[str, int],
) -> str:
    """Coverage sentence for a grouped comparison answer."""
    parts = [
        config.COVERAGE_GROUP_TEMPLATE.format(
            shown=int(shown_counts.get(item["label"], 0)),
            total=int(item["total"]),
            label=item.get("coverage_label") or item["label"],
        )
        for item in totals
    ]
    return f"{config.COVERAGE_PREFIX}{config.COVERAGE_JOINER.join(parts)}" \
        f"{config.COVERAGE_SUFFIX}"


def build_cross_report_coverage_line(shown: int, total: int) -> str:
    """Coverage sentence for cross_report answers without a comparison grouping."""
    return config.COVERAGE_CROSS_TEMPLATE.format(shown=int(shown), total=int(total))
