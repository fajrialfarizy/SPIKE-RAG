"""
Deterministic catalog answers (fix pack G5).

Catalog questions ask what is available, not for reasoning about document text.
The analyzer call still runs (it decides scope and filters), but the answer is
rendered here from Chroma metadata: counts by type, year and achievement group,
then a bounded list of reports. No answer-LLM call is made and no "Sumber" block
is attached, because the list itself is the source. This is an intentional
override of Brief 5.5 for catalog mode, recorded as a deviation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import config


def achievement_group_label(level: Any) -> Optional[str]:
    """Map a stored achievement level to its configured comparison group."""
    for group in config.COMPARISON_GROUPS.get("achievement_group", ()):
        levels = (group.get("filter") or {}).get("achievement_level") or ()
        if level in levels:
            return str(group.get("label"))
    return None


def _value(value: Any) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text or config.CATALOG_EMPTY_VALUE


def _count_pairs(counter: Dict[str, int]) -> str:
    parts = [f"{value} ({count})" for value, count in counter.items()]
    return config.CATALOG_COUNT_SEPARATOR.join(parts)


def summarize(reports: Sequence[Dict[str, Any]]) -> List[str]:
    """Counts by competition type, year and achievement group."""
    by_type: Dict[str, int] = {}
    by_year: Dict[str, int] = {}
    by_group: Dict[str, int] = {}

    for report in reports:
        competition_type = _value(report.get("competition_type"))
        year = _value(report.get("competition_year"))
        group = achievement_group_label(report.get("achievement_level"))
        group_text = _value(group)

        by_type[competition_type] = by_type.get(competition_type, 0) + 1
        by_year[year] = by_year.get(year, 0) + 1
        by_group[group_text] = by_group.get(group_text, 0) + 1

    ordered_types = dict(sorted(by_type.items()))
    ordered_years = dict(sorted(by_year.items()))
    ordered_groups = dict(sorted(by_group.items()))

    return [
        config.CATALOG_TYPE_LINE_TEMPLATE.format(value=_count_pairs(ordered_types)),
        config.CATALOG_YEAR_LINE_TEMPLATE.format(value=_count_pairs(ordered_years)),
        config.CATALOG_ACHIEVEMENT_LINE_TEMPLATE.format(
            value=_count_pairs(ordered_groups)
        ),
    ]


def render_catalog_answer(
    reports: Sequence[Dict[str, Any]],
    max_rows: int = config.CATALOG_MAX_ROWS,
) -> str:
    """Render a backend-owned catalog answer from report metadata."""
    if not reports:
        raise ValueError("A catalog answer needs at least one report.")

    ordered = sorted(reports, key=lambda report: str(report.get("report_id") or ""))
    shown = ordered[: int(max_rows)]

    lines: List[str] = [
        config.CATALOG_HEADER_TEMPLATE.format(total=len(ordered)),
        *summarize(ordered),
        "",
    ]
    for report in shown:
        lines.append(
            config.CATALOG_ROW_TEMPLATE.format(
                report_id=_value(report.get("report_id")),
                competition_name=_value(report.get("competition_name")),
                competition_type=_value(report.get("competition_type")),
                competition_year=_value(report.get("competition_year")),
                achievement_level=_value(report.get("achievement_level")),
            )
        )

    if len(ordered) > len(shown):
        lines.append("")
        lines.append(
            config.CATALOG_TRUNCATED_TEMPLATE.format(
                shown=len(shown), total=len(ordered)
            )
        )

    return "\n".join(lines).strip()
