"""
Backend bullet trimming for free-text answers (fix pack 7, M4b).

The addendum asks for at most six points and at most two points per report, but a
model can ignore both. Instead of spending a second model call (and more quota) on
a repair, the backend trims the finished answer:

1. keep up to ``ANSWER_MAX_BULLETS_PER_REPORT`` bullets per cited report, in the
   original order, so every report mentioned still has a representative point,
2. then fill the remaining slots up to ``ANSWER_MAX_BULLETS`` in original order,
3. drop the rest, keeping every non-bullet line (intro, closing paragraph).

This is a deliberate post-hoc deviation approved by the supervisor. The caller
recomputes citations, the Sumber block and the coverage line from the trimmed
text, so the answer never claims a source it no longer shows.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

import config

_BULLET_LINE = re.compile(r"^\s{0,3}(?:[-*]|\d{1,2}[.)])\s+\S")
_REPORT_ID = re.compile(r"RPT[0-9]{4,}")


def is_bullet(line: str) -> bool:
    return bool(_BULLET_LINE.match(line or ""))


def trim_bullets(
    text: str,
    max_bullets: Optional[int] = None,
    max_per_report: Optional[int] = None,
) -> Tuple[str, int]:
    """
    Return the trimmed text and how many bullets were dropped.

    A text with no more than ``max_bullets`` bullets is returned unchanged, so the
    common case pays nothing. Nothing is trimmed when no report id can be read
    from the bullets: the balanced pass has nothing to balance with.
    """
    limit = config.ANSWER_MAX_BULLETS if max_bullets is None else int(max_bullets)
    per_report = (
        config.ANSWER_MAX_BULLETS_PER_REPORT
        if max_per_report is None
        else int(max_per_report)
    )

    lines = (text or "").split("\n")
    bullet_indices = [index for index, line in enumerate(lines) if is_bullet(line)]
    if len(bullet_indices) <= limit:
        return text, 0

    counts: dict = {}
    keep: List[int] = []

    # Pass 1: a balanced sample, first bullets first, capped per cited report.
    for index in bullet_indices:
        if len(keep) >= limit:
            break
        reports = _REPORT_ID.findall(lines[index])
        if not reports:
            continue
        if any(counts.get(report, 0) >= per_report for report in reports):
            continue
        keep.append(index)
        for report in reports:
            counts[report] = counts.get(report, 0) + 1

    # Pass 2: fill the remaining slots in the original order.
    kept: Set[int] = set(keep)
    for index in bullet_indices:
        if len(kept) >= limit:
            break
        kept.add(index)

    dropped = len(bullet_indices) - len(kept)
    if dropped <= 0:
        return text, 0

    trimmed = "\n".join(
        line for index, line in enumerate(lines) if not is_bullet(line) or index in kept
    )
    return trimmed, dropped
