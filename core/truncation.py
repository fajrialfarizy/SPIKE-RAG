"""
Deterministic repair of an answer cut off by the completion cap (fix pack G2).

When Groq reports ``finish_reason == "length"`` the visible answer stops
mid-thought. The backend trims it back to the last complete sentence (or, when
there is no sentence end at all, the last complete line), drops dangling markup
such as ``**`` or an opened alias bracket, and then citation post-processing runs
again on that shorter text. No extra model call is made.
"""

from __future__ import annotations

import re

# A sentence terminator followed by whitespace or the end of the text.
_PATTERN_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
# Trailing pieces that must not survive the trim: bold/italic markers, an opened
# bracket, or an incomplete alias/report-id token.
_PATTERN_DANGLING = re.compile(
    r"(?:\*\*|__|\[|\()\s*(?:(?:LAP|RPT)[0-9]{0,4})?\s*$",
    re.IGNORECASE,
)


def _drop_dangling_markup(text: str) -> str:
    cleaned = text.rstrip()
    while True:
        updated = _PATTERN_DANGLING.sub("", cleaned).rstrip()
        if updated == cleaned:
            return updated
        cleaned = updated


def trim_to_last_complete_unit(text: str) -> str:
    """Return the longest prefix that ends on a complete sentence or line."""
    content = (text or "").strip()
    if not content:
        return content

    endings = list(_PATTERN_SENTENCE_END.finditer(content))
    if endings:
        trimmed = content[: endings[-1].end()]
    else:
        lines = content.splitlines()
        trimmed = "\n".join(lines[:-1]).strip()

    if not trimmed:
        # Nothing complete to keep: strip the dangling tail instead of returning
        # a half sentence as if it were finished.
        return _drop_dangling_markup(content)

    return _drop_dangling_markup(trimmed)
