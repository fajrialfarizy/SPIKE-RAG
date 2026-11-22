"""
Backend sanitizer for the final answer (fix pack F4d).

The model sometimes emits HTML line breaks inside Markdown tables. The app renders
Markdown only and never uses unsafe_allow_html, so the backend normalizes the text
first: <br> variants become newlines and any other HTML tag is removed. Markdown
(including ``**bold**``, lists and pipe tables) is left untouched.

Fix pack 8 (N5b): invisible and look-alike characters are normalized to ASCII
(non-breaking hyphen, no-break and thin spaces, zero-width characters). The real
model emitted U+2011 inside "end-to-end" and "terburu-buruan" and U+202F inside
"83 %"; the wording is otherwise left exactly as written.
"""

from __future__ import annotations

import re

import config

_CHARACTER_MAP = str.maketrans(config.SANITIZER_CHARACTER_MAP)

_PATTERN_BREAK = re.compile(r"(?i)<\s*br\s*/?\s*>")
_PATTERN_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_PATTERN_TRAILING_SPACE = re.compile(r"[ \t]+\n")


def sanitize_answer(text: str) -> str:
    """Normalize line breaks and strip HTML tags from an answer."""
    content = (text or "").translate(_CHARACTER_MAP)
    content = _PATTERN_BREAK.sub("\n", content)
    content = _PATTERN_TAG.sub("", content)
    content = _PATTERN_TRAILING_SPACE.sub("\n", content)
    content = re.sub(r"\n{3,}", config.SANITIZER_MAX_BLANK_LINES_PATTERN, content)
    return content.strip()


def contains_html(text: str) -> bool:
    """True when any HTML tag remains (used by tests and post-checks)."""
    return _PATTERN_TAG.search(text or "") is not None
