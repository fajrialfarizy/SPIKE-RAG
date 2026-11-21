"""
Non-blocking negation metric (fix pack K2).

Counts answer sentences or bullets that state a negation about a team or report
("Tim belum melakukan simulasi", "tidak ada anggota yang pasif"). It is a
measurement only: no answer is changed, trimmed or refused, because a pattern
cannot tell an invented negative from a legitimate one.

The sentence samples are returned for the debug object only. They are never
logged, never written to a file and never shown to a student.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

import config

_NEGATION = re.compile(
    r"(?i)\b(?:" + "|".join(re.escape(item) for item in config.NEGATION_MARKERS) + r")\b"
)
_SUBJECT = re.compile(
    r"(?i)\b(?:"
    + "|".join(re.escape(item) for item in config.NEGATION_SUBJECT_MARKERS)
    + r")\b"
)
_SENTENCE_SPLIT = re.compile(config.NEGATION_SENTENCE_SPLIT_PATTERN)


def split_sentences(text: str) -> List[str]:
    """Sentences of every line, with bullet markers stripped from the text."""
    sentences: List[str] = []
    for line in (text or "").splitlines():
        body = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line).strip()
        if not body:
            continue
        for part in _SENTENCE_SPLIT.split(body):
            cleaned = part.strip()
            if cleaned:
                sentences.append(cleaned)
    return sentences


def negation_sentences(text: str) -> Dict[str, object]:
    """
    Count negated sentences about a team or report and keep up to five samples.

    Returns ``{"count": n, "samples": [...]}``; samples are debug-only.
    """
    samples: List[str] = []
    count = 0
    for sentence in split_sentences(text):
        if _NEGATION.search(sentence) and _SUBJECT.search(sentence):
            count += 1
            if len(samples) < config.NEGATION_SAMPLE_LIMIT:
                samples.append(sentence)
    return {"count": count, "samples": samples}


def point_negation_counts(texts: Iterable[str]) -> List[int]:
    """One count per comparison point, in order."""
    return [
        int(negation_sentences(text)["count"])
        for text in texts
    ]
