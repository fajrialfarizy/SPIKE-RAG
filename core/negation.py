"""
Non-blocking negation metric (fix pack K2, refined in L4).

Counts answer sentences or bullets that state a negation about a team or report
("Tim belum melakukan simulasi", "tanpa cukup validasi pasar"). It is a
measurement only: no answer is changed, trimmed or refused, because a pattern
cannot tell an invented negative from a legitimate one.

Group labels are not claims. Before matching, configurable label phrases ("belum
juara", "belum juara (finalis)", "meraih penghargaan") are removed, and a line
that consists only of a heading/label, of the backend coverage sentence, or of a
Sumber block line is skipped entirely.

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
# Longest first, so "belum juara (finalis)" is removed before "belum juara".
_LABEL = re.compile(
    r"(?i)(?:"
    + "|".join(
        re.escape(item)
        for item in sorted(config.NEGATION_LABEL_PHRASES, key=len, reverse=True)
    )
    + r")"
)
_LABEL_LEFTOVER = " *_-–—()[]:,."
_BULLET_PREFIX = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
# A bullet may open with a prepositional negation about the cited team and name
# no subject at all ("tanpa cukup validasi pasar"). Other negations still need a
# subject, so a generic "Tidak tersedia." is not counted.
_NEGATION_OPENER = re.compile(r"(?i)^(?:tanpa|tanpa\s+adanya)\b")


def _strip_label(text: str) -> str:
    """Text without group labels, normalized whitespace."""
    return " ".join(_LABEL.sub(" ", text).split())


def _is_label_only(line: str) -> bool:
    """True when the line is just a heading or a group label."""
    stripped = line.strip()
    if not stripped:
        return True
    return not _strip_label(stripped).strip(_LABEL_LEFTOVER)


def _is_backend_line(line: str) -> bool:
    """True for the coverage sentence and for Sumber block lines."""
    stripped = line.strip()
    return stripped.startswith(config.NEGATION_SKIP_LINE_PREFIXES) or bool(
        config.PATTERN_NEGATION_SOURCE_ROW.match(stripped)
    )


def split_sentences(text: str) -> List[str]:
    """Sentences of every counted line, with bullet markers stripped."""
    sentences: List[str] = []
    for line in (text or "").splitlines():
        if _is_backend_line(line) or _is_label_only(line):
            continue
        body = _BULLET_PREFIX.sub("", line).strip()
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

    A sentence counts when, after label removal, it contains a negation marker
    and either names a subject (team, report, ...) or opens with "tanpa", as a
    bullet about the cited team does ("tanpa cukup validasi pasar").

    Returns ``{"count": n, "samples": [...]}``; samples are debug-only.
    """
    samples: List[str] = []
    count = 0
    for sentence in split_sentences(text):
        body = _strip_label(sentence)
        if not body:
            continue
        # The subject is looked up in the original sentence, because a label may
        # carry the noun ("Tim belum juara ..."); only the negation is matched on
        # the label-free body, so the label word itself never counts.
        if _NEGATION.search(body) and (
            _SUBJECT.search(sentence) or _NEGATION_OPENER.match(body)
        ):
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
