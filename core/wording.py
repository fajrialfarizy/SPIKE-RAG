"""
Non-blocking wording metrics (fix pack 10, R3b).

The real run showed two failure shapes that no rule can decide reliably:

- an interpretation the source does not state ("Pengalaman prototipe fisik
  mengajarkan tim tentang pentingnya uji ketahanan", where the source only says
  the prototype leaked in shipping),
- a verb stronger than the source ("mengabaikan kelayakan finansial" for "kurang
  menggali aspek kelayakan finansial").

These functions MEASURE. They never change, trim or refuse an answer:

- ``interpretive_phrases`` counts sentences with a configured lesson/interpretation
  phrase.
- ``strong_verbs`` counts sentences with a configured strong verb.
- ``support_overlap`` reports, per point, the share of its content words that
  appear in the text of the reports it cites, plus the minimum and the points
  below SUPPORT_OVERLAP_MIN.

The overlap heuristic is a rough lexical signal only. It CANNOT catch a
verb-strength error (the words are all in the source) or an invented causal link
(the words are too), and a faithful paraphrase can score low because it uses
different words. It exists to point a human rater at the suspicious points in
Phase 4; nothing here is a guardrail.

Samples stay in the debug object. They are never logged, never written to a file
and never shown to a student.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import config
from core.negation import split_sentences

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _phrase_pattern(phrases: Sequence[str]) -> "re.Pattern[str]":
    return re.compile(
        r"(?i)\b(?:" + "|".join(re.escape(item) for item in phrases) + r")\b"
    )


def _sentence_metric(
    text: str, phrases: Sequence[str]
) -> Dict[str, Any]:
    """Sentences carrying one of the phrases, with up to five samples."""
    if not phrases:
        return {"count": 0, "samples": []}
    pattern = _phrase_pattern(phrases)
    samples: List[str] = []
    count = 0
    for sentence in split_sentences(text):
        if not pattern.search(sentence):
            continue
        count += 1
        if len(samples) < config.WORDING_SAMPLE_LIMIT:
            samples.append(sentence)
    return {"count": count, "samples": samples}


def interpretive_phrases(text: str) -> Dict[str, Any]:
    """Sentences that state a lesson or interpretation (non-blocking)."""
    return _sentence_metric(text, config.INTERPRETIVE_PHRASES)


def strong_verbs(text: str) -> Dict[str, Any]:
    """Sentences whose verb is stronger than a source usually states."""
    return _sentence_metric(text, config.STRONG_VERBS)


def _content_words(text: str) -> set:
    """Lowercased tokens of at least four characters that are not stopwords."""
    words = {
        token.casefold()
        for token in _WORD.findall(text or "")
        if len(token) >= config.SUPPORT_MIN_WORD_CHARS
    }
    return {word for word in words if word not in config.SUPPORT_STOPWORDS}


def support_overlap(
    points: Sequence[Tuple[str, Sequence[str]]],
    report_texts: Mapping[str, str],
) -> Dict[str, Any]:
    """
    Lexical overlap between each point and the reports it cites (non-blocking).

    ``points`` holds ``(text, report_ids)`` in render order; ``report_texts``
    maps a report id to the text of its retrieved chunks. A point with no cited
    report text scores 0.0, and a point with no content words is skipped rather
    than scored.
    """
    per_point: List[Dict[str, Any]] = []
    overlaps: List[float] = []
    low = 0
    for text, report_ids in points:
        content = _content_words(text)
        if not content:
            per_point.append(
                {
                    "text": text,
                    "report_ids": list(report_ids),
                    "overlap": None,
                    "low_support": False,
                }
            )
            continue
        source = _content_words(
            " ".join(report_texts.get(report_id, "") for report_id in report_ids)
        )
        overlap = len(content & source) / len(content)
        is_low = overlap < config.SUPPORT_OVERLAP_MIN
        overlaps.append(overlap)
        if is_low:
            low += 1
        per_point.append(
            {
                "text": text,
                "report_ids": list(report_ids),
                "overlap": round(overlap, 3),
                "low_support": is_low,
            }
        )

    low_points = [item for item in per_point if item["low_support"]]
    return {
        "threshold": config.SUPPORT_OVERLAP_MIN,
        "minimum": round(min(overlaps), 3) if overlaps else None,
        "low_support_points": low,
        "samples": [
            {
                "report_ids": item["report_ids"],
                "overlap": item["overlap"],
                "text": item["text"][: config.SUPPORT_SAMPLE_TEXT_CHARS],
            }
            for item in low_points[: config.SUPPORT_SAMPLE_LIMIT]
        ],
        "per_point": per_point,
    }
