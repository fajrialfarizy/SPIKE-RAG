"""
Non-blocking causal-connector metric (fix pack 8, N2b).

The real model joined two separately sourced facts with "sehingga" and turned a
source's "kurang menggali aspek kelayakan finansial" into "mengabaikan kelayakan
finansial". Only the prompt can prevent that; a pattern cannot tell an invented
causal link from a legitimate one. So this module measures and changes nothing:
it counts the answer sentences or bullets that contain a configured connector and
keeps up to five samples in the debug object only.

Samples are never logged, never written to a file and never shown to a student.
"""

from __future__ import annotations

import re
from typing import Dict, List

import config
from core.negation import split_sentences

_CONNECTOR = re.compile(
    r"(?i)\b(?:"
    + "|".join(re.escape(item) for item in config.CAUSAL_CONNECTORS)
    + r")\b"
)


def causal_connectors(text: str) -> Dict[str, object]:
    """
    Count sentences or bullets with a causal connector and keep five samples.

    ``split_sentences`` already skips headings, the coverage sentence and the
    Sumber rows, so those backend-written lines are never counted.
    """
    samples: List[str] = []
    count = 0
    for sentence in split_sentences(text):
        if not _CONNECTOR.search(sentence):
            continue
        count += 1
        if len(samples) < config.CAUSAL_SAMPLE_LIMIT:
            samples.append(sentence)
    return {"count": count, "samples": samples}
