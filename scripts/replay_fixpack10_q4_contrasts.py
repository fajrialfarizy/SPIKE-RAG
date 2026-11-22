"""
Offline diagnosis of the Q4 contrast rejections (fix pack 10, R2a).

The real run reported three contrasts (dimensions "Kendala Keuangan", "Kendala
Teknis", "Kendala Waktu") and all three were rejected as generic, and Q3's
"Pembelajaran" likewise. The raw payload text was not persisted, so this script
takes the dimensions the run reported and replays them through (a) the rule the
validator used at commit 46656e1 (word containment) and (b) the current
pick-based validator on the same content converted to the new schema.

No model, no network, no API key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core import compare_answer

Q4_DIMENSIONS = ("Kendala Keuangan", "Kendala Teknis", "Kendala Waktu")
Q3_DIMENSIONS = ("Pembelajaran",)


def legacy_generic_words(dimension: str):
    """
    The word-containment rule used before fix pack 10 (commit 46656e1):

        words = [word.strip(",.;:").casefold() for word in dimension.split()]
        return any(word in config.COMPARE_GENERIC_DIMENSIONS for word in words)
    """
    words = [word.strip(",.;:").casefold() for word in dimension.split()]
    return [word for word in words if word in config.COMPARE_GENERIC_DIMENSIONS]


def main() -> int:
    print("payload_text_available: no (not persisted by the real run)")
    print("replay_source: the dimensions the run reported, converted to picks")
    print()

    print("=== legacy rule (word containment) ===")
    for dimension in Q4_DIMENSIONS + Q3_DIMENSIONS:
        matched = legacy_generic_words(dimension)
        verdict = "contrast_invalid" if matched else "accepted"
        print(
            f"- {dimension!r}: matched word(s)={matched or '-'} -> {verdict}"
        )

    print()
    print("=== current rule (whole normalized string) ===")
    for dimension in Q4_DIMENSIONS + Q3_DIMENSIONS:
        generic = compare_answer._is_generic_dimension(dimension)
        print(f"- {dimension!r}: generic={generic}")

    print()
    print("=== replay through the current validator (Q4 shape as picks) ===")
    group_keys = {"g1": "Business Plan", "g2": "Business Case"}
    alias_map = {
        "LAP1": "RPT0001",
        "LAP2": "RPT0002",
        "LAP3": "RPT0003",
        "LAP4": "RPT0004",
        "LAP5": "RPT0005",
        "LAP6": "RPT0014",
    }
    report_group = {
        "RPT0001": "Business Plan",
        "RPT0002": "Business Plan",
        "RPT0003": "Business Case",
        "RPT0004": "Business Case",
        "RPT0005": "Business Case",
        "RPT0014": "Business Case",
    }
    payload = {
        "groups": {
            "g1": [
                {"text": "Kelayakan finansial kurang digali.", "cite": ["LAP1"]},
                {"text": "Validasi pasar terbatas.", "cite": ["LAP1"]},
                {"text": "Waktu persiapan terbatas.", "cite": ["LAP1"]},
            ],
            "g2": [
                {"text": "Data sekunder dominan.", "cite": ["LAP4"]},
                {"text": "Regulasi OJK belum dipetakan.", "cite": ["LAP4"]},
                # This run put a Business Plan alias in the Business Case group.
                {"text": "Anggaran prototipe terbatas.", "cite": ["LAP2"]},
            ],
        },
        "contrasts": [
            {"dimension": "Kendala Keuangan", "pick": {"g1": 1, "g2": 3}},
            {"dimension": "Kendala Teknis", "pick": {"g1": 2, "g2": 1}},
            {"dimension": "Kendala Waktu", "pick": {"g1": 3, "g2": 2}},
        ],
    }

    validation = compare_answer.validate_payload(
        payload,
        group_keys=group_keys,
        alias_map=alias_map,
        report_group=report_group,
        retrieved=list(alias_map.values()),
    )
    print(f"contrasts_rendered={len(validation.contrasts)}")
    for contrast in validation.contrasts:
        print(f"- {contrast.dimension}")
        for side in contrast.sides:
            print(f"    {side.label}: {side.text} {side.report_ids}")
    print(f"incidents={json.dumps(validation.incidents, sort_keys=True)}")
    print(f"points_by_key={json.dumps(validation.points_by_key, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
