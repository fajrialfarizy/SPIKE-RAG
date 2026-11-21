"""
Exact-string tests for newline separation of source and catalog lines (fix H6).

The supervisor's terminal appeared to swallow newlines near its window width; these
tests settle that the backend emits real newline characters.
"""

import config
from core.catalog import render_catalog_answer
from core.citations import build_source_block, compose_answer

METADATA = {
    "RPT0001": {
        "competition_name": "Lomba Inovasi",
        "competition_type": "Business Plan",
        "competition_year": 2023,
        "achievement_level": "Juara 1",
    },
    "RPT0002": {
        "competition_name": "Lomba Esai",
        "competition_type": "Essay",
        "competition_year": 2024,
        "achievement_level": "Finalis",
    },
}


def test_source_block_lines_are_joined_by_newline_characters():
    block = build_source_block(["RPT0001", "RPT0002"], METADATA)

    assert block == (
        "**Sumber**\n"
        "- RPT0001 — Lomba Inovasi (Business Plan, 2023, Juara 1)\n"
        "- RPT0002 — Lomba Esai (Essay, 2024, Finalis)"
    )
    assert "\r" not in block
    assert block.splitlines()[1].startswith("- RPT0001")


def test_the_answer_and_source_block_are_separated_by_two_newlines():
    composed = compose_answer("Jawaban [RPT0001].", build_source_block(["RPT0001"], METADATA))

    assert composed == (
        "Jawaban [RPT0001].\n\n**Sumber**\n"
        "- RPT0001 — Lomba Inovasi (Business Plan, 2023, Juara 1)"
    )
    assert config.SOURCE_BLOCK_SEPARATOR == "\n\n"


def test_catalog_rows_are_joined_by_newline_characters():
    reports = [
        {
            "report_id": report_id,
            "competition_name": data["competition_name"],
            "competition_type": data["competition_type"],
            "competition_year": data["competition_year"],
            "achievement_level": data["achievement_level"],
        }
        for report_id, data in METADATA.items()
    ]

    text = render_catalog_answer(reports)
    rows = [line for line in text.splitlines() if line.startswith("- RPT")]

    assert rows == [
        "- RPT0001 — Lomba Inovasi (Business Plan, 2023, Juara 1)",
        "- RPT0002 — Lomba Esai (Essay, 2024, Finalis)",
    ]
    assert "\n- RPT0002 —" in text
    assert "Juara 1)- RPT0002" not in text
