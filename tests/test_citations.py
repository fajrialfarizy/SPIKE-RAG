"""
Offline tests for citation post-processing (Brief 5.6).
"""

from core.citations import (
    build_source_block,
    compose_answer,
    has_citation,
    process_citations,
)

ALIASES = {"S1": "RPT0002", "S2": "RPT0003"}
RETRIEVED = ["RPT0002", "RPT0003"]
METADATA = {
    "RPT0002": {
        "competition_name": "Inovasi Bisnis Mahasiswa Nasional (IBMN) 2023",
        "competition_type": "Business Plan",
        "competition_year": 2023,
        "achievement_level": "Finalis",
    },
    "RPT0003": {
        "competition_name": "Asean Consulting Business Case Challenge 2024",
        "competition_type": "Business Case",
        "competition_year": 2024,
        "achievement_level": "Juara 2",
    },
}


def test_valid_alias_becomes_report_id():
    result = process_citations("Kendala logistik muncul [S1].", ALIASES, RETRIEVED)

    assert result.text == "Kendala logistik muncul [RPT0002]."
    assert result.cited_report_ids == ["RPT0002"]
    assert result.incidents == {"unknown_alias": 0, "fabricated_report_id": 0}


def test_several_aliases_are_all_resolved_in_order():
    result = process_citations("A [S1] lalu B [S2] dan lagi [S1].", ALIASES, RETRIEVED)

    assert result.text == "A [RPT0002] lalu B [RPT0003] dan lagi [RPT0002]."
    assert result.cited_report_ids == ["RPT0002", "RPT0003"]


def test_unknown_alias_is_removed_and_counted():
    result = process_citations("Klaim tanpa dasar [S7].", ALIASES, RETRIEVED)

    assert "[S7]" not in result.text
    assert result.text == "Klaim tanpa dasar."
    assert result.incidents["unknown_alias"] == 1
    assert result.cited_report_ids == []


def test_alias_pointing_outside_the_retrieved_set_is_removed():
    result = process_citations("Klaim [S1].", {"S1": "RPT0099"}, ["RPT0002"])

    assert "RPT0099" not in result.text
    assert result.incidents["unknown_alias"] == 1


def test_report_id_in_the_retrieved_set_is_kept():
    result = process_citations("Kendala [RPT0003] muncul.", ALIASES, RETRIEVED)

    assert "[RPT0003]" in result.text
    assert result.cited_report_ids == ["RPT0003"]
    assert result.incidents["fabricated_report_id"] == 0


def test_fabricated_report_id_is_removed_and_counted():
    result = process_citations(
        "Menurut [RPT0007] dan [RPT0002] hal ini terjadi.", ALIASES, RETRIEVED
    )

    assert "RPT0007" not in result.text
    assert "[RPT0002]" in result.text
    assert result.incidents["fabricated_report_id"] == 1
    assert result.cited_report_ids == ["RPT0002"]


def test_bracketed_and_bare_report_ids_are_both_handled():
    result = process_citations("Lihat RPT0003.", ALIASES, RETRIEVED)

    assert result.text == "Lihat [RPT0003]."


def test_answer_without_any_citation_is_detected():
    assert has_citation("Jawaban tanpa sumber.") is False
    assert has_citation("Jawaban dengan sumber [S1].") is True

    result = process_citations("Jawaban tanpa sumber.", ALIASES, RETRIEVED)
    assert result.cited_report_ids == []
    assert has_citation(result.text) is False


def test_source_block_lists_only_cited_reports():
    block = build_source_block(["RPT0002"], METADATA)

    assert block == (
        "**Sumber**\n"
        "- RPT0002 — Inovasi Bisnis Mahasiswa Nasional (IBMN) 2023 "
        "(Business Plan, 2023, Finalis)"
    )


def test_source_block_never_invents_missing_metadata():
    block = build_source_block(["RPT0002", "RPT0404"], METADATA)

    assert "RPT0002" in block
    assert "RPT0404" not in block


def test_source_block_is_empty_when_nothing_is_cited():
    assert build_source_block([], METADATA) == ""


def test_compose_answer_attaches_the_block():
    text = compose_answer("Jawaban [RPT0002].", "**Sumber**\n- RPT0002 — X (Y, 2023, Finalis)")

    assert text.startswith("Jawaban [RPT0002].")
    assert text.endswith("(Y, 2023, Finalis)")
    assert compose_answer("Jawaban", "") == "Jawaban"
