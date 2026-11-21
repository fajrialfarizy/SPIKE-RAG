"""
Offline tests for citation post-processing (Brief 5.6, fix pack F4).
"""

import pytest

from core.citations import (
    CitationPostconditionError,
    build_source_block,
    compose_answer,
    find_uncited_mentions,
    has_citation,
    process_citations,
)

ALIASES = {"LAP1": "RPT0002", "LAP2": "RPT0003"}
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
    result = process_citations("Kendala logistik muncul [LAP1].", ALIASES, RETRIEVED)

    assert result.text == "Kendala logistik muncul [RPT0002]."
    assert result.cited_report_ids == ["RPT0002"]
    assert result.incidents == {"unknown_alias": 0, "fabricated_report_id": 0}


def test_several_aliases_are_all_resolved_in_order():
    result = process_citations(
        "A [LAP1] lalu B [LAP2] dan lagi [LAP1].", ALIASES, RETRIEVED
    )

    # Fix pack J3a: both mentions leave the middle of the sentence and are
    # merged, deduplicated, at the end.
    assert result.text == "A lalu B dan lagi [RPT0002] [RPT0003]."
    assert result.cited_report_ids == ["RPT0002", "RPT0003"]


@pytest.mark.parametrize(
    "answer",
    [
        "Klaim A [LAP1] dan klaim B [LAP2].",
        "Klaim A [LAP1, LAP2].",
        "Klaim A [LAP1][LAP2].",
        "Klaim A (LAP2).",
        "Klaim A LAP2.",
    ],
)
def test_every_citation_form_is_rewritten_to_report_ids(answer):
    result = process_citations(answer, ALIASES, RETRIEVED)

    assert "LAP" not in result.text
    assert "[RPT0002]" in result.text or "[RPT0003]" in result.text


def test_a_parenthesised_alias_becomes_a_bracketed_report_id():
    result = process_citations("Kendala waktu (LAP2).", ALIASES, RETRIEVED)

    assert result.text == "Kendala waktu [RPT0003]."


def test_a_comma_list_keeps_brackets_around_each_id():
    result = process_citations("Dua laporan [LAP1, LAP2] sepakat.", ALIASES, RETRIEVED)

    # Fix pack J3a: a citation followed by prose moves to the sentence end.
    assert result.text == "Dua laporan sepakat [RPT0002] [RPT0003]."
    assert result.cited_report_ids == ["RPT0002", "RPT0003"]


def test_unknown_alias_is_removed_and_counted():
    result = process_citations("Klaim tanpa dasar [LAP7].", ALIASES, RETRIEVED)

    assert "LAP7" not in result.text
    assert result.text == "Klaim tanpa dasar."
    assert result.incidents["unknown_alias"] == 1
    assert result.cited_report_ids == []


def test_alias_pointing_outside_the_retrieved_set_is_removed():
    result = process_citations("Klaim [LAP1].", {"LAP1": "RPT0099"}, ["RPT0002"])

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


def test_the_postcondition_rejects_a_surviving_alias():
    # The substitution resolves the documented uppercase form only; any other
    # alias-like token must fail loudly instead of reaching the student.
    with pytest.raises(CitationPostconditionError):
        process_citations("Klaim Lap1.", ALIASES, RETRIEVED)


def test_answer_without_any_citation_is_detected():
    assert has_citation("Jawaban tanpa sumber.") is False
    assert has_citation("Jawaban dengan sumber [LAP1].") is True

    result = process_citations("Jawaban tanpa sumber.", ALIASES, RETRIEVED)
    assert result.cited_report_ids == []
    assert has_citation(result.text) is False


def test_an_uncited_competition_name_is_reported():
    text = "Laporan Inovasi Bisnis Mahasiswa Nasional (IBMN) 2023 menunjukkan hal itu."

    missing = find_uncited_mentions(text, [], METADATA)

    assert missing == ["RPT0002"]


def test_a_mentioned_and_cited_report_is_not_reported():
    text = "Inovasi Bisnis Mahasiswa Nasional (IBMN) 2023 menunjukkan hal itu [RPT0002]."

    assert find_uncited_mentions(text, ["RPT0002"], METADATA) == []


def test_a_report_that_is_never_mentioned_is_not_reported():
    assert find_uncited_mentions("Jawaban umum tanpa nama.", [], METADATA) == []


def test_source_block_lists_only_cited_reports_with_newline_joins():
    block = build_source_block(["RPT0002"], METADATA)

    assert block == (
        "**Sumber**\n"
        "- RPT0002 — Inovasi Bisnis Mahasiswa Nasional (IBMN) 2023 "
        "(Business Plan, 2023, Finalis)"
    )
    assert "\n\n" not in block


def test_source_block_joins_several_items_with_a_single_newline():
    block = build_source_block(["RPT0002", "RPT0003"], METADATA)

    lines = block.split("\n")
    assert lines[0] == "**Sumber**"
    assert len(lines) == 3
    assert lines[1].startswith("- RPT0002 — ")
    assert lines[2].startswith("- RPT0003 — ")


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
