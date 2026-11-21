"""Offline tests for fix pack G4 (citation placement) and G6 (analyzer scope)."""

import config
from core import prompts
from core.citations import citation_style_metrics, process_citations

ALIASES = {"LAP1": "RPT0001", "LAP2": "RPT0002"}
RETRIEVED = ["RPT0001", "RPT0002"]


def test_the_addendum_is_version_1_3_and_states_the_placement_rules():
    addendum = prompts.CITATION_FORMAT_ADDENDUM

    assert prompts.ADDENDUM_VERSION == "1.3"
    assert "(v1.3)" in addendum
    assert "akhir kalimat atau akhir poin" in addendum
    assert "Jangan menaruh sitasi di tengah kalimat" in addendum
    assert "Paling banyak dua sitasi dalam satu kalimat" in addendum
    assert "tidak memuat sesuatu" in addendum
    # The verbatim PRD text is still the prefix of the system prompt.
    assert prompts.SYSTEM_PROMPT.startswith(prompts.PRD_9_1_SYSTEM_PROMPT)


def test_a_citation_glued_to_a_word_gets_a_separating_space():
    result = process_citations("Kendala utamanya di[LAP1].", ALIASES, RETRIEVED)

    assert result.text == "Kendala utamanya di [RPT0001]."


def test_mid_sentence_mentions_are_moved_and_the_parentheses_removed():
    result = process_citations(
        "Kendala utama adalah logistik [LAP1] dan koordinasi (LAP2).",
        ALIASES,
        RETRIEVED,
    )

    assert result.text == (
        "Kendala utama adalah logistik dan koordinasi [RPT0001], [RPT0002]."
    )
    assert "(" not in result.text
    assert result.cited_report_ids == ["RPT0001", "RPT0002"]


def test_spacing_normalization_does_not_move_a_sentence_final_citation():
    result = process_citations(
        "Validasi pasar menjadi fokus [LAP1].", ALIASES, RETRIEVED
    )

    assert result.text == "Validasi pasar menjadi fokus [RPT0001]."


def test_style_metrics_count_mid_sentence_citations_and_parentheses():
    text = (
        "Tim juara fokus pada validasi [RPT0007] sebelum final.\n"
        "- Belajar dari mentor (lihat catatan) [RPT0006].\n"
    )

    metrics = citation_style_metrics(text)

    assert metrics["mid_sentence_citations"] == 1
    assert metrics["unbalanced_parentheses"] == 0


def test_style_metrics_report_unbalanced_parentheses():
    metrics = citation_style_metrics("Catatan (belum selesai) dan (lainnya.")

    assert metrics["unbalanced_parentheses"] == 1


def test_the_pipeline_reports_style_metrics_in_the_debug_object():
    # Covered end to end in tests/test_pipeline.py; this pins the empty case.
    assert citation_style_metrics("") == {
        "mid_sentence_citations": 0,
        "unbalanced_parentheses": 0,
    }


def test_the_analyzer_treats_an_unknown_domain_as_in_scope():
    prompt = prompts.ANALYZER_SYSTEM_PROMPT

    normalized = " ".join(prompt.split())
    assert "belum ada di basis pengetahuan" in normalized
    assert "robotika bawah air" in normalized
    assert "bantuan menulis kode" in normalized
    # The unknown-domain example must be marked in scope.
    example = prompt.split('Pertanyaan: "Apa pengalaman tim pada kompetisi robotika')[1]
    assert '"in_scope": true' in example.splitlines()[1]


def test_the_answer_target_is_one_hundred_eighty_words_and_six_bullets():
    assert config.ANSWER_TARGET_WORDS == 180
    assert "180 kata" in prompts.CITATION_FORMAT_ADDENDUM
    assert config.ANSWER_MAX_BULLETS == 6
    assert "Paling banyak 6 poin" in prompts.CITATION_FORMAT_ADDENDUM
    assert "Parafrase" in prompts.CITATION_FORMAT_ADDENDUM
