"""
Offline tests for fix pack 8.

N1: paired contrasts replace the free differences list.
N2: source-fidelity addenda and the causal-connector metric.
N3: the per-report bullet cap always applies.
N4: post-processing stages are measured.
N5: the counting tokenizer is quiet and the sanitizer normalizes look-alike
    characters.

FakeLLM and the other doubles come from tests/stubs.py: no network, no API key,
no model weights (except the explicitly slow tokenizer test).
"""

import json
import logging

import pytest

import config
from core import compare_answer, prompts
from core.answer_trimmer import trim_bullets
from core.causal import causal_connectors
from core.knowledge_base import ChunkMatch
from core.pipeline import AnswerStatus, RagPipeline
from core.sanitizer import sanitize_answer
from tests.stubs import (
    FakeLLM,
    FilteringKnowledgeBase,
    RecordingEmbedder,
    WordCountTokenizer,
)

pytestmark = pytest.mark.usefixtures("legacy_answer_contract", "compare_contrast_pick")

AWARD_LABEL = "meraih penghargaan"
FINALIST_LABEL = "belum juara (finalis)"
AWARD_HEADING = (
    "Tim yang meraih penghargaan (Juara 1-3, Juara Harapan, Most Aspiring)"
)
FINALIST_HEADING = "Tim belum juara (finalis)"


# -- N1: paired contrasts --------------------------------------------------
# Fix pack 10 (R2b): a contrast picks the 1-based number of one point per group
# as the model wrote those points; it carries no text of its own.
def _points(text):
    return [{"text": text, "cite": ["LAP1"]}]


def _contrast(dimension):
    return {"dimension": dimension, "pick": {"g1": 1, "g2": 1}}


def _validation_payload(contrasts, *, award_text="Wawancara pengguna.", finalist_text="Data teknis."):
    return {
        "groups": {
            "g1": [
                {"text": award_text, "cite": ["LAP1"]},
            ],
            "g2": [
                {"text": finalist_text, "cite": ["LAP2"]},
            ],
        },
        "contrasts": contrasts,
    }


ALIAS_MAP = {"LAP1": "RPT0001", "LAP2": "RPT0002"}
REPORT_GROUP = {"RPT0001": AWARD_LABEL, "RPT0002": FINALIST_LABEL}
LABELS = [AWARD_LABEL, FINALIST_LABEL]
GROUP_KEYS = {"g1": AWARD_LABEL, "g2": FINALIST_LABEL}


def _validate(payload):
    return compare_answer.validate_payload(
        payload,
        group_keys=GROUP_KEYS,
        alias_map=ALIAS_MAP,
        report_group=REPORT_GROUP,
        retrieved=["RPT0001", "RPT0002"],
    )


def test_the_schema_has_paired_contrasts_instead_of_differences():
    schema = compare_answer.build_schema(list(GROUP_KEYS), "pick")

    assert set(schema["required"]) == {"groups", "contrasts"}
    assert "differences" not in schema["properties"]
    contrast = schema["properties"]["contrasts"]["items"]
    assert set(contrast["required"]) == {"dimension", "pick"}
    assert contrast["properties"]["pick"]["required"] == ["g1", "g2"]


def test_a_valid_contrast_keeps_one_side_per_group():
    validation = _validate(_validation_payload([_contrast("kedalaman validasi")]))

    assert len(validation.contrasts) == 1
    contrast = validation.contrasts[0]
    assert contrast.dimension == "kedalaman validasi"
    assert [side.label for side in contrast.sides] == LABELS
    assert validation.incidents == {}


def test_a_point_citing_the_other_groups_alias_is_cleaned():
    payload = _validation_payload([_contrast("kedalaman validasi")])
    payload["groups"]["g1"][0]["cite"] = ["LAP1", "LAP2"]

    validation = _validate(payload)

    assert validation.contrasts[0].sides[0].report_ids == ["RPT0001"]
    assert validation.incidents["cross_group_citation"] == 1


def test_a_point_left_without_its_own_group_takes_the_contrast_with_it():
    payload = _validation_payload([_contrast("kedalaman validasi")])
    payload["groups"]["g1"][0]["cite"] = ["LAP2"]

    validation = _validate(payload)

    assert validation.contrasts == []
    assert validation.incidents["cross_group_citation"] == 1
    assert validation.incidents["contrast_invalid_pick_dropped"] == 1


def test_a_contrast_missing_one_group_pick_is_dropped():
    contrast = _contrast("kedalaman validasi")
    contrast["pick"] = {"g1": 1}

    validation = _validate(_validation_payload([contrast]))

    assert validation.contrasts == []
    assert validation.incidents["contrast_invalid_pick_missing"] == 1


def test_a_pick_outside_the_group_points_is_dropped():
    contrast = _contrast("kedalaman validasi")
    contrast["pick"]["g2"] = 2

    validation = _validate(_validation_payload([contrast]))

    assert validation.contrasts == []
    assert validation.incidents["contrast_invalid_pick_range"] == 1


def test_an_unknown_pick_key_is_dropped():
    contrast = _contrast("kedalaman validasi")
    contrast["pick"]["g9"] = 1

    validation = _validate(_validation_payload([contrast]))

    assert validation.contrasts == []
    assert validation.incidents["contrast_invalid_pick_unknown"] == 1


def test_seven_contrasts_are_truncated_to_two_not_failed():
    contrasts = [_contrast(f"dimensi {index}") for index in range(7)]

    validation = _validate(_validation_payload(contrasts))

    assert len(validation.contrasts) == 2
    assert [item.dimension for item in validation.contrasts] == [
        "dimensi 0",
        "dimensi 1",
    ]
    assert validation.incidents["contrast_truncated"] == 5


def test_the_exact_rendered_form_has_bold_headings_blank_lines_and_nesting():
    validation = compare_answer.CompareValidation(
        sections=[
            compare_answer.CompareSection(
                label=AWARD_LABEL,
                points=[
                    compare_answer.ComparePoint(
                        text="Validasi dilakukan lewat wawancara.",
                        report_ids=["RPT0007"],
                    )
                ],
            ),
            compare_answer.CompareSection(
                label=FINALIST_LABEL,
                points=[
                    compare_answer.ComparePoint(
                        text="Riset pasar terbatas pada data sekunder.",
                        report_ids=["RPT0008"],
                    )
                ],
            ),
        ],
        contrasts=[
            compare_answer.CompareContrast(
                dimension="kedalaman validasi",
                sides=[
                    compare_answer.CompareContrastSide(
                        label=AWARD_LABEL,
                        text="Wawancara pengguna.",
                        report_ids=["RPT0007"],
                    ),
                    compare_answer.CompareContrastSide(
                        label=FINALIST_LABEL,
                        text="Data teknis.",
                        report_ids=["RPT0008"],
                    ),
                ],
            )
        ],
        incidents={},
        cited_report_ids=["RPT0007", "RPT0008"],
    )

    rendered = compare_answer.render_answer(validation, "achievement_group")

    assert rendered == (
        f"**{AWARD_HEADING}**\n\n"
        "- Validasi dilakukan lewat wawancara. [RPT0007]\n\n"
        f"**{FINALIST_HEADING}**\n\n"
        "- Riset pasar terbatas pada data sekunder. [RPT0008]\n\n"
        "**Perbedaan utama**\n\n"
        "- kedalaman validasi\n"
        f"  - {AWARD_HEADING}: Wawancara pengguna. [RPT0007]\n"
        f"  - {FINALIST_HEADING}: Data teknis. [RPT0008]"
    )


def test_two_contrasts_are_separated_by_one_blank_line():
    validation = compare_answer.CompareValidation(
        sections=[],
        contrasts=[
            compare_answer.CompareContrast(
                dimension="validasi",
                sides=[
                    compare_answer.CompareContrastSide(AWARD_LABEL, "A.", ["RPT0001"]),
                    compare_answer.CompareContrastSide(
                        FINALIST_LABEL, "B.", ["RPT0002"]
                    ),
                ],
            ),
            compare_answer.CompareContrast(
                dimension="presentasi",
                sides=[
                    compare_answer.CompareContrastSide(AWARD_LABEL, "C.", ["RPT0001"]),
                    compare_answer.CompareContrastSide(
                        FINALIST_LABEL, "D.", ["RPT0002"]
                    ),
                ],
            ),
        ],
        cited_report_ids=["RPT0001", "RPT0002"],
    )

    rendered = compare_answer.render_answer(validation, "achievement_group")

    assert "  - Tim belum juara (finalis): B. [RPT0002]\n\n- presentasi" in rendered


# -- N1/N4: the competition_type comparison through the pipeline -----------
def _metadata(report_id, competition_type):
    return {
        "report_id": report_id,
        "competition_name": f"Kompetisi {report_id}",
        "competition_type": competition_type,
        "competition_year": 2023,
        "achievement_level": "Finalis",
        "section_type": "evaluasi_tim",
        "chunk_id": f"{report_id}:evaluasi_tim:01",
    }


def _match(report_id, competition_type):
    meta = _metadata(report_id, competition_type)
    return ChunkMatch(
        chunk_id=meta["chunk_id"],
        report_id=report_id,
        document=f"Isi {report_id}.",
        metadata=meta,
        distance=0.1,
    )


def _type_pipeline(structured_payload):
    matches = [_match("RPT3001", "Business Plan"), _match("RPT3002", "Business Case")]
    metadatas = [_metadata(item.report_id, item.metadata["competition_type"]) for item in matches]
    llm = FakeLLM(
        json_responses=[
            json.dumps(
                {
                    "in_scope": True,
                    "mode": "cross_report",
                    "search_query": "kendala business plan dan business case",
                    "filters": {},
                    "compare_by": "competition_type",
                }
            )
        ],
        structured_responses=[json.dumps(structured_payload, ensure_ascii=False)],
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(matches=matches, metadatas=metadatas),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )
    return pipeline, llm


def test_a_competition_type_comparison_uses_the_two_type_labels():
    payload = {
        "groups": {
            "g1": [{"text": "Data primer terbatas.", "cite": ["LAP1"]}],
            "g2": [{"text": "Data sekunder dominan.", "cite": ["LAP2"]}],
        },
        "contrasts": [
            {"dimension": "kesiapan data", "pick": {"g1": 1, "g2": 1}}
        ],
    }
    pipeline, llm = _type_pipeline(payload)

    result = pipeline.run(
        "Bandingkan kendala yang dialami tim business plan dan business case."
    )

    assert result.status is AnswerStatus.ANSWERED
    lines = result.answer.splitlines()
    assert "**Business Plan**" in lines
    assert "**Business Case**" in lines
    assert "  - Business Case: Data sekunder dominan. [RPT3002]" in lines
    call = [item for item in llm.calls if item.get("response_format")][0]
    pick = call["response_format"]["json_schema"]["schema"]["properties"]["contrasts"][
        "items"
    ]["properties"]["pick"]
    # Fix pack 9/10: the schema carries keys for the two groups that have
    # documents (Business Plan, Business Case), never the empty configured types.
    assert pick["required"] == ["g1", "g2"]
    keys = call["response_format"]["json_schema"]["schema"]["properties"]["groups"][
        "required"
    ]
    assert keys == ["g1", "g2"]
    assert result.debug.incidents.get("contrast_invalid", 0) == 0
    assert "LAP" not in result.answer


# -- N2: source fidelity ---------------------------------------------------
def test_the_free_text_addendum_carries_the_fidelity_rules():
    addendum = prompts.CITATION_FORMAT_ADDENDUM

    assert prompts.ADDENDUM_VERSION == "1.5"
    assert "(v1.5)" in addendum
    assert '"kurang menggali" bukan "mengabaikan"' in addendum
    assert "kecuali kalimat sumber yang sama menyatakan hubungan itu" in addendum


def test_the_compare_addendum_carries_the_fidelity_rules():
    addendum = prompts.COMPARE_CITATION_ADDENDUM

    assert prompts.COMPARE_ADDENDUM_VERSION == "2.5"
    assert '"kurang menggali" bukan "mengabaikan"' in addendum
    assert "Jangan menyambung dua fakta" in addendum


def test_the_causal_metric_counts_connector_sentences():
    text = (
        "- Demo fitur dipadatkan sehingga waktu habis. [RPT0001]\n"
        "- Tim memadatkan demo karena durasi sempit. [RPT0001]\n"
        "- Riset pasar menjadi kendala. [RPT0001]"
    )

    result = causal_connectors(text)

    assert result["count"] == 2
    assert result["samples"][0].startswith("Demo fitur dipadatkan")


def test_the_causal_metric_skips_headings_and_backend_lines():
    text = (
        "**Perbedaan utama**\n\n"
        "Ringkasan ini didasarkan pada 1 dari 2 laporan, sehingga seimbang.\n"
        "**Sumber**\n"
        "- RPT0001 — Kompetisi (Business Plan, 2023, Juara 1)"
    )

    assert causal_connectors(text)["count"] == 0


def test_causal_samples_are_capped():
    text = "\n".join(f"Tim A membuat keputusan {index}." for index in range(8))

    result = causal_connectors(text)

    assert result["count"] == 8
    assert len(result["samples"]) == config.CAUSAL_SAMPLE_LIMIT


def _type_groups():
    return {
        "g1": [{"text": "Data primer terbatas.", "cite": ["LAP1"]}],
        "g2": [{"text": "Data sekunder dominan.", "cite": ["LAP2"]}],
    }


def test_the_pipeline_exposes_the_causal_metric_and_never_logs_it(caplog):
    pipeline, _ = _type_pipeline({"groups": _type_groups(), "contrasts": []})

    with caplog.at_level(logging.DEBUG):
        result = pipeline.run("Bandingkan business plan dan business case.")

    assert "causal" in result.debug.to_dict()
    assert all("Demo fitur" not in record.getMessage() for record in caplog.records)


# -- N3: the per-report cap always applies --------------------------------
def _bullets(report_counts):
    lines = ["Pendahuluan jawaban."]
    index = 0
    for report_id, count in report_counts:
        for _ in range(count):
            index += 1
            lines.append(f"- Poin {index} untuk laporan. [{report_id}]")
    return "\n".join(lines)


def test_this_runs_shape_keeps_two_bullets_for_the_overloaded_report():
    text = _bullets([("RPT0001", 3), ("RPT0002", 2), ("RPT0003", 1)])

    trimmed, dropped = trim_bullets(text)

    assert dropped == 1
    assert trimmed.count("[RPT0001]") == 2
    assert trimmed.count("[RPT0002]") == 2
    assert trimmed.count("[RPT0003]") == 1
    assert "Pendahuluan jawaban." in trimmed
    assert "Poin 3 untuk laporan." not in trimmed


def test_the_earlier_eleven_point_shape_keeps_every_report():
    text = _bullets([("RPT0001", 8), ("RPT0002", 2), ("RPT0003", 1)])

    trimmed, dropped = trim_bullets(text)

    assert dropped == 6
    bullets = [line for line in trimmed.splitlines() if line.startswith("- ")]
    assert len(bullets) == 5
    for report_id in ("RPT0001", "RPT0002", "RPT0003"):
        assert any(report_id in line for line in bullets)


def test_the_pipeline_trims_a_short_answer_that_breaks_the_per_report_cap():
    meta = _metadata("RPT3001", "Business Plan")
    chunk = ChunkMatch(
        chunk_id=meta["chunk_id"],
        report_id="RPT3001",
        document="Isi RPT3001.",
        metadata=meta,
        distance=0.1,
    )
    answer = (
        "Ringkasan pengalaman tim.\n"
        "- Poin satu. [LAP1]\n"
        "- Poin dua. [LAP1]\n"
        "- Poin tiga. [LAP1]"
    )
    llm = FakeLLM(
        json_responses=[
            json.dumps(
                {
                    "in_scope": True,
                    "mode": "specific",
                    "search_query": "pengalaman tim",
                    "filters": {},
                }
            )
        ],
        text_responses=[answer],
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(matches=[chunk], metadatas=[meta]),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run("Apa pengalaman tim pada business plan?")

    assert result.status is AnswerStatus.ANSWERED
    assert result.debug.dropped_bullets == 1
    assert result.answer.count("[RPT3001]") == 2
    assert "Poin tiga" not in result.answer


# -- N4: post-processing stages -------------------------------------------
def test_the_pipeline_reports_its_post_processing_stages():
    pipeline, _ = _type_pipeline({"groups": _type_groups(), "contrasts": []})

    result = pipeline.run("Bandingkan business plan dan business case.")

    phases = result.debug.post_phases
    assert {"compare_validate", "render", "metrics", "coverage", "sources"} <= set(phases)
    assert all(value >= 0 for value in phases.values())


def test_the_answer_path_reports_context_and_citation_stages():
    meta = _metadata("RPT3001", "Business Plan")
    chunk = ChunkMatch(
        chunk_id=meta["chunk_id"],
        report_id="RPT3001",
        document="Isi RPT3001.",
        metadata=meta,
        distance=0.1,
    )
    llm = FakeLLM(
        json_responses=[
            json.dumps(
                {
                    "in_scope": True,
                    "mode": "specific",
                    "search_query": "pengalaman tim",
                    "filters": {},
                }
            )
        ],
        text_responses=["Tim menyiapkan validasi [LAP1]."],
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(matches=[chunk], metadatas=[meta]),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run("Apa pengalaman tim pada business plan?")

    phases = result.debug.post_phases
    assert {"context", "sanitize", "citations", "guardrails", "trim", "metrics"} <= set(
        phases
    )


# -- N5b: look-alike characters -------------------------------------------
def test_non_breaking_and_zero_width_characters_are_normalized():
    text = "Demo end\u2011to\u2011end terburu\u2011buruan mendapat 83\u202f% nilai."

    assert sanitize_answer(text) == (
        "Demo end-to-end terburu-buruan mendapat 83 % nilai."
    )


def test_a_no_break_space_becomes_a_plain_space():
    assert sanitize_answer("83\u00a0%") == "83 %"
    assert sanitize_answer("a\u200bb\ufeffc") == "abc"


def test_normal_wording_is_untouched():
    text = "Tim kurang menggali kelayakan finansial, bukan mengabaikannya."

    assert sanitize_answer(text) == text


# -- N5a: the counting tokenizer stays quiet (slow: needs the weights) ----
@pytest.mark.slow
def test_counting_a_long_prompt_does_not_warn_about_the_512_token_limit(caplog):
    from core.tokens import E5TokenCounter

    counter = E5TokenCounter()
    counter.count_tokens("pemanasan")

    with caplog.at_level(logging.WARNING):
        count = counter.count_tokens("kata " * 1400)

    assert count > config.CHUNK_HARD_LIMIT_TOKENS
    assert not any(
        "longer than the specified maximum sequence length" in record.getMessage()
        for record in caplog.records
    )
    assert counter._ensure_tokenizer().model_max_length == config.TOKEN_COUNT_MAX_LENGTH
