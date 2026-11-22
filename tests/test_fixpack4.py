"""
Offline tests for fix pack J1 (structured comparison answers), J3 (citation
relocation and cited-only coverage) and J4 (queue message helpers).

FakeLLM, FilteringKnowledgeBase and WordCountTokenizer come from tests/stubs.py:
no network, no API key, no model weights.
"""

import json

import pytest

import config
from core import compare_answer, prompts
from core.citations import process_citations
from core.knowledge_base import ChunkMatch
from core.pipeline import AnswerStatus, RagPipeline
from tests.stubs import (
    FakeLLM,
    FilteringKnowledgeBase,
    RecordingEmbedder,
    WordCountTokenizer,
    keyed_compare_payload,
)

pytestmark = pytest.mark.usefixtures("legacy_answer_contract")

AWARD = "Juara 1"
FINALIST = "Finalis"


def metadata(report_id, level, section="evaluasi_tim"):
    return {
        "report_id": report_id,
        "competition_name": f"Kompetisi {report_id}",
        "competition_type": "Business Plan",
        "competition_year": 2023,
        "achievement_level": level,
        "section_type": section,
        "chunk_id": f"{report_id}:{section}:01",
    }


def chunk_match(report_id, level, score=0.9):
    meta = metadata(report_id, level)
    return ChunkMatch(
        chunk_id=meta["chunk_id"],
        report_id=report_id,
        document=f"Isi {report_id}.",
        metadata=meta,
        distance=1.0 - score,
    )


def analyzer_payload(compare_by="achievement_group"):
    return json.dumps(
        {
            "in_scope": True,
            "mode": "cross_report",
            "search_query": "perbandingan pembelajaran tim",
            "filters": {},
            "compare_by": compare_by,
        }
    )


def comparison_pipeline(
    structured_payload,
    *,
    compare_by="achievement_group",
    structured_finish_reasons=None,
):
    matches = [chunk_match(f"RPT100{i}", AWARD) for i in range(1, 4)]
    matches += [chunk_match(f"RPT200{i}", FINALIST) for i in range(1, 3)]
    metadatas = [metadata(f"RPT100{i}", AWARD) for i in range(1, 4)]
    metadatas += [metadata(f"RPT200{i}", FINALIST) for i in range(1, 3)]
    llm = FakeLLM(
        json_responses=[analyzer_payload(compare_by)],
        structured_responses=[structured_payload],
        structured_finish_reasons=structured_finish_reasons,
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(matches=matches, metadatas=metadatas),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )
    return pipeline, llm


QUESTION = "Apa perbedaan pembelajaran tim juara dan tim belum juara?"


def payload(groups, contrasts=None):
    return json.dumps(
        keyed_compare_payload(groups, contrasts), ensure_ascii=False
    )


# -- J1a: the schema -------------------------------------------------------
def test_the_schema_is_strict_and_keyed_by_group():
    schema = compare_answer.build_schema(["g1", "g2"])

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"groups", "contrasts"}
    contrast_item = schema["properties"]["contrasts"]["items"]
    assert contrast_item["additionalProperties"] is False
    assert set(contrast_item["required"]) == {"dimension", "sides"}
    side_item = contrast_item["properties"]["sides"]["items"]
    assert side_item["additionalProperties"] is False
    assert set(side_item["required"]) == {"group", "text", "cite"}
    assert side_item["properties"]["group"]["enum"] == ["g1", "g2"]
    # Fix pack 9: groups is a keyed object with every key required, so the model
    # never writes a group name and no group can be missing.
    group_object = schema["properties"]["groups"]
    assert group_object["additionalProperties"] is False
    assert set(group_object["required"]) == {"g1", "g2"}
    point_item = group_object["properties"]["g1"]["items"]
    assert point_item["additionalProperties"] is False
    assert set(point_item["required"]) == {"text", "cite"}


def test_the_structured_call_uses_json_schema_and_the_compare_cap():
    pipeline, llm = comparison_pipeline(
        payload(
            [
                {
                    "label": "meraih penghargaan",
                    "points": [{"text": "Tim juara memvalidasi masalah lewat wawancara.", "cite": ["LAP1"]}],
                },
                {
                    "label": "belum juara (finalis)",
                    "points": [{"text": "Tim finalis mengandalkan data teknis.", "cite": ["LAP4"]}],
                },
            ]
        )
    )

    pipeline.run(QUESTION)

    call = [item for item in llm.calls if item.get("response_format")][0]
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["strict"] is True
    assert call["max_tokens"] == config.LLM_MAX_OUTPUT_TOKENS_COMPARE
    assert call["reasoning_effort"] == config.ANSWER_REASONING_EFFORT


# -- J1c/J1d: rendering and validation -------------------------------------
def test_a_valid_payload_is_rendered_by_the_backend_with_citations_at_the_end():
    pipeline, _ = comparison_pipeline(
        payload(
            [
                {
                    "label": "meraih penghargaan",
                    "points": [
                        {"text": "Tim juara memvalidasi masalah lewat wawancara.", "cite": ["LAP1"]}
                    ],
                },
                {
                    "label": "belum juara (finalis)",
                    "points": [
                        {"text": "Tim finalis mengandalkan data teknis.", "cite": ["LAP4"]}
                    ],
                },
            ],
            [
                {
                    "dimension": "validasi lapangan",
                    "sides": [
                        {
                            "label": "meraih penghargaan",
                            "text": "Memvalidasi masalah lewat wawancara.",
                            "cite": ["LAP1"],
                        },
                        {
                            "label": "belum juara (finalis)",
                            "text": "Mengandalkan data teknis.",
                            "cite": ["LAP4"],
                        },
                    ],
                }
            ],
        )
    )

    result = pipeline.run(QUESTION)

    assert result.status is AnswerStatus.ANSWERED
    lines = result.answer.splitlines()
    assert "**Tim yang meraih penghargaan (Juara 1-3, Juara Harapan, Most Aspiring)**" in lines
    assert "**Tim belum juara (finalis)**" in lines
    assert "**Perbedaan utama**" in lines
    assert "- validasi lapangan" in lines
    side = next(line for line in lines if line.startswith("  - Tim belum juara"))
    assert side.endswith("[RPT2001]")
    bullet = next(line for line in lines if line.startswith("- Tim juara memvalidasi"))
    assert bullet.endswith("[RPT1001]")
    assert result.cited_report_ids == ["RPT1001", "RPT2001"]
    assert result.debug.coverage == (
        "Ringkasan ini didasarkan pada 1 dari 3 laporan meraih penghargaan dan "
        "1 dari 2 laporan belum juara (finalis)."
    )
    assert result.answer.index(result.debug.coverage) < result.answer.index("**Sumber**")


def test_a_cross_group_citation_is_removed_and_counted():
    pipeline, _ = comparison_pipeline(
        payload(
            [
                {
                    "label": "meraih penghargaan",
                    "points": [
                        {
                            "text": "Tim juara memvalidasi masalah.",
                            "cite": ["LAP1", "LAP4"],
                        }
                    ],
                },
                {"label": "belum juara (finalis)", "points": []},
            ]
        )
    )

    result = pipeline.run(QUESTION)

    bullet = next(
        line for line in result.answer.splitlines() if line.startswith("- Tim juara")
    )
    assert bullet.endswith("[RPT1001]")
    assert "RPT2001" not in bullet
    assert result.debug.incidents["cross_group_citation"] == 1


def test_a_point_with_only_unknown_or_cross_group_citations_is_dropped():
    pipeline, _ = comparison_pipeline(
        payload(
            [
                {
                    "label": "meraih penghargaan",
                    "points": [
                        {"text": "Klaim tanpa sumber valid.", "cite": ["LAP9", "LAP4"]}
                    ],
                },
                {
                    "label": "belum juara (finalis)",
                    "points": [{"text": "Tim finalis mengandalkan data.", "cite": ["LAP4"]}],
                },
            ]
        )
    )

    result = pipeline.run(QUESTION)

    assert result.status is AnswerStatus.ANSWERED
    assert "Klaim tanpa sumber valid." not in result.answer
    assert result.debug.incidents["unknown_alias"] == 1
    assert result.debug.incidents["cross_group_citation"] == 1
    assert result.debug.incidents["point_uncited"] == 1


def test_a_group_without_valid_points_says_the_attempt_failed():
    pipeline, _ = comparison_pipeline(
        payload(
            [
                {
                    "label": "meraih penghargaan",
                    "points": [{"text": "Tim juara memvalidasi masalah.", "cite": ["LAP1"]}],
                },
                {"label": "belum juara (finalis)", "points": []},
            ]
        )
    )

    result = pipeline.run(QUESTION)

    # Fix pack 9 (O2e): documents exist for the finalist group, so the answer may
    # not claim there is no evidence. The sentence says the attempt failed, the
    # contrasts are omitted, and the rendered group is counted as empty.
    assert "Tim belum juara (finalis)" in result.answer
    assert config.COMPARE_GROUP_EMPTY_RETRY in result.answer
    assert config.COMPARE_EMPTY_GROUP not in result.answer
    assert config.COMPARE_CONTRASTS_HEADING not in result.answer
    assert result.debug.incidents["group_empty"] == 1


def test_an_unknown_group_key_is_a_format_error():
    payload_with_unknown_key = json.dumps(
        {
            "groups": {
                "g1": [{"text": "Tim juara memvalidasi masalah.", "cite": ["LAP1"]}],
                "g2": [],
                "g9": [{"text": "Klaim.", "cite": ["LAP4"]}],
            },
            "contrasts": [],
        }
    )
    pipeline, _ = comparison_pipeline(payload_with_unknown_key)

    result = pipeline.run(QUESTION)

    # The schema only allows known keys; strict output makes an unknown key a
    # contract violation, never renderable text.
    assert result.status is AnswerStatus.TECHNICAL_ERROR
    assert "Klaim." not in result.answer


def test_no_alias_token_survives_in_the_rendered_text():
    pipeline, _ = comparison_pipeline(
        payload(
            [
                {
                    "label": "meraih penghargaan",
                    "points": [
                        {
                            "text": "Tim juara (LAP1) memvalidasi masalah.",
                            "cite": ["LAP1"],
                        }
                    ],
                },
                {"label": "belum juara (finalis)", "points": []},
            ]
        )
    )

    result = pipeline.run(QUESTION)

    assert "LAP" not in result.answer
    assert result.debug.incidents["alias_in_text"] == 1


def test_nothing_traceable_returns_no_source():
    pipeline, _ = comparison_pipeline(
        payload([{"label": "meraih penghargaan", "points": []}, {"label": "belum juara (finalis)", "points": []}])
    )

    result = pipeline.run(QUESTION)

    assert result.status is AnswerStatus.NO_SOURCE
    assert result.answer == prompts.NO_SOURCE_RESPONSE


def test_invalid_json_is_a_technical_error_without_a_retry_or_raw_text():
    pipeline, llm = comparison_pipeline('{"groups": [broken')

    result = pipeline.run(QUESTION)

    assert result.status is AnswerStatus.TECHNICAL_ERROR
    assert result.answer == prompts.TECHNICAL_ERROR_RESPONSE
    assert result.debug.error_type == "invalid_structured_output"
    assert llm.structured_call_count == 1
    assert "broken" not in result.answer
    assert result.debug.raw_answers[-1]["content"] == '{"groups": [broken'


def test_a_truncated_payload_is_a_technical_error():
    pipeline, llm = comparison_pipeline(
        payload([{"label": "meraih penghargaan", "points": []}, {"label": "belum juara (finalis)", "points": []}]),
        structured_finish_reasons=["length"],
    )

    result = pipeline.run(QUESTION)

    assert result.status is AnswerStatus.TECHNICAL_ERROR
    assert result.debug.error_type == "invalid_structured_output"
    assert llm.structured_call_count == 1


# -- J3a: citation relocation ----------------------------------------------
def test_parenthesised_alias_names_move_to_the_end_of_the_sentence():
    alias_map = {"LAP2": "RPT0002", "LAP3": "RPT0003"}
    text = (
        "Tim juara (LAP2, LAP3) menekankan validasi masalah lewat wawancara. "
        "Tim belum juara mengandalkan teori."
    )

    result = process_citations(text, alias_map, ["RPT0002", "RPT0003"])

    assert "LAP" not in result.text
    assert "(" not in result.text
    assert (
        "Tim juara menekankan validasi masalah lewat wawancara "
        "[RPT0002] [RPT0003]."
    ) in result.text
    assert result.cited_report_ids == ["RPT0002", "RPT0003"]


def test_a_mid_sentence_citation_moves_to_the_end_of_its_sentence():
    alias_map = {"LAP1": "RPT0001"}
    text = "Kendala logistik [LAP1] muncul pada dua tim. Tim lain fokus pada data."

    result = process_citations(text, alias_map, ["RPT0001"])

    assert result.text.startswith("Kendala logistik muncul pada dua tim [RPT0001].")


def test_a_citation_already_at_the_end_is_left_alone():
    alias_map = {"LAP1": "RPT0001"}

    result = process_citations("Kendala logistik [LAP1].", alias_map, ["RPT0001"])

    assert result.text == "Kendala logistik [RPT0001]."


def test_relocation_keeps_bullets_and_never_duplicates_a_citation():
    alias_map = {"LAP1": "RPT0001", "LAP2": "RPT0002"}
    text = "- Tim A [LAP1] dan tim B [LAP1] sama-sama mencatat kendala."

    result = process_citations(text, alias_map, ["RPT0001", "RPT0002"])

    assert result.text.startswith("- Tim A dan tim B sama-sama mencatat kendala")
    assert result.text.count("[RPT0001]") == 1


# -- J3b: coverage counts only cited reports -------------------------------
def test_generic_coverage_counts_cited_reports_only():
    matches = [chunk_match(f"RPT000{i}", FINALIST) for i in range(1, 4)]
    metadatas = [metadata(f"RPT000{i}", FINALIST) for i in range(1, 6)]
    llm = FakeLLM(
        json_responses=[
            json.dumps(
                {
                    "in_scope": True,
                    "mode": "cross_report",
                    "search_query": "kendala tim",
                    "filters": {},
                }
            )
        ],
        text_responses=["Kendala logistik muncul di beberapa tim [LAP1]."],
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(matches=matches, metadatas=metadatas),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run("Bandingkan kendala beberapa tim.")

    assert result.debug.coverage == (
        "Ringkasan ini didasarkan pada 1 dari 5 laporan yang sesuai."
    )


# -- J4: queue message helpers --------------------------------------------
class FakeLimiter:
    def __init__(self, wait):
        self.wait = wait
        self.calls = []

    def wait_estimate(self, tokens):
        self.calls.append(tokens)
        return self.wait


def test_the_queue_message_appears_only_for_a_long_wait():
    import app

    assert app.queue_message(FakeLimiter(1.5)) is None
    message = app.queue_message(FakeLimiter(config.QUEUE_MESSAGE_MIN_WAIT_SECONDS + 9.2))
    assert message == (
        "Sedang mengantre karena batas layanan, perkiraan tunggu ~13 detik."
    )


def test_the_queue_estimate_uses_the_configured_token_figure():
    import app

    limiter = FakeLimiter(30.0)
    app.queue_wait_seconds(limiter)

    assert limiter.calls == [config.QUEUE_MESSAGE_ESTIMATE_TOKENS]
