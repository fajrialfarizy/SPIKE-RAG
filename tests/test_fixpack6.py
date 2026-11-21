"""
Offline tests for fix pack L2-L6.

L2: comparison group names (model-facing and display).
L3: compare addendum v2.1 and the deterministic difference check.
L4: negation metric with label stripping and skipped backend lines.
L5: analyzer timing split and limiter reservation evidence.
L6: one citation format, [RPT0001] [RPT0002].

FakeLLM, FilteringKnowledgeBase and WordCountTokenizer come from tests/stubs.py:
no network, no API key, no model weights.
"""

import json

import config
from core import compare_answer, prompts
from core.analyzer import Analyzer
from core.citations import process_citations
from core.context_builder import ContextBuilder
from core.knowledge_base import ChunkMatch
from core.llm_client import GroqLLMClient, LLMResult
from core.negation import negation_sentences
from core.pipeline import AnswerStatus, RagPipeline
from core.retriever import RetrievedChunk
from tests.stubs import (
    FakeLLM,
    FilteringKnowledgeBase,
    RecordingEmbedder,
    WordCountTokenizer,
)

AWARD = "Juara 1"
FINALIST = "Finalis"
AWARD_LABEL = "meraih penghargaan"
FINALIST_LABEL = "belum juara (finalis)"
AWARD_HEADING = (
    "Tim yang meraih penghargaan (Juara 1-3, Juara Harapan, Most Aspiring)"
)
FINALIST_HEADING = "Tim belum juara (finalis)"
QUESTION = "Apa perbedaan pembelajaran tim juara dan tim belum juara?"


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


def comparison_pipeline(structured_payload):
    matches = [chunk_match(f"RPT100{i}", AWARD) for i in range(1, 4)]
    matches += [chunk_match(f"RPT200{i}", FINALIST) for i in range(1, 3)]
    metadatas = [metadata(f"RPT100{i}", AWARD) for i in range(1, 4)]
    metadatas += [metadata(f"RPT200{i}", FINALIST) for i in range(1, 3)]
    llm = FakeLLM(
        json_responses=[
            json.dumps(
                {
                    "in_scope": True,
                    "mode": "cross_report",
                    "search_query": "perbandingan pembelajaran tim",
                    "filters": {},
                    "compare_by": "achievement_group",
                }
            )
        ],
        structured_responses=[structured_payload],
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(matches=matches, metadatas=metadatas),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )
    return pipeline, llm


def payload(groups, differences=None):
    return json.dumps(
        {"groups": groups, "differences": differences or []}, ensure_ascii=False
    )


def groups_payload():
    return [
        {
            "label": AWARD_LABEL,
            "points": [
                {"text": "Validasi masalah lewat wawancara.", "cite": ["LAP1"]}
            ],
        },
        {
            "label": FINALIST_LABEL,
            "points": [
                {"text": "Riset rantai pasok menjadi kendala.", "cite": ["LAP4"]}
            ],
        },
    ]


# -- L2: group names -------------------------------------------------------
def test_the_configured_group_names_are_the_documented_ones():
    groups = config.COMPARISON_GROUPS["achievement_group"]

    assert [group["label"] for group in groups] == [AWARD_LABEL, FINALIST_LABEL]
    assert [group["display_label"] for group in groups] == [
        AWARD_HEADING,
        FINALIST_HEADING,
    ]
    assert config.COMPARISON_GROUPS["competition_type"][0]["label"] == "Business Plan"


def test_the_schema_enum_and_the_context_use_the_model_facing_names():
    pipeline, llm = comparison_pipeline(payload(groups_payload()))

    pipeline.run(QUESTION)

    call = [item for item in llm.calls if item.get("response_format")][0]
    enum = call["response_format"]["json_schema"]["schema"]["properties"]["groups"][
        "items"
    ]["properties"]["label"]["enum"]
    assert enum == [AWARD_LABEL, FINALIST_LABEL]
    assert f'<kelompok nama="{AWARD_LABEL}">' in call["user_prompt"]
    assert f'kelompok="{AWARD_LABEL}"' in call["user_prompt"]
    assert f'- {AWARD_LABEL}: [LAP1]' in call["user_prompt"]


def test_the_heading_is_the_plain_display_label_without_markdown_bold():
    pipeline, _ = comparison_pipeline(payload(groups_payload()))

    result = pipeline.run(QUESTION)

    lines = result.answer.splitlines()
    assert AWARD_HEADING in lines
    assert FINALIST_HEADING in lines
    assert not any(line.startswith("**") and "Sumber" not in line for line in lines)
    assert result.debug.coverage == (
        f"Ringkasan ini didasarkan pada 1 dari 3 laporan {AWARD_LABEL} dan "
        f"1 dari 2 laporan {FINALIST_LABEL}."
    )


def test_the_addendum_tells_the_model_not_to_restate_the_group():
    addendum = prompts.COMPARE_CITATION_ADDENDUM

    assert prompts.COMPARE_ADDENDUM_VERSION == "2.1"
    assert "(v2.1)" in addendum
    assert 'Jangan memulai poin dengan "Tim juara"' in addendum
    assert "berbeda pada dimensi yang SAMA" not in addendum
    assert "dimensi yang SAMA" in addendum


# -- L3: differences must cite both groups ---------------------------------
def test_a_difference_citing_both_groups_is_kept():
    pipeline, _ = comparison_pipeline(
        payload(
            groups_payload(),
            [
                {
                    "text": f"Kelompok {AWARD_LABEL} dan {FINALIST_LABEL} berbeda "
                    "pada kedalaman validasi pasar.",
                    "cite": ["LAP1", "LAP4"],
                }
            ],
        )
    )

    result = pipeline.run(QUESTION)

    assert f"**" not in result.answer or "**Sumber**" in result.answer
    assert "berbeda pada kedalaman validasi pasar" in result.answer
    assert "- " in result.answer
    assert "[RPT1001] [RPT2001]" in result.answer
    assert result.debug.negation["count"] == 0


def test_a_difference_citing_one_group_only_is_dropped_and_counted():
    pipeline, _ = comparison_pipeline(
        payload(
            groups_payload(),
            [
                {
                    "text": "Kedalaman validasi pasar menjadi pembeda utama.",
                    "cite": ["LAP1"],
                }
            ],
        )
    )

    result = pipeline.run(QUESTION)

    assert "Kedalaman validasi pasar menjadi pembeda utama." not in result.answer
    assert "Perbedaan utama" not in result.answer
    assert result.debug.incidents["difference_missing_group"] == 1


def test_all_dropped_differences_omit_the_section_entirely():
    pipeline, _ = comparison_pipeline(
        payload(
            groups_payload(),
            [
                {"text": "Pembeda pertama.", "cite": ["LAP1"]},
                {"text": "Pembeda kedua.", "cite": ["LAP4"]},
            ],
        )
    )

    result = pipeline.run(QUESTION)

    assert "Pembeda pertama" not in result.answer
    assert "Pembeda kedua" not in result.answer
    assert "Perbedaan utama" not in result.answer
    assert result.debug.incidents["difference_missing_group"] == 2


def test_the_compare_addendum_limits_differences_to_two():
    assert config.COMPARE_DIFFERENCES_MAX == 2
    assert "paling banyak 4 poin per kelompok, 2 butir differences" in (
        prompts.COMPARE_CITATION_ADDENDUM
    )


# -- L4: negation metric ---------------------------------------------------
def test_a_heading_label_alone_counts_nothing():
    assert negation_sentences("**belum juara**")["count"] == 0
    assert negation_sentences(FINALIST_HEADING)["count"] == 0


def test_a_label_only_sentence_is_not_a_negated_claim():
    text = "Tim belum juara terlalu fokus pada detail teknis."

    assert negation_sentences(text)["count"] == 0


def test_a_negation_next_to_a_label_is_counted_once():
    text = "Tim belum juara kurang menyiapkan riset rantai pasok."

    result = negation_sentences(text)

    assert result["count"] == 1
    assert result["samples"] == [text]


def test_a_bullet_starting_with_tanpa_counts():
    assert negation_sentences("tanpa cukup validasi pasar")["count"] == 1


def test_a_legitimate_negative_still_counts_and_blocks_nothing():
    assert negation_sentences("tidak ada anggota yang pasif")["count"] == 1


def test_the_backend_coverage_and_source_lines_are_skipped():
    text = (
        "Ringkasan ini didasarkan pada 1 dari 3 laporan meraih penghargaan dan "
        "1 dari 2 laporan belum juara (finalis).\n"
        "**Sumber**\n"
        "- RPT1001 — Kompetisi RPT1001 (Business Plan, 2023, Juara 1)"
    )

    assert negation_sentences(text)["count"] == 0


def test_the_two_new_markers_are_recognized():
    assert negation_sentences("Tim mengabaikan data pendukung.")["count"] == 1
    assert negation_sentences("Tim gagal menyiapkan data pendukung.")["count"] == 1


# -- L5: observability -----------------------------------------------------
class ScriptedMetricsLLM:
    """Analyzer-only double that reports a queue wait and a model latency."""

    def complete(self, *args, **kwargs):
        return LLMResult(
            content='{"in_scope": true, "mode": "specific", "search_query": "kendala tim"}',
            latency_ms=12_400.0,
            queue_wait_ms=11_000.0,
            model_latency_ms=1_400.0,
        )


def test_the_analyzer_reports_its_queue_wait_and_model_latency():
    analysis = Analyzer(ScriptedMetricsLLM()).analyze("Apa kendala tim?")

    assert analysis.queue_wait_ms == 11_000.0
    assert analysis.model_latency_ms == 1_400.0


def test_the_pipeline_exposes_the_analyzer_timing():
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(
            matches=[chunk_match("RPT1001", AWARD)],
            metadatas=[metadata("RPT1001", AWARD)],
        ),
        llm=FakeLLM(
            json_responses=[
                json.dumps(
                    {
                        "in_scope": True,
                        "mode": "specific",
                        "search_query": "kendala tim",
                        "filters": {},
                    }
                )
            ],
            text_responses=["Tim fokus pada validasi [LAP1]."],
        ),
        embedder=RecordingEmbedder(),
        analyzer=Analyzer(ScriptedMetricsLLM()),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    analyzer_call = next(
        call for call in result.debug.calls if call["name"] == "analyzer"
    )
    assert analyzer_call["queue_wait_ms"] == 11_000.0
    assert analyzer_call["model_latency_ms"] == 1_400.0
    assert result.debug.analyzer["queue_wait_ms"] == 11_000.0
    assert result.debug.analyzer["model_latency_ms"] == 1_400.0


def test_a_call_reports_the_limiter_state_at_its_reservation():
    from tests.test_llm_client import FakeGroqClient, FakeResponse

    fake = FakeGroqClient([FakeResponse("ok")])
    client = GroqLLMClient(api_key="dummy-key", client=fake)

    result = client.complete(
        "instruksi sistem",
        "pertanyaan mahasiswa",
        temperature=config.LLM_TEMPERATURE_CHAT,
        max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
        reasoning_effort=config.ANSWER_REASONING_EFFORT,
    )

    reservation = result.reservation
    assert reservation["tokens_used_last_60s"] == 0
    assert reservation["estimated_tokens"] > 0
    assert reservation["wait_seconds"] == 0
    assert reservation["capacity_tokens_per_minute"] == config.GROQ_LIMITS[
        "tokens_per_minute"
    ]
    assert reservation["actual_tokens"] == result.usage.total
    assert reservation["unused_released"] == max(
        0, reservation["reserved_tokens"] - reservation["actual_tokens"]
    )
    metrics = result.to_metrics("answer")
    assert metrics["reservation"]["estimated_tokens"] == reservation["estimated_tokens"]


# -- L6: one citation format ----------------------------------------------
def test_free_text_uses_separate_brackets_with_one_space():
    alias_map = {"LAP1": "RPT0010", "LAP2": "RPT0004"}
    result = process_citations(
        "Tim juara dan finalis menekankan hal berbeda [LAP1, LAP2].",
        alias_map,
        ["RPT0010", "RPT0004"],
    )

    assert result.text == (
        "Tim juara dan finalis menekankan hal berbeda [RPT0010] [RPT0004]."
    )
    assert "[RPT0010, RPT0004]" not in result.text


def test_the_compare_renderer_uses_the_same_format():
    validation = compare_answer.CompareValidation(
        sections=[],
        differences=[
            compare_answer.ComparePoint(
                text="Kedalaman validasi pasar berbeda.", report_ids=["RPT1001", "RPT2001"]
            )
        ],
        incidents={},
        cited_report_ids=["RPT1001", "RPT2001"],
    )

    rendered = compare_answer.render_answer(validation)

    assert "- Kedalaman validasi pasar berbeda. [RPT1001] [RPT2001]" in rendered


# -- L1d: context normalization -------------------------------------------
def _chunk(document):
    return RetrievedChunk(
        chunk_id="RPT1001:evaluasi_tim:01",
        report_id="RPT1001",
        document=document,
        metadata=metadata("RPT1001", AWARD),
        score=0.9,
    )


def test_the_context_never_keeps_a_blank_line_inside_a_paragraph():
    chunk = _chunk(
        "Kompetisi: Contoh (Essay, 2024) | Judul ide: Ide | Bagian: Evaluasi Tim\n\n"
        "Tim menyiapkan dokumen serah terima\n\nberisi daftar asumsi.\n\n"
        "Paragraf kedua utuh."
    )

    bundle = ContextBuilder().build_chunks([chunk])

    assert "serah terima\n\nberisi" not in bundle.text
    assert "serah terima berisi daftar asumsi." in bundle.text
    assert bundle.raw_text != bundle.text
