"""
Offline tests for fix pack H1 (identity route), H2 (model-declared not found)
and H3 (comparison coverage).

FakeLLM, FilteringKnowledgeBase and WordCountTokenizer come from tests/stubs.py:
no network, no API key, no model weights.
"""

import json

import pytest

import config
from core import prompts
from core.context_builder import ContextBuilder
from core.knowledge_base import ChunkMatch, StoredChunk
from core.llm_client import LLMRateLimitError
from core.pipeline import AnswerStatus, RagPipeline
from core.prompts import NOT_FOUND_RESPONSE, is_not_found_refusal
from core.retriever import RetrievedChunk
from tests.stubs import (
    FakeKnowledgeBase,
    FakeLLM,
    FilteringKnowledgeBase,
    RecordingEmbedder,
    WordCountTokenizer,
)

pytestmark = pytest.mark.usefixtures("legacy_answer_contract")

NAME = "Inovasi Bisnis Mahasiswa Nasional (IBMN) 2023"


def metadata(report_id, level="Finalis", section="saran_pengembangan", **overrides):
    data = {
        "report_id": report_id,
        "competition_name": f"Kompetisi {report_id}",
        "competition_type": "Business Plan",
        "competition_year": 2023,
        "achievement_level": level,
        "section_type": section,
        "chunk_id": f"{report_id}:{section}:01",
    }
    data.update(overrides)
    return data


def stored(report_id="RPT0002", section="saran_pengembangan"):
    return StoredChunk(
        chunk_id=f"{report_id}:{section}:01",
        report_id=report_id,
        document="Validasi pasar dan penguatan model bisnis.",
        metadata=metadata(report_id, section=section),
    )


def build_id_route(chunks=None, text_responses=None, question=None, llm=None):
    llm = llm or FakeLLM(text_responses=text_responses)
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(chunks=chunks or [stored()]),
        llm=llm,
        embedder=RecordingEmbedder(),
    )
    return pipeline, llm


# -- H1: the identity route ------------------------------------------------
ID_QUESTION = "Apa saja saran pengembangan yang tercantum pada RPT0002?"


def test_the_id_route_question_sent_to_the_model_uses_the_alias_not_the_raw_id():
    pipeline, llm = build_id_route(
        text_responses=["Saran utamanya adalah validasi pasar [LAP1]."]
    )

    result = pipeline.run(ID_QUESTION)

    assert result.status is AnswerStatus.ANSWERED
    prompt = llm.calls[0]["user_prompt"]
    question_block = prompt.split("<pertanyaan>")[1].split("</pertanyaan>")[0]
    assert "pada LAP1?" in question_block
    assert "RPT0002" not in question_block
    assert "Laporan pada konteks: [LAP1]." in prompt
    assert "isi lengkap laporan yang ditanyakan" in prompt


def test_the_id_route_hint_states_the_missing_report_and_keeps_the_found_alias():
    pipeline, llm = build_id_route(
        chunks=[stored("RPT0002")],
        text_responses=["Saran utamanya adalah validasi pasar [LAP1]."],
    )

    result = pipeline.run(
        "Bandingkan saran pengembangan pada RPT0002 dan RPT0088."
    )

    assert result.debug.missing_report_ids == ["RPT0088"]
    prompt = llm.calls[0]["user_prompt"]
    assert "pada LAP1" in prompt
    assert "Laporan lain yang diminta tidak tersedia: RPT0088." in prompt
    assert result.status is AnswerStatus.ANSWERED


def test_an_alias_citation_on_the_id_route_is_accepted():
    pipeline, _ = build_id_route(
        text_responses=["Saran utamanya adalah validasi pasar [LAP1]."]
    )

    result = pipeline.run(ID_QUESTION)

    assert result.cited_report_ids == ["RPT0002"]


def test_a_raw_report_id_citation_on_the_id_route_is_accepted_when_retrieved():
    pipeline, _ = build_id_route(
        text_responses=["Saran utamanya adalah validasi pasar [RPT0002]."]
    )

    result = pipeline.run(ID_QUESTION)

    assert result.status is AnswerStatus.ANSWERED
    assert result.cited_report_ids == ["RPT0002"]


def test_an_uncited_id_route_answer_still_gets_one_stricter_retry():
    pipeline, llm = build_id_route(
        text_responses=["Saran utamanya adalah validasi pasar."] * 2
    )

    result = pipeline.run(ID_QUESTION)

    assert result.status is AnswerStatus.NO_SOURCE
    assert llm.text_call_count == 2


# -- H2: a model-declared refusal ------------------------------------------
def build_semantic(text_responses, llm=None):
    llm = llm or FakeLLM(
        json_responses=[
            json.dumps(
                {
                    "in_scope": True,
                    "mode": "specific",
                    "search_query": "kendala business plan",
                    "filters": {},
                }
            )
        ],
        text_responses=text_responses,
    )
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(matches=[chunk_match("RPT0002")]),
        llm=llm,
        embedder=RecordingEmbedder(),
    )
    return pipeline, llm


def chunk_match(report_id, level="Finalis", score=0.9, section="evaluasi_tim"):
    meta = metadata(report_id, level=level, section=section)
    return ChunkMatch(
        chunk_id=meta["chunk_id"],
        report_id=report_id,
        document=f"Isi {report_id}.",
        metadata=meta,
        distance=1.0 - score,
    )


def test_a_model_declared_refusal_is_not_found_without_a_retry():
    pipeline, llm = build_semantic([NOT_FOUND_RESPONSE])

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.status is AnswerStatus.NOT_FOUND
    assert result.answer == NOT_FOUND_RESPONSE
    assert llm.text_call_count == 1


def test_a_first_sentence_refusal_with_markdown_still_matches():
    first_sentence = NOT_FOUND_RESPONSE.split(".")[0] + "."
    pipeline, llm = build_semantic([f"**{first_sentence}**"])

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.status is AnswerStatus.NOT_FOUND
    assert llm.text_call_count == 1


class RateLimitedOnRetryLLM(FakeLLM):
    """Analyzer works; any second text call is rate limited."""

    def complete(self, *args, **kwargs):
        if any(not call["json_mode"] for call in self.calls):
            raise LLMRateLimitError("minute token budget exhausted")
        return super().complete(*args, **kwargs)


def test_a_rate_limited_retry_cannot_overwrite_a_decisive_not_found():
    llm = RateLimitedOnRetryLLM(
        json_responses=[
            json.dumps(
                {
                    "in_scope": True,
                    "mode": "specific",
                    "search_query": "kendala business plan",
                    "filters": {},
                }
            )
        ],
        text_responses=[NOT_FOUND_RESPONSE],
    )
    pipeline, _ = build_semantic([NOT_FOUND_RESPONSE], llm=llm)

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.status is AnswerStatus.NOT_FOUND
    assert result.answer == NOT_FOUND_RESPONSE


def test_a_declared_refusal_on_the_id_route_is_no_source_without_a_retry():
    pipeline, llm = build_id_route(text_responses=[NOT_FOUND_RESPONSE])

    result = pipeline.run(ID_QUESTION)

    assert result.status is AnswerStatus.NO_SOURCE
    assert result.answer == prompts.NO_SOURCE_RESPONSE
    assert llm.text_call_count == 1
    assert result.debug.incidents["model_declared_not_found_with_context"] == 1


def test_the_refusal_detector_ignores_normal_answers():
    assert is_not_found_refusal(NOT_FOUND_RESPONSE)
    assert is_not_found_refusal(NOT_FOUND_RESPONSE.split(".")[0] + ".")
    assert is_not_found_refusal("  MAaf,   informasi yang kamu cari  " + "\n" + "belum tersedia dalam basis pengetahuan Lab Prestasi.")
    assert not is_not_found_refusal("Tim juara fokus pada validasi pasar [LAP1].")
    assert not is_not_found_refusal("")
    assert not is_not_found_refusal("Maaf, saya tidak dapat memberikan data pribadi.")


# -- H3: comparison coverage ----------------------------------------------
def group_payload(compare_by="achievement_group"):
    return json.dumps(
        {
            "in_scope": True,
            "mode": "cross_report",
            "search_query": "perbandingan pembelajaran tim",
            "filters": {},
            "compare_by": compare_by,
        }
    )


def comparison_base(
    matches, metadatas, cites=None, compare_by="achievement_group", groups=None
):
    """Grouped comparison pipeline with a structured answer (fix pack J1)."""
    payload = groups or {
        "g1": [
            {
                "text": "Tim juara memvalidasi masalah sebelum final.",
                "cite": list(cites or ("LAP1", "LAP2")),
            }
        ],
        "g2": [],
    }
    structured = json.dumps({"groups": payload, "contrasts": []})
    llm = FakeLLM(
        json_responses=[group_payload(compare_by)],
        structured_responses=[structured],
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(matches=matches, metadatas=metadatas),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )
    return pipeline, llm


def test_each_group_contributes_one_chunk_per_distinct_report():
    matches = [chunk_match(f"RPT100{i}", level="Juara 1") for i in range(1, 6)]
    matches += [chunk_match(f"RPT200{i}") for i in range(1, 6)]
    pipeline, _ = comparison_base(matches, metadatas=[])

    result = pipeline.run("Bandingkan tim juara dan belum juara.")

    assert result.debug.groups == [
        {"label": "meraih penghargaan", "count": config.COMPARE_REPORTS_PER_GROUP},
        {"label": "belum juara (finalis)", "count": config.COMPARE_REPORTS_PER_GROUP},
    ]
    report_ids = [item["chunk_id"] for item in result.debug.retrieved]
    assert len(report_ids) == len(set(report_ids))
    sections = {item["section_type"] for item in result.debug.retrieved}
    assert sections == {"evaluasi_tim"}


def test_a_group_uses_at_most_the_configured_reports_per_group():
    matches = [chunk_match(f"RPT100{i}", level="Juara 1") for i in range(1, 6)]
    matches += [chunk_match("RPT2001")]
    pipeline, _ = comparison_base(matches, metadatas=[])

    result = pipeline.run("Bandingkan tim juara dan belum juara.")

    assert [group["count"] for group in result.debug.groups] == [
        config.COMPARE_REPORTS_PER_GROUP,
        1,
    ]


def test_the_preferred_section_wins_inside_one_report():
    preferred = chunk_match("RPT0001", level="Juara 1", score=0.60, section="evaluasi_tim")
    other = chunk_match("RPT0001", level="Juara 1", score=0.95, section="executive_summary")
    other = ChunkMatch(
        chunk_id="RPT0001:executive_summary:01",
        report_id=other.report_id,
        document=other.document,
        metadata=other.metadata,
        distance=other.distance,
    )
    pipeline, _ = comparison_base([preferred, other], metadatas=[])

    result = pipeline.run("Bandingkan tim juara dan belum juara.")

    assert [item["section_type"] for item in result.debug.retrieved] == [
        "evaluasi_tim"
    ]


def test_trimming_drops_whole_reports_and_keeps_two_per_group():
    builder = ContextBuilder()
    counter = WordCountTokenizer()

    def grouped(report_id, score, label):
        return RetrievedChunk(
            chunk_id=f"{report_id}:evaluasi_tim:01",
            report_id=report_id,
            document=" ".join(["kata"] * 40),
            metadata=metadata(report_id, level="Juara 1", section="evaluasi_tim"),
            score=score,
            group_label=label,
        )

    chunks = [
        grouped("RPT1001", 0.95, "meraih penghargaan"),
        grouped("RPT1002", 0.90, "meraih penghargaan"),
        grouped("RPT1003", 0.85, "meraih penghargaan"),
        grouped("RPT2001", 0.60, "belum juara (finalis)"),
        grouped("RPT2002", 0.55, "belum juara (finalis)"),
    ]
    limit = counter.count_tokens(
        builder.build_chunks([chunks[0], chunks[1], chunks[3], chunks[4]]).text
    )

    bundle, dropped = builder.build_chunks_within_budget(chunks, counter, limit)

    assert dropped == ["RPT1003:evaluasi_tim:01"]
    assert set(bundle.report_metadata) == {"RPT1001", "RPT1002", "RPT2001", "RPT2002"}
    assert "<kelompok nama=\"meraih penghargaan\">" in bundle.text
    assert "<kelompok nama=\"belum juara (finalis)\">" in bundle.text


def test_the_coverage_line_reports_shown_against_available_reports():
    matches = [chunk_match(f"RPT100{i}", level="Juara 1") for i in range(1, 4)]
    matches += [chunk_match(f"RPT200{i}") for i in range(1, 3)]
    metadatas = [metadata(f"RPT100{i}", level="Juara 1") for i in range(1, 5)]
    metadatas += [metadata(f"RPT200{i}") for i in range(1, 4)]
    pipeline, _ = comparison_base(matches, metadatas)

    result = pipeline.run("Bandingkan tim juara dan belum juara.")

    # Fix pack J3b: only the two cited reports count, so the sentence and the
    # Sumber block agree. The answer cites LAP1 and LAP2, both in the award group.
    assert result.debug.coverage == (
        "Ringkasan ini didasarkan pada 2 dari 4 laporan meraih penghargaan dan "
        "0 dari 3 laporan belum juara (finalis)."
    )
    assert result.answer.index(result.debug.coverage) < result.answer.index("**Sumber**")


def test_a_group_without_data_is_not_counted_at_all():
    matches = [chunk_match(f"RPT100{i}", level="Juara 1") for i in range(1, 3)]
    metadatas = [metadata(f"RPT100{i}", level="Juara 1") for i in range(1, 3)]
    pipeline, _ = comparison_base(
        matches,
        metadatas,
        groups={
            "g1": [
                {
                    "text": "Tim juara memvalidasi masalah sebelum final.",
                    "cite": ["LAP1", "LAP2"],
                }
            ]
        },
    )

    result = pipeline.run("Bandingkan tim juara dan belum juara.")

    # Fix pack 9 (O2a): a group with no document is never rendered and never
    # counted, so the sentence cannot claim "0 dari 0 laporan".
    assert result.debug.coverage == (
        "Ringkasan ini didasarkan pada 2 dari 2 laporan meraih penghargaan."
    )
    assert "belum juara" not in result.answer


def test_cross_report_without_grouping_uses_the_generic_coverage_line():
    matches = [chunk_match(f"RPT000{i}") for i in range(1, 4)]
    metadatas = [metadata(f"RPT000{i}") for i in range(1, 6)]
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


def test_a_specific_answer_carries_no_coverage_line():
    pipeline, _ = build_semantic(["Kendala logistik [LAP1]."])

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.debug.coverage == ""
