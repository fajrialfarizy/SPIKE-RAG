"""
Offline tests for the context token budget (report F1d).
"""

import pytest

from core.context_builder import ContextBudgetError, ContextBuilder
from core.knowledge_base import ChunkMatch
from core.retriever import RetrievedChunk
from core.pipeline import AnswerStatus, RagPipeline
from tests.stubs import (
    FakeKnowledgeBase,
    FakeLLM,
    RecordingEmbedder,
    WordCountTokenizer,
)


def metadata(report_id, section="evaluasi_tim"):
    return {
        "report_id": report_id,
        "competition_name": f"Kompetisi {report_id}",
        "competition_type": "Business Plan",
        "competition_year": 2023,
        "achievement_level": "Finalis",
        "section_type": section,
    }


def chunk(report_id, index, score, words=40):
    return RetrievedChunk(
        chunk_id=f"{report_id}:{index}",
        report_id=report_id,
        document=" ".join(["kata"] * words),
        metadata=metadata(report_id),
        score=score,
    )


def test_the_lowest_score_is_dropped_until_the_context_fits():
    builder = ContextBuilder()
    chunks = [chunk("RPT0001", 1, 0.9), chunk("RPT0001", 2, 0.5), chunk("RPT0002", 1, 0.8)]
    counter = WordCountTokenizer()

    bundle, dropped = builder.build_chunks_within_budget(
        chunks, counter, max_tokens=120
    )

    assert dropped == ["RPT0001:2"]
    assert counter.count_tokens(bundle.text) <= 120
    assert set(bundle.report_metadata) == {"RPT0001", "RPT0002"}


def test_one_chunk_per_report_is_kept_while_another_report_has_spares():
    builder = ContextBuilder()
    chunks = [
        chunk("RPT0001", 1, 0.99),
        chunk("RPT0001", 2, 0.98),
        chunk("RPT0001", 3, 0.97),
        chunk("RPT0002", 1, 0.50),
    ]
    counter = WordCountTokenizer()

    bundle, dropped = builder.build_chunks_within_budget(
        chunks, counter, max_tokens=200
    )

    # The weak single-chunk report survives; the multi-chunk report is trimmed.
    assert "RPT0002" in bundle.report_metadata
    assert not dropped or all(item.startswith("RPT0001") for item in dropped)


def test_a_single_oversized_chunk_fails_loudly_instead_of_being_truncated():
    builder = ContextBuilder()
    chunks = [chunk("RPT0001", 1, 0.9, words=500)]

    with pytest.raises(ContextBudgetError):
        builder.build_chunks_within_budget(
            chunks, WordCountTokenizer(), max_tokens=100
        )


def test_a_context_that_already_fits_is_untouched():
    builder = ContextBuilder()
    chunks = [chunk("RPT0001", 1, 0.9)]

    bundle, dropped = builder.build_chunks_within_budget(
        chunks, WordCountTokenizer(), max_tokens=5000
    )

    assert dropped == []
    assert list(bundle.alias_map.values()) == ["RPT0001"]


def test_the_pipeline_reports_dropped_chunks_in_the_debug_object():
    matches = [
        ChunkMatch(
            chunk_id=f"RPT0001:{index}",
            report_id="RPT0001",
            document=" ".join(["kata"] * 40),
            metadata=metadata("RPT0001"),
            distance=1.0 - score,
        )
        for index, score in ((1, 0.95), (2, 0.80))
    ]
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(matches=matches),
        llm=FakeLLM(
            json_responses=[
                '{"in_scope": true, "mode": "specific", "search_query": "x", "filters": {}}'
            ],
            text_responses=["Kendala utama adalah logistik [LAP1]."],
        ),
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
        max_context_tokens=90,
    )

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.status is AnswerStatus.ANSWERED
    assert result.debug.context_dropped_chunk_ids == ["RPT0001:2"]
    assert result.debug.token_usage["total_tokens"] >= 0
