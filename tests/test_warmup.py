"""
Offline tests for the embedder warm-up and the per-call latency evidence (F5).
"""

import pytest

import config
from core.pipeline import AnswerStatus, RagPipeline
from core.warmup import WarmupError, warm_up_embedder
from tests.stubs import FakeKnowledgeBase, FakeLLM, RecordingEmbedder, WordCountTokenizer


class TimedEmbedder(RecordingEmbedder):
    """Recording embedder whose clock advances by a fixed amount."""

    def __init__(self, step_ms=5.0):
        super().__init__()
        self.step_ms = step_ms

    def embed_query(self, query):
        self.queries.append(f"{config.EMBEDDING_QUERY_PREFIX}{query}")
        return [0.0] * config.EMBEDDING_DIMENSION


def test_warm_up_uses_the_production_query_path_once():
    embedder = RecordingEmbedder()
    clock_values = iter([100.0, 100.25])

    elapsed = warm_up_embedder(embedder, clock=lambda: next(clock_values))

    assert elapsed == pytest.approx(250.0)
    assert embedder.queries == [
        f"{config.EMBEDDING_QUERY_PREFIX}{config.EMBEDDING_WARMUP_QUERY}"
    ]


def test_a_wrong_dimension_during_warm_up_fails_loudly():
    class BrokenEmbedder:
        def embed_query(self, query):
            return [0.0] * 10

    with pytest.raises(WarmupError):
        warm_up_embedder(BrokenEmbedder())


def test_the_debug_object_carries_latency_and_token_evidence():
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(),
        llm=FakeLLM(json_responses=["not json"]),
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.status is AnswerStatus.NOT_FOUND
    assert result.debug.embedding_latency_ms >= 0.0
    assert result.debug.calls[0]["name"] == "analyzer"
    assert "latency_ms" in result.debug.calls[0]
    assert "usage" in result.debug.calls[0]
    assert result.debug.token_usage == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
