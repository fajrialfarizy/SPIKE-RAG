"""
Retrieval determinism (fix pack K1c).

An identical query must produce the identical ordered chunk set, even when the
vector store returns equal-distance hits in a different order. The default-suite
test drives that case with a double that flips its result order on every call; the
slow test repeats the real embedder and the dummy corpus twice.
"""

from pathlib import Path

import pytest

import config
from core.knowledge_base import ChunkMatch
from core.retriever import Retriever
from enums import QueryMode
from tests.stubs import FilteringKnowledgeBase, RecordingEmbedder, _matches_where


def chunk(report_id, level="Finalis", score=0.9, section="evaluasi_tim", part=1):
    metadata = {
        "report_id": report_id,
        "competition_name": f"Kompetisi {report_id}",
        "competition_type": "Business Plan",
        "competition_year": 2023,
        "achievement_level": level,
        "section_type": section,
        "chunk_id": f"{report_id}:{section}:0{part}",
    }
    return ChunkMatch(
        chunk_id=metadata["chunk_id"],
        report_id=report_id,
        document=f"Isi {report_id}.",
        metadata=metadata,
        distance=1.0 - score,
    )


class FlippingKnowledgeBase(FilteringKnowledgeBase):
    """Returns its matches in reverse order on every second query call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flips = 0

    def query(self, query_embedding, where=None, n_results=1):
        self.query_calls.append(
            {
                "query_embedding": list(query_embedding),
                "where": where,
                "n_results": n_results,
            }
        )
        matching = [
            match for match in self.matches if _matches_where(match.metadata, where)
        ][:n_results]
        self.flips += 1
        return list(reversed(matching)) if self.flips % 2 == 0 else matching


def signature(chunks):
    return [
        (item.chunk_id, item.report_id, None if item.score is None else round(item.score, 6))
        for item in chunks
    ]


def flipping_retriever(matches):
    knowledge_base = FlippingKnowledgeBase(matches=matches, total=len(matches))
    return Retriever(knowledge_base, embedder=RecordingEmbedder())


def test_specific_retrieval_is_identical_when_the_store_order_flips():
    matches = [
        chunk("RPT0001", score=0.90),
        chunk("RPT0002", score=0.90),
        chunk("RPT0003", score=0.95),
    ]
    retriever = flipping_retriever(matches)

    first = retriever.retrieve("kendala tim", QueryMode.SPECIFIC, {})
    second = retriever.retrieve("kendala tim", QueryMode.SPECIFIC, {})

    assert signature(first) == signature(second)


def test_equal_scores_break_ties_by_report_id_then_chunk_id():
    matches = [
        chunk("RPT0003", score=0.90),
        chunk("RPT0001", score=0.90),
        chunk("RPT0002", score=0.90),
    ]
    retriever = flipping_retriever(matches)

    result = retriever.retrieve("kendala tim", QueryMode.SPECIFIC, {})

    assert [item.report_id for item in result] == ["RPT0001", "RPT0002", "RPT0003"]


def test_grouped_retrieval_is_identical_when_the_store_order_flips():
    matches = [chunk(f"RPT100{i}", level="Juara 1", score=0.90) for i in range(1, 4)]
    matches += [chunk(f"RPT200{i}", score=0.90) for i in range(1, 4)]
    retriever = flipping_retriever(matches)

    first, first_groups = retriever.retrieve_grouped(
        "perbandingan pembelajaran tim", QueryMode.CROSS_REPORT, {}, "achievement_group"
    )
    second, second_groups = retriever.retrieve_grouped(
        "perbandingan pembelajaran tim", QueryMode.CROSS_REPORT, {}, "achievement_group"
    )

    assert signature(first) == signature(second)
    assert first_groups == second_groups


@pytest.mark.slow
def test_the_real_embedder_returns_the_same_dummy_corpus_result_twice():
    from core.knowledge_base import KnowledgeBase
    from ingestion.embedder import E5Embedder

    if not Path(config.CHROMA_PERSIST_DIR).exists():
        pytest.skip("chroma_db is not present; run ingestion first")

    knowledge_base = KnowledgeBase()
    if knowledge_base.count_chunks() == 0:
        pytest.skip("the collection is empty; run ingestion first")

    retriever = Retriever(knowledge_base, embedder=E5Embedder())
    query = "perbandingan pembelajaran tim juara dan finalis"

    first, first_groups = retriever.retrieve_grouped(
        query, QueryMode.CROSS_REPORT, {}, "achievement_group"
    )
    second, second_groups = retriever.retrieve_grouped(
        query, QueryMode.CROSS_REPORT, {}, "achievement_group"
    )

    assert signature(first) == signature(second)
    assert first_groups == second_groups
    assert retriever.retrieve("kendala tim", QueryMode.SPECIFIC, {}) != []
