"""
Offline tests for retrieval plumbing: where-filter translation, thresholding,
the per-report cap, the exact ID route and the catalog mode.
"""

import pytest

import config
from core.knowledge_base import ChunkMatch, StoredChunk
from core.retriever import (
    FILTER_FIELDS,
    RetrievalError,
    Retriever,
    build_where_filter,
)
from enums import QueryMode
from ingestion.embedder import E5Embedder
from tests.stubs import FakeKnowledgeBase, RecordingEmbedder


def match(report_id, chunk_id, distance):
    return ChunkMatch(
        chunk_id=chunk_id,
        report_id=report_id,
        document=f"isi {chunk_id}",
        metadata={
            "report_id": report_id,
            "competition_name": f"Kompetisi {report_id}",
            "competition_type": "Business Plan",
            "competition_year": 2024,
            "achievement_level": "Juara 1",
            "section_type": "evaluasi_tim",
        },
        distance=distance,
    )


def make_retriever(matches, total=None, threshold=config.RETRIEVAL_SIMILARITY_THRESHOLD):
    kb = FakeKnowledgeBase(matches=matches, total=total)
    embedder = RecordingEmbedder()
    return Retriever(kb, embedder=embedder, similarity_threshold=threshold), kb, embedder


# -- where filter ---------------------------------------------------------
def test_no_filters_means_no_where_clause():
    assert build_where_filter(None) is None
    assert build_where_filter({}) is None
    assert build_where_filter({field: [] for field in FILTER_FIELDS}) is None


def test_single_condition_has_no_and_operator():
    where = build_where_filter({"competition_type": ["Essay"]})

    assert where == {"competition_type": {"$in": ["Essay"]}}
    assert "$and" not in where


def test_multiple_conditions_use_and_with_in():
    where = build_where_filter(
        {
            "competition_type": ["Essay", "Business Plan"],
            "achievement_level": ["Finalis"],
        }
    )

    assert where == {
        "$and": [
            {"competition_type": {"$in": ["Business Plan", "Essay"]}},
            {"achievement_level": {"$in": ["Finalis"]}},
        ]
    }


def test_year_filter_is_an_integer():
    where = build_where_filter({"competition_year": ["2024", 2023]})

    assert where == {"competition_year": {"$in": [2023, 2024]}}
    assert all(isinstance(value, int) for value in where["competition_year"]["$in"])


def test_all_three_conditions_produce_three_branches():
    where = build_where_filter(
        {
            "competition_type": ["Essay"],
            "competition_year": [2024],
            "achievement_level": ["Juara 1"],
        }
    )

    assert len(where["$and"]) == 3


# -- semantic retrieval ---------------------------------------------------
def test_default_embedder_is_the_ingestion_embedder():
    retriever = Retriever(FakeKnowledgeBase())

    assert isinstance(retriever.embedder, E5Embedder)
    assert retriever.embedder.model_name == config.EMBEDDING_MODEL_NAME


def test_query_uses_the_ingestion_prefix_exactly_once():
    retriever, kb, embedder = make_retriever([match("RPT0001", "RPT0001:evaluasi_tim:01", 0.1)])

    retriever.retrieve("kendala presentasi", QueryMode.SPECIFIC, {})

    assert embedder.queries == ["query: kendala presentasi"]
    assert embedder.model_name == config.EMBEDDING_MODEL_NAME


def test_filters_are_passed_to_chroma():
    retriever, kb, _ = make_retriever([], total=10)

    retriever.retrieve(
        "kendala", QueryMode.SPECIFIC, {"competition_type": ["Essay"]}
    )

    assert kb.query_calls[0]["where"] == {"competition_type": {"$in": ["Essay"]}}


def test_candidates_below_threshold_are_dropped():
    retriever, _, _ = make_retriever(
        [
            match("RPT0001", "RPT0001:evaluasi_tim:01", 0.10),  # 0.90
            match("RPT0002", "RPT0002:evaluasi_tim:01", 0.60),  # 0.40 < 0.50
        ]
    )

    selected = retriever.retrieve("pertanyaan", QueryMode.SPECIFIC, {})

    assert [chunk.report_id for chunk in selected] == ["RPT0001"]
    assert selected[0].score == pytest.approx(0.90)


def test_specific_mode_caps_two_chunks_per_report():
    # Fix pack H4b: two chunks per report, so at least three distinct reports can
    # appear in a six-chunk specific answer.
    matches = [
        match("RPT0001", f"RPT0001:evaluasi_tim:{index:02d}", 0.10 + index * 0.01)
        for index in range(1, 6)
    ]
    retriever, _, _ = make_retriever(matches)

    selected = retriever.retrieve("pertanyaan", QueryMode.SPECIFIC, {})

    assert config.RETRIEVAL_SPECIFIC_MAX_CHUNKS_PER_REPORT == 2
    assert len(selected) == config.RETRIEVAL_SPECIFIC_MAX_CHUNKS_PER_REPORT
    assert {chunk.report_id for chunk in selected} == {"RPT0001"}
    # Highest scores win.
    assert [chunk.chunk_id for chunk in selected] == [
        "RPT0001:evaluasi_tim:01",
        "RPT0001:evaluasi_tim:02",
    ]


def test_specific_mode_fills_k_across_reports():
    matches = [
        match("RPT0001", "RPT0001:evaluasi_tim:01", 0.05),
        match("RPT0001", "RPT0001:evaluasi_tim:02", 0.06),
        match("RPT0001", "RPT0001:evaluasi_tim:03", 0.07),
        match("RPT0001", "RPT0001:evaluasi_tim:04", 0.08),
        match("RPT0002", "RPT0002:executive_summary:01", 0.09),
    ]
    retriever, _, _ = make_retriever(matches)

    selected = retriever.retrieve("pertanyaan", QueryMode.SPECIFIC, {})

    assert [chunk.report_id for chunk in selected] == [
        "RPT0001",
        "RPT0001",
        "RPT0002",
    ]


def test_cross_report_mode_caps_two_chunks_per_report():
    matches = [
        match("RPT0001", f"RPT0001:evaluasi_tim:{index:02d}", 0.10 + index * 0.01)
        for index in range(1, 5)
    ] + [
        match("RPT0002", f"RPT0002:evaluasi_tim:{index:02d}", 0.20 + index * 0.01)
        for index in range(1, 4)
    ]
    retriever, _, _ = make_retriever(matches)

    selected = retriever.retrieve("bandingkan", QueryMode.CROSS_REPORT, {})

    per_report = {}
    for chunk in selected:
        per_report[chunk.report_id] = per_report.get(chunk.report_id, 0) + 1
    assert all(count <= 2 for count in per_report.values())
    assert len(selected) == 4


def test_candidate_pool_is_three_times_k():
    retriever, kb, _ = make_retriever([], total=100)

    retriever.retrieve("pertanyaan", QueryMode.SPECIFIC, {})
    assert kb.query_calls[0]["n_results"] == (
        config.RETRIEVAL_SPECIFIC_K * config.RETRIEVAL_CANDIDATE_MULTIPLIER
    )

    retriever.retrieve("pertanyaan", QueryMode.CROSS_REPORT, {})
    assert kb.query_calls[1]["n_results"] == (
        config.RETRIEVAL_CROSS_REPORT_K * config.RETRIEVAL_CANDIDATE_MULTIPLIER
    )


def test_empty_collection_returns_nothing_without_embedding_a_query():
    retriever, _, embedder = make_retriever([], total=0)

    assert retriever.retrieve("pertanyaan", QueryMode.SPECIFIC, {}) == []
    assert embedder.queries == []


def test_catalog_mode_cannot_be_retrieved_semantically():
    retriever, _, _ = make_retriever([])

    with pytest.raises(RetrievalError):
        retriever.retrieve("daftar laporan", QueryMode.CATALOG, {})


# -- ID route -------------------------------------------------------------
def test_id_route_fetches_by_filter_and_never_embeds():
    chunks = [
        StoredChunk("RPT0003:executive_summary:01", "RPT0003", "isi a", {"report_id": "RPT0003"}),
        StoredChunk("RPT0003:evaluasi_tim:01", "RPT0003", "isi b", {"report_id": "RPT0003"}),
        StoredChunk("RPT0009:evaluasi_tim:01", "RPT0009", "isi c", {"report_id": "RPT0009"}),
    ]
    kb = FakeKnowledgeBase(chunks=chunks)
    embedder = RecordingEmbedder()
    retriever = Retriever(kb, embedder=embedder)

    selected = retriever.retrieve_by_report_ids(["RPT0003"])

    assert [chunk.chunk_id for chunk in selected] == [
        "RPT0003:executive_summary:01",
        "RPT0003:evaluasi_tim:01",
    ]
    assert all(chunk.score is None for chunk in selected)
    assert kb.id_calls == [["RPT0003"]]
    assert embedder.queries == []


# -- catalog --------------------------------------------------------------
def test_catalog_dedupes_reports_and_respects_the_cap():
    metadatas = []
    for report_id in ("RPT0003", "RPT0001", "RPT0002"):
        for chunk in range(4):
            metadatas.append(
                {
                    "report_id": report_id,
                    "competition_name": f"Kompetisi {report_id}",
                    "competition_type": "Essay",
                    "competition_year": 2024,
                    "achievement_level": "Finalis",
                    "section_type": "evaluasi_tim",
                    "chunk_id": f"{report_id}:evaluasi_tim:{chunk:02d}",
                }
            )
    kb = FakeKnowledgeBase(metadatas=metadatas)
    embedder = RecordingEmbedder()
    retriever = Retriever(kb, embedder=embedder)

    reports = retriever.list_catalog(max_reports=2)

    assert [report["report_id"] for report in reports] == ["RPT0001", "RPT0002"]
    assert embedder.queries == []
    # Unused ingestion fields never leak into the catalog payload.
    assert "chunk_id" not in reports[0]
    assert set(reports[0]) == {
        "report_id",
        "competition_name",
        "competition_type",
        "competition_year",
        "achievement_level",
    }
