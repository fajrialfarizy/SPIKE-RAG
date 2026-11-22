"""
Offline tests for group-balanced comparison retrieval (fix pack F3).
"""

import json

import pytest

import config
from core.analyzer import validate_analysis
from core.knowledge_base import ChunkMatch
from core.pipeline import AnswerStatus, RagPipeline
from core.retriever import Retriever, RetrievalError, merge_filter_dicts
from enums import AWARD_WINNER_LEVELS, QueryMode
from tests.stubs import FakeLLM, FilteringKnowledgeBase, RecordingEmbedder, WordCountTokenizer

pytestmark = pytest.mark.usefixtures("legacy_answer_contract")


def metadata(report_id, level="Finalis", competition_type="Business Plan"):
    return {
        "report_id": report_id,
        "competition_name": f"Kompetisi {report_id}",
        "competition_type": competition_type,
        "competition_year": 2023,
        "achievement_level": level,
        "section_type": "evaluasi_tim",
    }


def match(report_id, level="Finalis", competition_type="Business Plan", score=0.9):
    meta = metadata(report_id, level, competition_type)
    return ChunkMatch(
        chunk_id=f"{report_id}:{level}:{competition_type}:{score}",
        report_id=report_id,
        document=f"Isi {report_id} sebagai {level} pada {competition_type}.",
        metadata=meta,
        distance=1.0 - score,
    )


def cross_payload(compare_by="achievement_group", **filters):
    return json.dumps(
        {
            "in_scope": True,
            "mode": "cross_report",
            "search_query": "perbandingan pembelajaran tim",
            "filters": filters,
            "compare_by": compare_by,
        }
    )


# -- analyzer contract ----------------------------------------------------
def test_compare_by_is_accepted_only_for_cross_report():
    accepted = validate_analysis(
        json.loads(cross_payload("achievement_group")), "Bandingkan tim juara dan belum juara."
    )
    assert accepted.compare_by == "achievement_group"

    rejected = validate_analysis(
        {
            "in_scope": True,
            "mode": "specific",
            "search_query": "x",
            "filters": {},
            "compare_by": "achievement_group",
        },
        "Apa kendala tim?",
    )
    assert rejected.compare_by is None
    assert "compare_by:not_cross_report:achievement_group" in rejected.dropped_reasons


def test_an_unknown_grouping_is_dropped_with_a_reason():
    result = validate_analysis(
        json.loads(cross_payload("winner_vs_loser")),
        "Bandingkan tim juara dan belum juara.",
    )

    assert result.compare_by is None
    assert "compare_by:unknown:winner_vs_loser" in result.dropped_reasons


def test_the_grouping_value_tolerates_spacing_and_case():
    result = validate_analysis(
        json.loads(cross_payload("Achievement Group")),
        "Bandingkan tim juara dan belum juara.",
    )

    assert result.compare_by == "achievement_group"


def test_the_award_group_covers_exactly_the_award_levels():
    group = config.COMPARISON_GROUPS["achievement_group"][0]
    assert set(group["filter"]["achievement_level"]) == set(AWARD_WINNER_LEVELS)
    assert config.COMPARISON_GROUPS["achievement_group"][1]["filter"][
        "achievement_level"
    ] == ("Finalis",)


def test_merge_filter_dicts_intersects_shared_fields():
    merged = merge_filter_dicts(
        {"achievement_level": ["Finalis"], "competition_year": [2023]},
        {"achievement_level": ["Juara 1", "Finalis"]},
    )

    assert merged["achievement_level"] == ["Finalis"]
    assert merged["competition_year"] == [2023]

    contradictory = merge_filter_dicts(
        {"achievement_level": ["Juara 1"]},
        {"achievement_level": ["Finalis"]},
    )
    assert contradictory["achievement_level"] == []


# -- grouped retrieval ----------------------------------------------------
def build_retriever(matches):
    return Retriever(
        knowledge_base=FilteringKnowledgeBase(matches=matches),
        embedder=RecordingEmbedder(),
    )


def test_both_groups_are_represented_in_a_comparison():
    matches = [
        match(f"RPT000{i}", level="Juara 2") for i in range(1, 4)
    ] + [match("RPT0009", level="Finalis")]

    chunks, groups = build_retriever(matches).retrieve_grouped(
        "perbandingan pembelajaran", QueryMode.CROSS_REPORT, {}, "achievement_group"
    )

    levels = {chunk.report_id: chunk.metadata["achievement_level"] for chunk in chunks}
    assert groups == [
        {"label": "meraih penghargaan", "count": 3},
        {"label": "belum juara (finalis)", "count": 1},
    ]
    assert levels["RPT0001"] == "Juara 2"
    assert levels["RPT0009"] == "Finalis"


def test_no_group_dominates_because_each_group_is_capped_by_reports():
    matches = [match(f"RPT1{i:03d}", level="Juara 1") for i in range(8)] + [
        match("RPT2001", level="Finalis"),
        match("RPT2002", level="Finalis"),
        match("RPT2003", level="Finalis"),
    ]

    chunks, groups = build_retriever(matches).retrieve_grouped(
        "perbandingan", QueryMode.CROSS_REPORT, {}, "achievement_group"
    )

    limit = config.COMPARE_REPORTS_PER_GROUP
    counts = [group["count"] for group in groups]
    assert counts == [limit, limit]
    assert len(chunks) == 2 * limit
    assert max(counts) < len(matches)


def test_an_empty_group_is_reported_explicitly():
    matches = [match(f"RPT1{i:03d}", level="Juara 1") for i in range(3)]

    chunks, groups = build_retriever(matches).retrieve_grouped(
        "perbandingan", QueryMode.CROSS_REPORT, {}, "achievement_group"
    )

    assert chunks
    assert groups[1] == {"label": "belum juara (finalis)", "count": 0}


def test_a_contradictory_filter_makes_the_group_empty_without_dropping_it():
    matches = [match("RPT0001", level="Juara 1"), match("RPT0002", level="Finalis")]

    chunks, groups = build_retriever(matches).retrieve_grouped(
        "perbandingan",
        QueryMode.CROSS_REPORT,
        {"achievement_level": ["Finalis"]},
        "achievement_group",
    )

    assert groups == [
        {"label": "meraih penghargaan", "count": 0},
        {"label": "belum juara (finalis)", "count": 1},
    ]
    assert [chunk.report_id for chunk in chunks] == ["RPT0002"]


def test_competition_type_grouping_uses_one_group_per_type():
    matches = [
        match("RPT0001", competition_type="Business Plan"),
        match("RPT0002", competition_type="Business Case"),
    ]

    chunks, groups = build_retriever(matches).retrieve_grouped(
        "bandingkan", QueryMode.CROSS_REPORT, {}, "competition_type"
    )

    assert [group["count"] for group in groups] == [1, 1, 0, 0]
    assert {chunk.metadata["competition_type"] for chunk in chunks} == {
        "Business Plan",
        "Business Case",
    }


def test_grouping_rejects_a_non_comparison_mode():
    with pytest.raises(RetrievalError):
        build_retriever([]).retrieve_grouped(
            "x", QueryMode.SPECIFIC, {}, "achievement_group"
        )


# -- pipeline -------------------------------------------------------------
def structured_payload():
    """One valid comparison answer citing both groups (fix pack J1)."""
    return json.dumps(
        {
            "groups": {
                "g1": [
                    {"text": "Tim juara fokus pada validasi pasar.", "cite": ["LAP1"]}
                ],
                "g2": [
                    {"text": "Tim finalis kekurangan waktu uji coba.", "cite": ["LAP2"]}
                ],
            },
            "contrasts": [],
        }
    )


def test_the_pipeline_passes_the_group_status_to_the_answer_model():
    matches = [
        match("RPT0001", level="Juara 1"),
        match("RPT0002", level="Finalis"),
    ]
    llm = FakeLLM(
        json_responses=[cross_payload("achievement_group")],
        structured_responses=[structured_payload()],
        text_responses=["Tim juara fokus pada validasi pasar [LAP1], tim finalis pada waktu [LAP2]."],
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(matches=matches),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run("Bandingkan pembelajaran tim yang juara dan belum juara.")

    assert result.status is AnswerStatus.ANSWERED
    assert result.debug.compare_by == "achievement_group"
    assert [group["count"] for group in result.debug.groups] == [1, 1]
    assert result.debug.route == "semantic_grouped"

    answer_prompt = llm.calls[1]["user_prompt"]
    assert "Kelompok pembanding yang dipakai" in answer_prompt
    assert "- meraih penghargaan: 1 laporan" in answer_prompt
    assert "- belum juara (finalis): 1 laporan" in answer_prompt


def test_an_empty_group_is_not_in_the_answer_contract():
    matches = [match("RPT0001", level="Juara 1")]
    llm = FakeLLM(
        json_responses=[cross_payload("achievement_group")],
        structured_responses=[
            json.dumps(
                {
                    "groups": {
                        "g1": [
                            {"text": "Tim juara fokus pada validasi pasar.", "cite": ["LAP1"]}
                        ]
                    },
                    "contrasts": [],
                }
            )
        ],
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(matches=matches),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run("Bandingkan pembelajaran tim yang juara dan belum juara.")

    assert result.status is AnswerStatus.ANSWERED
    # Fix pack 9 (O2a): the group without documents is not in the schema, not in
    # the hint, and not in the answer.
    assert "tidak ada dokumen yang cocok" not in llm.calls[1]["user_prompt"]
    schema = llm.calls[1]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["groups"]["required"] == ["g1"]
    assert "belum juara" not in result.answer
