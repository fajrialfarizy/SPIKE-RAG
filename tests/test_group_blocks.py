"""Offline tests for explicit comparison groups in the answer context (G3)."""

import json

import pytest

import config
from core.context_builder import ContextBuilder
from core.knowledge_base import ChunkMatch
from core.pipeline import AnswerStatus, RagPipeline
from core.retriever import RetrievedChunk
from enums import QueryMode
from tests.stubs import FakeLLM, FilteringKnowledgeBase, RecordingEmbedder, WordCountTokenizer


def metadata(report_id, level="Finalis", competition_type="Business Plan"):
    return {
        "report_id": report_id,
        "competition_name": f"Kompetisi {report_id}",
        "competition_type": competition_type,
        "competition_year": 2023,
        "achievement_level": level,
        "section_type": "evaluasi_tim",
    }


def match(report_id, level="Finalis", score=0.9):
    meta = metadata(report_id, level)
    return ChunkMatch(
        chunk_id=f"{report_id}:evaluasi_tim:01",
        report_id=report_id,
        document=f"Isi {report_id} sebagai {level}.",
        metadata=meta,
        distance=1.0 - score,
    )


def chunk(report_id, index, score, group, words=40):
    return RetrievedChunk(
        chunk_id=f"{report_id}:{index}",
        report_id=report_id,
        document=" ".join(["kata"] * words),
        metadata=metadata(report_id),
        score=score,
        group_label=group,
    )


def cross_payload():
    return json.dumps(
        {
            "in_scope": True,
            "mode": "cross_report",
            "search_query": "perbandingan pembelajaran tim",
            "filters": {},
            "compare_by": "achievement_group",
        }
    )


def structured_payload():
    """One valid comparison answer citing both groups (fix pack J1)."""
    return json.dumps(
        {
            "groups": [
                {
                    "label": "juara",
                    "points": [
                        {"text": "Tim juara fokus pada validasi pasar.", "cite": ["LAP1"]}
                    ],
                },
                {
                    "label": "belum juara",
                    "points": [
                        {"text": "Tim finalis kekurangan waktu uji coba.", "cite": ["LAP2"]}
                    ],
                },
            ],
            "differences": [],
        }
    )


def build_pipeline(matches, structured=None):
    llm = FakeLLM(
        json_responses=[cross_payload()],
        structured_responses=[structured or structured_payload()],
        text_responses=[
            "Tim juara fokus pada validasi pasar [LAP1], tim finalis pada waktu [LAP2]."
        ],
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(matches=matches),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )
    return pipeline, llm


def test_each_retrieved_chunk_carries_its_group_label():
    pipeline, _ = build_pipeline(
        [match("RPT0001", "Juara 1"), match("RPT0009", "Finalis")]
    )

    result = pipeline.run("Bandingkan pembelajaran tim yang juara dan belum juara.")

    # retrieve_grouped tags chunks before de-duplication.
    from core.retriever import Retriever

    retriever = Retriever(
        FilteringKnowledgeBase(
            matches=[match("RPT0001", "Juara 1"), match("RPT0009", "Finalis")]
        ),
        embedder=RecordingEmbedder(),
    )
    chunks, _ = retriever.retrieve_grouped(
        "perbandingan", QueryMode.CROSS_REPORT, {}, "achievement_group"
    )
    assert {item.report_id: item.group_label for item in chunks} == {
        "RPT0001": "juara",
        "RPT0009": "belum juara",
    }


def test_the_answer_context_wraps_each_group_in_a_named_block():
    pipeline, llm = build_pipeline(
        [match("RPT0001", "Juara 1"), match("RPT0009", "Finalis")]
    )

    result = pipeline.run("Bandingkan pembelajaran tim yang juara dan belum juara.")

    assert result.status is AnswerStatus.ANSWERED
    prompt = llm.calls[1]["user_prompt"]
    assert '<kelompok nama="juara">' in prompt
    assert '<kelompok nama="belum juara">' in prompt
    assert 'kelompok="juara"' in prompt
    assert 'kelompok="belum juara"' in prompt
    assert prompt.index('<kelompok nama="juara">') < prompt.index(
        '<kelompok nama="belum juara">'
    )


def test_the_group_hint_states_the_membership_and_forbids_regrouping():
    pipeline, llm = build_pipeline(
        [match("RPT0001", "Juara 1"), match("RPT0009", "Finalis")]
    )

    pipeline.run("Bandingkan pembelajaran tim yang juara dan belum juara.")

    prompt = llm.calls[1]["user_prompt"]
    assert "jangan mengelompokkan ulang" in prompt
    assert '"Juara Harapan"' in prompt
    assert '"Most Aspiring dan penghargaan serupa"' in prompt
    assert '"Finalis"' in prompt
    assert "Keanggotaan:" in prompt


def test_every_award_level_is_inside_the_award_group_block():
    # (end-to-end group membership per level is tested below)
    award = config.COMPARISON_GROUPS["achievement_group"][0]["filter"][
        "achievement_level"
    ]
    assert "Juara Harapan" in award
    assert "Most Aspiring dan penghargaan serupa" in award
    assert config.COMPARISON_GROUPS["achievement_group"][1]["filter"][
        "achievement_level"
    ] == ("Finalis",)


@pytest.mark.parametrize(
    "level,label",
    [
        ("Juara 1", "juara"),
        ("Juara 2", "juara"),
        ("Juara 3", "juara"),
        ("Juara Harapan", "juara"),
        ("Most Aspiring dan penghargaan serupa", "juara"),
        ("Finalis", "belum juara"),
    ],
)
def test_every_achievement_level_is_retrieved_inside_its_group(level, label):
    from core.retriever import Retriever

    retriever = Retriever(
        FilteringKnowledgeBase(matches=[match("RPT0001", level)]),
        embedder=RecordingEmbedder(),
    )

    chunks, _ = retriever.retrieve_grouped(
        "perbandingan", QueryMode.CROSS_REPORT, {}, "achievement_group"
    )

    assert [item.group_label for item in chunks] == [label]


def test_group_aware_trimming_keeps_at_least_one_chunk_per_group():
    builder = ContextBuilder()
    counter = WordCountTokenizer()
    chunks = [
        chunk("RPT0001", 1, 0.95, "juara"),
        chunk("RPT0001", 2, 0.90, "juara"),
        chunk("RPT0001", 3, 0.85, "juara"),
        chunk("RPT0009", 1, 0.50, "belum juara"),
    ]
    limit = counter.count_tokens(builder.build_chunks([chunks[0], chunks[3]]).text)

    bundle, dropped = builder.build_chunks_within_budget(chunks, counter, limit)

    assert dropped
    assert '<kelompok nama="juara">' in bundle.text
    assert '<kelompok nama="belum juara">' in bundle.text
    kept = [item for item in chunks if item.chunk_id not in dropped]
    assert {item.group_label for item in kept} == {"juara", "belum juara"}
