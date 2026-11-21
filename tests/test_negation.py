"""
Non-blocking negation metric (fix pack K2).

The metric only measures. These tests prove it counts the intended sentences,
keeps a legitimate sentence as a hit, and never changes an answer.
"""

import json

import config
from core.knowledge_base import ChunkMatch
from core.negation import negation_sentences, point_negation_counts, split_sentences
from core.pipeline import AnswerStatus, RagPipeline
from tests.stubs import FakeKnowledgeBase, FakeLLM, RecordingEmbedder


def test_a_negated_statement_about_a_team_is_counted():
    result = negation_sentences("Tim belum juara tidak melakukan simulasi intensif.")

    # One sentence counts once, however many markers it contains.
    assert result["count"] == 1
    assert result["samples"] == ["Tim belum juara tidak melakukan simulasi intensif."]


def test_a_legitimate_sentence_still_counts_and_blocks_nothing():
    result = negation_sentences("Tidak ada anggota yang pasif saat sesi tanya jawab.")

    assert result["count"] == 1
    assert "tidak ada anggota yang pasif" in result["samples"][0].lower()


def test_a_sentence_without_a_subject_marker_is_ignored():
    assert negation_sentences("Tidak tersedia.")["count"] == 0


def test_a_positive_sentence_is_not_counted():
    assert negation_sentences("Tim juara memvalidasi masalah lewat wawancara.")["count"] == 0


def test_every_configured_marker_is_recognized():
    for marker in config.NEGATION_MARKERS:
        text = f"Tim {marker} menyiapkan data pendukung."
        assert negation_sentences(text)["count"] == 1, marker


def test_bullets_are_split_into_sentences():
    text = "- Tim A kurang persiapan.\n- Laporan B tanpa data pendukung."
    assert len(split_sentences(text)) == 2
    assert negation_sentences(text)["count"] == 2


def test_only_five_samples_are_kept():
    text = " ".join("Tim tidak menyiapkan data." for _ in range(7))

    result = negation_sentences(text)

    assert result["count"] == 7
    assert len(result["samples"]) == config.NEGATION_SAMPLE_LIMIT


def test_per_point_counts_keep_order():
    assert point_negation_counts(
        ["Tim juara fokus validasi.", "Tim finalis tidak melakukan simulasi."]
    ) == [0, 1]


def test_the_metric_never_changes_the_answer():
    metadata = {
        "report_id": "RPT0002",
        "competition_name": "Kompetisi RPT0002",
        "competition_type": "Business Plan",
        "competition_year": 2023,
        "achievement_level": "Finalis",
        "section_type": "evaluasi_tim",
        "chunk_id": "RPT0002:evaluasi_tim:01",
    }
    llm = FakeLLM(
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
        text_responses=["Tim finalis tidak melakukan simulasi intensif [LAP1]."],
    )
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(
            matches=[
                ChunkMatch(
                    chunk_id=metadata["chunk_id"],
                    report_id="RPT0002",
                    document="Isi RPT0002.",
                    metadata=metadata,
                    distance=0.1,
                )
            ]
        ),
        llm=llm,
        embedder=RecordingEmbedder(),
    )

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.status is AnswerStatus.ANSWERED
    assert result.answer.startswith("Tim finalis tidak melakukan simulasi intensif [RPT0002].")
    assert result.debug.negation["count"] == 1
    assert result.debug.negation["samples"][0] in result.answer
