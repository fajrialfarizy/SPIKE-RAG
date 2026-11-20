"""
Offline tests for analyzer normalization, the deterministic extractor and the
analyzer debug payload (fix pack F2).

The two questions from the real run are used verbatim so the reproduction is the
same one reported to the supervisor.
"""

import json

import pytest

from core.analyzer import (
    Analyzer,
    extract_deterministic_filters,
    normalize_achievement_level,
    normalize_competition_type,
    validate_analysis,
)
from core.pipeline import AnswerStatus, RagPipeline
from tests.stubs import FakeKnowledgeBase, FakeLLM, RecordingEmbedder, WordCountTokenizer

BUSINESS_PLAN_QUESTION = (
    "Apa saja pengalaman tim Lab Prestasi ketika mengikuti kompetisi business plan?"
)
FINTECH_QUESTION = "Apakah ada pengalaman tim dalam business case bertema fintech?"


def payload(**filters):
    return {
        "in_scope": True,
        "mode": "specific",
        "search_query": "pengalaman tim",
        "filters": filters,
    }


# -- normalization --------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        ("Business Plan", "Business Plan"),
        ("business plan", "Business Plan"),
        ("  BUSINESS   PLAN!! ", "Business Plan"),
        ("bp", "Business Plan"),
        ("BP", "Business Plan"),
        ("rencana bisnis", "Business Plan"),
        ("case", "Business Case"),
        ("studi kasus", "Business Case"),
        ("esai", "Essay"),
        ("kti", "Karya Tulis Ilmiah"),
        ("Karya Tulis", "Karya Tulis Ilmiah"),
        ("Hackathon", None),
        ("Lainnya", None),
    ],
)
def test_competition_type_normalization(value, expected):
    assert normalize_competition_type(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Juara 1", "Juara 1"),
        ("juara i", "Juara 1"),
        ("JUARA PERTAMA", "Juara 1"),
        ("runner up", "Juara 2"),
        ("juara iii", "Juara 3"),
        ("harapan", "Juara Harapan"),
        ("most aspiring", "Most Aspiring dan penghargaan serupa"),
        ("finalist", "Finalis"),
        ("belum juara", "Finalis"),
        ("Juara 9", None),
    ],
)
def test_achievement_level_normalization(value, expected):
    assert normalize_achievement_level(value) == expected


def test_lowercase_model_values_survive_validation():
    result = validate_analysis(
        payload(competition_type=["business plan"], achievement_level=["finalis"]),
        "Apa kendala tim?",
    )

    assert result.filters["competition_type"] == ["Business Plan"]
    assert result.filters["achievement_level"] == ["Finalis"]
    assert result.dropped_reasons == []


def test_year_strings_are_accepted_and_out_of_range_years_are_dropped():
    result = validate_analysis(
        payload(competition_year=["2023", "1999", "2101", "dua ribu"]),
        "Apa kendala tim?",
    )

    assert result.filters["competition_year"] == [2023]
    assert "competition_year:out_of_range:1999" in result.dropped_reasons
    assert "competition_year:out_of_range:2101" in result.dropped_reasons
    assert "competition_year:not_a_year:dua ribu" in result.dropped_reasons


def test_unknown_values_are_recorded_with_a_reason():
    result = validate_analysis(
        payload(competition_type=["Hackathon"]), "Apa kendala tim?"
    )

    assert result.filters == {}
    assert result.dropped_reasons == ["competition_type:unknown:hackathon"]


# -- deterministic extractor ---------------------------------------------
def test_type_keywords_are_extracted_from_the_question():
    assert extract_deterministic_filters(BUSINESS_PLAN_QUESTION) == {
        "competition_type": ["Business Plan"]
    }
    assert extract_deterministic_filters(FINTECH_QUESTION) == {
        "competition_type": ["Business Case"]
    }
    assert extract_deterministic_filters("Ceritakan pengalaman lomba KTI.") == {
        "competition_type": ["Karya Tulis Ilmiah"]
    }


def test_the_year_is_extracted_from_the_question():
    assert extract_deterministic_filters("Apa kendala tim pada kompetisi tahun 2023?") == {
        "competition_year": [2023]
    }


@pytest.mark.parametrize(
    "question",
    [
        "Tim mendapat hadiah Rp 20.230.000 pada kompetisi itu.",
        "Dana sebesar 2023 juta rupiah dialokasikan untuk proposal.",
        "Anggaran 2024 ribu dipakai untuk transportasi.",
        "Kompetisi tahun 1999 dan 2101 tidak relevan.",
        "Tidak ada angka tahun di sini.",
    ],
)
def test_a_number_that_is_not_a_competition_year_is_ignored(question):
    assert "competition_year" not in extract_deterministic_filters(question)


def test_the_extractor_never_infers_an_achievement_level():
    filters = extract_deterministic_filters(
        "Bandingkan tim yang juara dengan tim yang belum juara."
    )

    assert "achievement_level" not in filters


def test_deterministic_and_llm_filters_are_unioned():
    result = validate_analysis(
        payload(competition_type=["Business Plan"], competition_year=[2022]),
        "Bandingkan business case tahun 2023 dengan laporan sebelumnya.",
    )

    assert result.filters["competition_type"] == ["Business Case", "Business Plan"]
    assert result.filters["competition_year"] == [2022, 2023]
    assert result.deterministic_filters == {
        "competition_type": ["Business Case"],
        "competition_year": [2023],
    }


# -- debug payload --------------------------------------------------------
def test_the_pipeline_debug_object_exposes_the_analyzer_evidence():
    raw = json.dumps(
        payload(competition_type=["business plan"], competition_year=["2023"])
    )
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(),
        llm=FakeLLM(json_responses=[raw]),
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run(BUSINESS_PLAN_QUESTION)

    assert result.status is AnswerStatus.NOT_FOUND  # no chunks in the fake
    analyzer = result.debug.analyzer
    assert analyzer["raw_output"] == raw
    assert analyzer["filters_before_validation"]["competition_type"] == [
        "business plan"
    ]
    assert analyzer["filters"]["competition_type"] == ["Business Plan"]
    assert analyzer["deterministic_filters"]["competition_type"] == ["Business Plan"]
    assert analyzer["used_fallback"] is False
    assert "latency_ms" in analyzer
    assert analyzer["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }


def test_a_fallback_is_visible_in_the_debug_object():
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(),
        llm=FakeLLM(json_responses=["not json at all"]),
        embedder=RecordingEmbedder(),
    )

    result = pipeline.run(BUSINESS_PLAN_QUESTION)

    assert result.debug.used_fallback is True
    assert result.debug.analyzer["error_type"] == "invalid_json"
    assert result.debug.analyzer["raw_output"] == "not json at all"


def test_the_analyzer_prompt_carries_few_shot_examples():
    from core import prompts

    text = prompts.ANALYZER_SYSTEM_PROMPT
    assert text.count("Pertanyaan:") >= 6
    assert "business plan" in text.lower()
    assert "Bandingkan pembelajaran tim yang juara" in text
