"""
Offline tests for the analyzer (LLM call #1): tolerant parsing, strict validation
against the enum sets, and the mandated fallback on broken output.
"""

import json
import logging

import config
from core.analyzer import (
    ALLOWED_ACHIEVEMENT_LEVELS,
    ALLOWED_COMPETITION_TYPES,
    Analyzer,
    fallback_result,
    parse_analyzer_output,
    validate_analysis,
)
from core.llm_client import LLMUnavailableError
from enums import QueryMode
from tests.stubs import FakeLLM

QUESTION = "Apa saja kendala tim pada kompetisi business plan 2024?"


def valid_payload(**overrides):
    payload = {
        "in_scope": True,
        "mode": "specific",
        "search_query": "kendala business plan 2024",
        "filters": {
            "competition_type": ["Business Plan"],
            "competition_year": [2024],
            "achievement_level": ["Finalis"],
        },
    }
    payload.update(overrides)
    return payload


def test_valid_payload_is_accepted():
    result = validate_analysis(valid_payload(), QUESTION)

    assert result.in_scope is True
    assert result.mode is QueryMode.SPECIFIC
    assert result.search_query == "kendala business plan 2024"
    assert result.filters == {
        "competition_type": ["Business Plan"],
        "competition_year": [2024],
        "achievement_level": ["Finalis"],
    }
    assert result.used_fallback is False
    assert result.dropped_invalid_values == 0


def test_invalid_enum_values_are_dropped_and_counted():
    result = validate_analysis(
        valid_payload(
            filters={
                "competition_type": ["Business Plan", "Hackathon", "Lainnya"],
                "competition_year": [2024, "dua ribu", 1800],
                "achievement_level": ["Juara 1", "Juara 9"],
            }
        ),
        QUESTION,
    )

    assert result.filters["competition_type"] == ["Business Plan"]
    assert result.filters["competition_year"] == [2024]
    assert result.filters["achievement_level"] == ["Juara 1"]
    assert result.dropped_invalid_values == 5


def test_year_strings_are_cast_to_int_and_deduplicated():
    result = validate_analysis(
        valid_payload(filters={"competition_year": ["2024", 2024, 2023]}), QUESTION
    )

    assert result.filters["competition_year"] == [2023, 2024]
    assert all(isinstance(year, int) for year in result.filters["competition_year"])


def test_empty_filters_are_omitted_entirely():
    result = validate_analysis(
        valid_payload(filters={"competition_type": [], "competition_year": []}), QUESTION
    )

    assert result.filters == {}


def test_invalid_mode_falls_back_to_specific():
    result = validate_analysis(valid_payload(mode="bertanya"), QUESTION)

    assert result.mode is QueryMode.SPECIFIC
    assert result.dropped_invalid_values >= 1


def test_valid_cross_report_and_catalog_modes_are_preserved():
    assert validate_analysis(valid_payload(mode="cross_report"), QUESTION).mode is (
        QueryMode.CROSS_REPORT
    )
    assert validate_analysis(valid_payload(mode="catalog"), QUESTION).mode is (
        QueryMode.CATALOG
    )


def test_oversized_search_query_is_capped():
    result = validate_analysis(
        valid_payload(search_query="kata " * 500), QUESTION
    )

    assert len(result.search_query) <= config.ANALYZER_MAX_SEARCH_QUERY_CHARS


def test_empty_search_query_falls_back_to_the_question():
    result = validate_analysis(valid_payload(search_query="   "), QUESTION)

    assert result.search_query == QUESTION


def test_non_bool_in_scope_stays_in_scope():
    result = validate_analysis(valid_payload(in_scope="no"), QUESTION)

    assert result.in_scope is True
    assert result.dropped_invalid_values >= 1


def test_out_of_scope_flag_is_preserved():
    result = validate_analysis(valid_payload(in_scope=False), QUESTION)

    assert result.in_scope is False


def test_filters_that_are_not_a_dict_are_dropped():
    result = validate_analysis(valid_payload(filters=["Business Plan"]), QUESTION)

    assert result.filters == {}


def test_parse_handles_fenced_json_and_surrounding_text():
    raw = "```json\n" + json.dumps(valid_payload()) + "\n```"
    assert parse_analyzer_output(raw)["mode"] == "specific"

    raw = "Berikut hasilnya:\n" + json.dumps(valid_payload()) + "\nSelesai."
    assert parse_analyzer_output(raw)["in_scope"] is True


def test_broken_json_falls_back_without_raising(caplog):
    analyzer = Analyzer(FakeLLM(json_responses=['{"in_scope": true, "mode": ']))

    with caplog.at_level(logging.WARNING, logger="core_analyzer"):
        result = analyzer.analyze(QUESTION)

    assert result.in_scope is True
    assert result.mode is QueryMode.SPECIFIC
    assert result.filters == {}
    assert result.used_fallback is True
    assert result.search_query == QUESTION
    assert "invalid_json" in caplog.text
    # The question itself never reaches the log.
    assert QUESTION not in caplog.text


def test_non_object_json_falls_back():
    analyzer = Analyzer(FakeLLM(json_responses=["[1, 2, 3]"]))

    assert analyzer.analyze(QUESTION).used_fallback is True


def test_empty_analyzer_response_falls_back():
    analyzer = Analyzer(FakeLLM(json_responses=["   "]))

    assert analyzer.analyze(QUESTION).used_fallback is True


def test_provider_failure_falls_back():
    analyzer = Analyzer(FakeLLM(fail_with=LLMUnavailableError("provider down")))

    result = analyzer.analyze(QUESTION)

    assert result.used_fallback is True
    assert result.in_scope is True
    assert result.mode is QueryMode.SPECIFIC


def test_analyzer_uses_low_reasoning_and_json_mode():
    llm = FakeLLM(json_responses=[json.dumps(valid_payload())])
    Analyzer(llm).analyze(QUESTION)

    call = llm.calls[0]
    assert call["json_mode"] is True
    assert call["temperature"] == config.LLM_TEMPERATURE_ANALYZER == 0.0
    assert call["reasoning_effort"] == config.LLM_REASONING_EFFORT_ANALYZER == "low"
    assert call["max_tokens"] == config.LLM_ANALYZER_MAX_OUTPUT_TOKENS
    assert "<pertanyaan>" in call["user_prompt"]
    assert QUESTION in call["user_prompt"]


def test_fallback_result_shape_is_fixed():
    result = fallback_result(QUESTION, reason="test")

    assert result.in_scope is True
    assert result.mode is QueryMode.SPECIFIC
    assert result.filters == {}
    assert len(result.search_query) <= config.ANALYZER_MAX_SEARCH_QUERY_CHARS


def test_allowed_enum_sets_exclude_lainnya():
    assert "Lainnya" not in ALLOWED_COMPETITION_TYPES
    assert "Finalis" in ALLOWED_ACHIEVEMENT_LEVELS
