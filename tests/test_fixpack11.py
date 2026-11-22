"""
Offline tests for fix pack 11.

S1: the HTTP 400 is diagnosed offline. The provider error body is parsed and
    sanitized, the documented Groq strict-schema rules are enforced by
    ``core.schema_check``, and the contrast payload is available in two shapes
    selected by config only ("pick" with string enums, or the "sides" shape that
    is known to work on the real model).
S2: failure semantics. A non-retryable 4xx releases its token reservation,
    429/5xx/timeouts keep it, a failed call appears in model_calls, and the
    offline replay shows what the real retry_after=60 came from.
S3: the support_overlap argument bug is fixed and a missing cited text is
    reported as such.
S5: both comparison questions run end to end through the pipeline in both
    contrast modes.

FakeLLM and the other doubles come from tests/stubs.py: no network, no API key.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from core import compare_answer, schema_check, structured_answer, wording
from core.knowledge_base import ChunkMatch
from core.llm_client import (
    GroqLLMClient,
    LLMConcurrencyLimiter,
    LLMRequestError,
    LLMUnavailableError,
    TokenUsage,
)
from core.pipeline import AnswerStatus, RagPipeline
from core.rate_limiter import TokenAwareRateLimiter
from tests.stubs import (
    FakeLLM,
    FilteringKnowledgeBase,
    RecordingEmbedder,
    WordCountTokenizer,
)


# -- helpers ---------------------------------------------------------------
class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


class _ProviderError(Exception):
    def __init__(self, status, body=None, headers=None):
        super().__init__("provider rejected the request")
        self.status_code = status
        self.body = body or {}
        self.response = SimpleNamespace(headers=headers or {})


class _Client:
    def __init__(self, error):
        self.error = error
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        self.calls = 0

    def _create(self, **kwargs):
        self.calls += 1
        raise self.error


def _limiter(clock):
    return TokenAwareRateLimiter(
        limits={
            "tokens_per_minute": 100_000,
            "tokens_per_day": 1_000_000,
            "requests_per_minute": 60,
            "requests_per_day": 1000,
        },
        clock=clock,
        sleep=clock.advance,
        token_counter=WordCountTokenizer(),
    )


def _groq_client(error, limiter):
    return GroqLLMClient(
        api_key="test-key",
        client=_Client(error),
        limiter=LLMConcurrencyLimiter(1),
        token_limiter=limiter,
        clock=limiter._clock,
        sleep_fn=limiter._sleep,
    )


def _metadata(report_id, level="Juara 1", competition_type="Business Plan"):
    return {
        "report_id": report_id,
        "competition_name": f"Kompetisi {report_id}",
        "competition_type": competition_type,
        "competition_year": 2023,
        "achievement_level": level,
        "section_type": "evaluasi_tim",
        "chunk_id": f"{report_id}:evaluasi_tim:01",
    }


def _match(report_id, level="Juara 1", competition_type="Business Plan"):
    meta = _metadata(report_id, level, competition_type)
    return ChunkMatch(
        chunk_id=meta["chunk_id"],
        report_id=report_id,
        document=f"Isi {report_id}.",
        metadata=meta,
        distance=0.1,
    )


def _analyzer(mode="specific", compare_by=None):
    return json.dumps(
        {
            "in_scope": True,
            "mode": mode,
            "search_query": "pengalaman tim",
            "filters": {},
            "compare_by": compare_by,
        }
    )


def _pick_payload():
    return {
        "groups": {
            "g1": [{"text": "Validasi lewat wawancara.", "cite": ["LAP1"]}],
            "g2": [{"text": "Data sekunder dominan.", "cite": ["LAP2"]}],
        },
        "contrasts": [
            {"dimension": "Kedalaman validasi", "pick": {"g1": "1", "g2": 1}}
        ],
    }


ALIAS_MAP = {"LAP1": "RPT0001", "LAP2": "RPT0002"}
REPORT_GROUP = {"RPT0001": "Business Plan", "RPT0002": "Business Case"}
GROUP_KEYS = {"g1": "Business Plan", "g2": "Business Case"}


def _validate(payload, mode="pick", alias_map=None, report_group=None, retrieved=None):
    return compare_answer.validate_payload(
        payload,
        group_keys=GROUP_KEYS,
        alias_map=alias_map or ALIAS_MAP,
        report_group=report_group or REPORT_GROUP,
        retrieved=list(retrieved or ["RPT0001", "RPT0002"]),
        mode=mode,
    )


# -- S1b/S1c: the documented schema subset --------------------------------
def test_every_schema_the_app_sends_is_within_the_documented_subset():
    for name, response_format in schema_check.app_response_formats().items():
        assert schema_check.check_response_format(response_format) == [], name


def test_the_pick_value_is_a_documented_string_enum():
    schema = compare_answer.build_schema(["g1", "g2"])
    pick = schema["properties"]["contrasts"]["items"]["properties"]["pick"]

    assert pick["properties"]["g1"] == {
        "type": "string",
        "enum": ["1", "2", "3"],
    }
    assert "minimum" not in pick["properties"]["g1"]
    assert schema_check.check_schema(schema) == []


def test_a_missing_required_property_is_rejected():
    schema = structured_answer.build_schema()
    schema["required"] = ["found"]

    violations = schema_check.check_schema(schema)

    assert any("requires every property" in item for item in violations)


def test_an_open_object_is_rejected():
    schema = structured_answer.build_schema()
    schema["additionalProperties"] = True

    violations = schema_check.check_schema(schema)

    assert any("additionalProperties: false" in item for item in violations)


def test_an_undocumented_keyword_is_rejected():
    schema = structured_answer.build_schema()
    schema["properties"]["points"]["minItems"] = 1

    violations = schema_check.check_schema(schema)

    assert any("minItems" in item for item in violations)


def test_an_array_without_items_is_rejected():
    schema = {"type": "object", "properties": {"x": {"type": "array"}},
              "required": ["x"], "additionalProperties": False}

    violations = schema_check.check_schema(schema)

    assert any("must declare 'items'" in item for item in violations)


def test_an_undocumented_type_is_rejected():
    schema = {"type": "object", "properties": {"x": {"type": "date"}},
              "required": ["x"], "additionalProperties": False}

    violations = schema_check.check_schema(schema)

    assert any("'date' is not documented" in item for item in violations)


def test_the_validator_does_not_invent_depth_or_property_name_rules():
    # Neither rule is documented by Groq, so a deep nest and a spaced property
    # name must both pass: the validator only enforces what the docs state.
    node = {"type": "string"}
    for _ in range(20):
        node = {
            "type": "object",
            "properties": {"a b": node},
            "required": ["a b"],
            "additionalProperties": False,
        }

    assert schema_check.check_schema(node) == []
    assert "nesting depth limits" in schema_check.undocumented_rules()


def test_a_documented_integer_without_bounds_is_accepted():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
        "additionalProperties": False,
    }

    assert schema_check.check_schema(schema) == []


# -- S1a: the provider error body -----------------------------------------
def test_a_rejected_request_reports_the_sanitized_provider_body():
    clock = _Clock()
    limiter = _limiter(clock)
    body = {
        "error": {
            "message": "Invalid schema for response_format 'compare_answer'",
            "type": "invalid_request_error",
            "param": "response_format",
            "code": "json_validate_failed",
        }
    }
    client = _groq_client(_ProviderError(400, body), limiter)

    with pytest.raises(LLMRequestError) as caught:
        client.complete(
            "system",
            "question text",
            temperature=0.0,
            max_tokens=100,
            reasoning_effort="low",
        )

    error = caught.value
    assert error.error_type == "invalid_request"
    assert error.http_status == 400
    assert error.provider_error["type"] == "invalid_request_error"
    assert error.provider_error["param"] == "response_format"
    assert error.provider_error["code"] == "json_validate_failed"
    assert error.provider_error["message"].startswith("Invalid schema")
    # S2a: a rejected request never used the allowance, so it is released.
    assert limiter.snapshot()["tokens_used_this_minute"] == 0
    assert error.call_metrics["reservation_outcome"] == "released"
    assert error.call_metrics["http_status"] == 400


def test_a_provider_message_that_echoes_the_request_is_suppressed():
    clock = _Clock()
    limiter = _limiter(clock)
    user_prompt = "Apa saja saran pengembangan yang tercantum pada RPT0002?"
    body = {
        "error": {
            "message": f"Could not parse request with messages {user_prompt}",
            "type": "invalid_request_error",
        }
    }
    client = _groq_client(_ProviderError(400, body), limiter)

    with pytest.raises(LLMRequestError) as caught:
        client.complete(
            "system",
            user_prompt,
            temperature=0.0,
            max_tokens=100,
            reasoning_effort="low",
        )

    assert caught.value.provider_error["type"] == "invalid_request_error"
    assert "message" not in caught.value.provider_error
    assert (
        caught.value.provider_error["message_suppressed"]
        == "echoes_request_content"
    )


def test_the_provider_message_is_truncated():
    clock = _Clock()
    limiter = _limiter(clock)
    body = {"error": {"message": "x" * 2000, "type": "invalid_request_error"}}
    client = _groq_client(_ProviderError(400, body), limiter)

    with pytest.raises(LLMRequestError) as caught:
        client.complete(
            "system", "q", temperature=0.0, max_tokens=100, reasoning_effort="low"
        )

    assert len(caught.value.provider_error["message"]) == (
        config.LLM_PROVIDER_ERROR_MAX_CHARS
    )


# -- S2a: reservation settlement per failure class -------------------------
def test_a_429_releases_the_estimate_because_the_headers_are_authoritative(monkeypatch):
    # A 429 rejects the request before generation and carries the provider's own
    # window headers, so keeping our estimate would double-count and could block
    # the retry the provider asked for.
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 0)
    clock = _Clock()
    limiter = _limiter(clock)
    client = _groq_client(
        _ProviderError(429, {}, {"retry-after": "1"}), limiter
    )

    with pytest.raises(Exception) as caught:
        client.complete(
            "system", "q", temperature=0.0, max_tokens=100, reasoning_effort="low"
        )

    assert limiter.snapshot()["tokens_used_this_minute"] == 0
    assert caught.value.call_metrics["reservation_outcome"] == "released"
    assert caught.value.call_metrics["http_status"] == 429


def test_a_500_keeps_the_estimate_in_the_window(monkeypatch):
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 0)
    clock = _Clock()
    limiter = _limiter(clock)
    client = _groq_client(_ProviderError(500, {}), limiter)

    with pytest.raises(Exception) as caught:
        client.complete(
            "system", "q", temperature=0.0, max_tokens=100, reasoning_effort="low"
        )

    assert limiter.snapshot()["tokens_used_this_minute"] > 0
    assert caught.value.call_metrics["reservation_outcome"] == "kept"


def test_an_unexpected_error_also_reports_a_failed_call():
    clock = _Clock()
    limiter = _limiter(clock)
    client = _groq_client(ValueError("boom"), limiter)

    with pytest.raises(LLMUnavailableError) as caught:
        client.complete(
            "system", "q", temperature=0.0, max_tokens=100, reasoning_effort="low"
        )

    metrics = caught.value.call_metrics
    assert metrics["error_type"] == "unexpected_error"
    assert metrics["reservation_outcome"] == "released"
    assert metrics["attempts"] == 1


# -- S2b/S2d: the pipeline reports a failed call and an invalid request ----
def test_a_failed_call_appears_in_model_calls():
    class _Failing(FakeLLM):
        def complete(self, *args, **kwargs):
            super().complete(*args, **kwargs)
            error = LLMUnavailableError("rejected")
            error.call_metrics = {
                "latency_ms": 1234.0,
                "queue_wait_ms": 900.0,
                "model_latency_ms": 10.0,
                "attempts": 1,
                "usage": TokenUsage().to_dict(),
                "error_type": "http_400",
                "http_status": 400,
                "reservation_outcome": "released",
                "reservation": {},
            }
            raise error

    matches = [_match("RPT0002", level="Finalis")]
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(
            matches=matches,
            chunks=matches,
            metadatas=[item.metadata for item in matches],
        ),
        llm=_Failing(
            structured_responses=[json.dumps({"found": True, "points": []})]
        ),
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run("Apa saja saran pengembangan pada RPT0002?")

    assert result.status is AnswerStatus.TECHNICAL_ERROR
    call = result.debug.calls[0]
    assert call["name"] == "answer_structured"
    assert call["queue_wait_ms"] == 900.0
    assert call["http_status"] == 400
    assert call["reservation_outcome"] == "released"


def test_a_request_level_400_is_invalid_request_and_is_not_retried():
    matches = [_match("RPT0002")]
    llm = FakeLLM(
        json_responses=[_analyzer()],
        fail_with=LLMRequestError(
            "rejected",
            http_status=400,
            provider_error={"type": "invalid_request_error", "param": "response_format"},
        ),
    )
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(
            matches=matches, metadatas=[item.metadata for item in matches]
        ),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run("Apa pengalaman tim pada business plan?")

    assert result.status is AnswerStatus.TECHNICAL_ERROR
    assert result.debug.error_type == "invalid_request"
    assert result.debug.incidents["provider_400"] == 1
    assert result.debug.provider_error["param"] == "response_format"
    assert len(llm.calls) == 1


# -- S2c: the offline replay ----------------------------------------------
def _load_replay():
    path = Path(__file__).resolve().parent.parent / "scripts" / "replay_fixpack11_limiter.py"
    spec = importlib.util.spec_from_file_location("replay_fixpack11_limiter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_leak_does_not_explain_retry_after_60_but_the_stale_clamp_does():
    results = _load_replay().run_scenarios()

    # The current code releases the failed reservation, so Q3's answer fits.
    assert results["released"]["retry_after"] is None
    # The hypothesis from the run report would have waited ~24 s and succeeded,
    # so it cannot produce RATE_LIMITED with retry_after=60.
    assert results["leaked"]["retry_after"] is None
    # A provider clamp that never expires is the only behaviour that yields 60.
    assert results["clamp-never"]["retry_after"] == 60
    # With its documented reset, the clamp expires and the call fits again.
    assert results["clamp-expires"]["retry_after"] is None


def test_a_provider_clamp_expires_with_its_reset():
    clock = _Clock()
    limiter = _limiter(clock)
    limiter.observe_headers(
        {"x-ratelimit-remaining-tokens": "1500", "x-ratelimit-reset-tokens": "10s"}
    )

    clock.advance(5.0)
    assert limiter.wait_estimate(2000) > 0

    clock.advance(6.0)
    assert limiter.wait_estimate(2000) == 0


# -- S1e: the two contrast modes -----------------------------------------
def test_the_sides_mode_keeps_the_known_good_schema():
    schema = compare_answer.build_schema(["g1", "g2"], "sides")
    contrast = schema["properties"]["contrasts"]["items"]

    assert set(contrast["required"]) == {"dimension", "sides"}
    side = contrast["properties"]["sides"]["items"]
    assert set(side["required"]) == {"group", "text", "cite"}
    assert side["properties"]["group"]["enum"] == ["g1", "g2"]
    assert schema_check.check_schema(schema) == []


def test_the_sides_mode_keeps_the_fix_pack_10_dimension_rules():
    payload = {
        "groups": {
            "g1": [{"text": "Validasi lewat wawancara.", "cite": ["LAP1"]}],
            "g2": [{"text": "Data sekunder dominan.", "cite": ["LAP2"]}],
        },
        "contrasts": [
            {
                "dimension": "Pembelajaran",
                "sides": [
                    {"group": "g1", "text": "Wawancara pengguna.", "cite": ["LAP1"]},
                    {"group": "g2", "text": "Data sekunder.", "cite": ["LAP2"]},
                ],
            }
        ],
    }

    validation = _validate(payload, mode="sides")

    assert len(validation.contrasts) == 1
    contrast = validation.contrasts[0]
    assert contrast.dimension == config.COMPARE_NEUTRAL_DIMENSION
    assert validation.incidents["generic_dimension"] == 1
    assert [side.label for side in contrast.sides] == ["Business Plan", "Business Case"]
    assert contrast.sides[0].report_ids == ["RPT0001"]


def test_the_sides_mode_applies_the_citation_ceiling():
    payload = {
        "groups": {
            "g1": [{"text": "Validasi lewat wawancara.", "cite": ["LAP1"]}],
            "g2": [{"text": "Data sekunder dominan.", "cite": ["LAP2"]}],
        },
        "contrasts": [
            {
                "dimension": "Kedalaman validasi",
                "sides": [
                    {
                        "group": "g1",
                        "text": "Terlalu banyak sitasi.",
                        "cite": ["LAP1", "LAP2", "LAP3"],
                    },
                    {"group": "g2", "text": "Data sekunder.", "cite": ["LAP2"]},
                ],
            }
        ],
    }

    validation = _validate(
        payload,
        mode="sides",
        alias_map={"LAP1": "RPT0001", "LAP2": "RPT0002", "LAP3": "RPT0003"},
        report_group={
            "RPT0001": "Business Plan",
            "RPT0002": "Business Plan",
            "RPT0003": "Business Plan",
        },
        retrieved=["RPT0001", "RPT0002", "RPT0003"],
    )

    assert validation.contrasts == []
    assert validation.incidents["contrast_invalid_side_cite_over_limit"] == 1


def test_a_pick_accepts_a_number_string_and_a_plain_integer():
    payload = _pick_payload()

    validation = _validate(payload)

    assert len(validation.contrasts) == 1
    assert validation.contrasts[0].sides[0].report_ids == ["RPT0001"]


def test_an_unknown_contrast_mode_fails_loudly():
    with pytest.raises(ValueError, match="COMPARE_CONTRAST_MODE"):
        compare_answer.build_schema(["g1", "g2"], "guess")


# -- S3: the support overlap ----------------------------------------------
RPT0002_CHUNK = (
    "1. Pastikan validasi operasional supply chain dihitung hingga skala terkecil "
    "sebelum membawanya ke kompetisi bisnis plan manufaktur.\n"
    "2. Buat simulasi pitch deck berulang kali dengan timer ketat agar tidak "
    "terpotong oleh time keeper saat presentasi final.\n"
    "3. Jangan abaikan regulasi lingkungan hidup dan sertifikasi standar mutu "
    "bahan bakar padat."
)

RPT0002_RECOMMENDATIONS = (
    "Pastikan validasi operasional supply chain dihitung hingga skala terkecil "
    "sebelum membawanya ke kompetisi bisnis plan manufaktur.",
    "Buat simulasi pitch deck berulang kali dengan timer ketat agar tidak "
    "terpotong oleh time keeper saat presentasi final.",
    "Jangan abaikan regulasi lingkungan hidup dan sertifikasi standar mutu "
    "bahan bakar padat.",
)


def test_the_three_rpt0002_recommendations_are_supported_by_their_own_chunk():
    result = wording.support_overlap(
        [(text, ["RPT0002"]) for text in RPT0002_RECOMMENDATIONS],
        {"RPT0002": RPT0002_CHUNK},
    )

    assert result["low_support_points"] == 0
    assert result["cited_text_missing_points"] == 0
    assert result["minimum"] >= 0.8
    for item in result["per_point"]:
        assert item["report_ids"] == ["RPT0002"]


def test_an_invented_sentence_scores_low():
    invented = (
        "Tim memenangkan hadiah utama karena robotika bawah air mereka "
        "mengalahkan seluruh peserta internasional."
    )
    result = wording.support_overlap([(invented, ["RPT0002"])], {"RPT0002": RPT0002_CHUNK})

    assert result["low_support_points"] == 1
    assert result["samples"][0]["report_ids"] == ["RPT0002"]


def test_a_point_citing_a_missing_report_is_flagged_distinctly():
    result = wording.support_overlap(
        [("Validasi operasional supply chain belum dihitung.", ["RPT9999"])],
        {"RPT0002": RPT0002_CHUNK},
    )

    assert result["cited_text_missing_points"] == 1
    assert result["low_support_points"] == 0
    assert result["missing_samples"][0]["report_ids"] == ["RPT9999"]


def test_a_dict_point_fails_loudly_instead_of_reading_its_keys():
    with pytest.raises(TypeError, match="report_ids"):
        wording.support_overlap(
            [{"text": "apa pun", "report_ids": ["RPT0002"]}],  # type: ignore[list-item]
            {"RPT0002": RPT0002_CHUNK},
        )


def test_the_pipeline_passes_real_report_ids_to_the_support_metric():
    matches = [_match("RPT0002", level="Finalis")]
    payload = json.dumps(
        {
            "found": True,
            "points": [
                {"text": "Saran pengembangan dari dokumen.", "cite": ["LAP1"]}
            ],
        }
    )
    llm = FakeLLM(json_responses=[_analyzer()], structured_responses=[payload])
    pipeline = RagPipeline(
        knowledge_base=FilteringKnowledgeBase(
            matches=matches, metadatas=[item.metadata for item in matches]
        ),
        llm=llm,
        embedder=RecordingEmbedder(),
        token_counter=WordCountTokenizer(),
    )

    result = pipeline.run("Apa pengalaman tim pada business plan?")

    per_point = result.debug.support["per_point"]
    assert per_point[0]["report_ids"] == ["RPT0002"]
    assert per_point[0]["overlap"] is not None


# -- S5: both comparison questions, both modes ----------------------------
def _e2e():
    return _load_script("offline_compare_endtoend")


def _load_script(name: str):
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["pick", "sides"])
@pytest.mark.parametrize("case", ["juara_vs_belum_juara", "compare_by_type"])
def test_both_compare_questions_answer_end_to_end(mode, case):
    result, llm = _e2e().run_case(case, mode)

    assert result.status is AnswerStatus.ANSWERED, result.debug.error_type
    assert llm.schemas_checked == ["compare_answer"]
    assert result.cited_report_ids
    assert config.COMPARE_CONTRASTS_HEADING in result.answer
    assert "LAP" not in result.answer


def test_the_pick_mode_renders_the_validated_points_not_model_contrast_text():
    result, _ = _e2e().run_case("compare_by_type", "pick")

    assert "- Kendala Keuangan" in result.answer
    assert "  - Business Plan: Kelayakan finansial kurang digali. [RPT9101]" in (
        result.answer
    )


def test_a_generic_dimension_is_neutral_in_both_modes():
    for mode in ("pick", "sides"):
        result, _ = _e2e().run_case("juara_vs_belum_juara", mode)
        assert f"- {config.COMPARE_NEUTRAL_DIMENSION}" in result.answer
        assert result.debug.incidents["generic_dimension"] == 1
