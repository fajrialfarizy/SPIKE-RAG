"""
Offline tests for the token-aware limiter and the two RATE_LIMITED texts
(fix pack G7).

The clock and sleep function are injected, so the rolling windows are tested
without waiting.
"""

from types import SimpleNamespace

import pytest

import config
from core.deadline import Deadline
from core.llm_client import GroqLLMClient, LLMRateLimitError
from core import prompts
from core.pipeline import AnswerStatus, RagPipeline
from core.rate_limiter import (
    DailyTokenBudgetExhausted,
    MinuteTokenBudgetExhausted,
    TokenAwareRateLimiter,
)
from tests.stubs import (
    FakeKnowledgeBase,
    FakeLLM,
    RecordingEmbedder,
    WordCountTokenizer,
)
from tests.test_llm_client import PROMPT, USER, FakeGroqClient, FakeResponse, call

LIMITS = {
    "model": "test-model",
    "requests_per_minute": 10,
    "requests_per_day": 100,
    "tokens_per_minute": 100,
    "tokens_per_day": 250,
    "source": "test double",
    "date": "2026-11-21",
}


class FakeClock:
    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_limiter():
    clock = FakeClock()
    limiter = TokenAwareRateLimiter(
        limits=LIMITS,
        clock=clock,
        sleep=clock.advance,
        token_counter=WordCountTokenizer(),
    )
    return limiter, clock


# -- estimation -------------------------------------------------------------
def test_the_estimate_counts_static_text_exactly_and_dynamic_chars_by_ratio():
    limiter = TokenAwareRateLimiter(
        limits=LIMITS,
        clock=FakeClock(),
        sleep=lambda _: None,
        token_counter=WordCountTokenizer(),
        dynamic_chars_per_token=4.0,
        margin_ratio=0.0,
    )

    estimate = limiter.estimate_tokens("satu dua tiga", "x" * 400, 120)

    assert estimate == 3 + 100 + 120


def test_the_estimate_adds_the_configured_margin_to_the_prompt_only():
    limiter = TokenAwareRateLimiter(
        limits=LIMITS,
        clock=FakeClock(),
        sleep=lambda _: None,
        token_counter=WordCountTokenizer(),
        dynamic_chars_per_token=4.0,
        margin_ratio=0.5,
    )

    # Prompt 3 + 100, its 50% margin 52, plus the completion 120 (never margined).
    assert limiter.estimate_tokens("satu dua tiga", "x" * 400, 120) == 103 + 52 + 120


def test_the_static_count_is_computed_once_per_distinct_text():
    class CountingTokenizer:
        def __init__(self):
            self.calls = 0

        def count_tokens(self, text):
            self.calls += 1
            return len(text.split())

    counter = CountingTokenizer()
    limiter = TokenAwareRateLimiter(
        limits=LIMITS, clock=FakeClock(), sleep=lambda _: None, token_counter=counter
    )

    limiter.estimate_tokens("instruksi statis", "a", 0)
    limiter.estimate_tokens("instruksi statis", "b", 0)

    assert counter.calls == 1


# -- windows ----------------------------------------------------------------
def test_a_reservation_is_settled_to_the_reported_usage():
    limiter, _ = make_limiter()

    entry = limiter.reserve(80)
    assert limiter.snapshot()["tokens_used_today"] == 80

    limiter.record(entry, 50)
    snapshot = limiter.snapshot()
    assert snapshot["tokens_used_today"] == 50
    assert snapshot["tokens_used_this_minute"] == 50


def test_a_released_reservation_does_not_spend_budget():
    limiter, _ = make_limiter()

    entry = limiter.reserve(80)
    limiter.release(entry)

    assert limiter.snapshot()["tokens_used_today"] == 0
    assert limiter.snapshot()["tokens_used_this_minute"] == 0


def test_a_minute_overflow_waits_inside_the_queue_limit():
    limiter, clock = make_limiter()
    limiter.reserve(60)

    limiter.reserve(60, max_wait_seconds=120)

    assert clock.now == 60
    assert limiter.waits == 1


def test_a_wait_beyond_the_queue_limit_is_refused_with_the_computed_wait():
    limiter, clock = make_limiter()
    limiter.reserve(60)

    with pytest.raises(MinuteTokenBudgetExhausted) as excinfo:
        limiter.reserve(60, max_wait_seconds=5)

    assert clock.now == 0
    assert limiter.snapshot()["tokens_used_this_minute"] == 60
    assert excinfo.value.retry_after_seconds == 60
    assert limiter.waits == 0


def test_the_configured_queue_limit_is_used_by_default():
    limiter, clock = make_limiter()
    limiter.queue_max_wait_seconds = 30.0
    limiter.reserve(60)

    with pytest.raises(MinuteTokenBudgetExhausted) as excinfo:
        limiter.reserve(60)

    assert excinfo.value.retry_after_seconds == 60
    assert config.QUEUE_MAX_WAIT_SECONDS == 45.0


def test_a_single_request_larger_than_the_minute_limit_fails_loudly():
    limiter, clock = make_limiter()

    with pytest.raises(MinuteTokenBudgetExhausted) as excinfo:
        limiter.reserve(200, max_wait_seconds=120)

    assert clock.now == 0
    assert excinfo.value.retry_after_seconds is None


def test_the_daily_budget_refuses_before_the_provider_call():
    clock = FakeClock()
    limiter = TokenAwareRateLimiter(
        limits=dict(LIMITS, tokens_per_minute=1000), clock=clock, sleep=clock.advance
    )
    limiter.reserve(200)

    with pytest.raises(DailyTokenBudgetExhausted):
        limiter.reserve(60, max_wait_seconds=120)

    assert clock.now == 0


def test_the_daily_window_rolls_over_after_24_hours():
    clock = FakeClock()
    limiter = TokenAwareRateLimiter(
        limits=dict(LIMITS, tokens_per_minute=1000), clock=clock, sleep=clock.advance
    )
    limiter.reserve(200)

    clock.advance(config.TOKEN_LIMITER_DAY_WINDOW_SECONDS)

    entry = limiter.reserve(60, max_wait_seconds=120)
    assert entry[1] == 60.0
    assert limiter.snapshot()["tokens_used_today"] == 60


def test_the_request_per_minute_cap_also_queues_calls():
    limits = dict(LIMITS, requests_per_minute=1, tokens_per_minute=1000)
    clock = FakeClock()
    limiter = TokenAwareRateLimiter(limits=limits, clock=clock, sleep=clock.advance)

    limiter.reserve(10)
    limiter.reserve(10, max_wait_seconds=120)

    assert clock.now == 60


def test_response_headers_refine_the_minute_budget():
    limiter, clock = make_limiter()
    limiter.observe_headers(
        {
            "x-ratelimit-remaining-tokens": "10",
            "x-ratelimit-reset-tokens": "5.5s",
            "x-ratelimit-remaining-requests": "999",
        }
    )

    snapshot = limiter.snapshot()
    assert snapshot["provider_tpm_remaining"] == 10
    assert snapshot["provider_tpm_reset_seconds"] == 5.5

    with pytest.raises(MinuteTokenBudgetExhausted):
        limiter.reserve(20, max_wait_seconds=3)


@pytest.mark.parametrize("seconds,expected", [("7.66s", 7.66), ("1m2s", 62.0)])
def test_the_header_duration_parser_handles_groq_spellings(seconds, expected):
    from core.rate_limiter import _seconds_header

    assert _seconds_header({"x": seconds}, "x") == pytest.approx(expected)


# -- capacity ---------------------------------------------------------------
def test_capacity_estimate_uses_the_real_limits_and_the_binding_window():
    limiter = TokenAwareRateLimiter()

    capacity = limiter.capacity_estimate(4500)

    assert capacity["tokens_per_minute"] == config.GROQ_LIMITS["tokens_per_minute"]
    assert capacity["tokens_per_day"] == config.GROQ_LIMITS["tokens_per_day"]
    # 8,000 TPM / 4,500 tokens is the tighter bound (not 30 RPM).
    assert capacity["questions_per_minute"] == pytest.approx(1.78)
    # 200,000 TPD / 4,500 tokens, still below the 1,000 RPD cap.
    assert capacity["questions_per_day"] == pytest.approx(44.4)


def test_the_configured_limits_are_the_ones_the_supervisor_reported():
    assert config.GROQ_LIMITS["requests_per_minute"] == 30
    assert config.GROQ_LIMITS["requests_per_day"] == 1000
    assert config.GROQ_LIMITS["tokens_per_minute"] == 8000
    assert config.GROQ_LIMITS["tokens_per_day"] == 200000
    assert config.GROQ_LIMITS["model"] == config.LLM_MODEL_NAME


# -- client integration -----------------------------------------------------
def test_the_client_refuses_a_daily_exhausted_budget_without_a_provider_call():
    limiter = TokenAwareRateLimiter(
        limits=dict(LIMITS, tokens_per_day=10, tokens_per_minute=100000),
        sleep=lambda _: None,
        token_counter=WordCountTokenizer(),
    )
    fake = FakeGroqClient([FakeResponse("ok")])
    client = GroqLLMClient(
        api_key="dummy-key", client=fake, token_limiter=limiter
    )

    with pytest.raises(LLMRateLimitError) as excinfo:
        call(client)

    assert excinfo.value.is_daily_limit is True
    assert fake.completions.requests == []


def test_a_minute_budget_that_cannot_fit_the_request_is_rate_limited():
    limiter = TokenAwareRateLimiter(
        limits=dict(LIMITS, tokens_per_day=100000, tokens_per_minute=10),
        sleep=lambda _: None,
        token_counter=WordCountTokenizer(),
    )
    fake = FakeGroqClient([FakeResponse("ok")])
    client = GroqLLMClient(
        api_key="dummy-key", client=fake, token_limiter=limiter
    )

    with pytest.raises(LLMRateLimitError) as excinfo:
        call(client, deadline=Deadline(1.0))

    assert excinfo.value.is_daily_limit is False
    assert fake.completions.requests == []


# -- two RATE_LIMITED texts -------------------------------------------------
def test_a_daily_limit_turns_into_the_daily_quota_text():
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(),
        llm=FakeLLM(
            fail_with=LLMRateLimitError("provider text", is_daily_limit=True)
        ),
        embedder=RecordingEmbedder(),
    )

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.status is AnswerStatus.RATE_LIMITED
    assert result.answer == prompts.DAILY_QUOTA_RESPONSE
    assert result.debug.incidents["daily_limit"] == 1


def test_a_minute_limit_keeps_the_generic_rate_limited_text():
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(),
        llm=FakeLLM(fail_with=LLMRateLimitError("provider text")),
        embedder=RecordingEmbedder(),
    )

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.status is AnswerStatus.RATE_LIMITED
    assert result.answer == prompts.RATE_LIMITED_RESPONSE
