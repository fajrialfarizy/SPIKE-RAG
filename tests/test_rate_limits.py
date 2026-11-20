"""
Offline tests for 429 handling (report F1a/F1b).

The provider double carries the response headers Groq documents: ``retry-after``
(on 429 only), ``x-ratelimit-remaining-requests`` and
``x-ratelimit-reset-requests``. Nothing here touches the network.
"""

import threading
import time
from types import SimpleNamespace

import pytest

import config
from core.deadline import Deadline
from core.llm_client import (
    GroqLLMClient,
    LLMConcurrencyLimiter,
    LLMRateLimitError,
    TokenUsage,
)
from core.pipeline import AnswerStatus, RagPipeline
from tests.stubs import FakeKnowledgeBase, FakeLLM, RecordingEmbedder
from tests.test_llm_client import FakeGroqClient, FakeResponse, call

RAW_PROVIDER_TEXT = "raw-provider-detail-MUST-NOT-LEAK"


class RateLimitError(Exception):
    """Duck-typed stand-in for groq's APIStatusError on a 429."""

    def __init__(self, headers=None, status_code=429):
        super().__init__(RAW_PROVIDER_TEXT)
        self.status_code = status_code
        self.response = SimpleNamespace(
            status_code=status_code, headers=headers or {}
        )


class FakeClock:
    def __init__(self):
        self.now = 500.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make(headers=None, status_code=429, budget=60.0):
    clock = FakeClock()
    sleeps = []

    def sleep_and_advance(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)

    fake = FakeGroqClient([RateLimitError(headers, status_code)] * 4)
    client = GroqLLMClient(api_key="dummy-key", client=fake, sleep_fn=sleep_and_advance)
    deadline = Deadline(budget, clock=clock)
    return client, fake, sleeps, deadline


def test_retry_after_is_honoured_instead_of_the_exponential_backoff():
    client, fake, sleeps, deadline = make({"retry-after": "3"})

    with pytest.raises(LLMRateLimitError):
        call(client, deadline=deadline)

    assert sleeps == [3.0, 3.0, 3.0]

    # A success after the first wait proves the header was used, not the backoff.
    fake2 = FakeGroqClient([RateLimitError({"retry-after": "2"}), FakeResponse("ok")])
    sleeps2 = []
    clock = FakeClock()
    client2 = GroqLLMClient(
        api_key="dummy-key",
        client=fake2,
        sleep_fn=lambda value: (sleeps2.append(value), clock.advance(value)),
    )
    assert call(client2, deadline=Deadline(60.0, clock=clock)) == "ok"
    assert sleeps2 == [2.0]


def test_retry_after_longer_than_the_remaining_budget_fails_fast():
    client, fake, sleeps, deadline = make({"retry-after": "120"}, budget=5.0)

    with pytest.raises(LLMRateLimitError) as excinfo:
        call(client, deadline=deadline)

    assert len(fake.completions.requests) == 1
    assert sleeps == []
    assert excinfo.value.retry_after_seconds == 120.0
    assert excinfo.value.is_daily_limit is True
    assert RAW_PROVIDER_TEXT not in str(excinfo.value)


def test_a_daily_request_limit_without_retry_after_fails_fast():
    client, fake, sleeps, deadline = make(
        {
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "6h12m3.5s",
        }
    )

    with pytest.raises(LLMRateLimitError) as excinfo:
        call(client, deadline=deadline)

    assert len(fake.completions.requests) == 1
    assert sleeps == []
    assert excinfo.value.is_daily_limit is True


def test_without_headers_the_exponential_backoff_still_applies():
    client, fake, sleeps, deadline = make(None, budget=60.0)
    fake.completions.outcomes = [
        RateLimitError(None),
        RateLimitError(None),
        FakeResponse("akhirnya berhasil"),
    ]

    assert call(client, deadline=deadline) == "akhirnya berhasil"
    assert sleeps == [1.0, 2.0]


def test_an_exhausted_attempt_budget_raises_rate_limited_not_a_generic_error():
    client, fake, sleeps, deadline = make({"retry-after": "1"}, budget=60.0)

    with pytest.raises(LLMRateLimitError):
        call(client, deadline=deadline)

    assert len(sleeps) == config.LLM_MAX_RETRIES


def test_a_rate_limited_pipeline_returns_the_fixed_text():
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(),
        llm=FakeLLM(fail_with=LLMRateLimitError("provider words must not leak")),
        embedder=RecordingEmbedder(),
    )

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.status is AnswerStatus.RATE_LIMITED
    assert result.debug.error_type == "LLMRateLimitError"
    assert "provider words" not in result.answer
    assert "terlalu banyak permintaan" in result.answer


# -- concurrency ----------------------------------------------------------
class SlowCompletions:
    """Records how many provider calls overlapped."""

    def __init__(self, delay=0.05):
        self.delay = delay
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def create(self, **kwargs):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            return FakeResponse(str(kwargs["temperature"]))
        finally:
            with self.lock:
                self.active -= 1


class SlowGroqClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def test_the_concurrency_limit_queues_extra_threads():
    completions = SlowCompletions()
    limiter = LLMConcurrencyLimiter(limit=2)
    clients = [
        GroqLLMClient(
            api_key="dummy-key",
            client=SlowGroqClient(completions),
            sleep_fn=lambda _: None,
            limiter=limiter,
            clock=lambda: 0.0,
        )
        for _ in range(4)
    ]

    results = []
    errors = []

    def worker(client):
        try:
            results.append(call(client))
        except BaseException as exc:  # pragma: no cover - failure diagnostics
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(client,)) for client in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert len(results) == 4
    assert limiter.peak == 2
    assert completions.peak == 2
