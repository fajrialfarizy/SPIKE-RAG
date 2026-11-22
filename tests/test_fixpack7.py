"""
Offline tests for fix pack 7: the queue policy, the calibrated estimator and the
backend bullet trim.

The failing sequence from the supervisor's real run is reproduced with a fake
clock: Q1 spends 1,239 + 3,451 tokens, Q2's analyzer spends 1,257, and the
structured comparison call then needs more than the 8,000-token minute window
allows. Before the fix the limiter slept the whole pipeline deadline and the
concurrency-slot guard raised ``DeadlineExceeded``, which the pipeline reported as
TECHNICAL_ERROR. Now the wait is bounded by ``QUEUE_MAX_WAIT_SECONDS``, refunded
to the model deadline, and a too-long wait is RATE_LIMITED with a retry hint.

Nothing here touches the network: the provider is a scripted double.
"""

import json
import threading
from types import SimpleNamespace

import pytest

import config
from core import prompts
from core.answer_trimmer import trim_bullets
from core.compare_answer import response_format
from core.deadline import Deadline, DeadlineExceeded, QueueDeadlineExceeded
from core.knowledge_base import StoredChunk
from core.llm_client import GroqLLMClient, LLMConcurrencyLimiter, LLMRateLimitError
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
from tests.test_pipeline import build, metadata

# The comparison user prompt of the real run (system prompt + context + hint +
# question): 9,252 characters, of which 8,156 are the rendered context.
COMPARE_PROMPT = ("Konteks dokumen evaluasi kompetisi. " * 300)[:9252]
LABELS = [
    group["label"] for group in config.COMPARISON_GROUPS["achievement_group"]
]


class ProseTokenizer:
    """Fast stand-in for the e5 tokenizer in the default suite (about 5 chars/token).

    The real tokenizer is exercised by the slow calibration test below and by
    scripts/replay_limiter_sequence.py.
    """

    def count_tokens(self, text: str) -> int:
        return max(1, round(len(text) / 5.0))


class FakeClock:
    """Thread-safe fake monotonic clock shared by deadline, limiter and client."""

    def __init__(self, start: float = 0.0):
        self._now = float(start)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        return self._now

    @property
    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += float(seconds)

    def jump_to(self, value: float) -> None:
        with self._lock:
            self._now = float(value)


class FakeUsage:
    def __init__(self, prompt_tokens=3600, completion_tokens=538):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.completion_tokens_details = SimpleNamespace(reasoning_tokens=0)
        self.prompt_tokens_details = SimpleNamespace(cached_tokens=None)


class FakeResponse:
    def __init__(self, content, usage=None):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason="stop"
            )
        ]
        self.usage = usage or FakeUsage()


class Provider429(Exception):
    """Duck-typed 429 with the retry-after header Groq documents."""

    def __init__(self, retry_after="2"):
        super().__init__("raw provider text")
        self.status_code = 429
        self.response = SimpleNamespace(
            status_code=429, headers={"retry-after": retry_after}
        )


class ScriptedProvider:
    """Returns scripted outcomes and spends a second of the fake clock per call."""

    def __init__(self, clock: FakeClock, outcomes):
        self.clock = clock
        self.outcomes = list(outcomes)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        self.clock.advance(1.0)
        return outcome


def make_client(clock, limiter, outcomes, slots=2):
    provider = ScriptedProvider(clock, outcomes)
    client = GroqLLMClient(
        api_key="offline-test",
        client=SimpleNamespace(chat=SimpleNamespace(completions=provider)),
        token_limiter=limiter,
        limiter=LLMConcurrencyLimiter(slots),
        sleep_fn=clock.advance,
        clock=clock,
    )
    return client, provider


def compare_call(client, deadline):
    return client.complete(
        prompts.SYSTEM_PROMPT_COMPARE,
        COMPARE_PROMPT,
        temperature=config.LLM_TEMPERATURE_CHAT,
        max_tokens=config.LLM_MAX_OUTPUT_TOKENS_COMPARE,
        reasoning_effort="low",
        response_format=response_format(LABELS),
        deadline=deadline,
    )


def seed_q1(limiter, clock):
    """Q1 of the real run: analyzer 1,239 at t=0, answer 3,451 reserved at t=1."""
    entry = limiter.reserve(1239)
    limiter.record(entry, 1239)
    clock.advance(1.0)
    entry = limiter.reserve(5535)
    clock.advance(5.0)
    limiter.record(entry, 3451)


def seed_q2_analyzer(limiter):
    entry = limiter.reserve(1257)
    limiter.record(entry, 1257)


# -- M1: the reproduced sequence -------------------------------------------
def test_the_failing_sequence_now_waits_inside_the_limit_and_is_served():
    clock = FakeClock()
    limiter = TokenAwareRateLimiter(
        clock=clock, sleep=clock.advance, token_counter=ProseTokenizer()
    )
    seed_q1(limiter, clock)
    clock.jump_to(33.0)  # Q2 starts about 33 s after Q1 finished
    seed_q2_analyzer(limiter)

    client, provider = make_client(clock, limiter, [FakeResponse("{}")])
    deadline = Deadline(config.PIPELINE_TOTAL_DEADLINE_SECONDS, clock=clock)

    result = compare_call(client, deadline)

    assert result.content == "{}"
    assert len(provider.requests) == 1
    assert limiter.waits == 1
    waited = result.queue_wait_ms / 1000.0
    assert 0 < waited <= config.QUEUE_MAX_WAIT_SECONDS
    # The answer call ran with a live deadline: the wait was refunded, not spent.
    assert deadline.remaining() > 0
    assert clock.now > 33.0 + waited


def test_a_wait_beyond_the_queue_limit_is_rate_limited_without_a_call():
    clock = FakeClock()
    limiter = TokenAwareRateLimiter(
        clock=clock, sleep=clock.advance, token_counter=ProseTokenizer()
    )
    seed_q1(limiter, clock)
    clock.jump_to(3.0)  # too early: the window cannot free the tokens in time
    seed_q2_analyzer(limiter)

    client, provider = make_client(clock, limiter, [FakeResponse("{}")])

    with pytest.raises(LLMRateLimitError) as excinfo:
        compare_call(client, Deadline(config.PIPELINE_TOTAL_DEADLINE_SECONDS,
                                      clock=clock))

    assert provider.requests == []
    assert excinfo.value.is_daily_limit is False
    assert excinfo.value.retry_after_seconds > config.QUEUE_MAX_WAIT_SECONDS


def test_a_queue_deadline_expiry_is_rate_limited_never_technical():
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(),
        llm=FakeLLM(
            fail_with=QueueDeadlineExceeded("slot", retry_after_seconds=30)
        ),
        embedder=RecordingEmbedder(),
    )

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.status is AnswerStatus.RATE_LIMITED
    assert result.debug.error_type == "QueueDeadlineExceeded"
    assert result.retry_after_seconds == 30
    assert result.answer == prompts.RATE_LIMITED_RESPONSE


def test_a_real_deadline_overrun_is_still_a_technical_error():
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(),
        llm=FakeLLM(fail_with=DeadlineExceeded("model overrun")),
        embedder=RecordingEmbedder(),
    )

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.status is AnswerStatus.TECHNICAL_ERROR
    assert result.retry_after_seconds is None


# -- M2: estimator and provider accounting ---------------------------------
@pytest.mark.slow
def test_the_estimator_keeps_the_real_prompts_inside_the_minute_budget():
    limiter = TokenAwareRateLimiter()

    answer = limiter.estimate_tokens(
        prompts.SYSTEM_PROMPT,
        COMPARE_PROMPT,
        config.COMPLETION_RESERVATION_TOKENS,
    )
    compare = limiter.estimate_tokens(
        prompts.SYSTEM_PROMPT_COMPARE,
        COMPARE_PROMPT,
        config.COMPLETION_RESERVATION_TOKENS_COMPARE,
        static_extra=json.dumps(response_format(LABELS), sort_keys=True),
    )
    analyzer = limiter.estimate_tokens(
        prompts.ANALYZER_SYSTEM_PROMPT, "<pertanyaan>tim juara</pertanyaan>", 120
    )

    # A single answer reservation plus the cap it declares must fit the window
    # (the provider checks prompt + declared cap, so this is the real 413 guard).
    assert compare + config.LLM_MAX_OUTPUT_TOKENS_COMPARE <= limiter.tokens_per_minute
    assert answer + config.LLM_MAX_OUTPUT_TOKENS <= limiter.tokens_per_minute
    assert analyzer < 2000
    # The old rule estimated 7,014 for this prompt; the calibration must be
    # closer to the measured 3,656 prompt without dropping below it.
    assert 4_000 < compare < 5_600


def test_the_completion_reservation_uses_the_expected_completion_not_the_cap():
    client = GroqLLMClient(api_key="offline-test", client=SimpleNamespace())

    assert client._completion_reservation(800, False, None) == 600
    assert client._completion_reservation(900, False, response_format(LABELS)) == 700
    assert client._completion_reservation(400, True, {"type": "json_object"}) == 120
    # A tighter cap still wins: the reservation never exceeds what may be spent.
    assert client._completion_reservation(50, False, None) == 50


def test_a_real_429_is_retried_even_though_the_local_estimate_allowed_the_call():
    clock = FakeClock()
    limiter = TokenAwareRateLimiter(
        clock=clock, sleep=clock.advance, token_counter=ProseTokenizer()
    )
    client, provider = make_client(
        clock, limiter, [Provider429(), FakeResponse("akhirnya berhasil")]
    )

    result = compare_call(client, Deadline(60.0, clock=clock))

    assert result.content == "akhirnya berhasil"
    assert len(provider.requests) == 2
    # 2 s of retry-after plus the 1 s the successful call spends.
    assert clock.now == 3.0


# -- M3: queue policy -------------------------------------------------------
def test_a_wait_inside_the_limit_is_served_and_beyond_it_is_refused():
    clock = FakeClock()
    limiter = TokenAwareRateLimiter(
        limits=dict(config.GROQ_LIMITS, tokens_per_minute=100),
        clock=clock,
        sleep=clock.advance,
        token_counter=WordCountTokenizer(),
    )
    limiter.reserve(60)

    served = limiter.reserve(60, max_wait_seconds=90)
    assert served[1] == 60.0
    assert clock.now == 60

    with pytest.raises(MinuteTokenBudgetExhausted) as excinfo:
        limiter.reserve(60, max_wait_seconds=30)
    assert excinfo.value.retry_after_seconds == 60


def test_an_exhausted_daily_budget_keeps_the_daily_quota_text():
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(),
        llm=FakeLLM(
            fail_with=LLMRateLimitError("daily", is_daily_limit=True)
        ),
        embedder=RecordingEmbedder(),
    )

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.status is AnswerStatus.RATE_LIMITED
    assert result.answer == prompts.DAILY_QUOTA_RESPONSE
    assert result.retry_after_seconds is None


def test_two_concurrent_comparison_questions_are_served_never_technical():
    clock = FakeClock()
    limiter = TokenAwareRateLimiter(
        clock=clock,
        sleep=clock.advance,
        token_counter=ProseTokenizer(),
        queue_max_wait_seconds=180,
    )
    seed_q1(limiter, clock)
    clock.jump_to(33.0)

    results, errors = [], []

    def worker():
        client, _ = make_client(clock, limiter, [FakeResponse("ok")])
        try:
            results.append(
                compare_call(client, Deadline(60.0, clock=clock))
            )
        except BaseException as exc:  # pragma: no cover - failure diagnostics
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert not errors
    assert [item.content for item in results] == ["ok", "ok"]
    assert limiter.waits >= 1


# -- M4: answer length ------------------------------------------------------
def _bullets(count, report_id, prefix):
    return [f"- {prefix} {index} [RPT{report_id}]" for index in range(count)]


def test_the_addendum_v1_4_caps_points_per_report_and_keeps_the_prd_core():
    assert prompts.ADDENDUM_VERSION == "1.4"
    assert prompts.SYSTEM_PROMPT.startswith(prompts.PRD_9_1_SYSTEM_PROMPT)
    assert (
        f"paling banyak {config.ANSWER_MAX_BULLETS_PER_REPORT} poin untuk setiap "
        "laporan" in prompts.CITATION_FORMAT_ADDENDUM
    )


def test_trim_bullets_keeps_the_real_shape_balanced_across_reports():
    body = (
        ["Pembelajaran utama tim:", ""]
        + _bullets(8, "0001", "Poin")
        + _bullets(2, "0002", "Poin")
        + _bullets(1, "0003", "Poin")
        + ["", "Secara ringkas, ketiga laporan menekankan persiapan riset."]
    )
    trimmed, dropped = trim_bullets("\n".join(body))

    lines = trimmed.splitlines()
    kept = [line for line in lines if line.startswith("- ")]
    assert len(kept) == config.ANSWER_MAX_BULLETS
    assert dropped == 5
    for report_id in ("RPT0001", "RPT0002", "RPT0003"):
        assert report_id in trimmed
    assert "Pembelajaran utama tim:" in lines
    assert "Secara ringkas, ketiga laporan menekankan persiapan riset." in lines


def test_trim_bullets_leaves_a_six_bullet_answer_untouched():
    text = "Intro.\n" + "\n".join(_bullets(6, "0001", "Poin"))

    trimmed, dropped = trim_bullets(text)

    assert dropped == 0
    assert trimmed == text


def test_the_pipeline_trims_an_overlong_answer_and_rebuilds_the_sources():
    chunks = [
        StoredChunk(
            chunk_id=f"RPT{report_id}:evaluasi_tim:01",
            report_id=f"RPT{report_id}",
            document=f"Isi evaluasi RPT{report_id}.",
            metadata=metadata(f"RPT{report_id}"),
        )
        for report_id in ("0001", "0002", "0003")
    ]
    answer = "\n".join(
        ["Pembelajaran dari ketiga laporan:", ""]
        + _bullets(8, "0001", "Poin")
        + _bullets(2, "0002", "Poin")
        + _bullets(1, "0003", "Poin")
        + ["", "Secara ringkas, semua tim menekankan persiapan."]
    )
    pipeline, _, _, _ = build(
        chunks=chunks, json_responses=[], text_responses=[answer]
    )

    result = pipeline.run("Bandingkan evaluasi RPT0001, RPT0002, dan RPT0003.")

    assert result.status is AnswerStatus.ANSWERED
    assert result.debug.dropped_bullets == 5
    assert result.debug.incidents["bullets_trimmed"] == 5
    assert result.cited_report_ids == ["RPT0001", "RPT0002", "RPT0003"]
    body, _, sources = result.answer.partition("**Sumber**")
    assert len([line for line in body.splitlines() if line.startswith("- ")]) == 6
    for report_id in ("RPT0001", "RPT0002", "RPT0003"):
        assert f"- {report_id} " in sources
