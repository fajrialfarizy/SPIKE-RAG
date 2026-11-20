"""
Offline tests for the total pipeline deadline (report K2).

Time is injected everywhere: no test sleeps, and the provider is a fake object.
"""

import pytest

import config
from core.analyzer import Analyzer
from core.deadline import Deadline, DeadlineExceeded
from core.llm_client import GroqLLMClient
from core.pipeline import AnswerStatus, RagPipeline
from tests.stubs import FakeLLM, FakeKnowledgeBase, RecordingEmbedder
from tests.test_llm_client import FakeGroqClient, FakeResponse, ProviderError, call


class FakeClock:
    """A clock that only moves when a test says so."""

    def __init__(self, start=1000.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


class SteppingClock:
    """A clock that jumps on every read, to simulate an exhausted budget."""

    def __init__(self, step):
        self.step = float(step)
        self.now = 0.0
        self.reads = 0

    def __call__(self):
        self.reads += 1
        value = self.now
        self.now += self.step
        return value


# -- Deadline value object -------------------------------------------------
def test_remaining_and_elapsed_follow_the_injected_clock():
    clock = FakeClock()
    deadline = Deadline(10.0, clock=clock)

    clock.advance(4.0)

    assert deadline.remaining() == pytest.approx(6.0)
    assert deadline.elapsed() == pytest.approx(4.0)
    assert deadline.expired is False

    clock.advance(6.0)
    assert deadline.expired is True
    with pytest.raises(DeadlineExceeded):
        deadline.require_remaining()


def test_a_non_positive_budget_is_a_configuration_error():
    with pytest.raises(ValueError):
        Deadline(0.0)


def test_call_timeout_is_the_per_call_cap_bounded_by_the_budget():
    clock = FakeClock()
    generous = Deadline(60.0, clock=clock)
    tight = Deadline(5.0, clock=clock)

    assert generous.call_timeout(config.LLM_TIMEOUT_SECONDS) == config.LLM_TIMEOUT_SECONDS
    assert tight.call_timeout(config.LLM_TIMEOUT_SECONDS) == pytest.approx(5.0)

    clock.advance(5.0)
    with pytest.raises(DeadlineExceeded):
        tight.call_timeout(config.LLM_TIMEOUT_SECONDS)


def test_sleep_beyond_the_remaining_budget_raises_without_sleeping():
    clock = FakeClock()
    deadline = Deadline(3.0, clock=clock)
    slept = []

    def sleep_and_advance(seconds):
        slept.append(seconds)
        clock.advance(seconds)

    deadline.sleep(2.0, sleep_and_advance)
    assert slept == [2.0]

    with pytest.raises(DeadlineExceeded):
        deadline.sleep(2.0, sleep_and_advance)
    assert slept == [2.0]


# -- LLM client honours the deadline --------------------------------------
def test_per_call_timeout_is_what_the_client_sends():
    clock = FakeClock()
    fake = FakeGroqClient([FakeResponse("ok")])
    client = GroqLLMClient(api_key="dummy-key", client=fake, sleep_fn=lambda _: None)

    client.complete(
        "sistem",
        "user",
        temperature=0.0,
        max_tokens=10,
        reasoning_effort="low",
        deadline=Deadline(8.0, clock=clock),
    )

    assert fake.completions.requests[0]["timeout"] == pytest.approx(8.0)


def test_backoff_stops_when_the_next_wait_does_not_fit_the_deadline():
    clock = FakeClock()
    sleeps = []

    def sleep_and_advance(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)

    fake = FakeGroqClient([ProviderError(429)] * 4)
    client = GroqLLMClient(api_key="dummy-key", client=fake, sleep_fn=sleep_and_advance)
    deadline = Deadline(5.0, clock=clock)

    with pytest.raises(DeadlineExceeded):
        client.complete(
            "sistem",
            "user",
            temperature=0.0,
            max_tokens=10,
            reasoning_effort="low",
            deadline=deadline,
        )

    # 1s + 2s fit; the next 4s wait does not.
    assert sleeps == [1.0, 2.0]
    assert len(fake.completions.requests) == 3


# -- pipeline boundary ----------------------------------------------------
def build_pipeline(**kwargs):
    llm = FakeLLM(
        json_responses=kwargs.pop("json_responses", []),
        text_responses=kwargs.pop("text_responses", []),
        fail_with=kwargs.pop("fail_with", None),
    )
    pipeline = RagPipeline(
        knowledge_base=FakeKnowledgeBase(),
        llm=llm,
        embedder=RecordingEmbedder(),
        **kwargs,
    )
    return pipeline, llm


def test_exhausted_budget_becomes_technical_error_without_any_llm_call():
    pipeline, llm = build_pipeline(deadline_budget_seconds=5.0, clock=SteppingClock(10.0))

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.status is AnswerStatus.TECHNICAL_ERROR
    assert result.debug.error_type == "DeadlineExceeded"
    assert llm.calls == []


def test_deadline_raised_during_the_analyzer_is_not_turned_into_a_fallback():
    pipeline, llm = build_pipeline(
        fail_with=DeadlineExceeded("budget gone"),
        deadline_budget_seconds=60.0,
    )

    result = pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert result.status is AnswerStatus.TECHNICAL_ERROR
    assert result.debug.error_type == "DeadlineExceeded"
    assert result.debug.used_fallback is False
    assert len(llm.calls) == 1


def test_the_pipeline_hands_one_deadline_object_to_the_analyzer():
    pipeline, llm = build_pipeline(
        json_responses=['{"in_scope": true, "mode": "specific", "search_query": "x", "filters": {}}'],
        text_responses=["Jawaban [LAP1]."],
        deadline_budget_seconds=60.0,
    )
    # No chunks means NOT_FOUND before the answer call; the analyzer still runs.
    pipeline.run("Apa kendala tim pada kompetisi business plan?")

    assert isinstance(llm.calls[0]["deadline"], Deadline)
    assert llm.calls[0]["deadline"].budget_seconds == 60.0
