"""Offline tests for the per-call debug evidence added in fix pack H5."""

import config
from core.llm_client import GroqLLMClient, LLMResult, TokenUsage

PROMPT = "instruksi sistem"
USER = "pertanyaan mahasiswa"


class FakeCachedDetails:
    def __init__(self, cached_tokens):
        self.cached_tokens = cached_tokens


class FakeUsage:
    def __init__(self, cached_tokens=None):
        self.prompt_tokens = 100
        self.completion_tokens = 20
        self.total_tokens = 120
        self.completion_tokens_details = None
        if cached_tokens is not None:
            self.prompt_tokens_details = FakeCachedDetails(cached_tokens)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)
        self.finish_reason = "stop"


class FakeResponse:
    def __init__(self, content, usage):
        self.choices = [FakeChoice(content)]
        self.usage = usage


class FakeCompletions:
    def __init__(self, response):
        self.response = response

    def create(self, **kwargs):
        return self.response


class FakeGroqClient:
    def __init__(self, response):
        self.chat = type(
            "Chat", (), {"completions": FakeCompletions(response)}
        )()


def call_client(usage):
    client = GroqLLMClient(
        api_key="dummy-key", client=FakeGroqClient(FakeResponse("ok", usage))
    )
    return client.complete(
        PROMPT,
        USER,
        temperature=config.LLM_TEMPERATURE_CHAT,
        max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
        reasoning_effort=config.ANSWER_REASONING_EFFORT,
    )


def test_cached_prompt_tokens_are_recorded_when_the_provider_reports_them():
    result = call_client(FakeUsage(cached_tokens=64))

    assert result.usage.cached_tokens == 64
    assert result.usage.to_dict()["cached_tokens"] == 64


def test_a_provider_without_the_cached_field_reports_nothing():
    result = call_client(FakeUsage())

    assert result.usage.cached_tokens is None
    assert "cached_tokens" not in result.usage.to_dict()


def test_each_call_metric_separates_queue_wait_from_model_latency():
    metrics = LLMResult(
        content="ok", latency_ms=1200.0, queue_wait_ms=900.0, model_latency_ms=250.0
    ).to_metrics("answer")

    assert metrics["queue_wait_ms"] == 900.0
    assert metrics["model_latency_ms"] == 250.0
    assert metrics["latency_ms"] == 1200.0
    assert metrics["usage"] == TokenUsage().to_dict()
