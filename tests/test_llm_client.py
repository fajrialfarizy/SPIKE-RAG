"""
Offline tests for the Groq client wrapper.

No network, no API key: a fake Groq client object records the request payload and
returns a scripted response or raises a duck-typed provider error.
"""

import pytest

import config
from core.llm_client import (
    GroqLLMClient,
    LLMConfigurationError,
    LLMResponseError,
    LLMUnavailableError,
)

PROMPT = "instruksi sistem"
USER = "pertanyaan mahasiswa"


class ProviderError(Exception):
    """Duck-typed stand-in for groq APIStatusError / APITimeoutError."""

    def __init__(self, status_code=None, detail="raw-provider-detail-MUST-NOT-LEAK"):
        super().__init__(detail)
        if status_code is not None:
            self.status_code = status_code


class APITimeoutError(Exception):
    """Class name matches the groq SDK timeout error the classifier looks for."""


class FakeReasoningDetails:
    def __init__(self, reasoning_tokens):
        self.reasoning_tokens = reasoning_tokens


class FakeUsage:
    def __init__(self, prompt_tokens=11, completion_tokens=7, reasoning_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.completion_tokens_details = FakeReasoningDetails(reasoning_tokens)


class FakeMessage:
    def __init__(self, content, reasoning="rahasia penalaran internal"):
        self.content = content
        self.reasoning = reasoning


class FakeChoice:
    def __init__(self, content, reasoning=None):
        self.message = FakeMessage(content, reasoning) if reasoning is not None else FakeMessage(content)


class FakeResponse:
    def __init__(self, content, usage=None):
        self.choices = [FakeChoice(content)]
        self.usage = usage


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeGroqClient:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = type("Chat", (), {"completions": self.completions})()


def make_client(outcomes, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    fake = FakeGroqClient(outcomes)
    client = GroqLLMClient(api_key="dummy-key", client=fake, sleep_fn=sleeps.append)
    return client, fake, sleeps


def call_result(client, json_mode=False, deadline=None):
    return client.complete(
        PROMPT,
        USER,
        temperature=config.LLM_TEMPERATURE_CHAT,
        max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
        reasoning_effort=config.ANSWER_REASONING_EFFORT,
        json_mode=json_mode,
        deadline=deadline,
    )


def call(client, json_mode=False, deadline=None):
    return call_result(client, json_mode=json_mode, deadline=deadline).content


def test_returns_final_content_only_and_never_reasoning():
    client, fake, _ = make_client([FakeResponse("Jawaban final [LAP1].")])

    answer = call(client)

    assert answer == "Jawaban final [LAP1]."
    assert "penalaran" not in answer


def test_token_usage_is_reported_per_call():
    response = FakeResponse("ok", usage=FakeUsage(120, 40, 25))
    client, _, _ = make_client([response])

    result = call_result(client)

    assert result.usage.prompt_tokens == 120
    assert result.usage.completion_tokens == 40
    assert result.usage.reasoning_tokens == 25
    assert result.usage.to_dict()["total_tokens"] == 160
    assert result.attempts == 1
    assert result.latency_ms >= 0.0


def test_a_response_without_usage_reports_zeroes():
    client, _, _ = make_client([FakeResponse("ok")])

    usage = call_result(client).usage.to_dict()

    assert usage == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }


def test_request_payload_matches_verified_groq_contract():
    client, fake, _ = make_client([FakeResponse("ok")])
    call(client)

    request = fake.completions.requests[0]
    assert request["model"] == config.LLM_MODEL_NAME == "openai/gpt-oss-20b"
    assert request["messages"][0] == {"role": "system", "content": PROMPT}
    assert request["messages"][1] == {"role": "user", "content": USER}
    assert request["max_completion_tokens"] == config.LLM_MAX_OUTPUT_TOKENS
    assert request["reasoning_effort"] == config.ANSWER_REASONING_EFFORT
    assert request["timeout"] == config.LLM_TIMEOUT_SECONDS
    # GPT-OSS: reasoning_format is unsupported; include_reasoning=False hides it.
    assert request["include_reasoning"] is False
    assert "reasoning_format" not in request
    assert request["stream"] is False
    assert "response_format" not in request


def test_json_mode_sets_response_format():
    client, fake, _ = make_client([FakeResponse('{"in_scope": true}')])
    call(client, json_mode=True)

    assert fake.completions.requests[0]["response_format"] == {"type": "json_object"}


def test_retryable_429_backs_off_exponentially_then_succeeds():
    client, fake, sleeps = make_client(
        [ProviderError(429), ProviderError(429), FakeResponse("akhirnya berhasil")]
    )

    assert call(client) == "akhirnya berhasil"
    assert len(fake.completions.requests) == 3
    assert sleeps == [config.LLM_RETRY_BASE_DELAY_SECONDS, config.LLM_RETRY_BASE_DELAY_SECONDS * config.LLM_RETRY_BACKOFF_FACTOR]


def test_retries_are_capped_and_raise_sanitized_error():
    client, fake, sleeps = make_client([ProviderError(429)] * (config.LLM_MAX_RETRIES + 1))

    with pytest.raises(LLMUnavailableError) as excinfo:
        call(client)

    assert len(fake.completions.requests) == config.LLM_MAX_RETRIES + 1
    assert len(sleeps) == config.LLM_MAX_RETRIES
    message = str(excinfo.value)
    assert "raw-provider-detail" not in message
    assert "rate limited" in message


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_5xx_is_retried(status):
    client, fake, _ = make_client([ProviderError(status), FakeResponse("ok")])

    assert call(client) == "ok"
    assert len(fake.completions.requests) == 2


def test_timeout_is_retried():
    client, fake, sleeps = make_client([APITimeoutError("connection timed out"), FakeResponse("ok")])

    assert call(client) == "ok"
    assert len(fake.completions.requests) == 2
    assert len(sleeps) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_other_4xx_is_not_retried(status):
    client, fake, sleeps = make_client([ProviderError(status)])

    with pytest.raises(LLMUnavailableError) as excinfo:
        call(client)

    assert len(fake.completions.requests) == 1
    assert sleeps == []
    assert "raw-provider-detail" not in str(excinfo.value)


def test_missing_api_key_fails_loudly_without_leaking(monkeypatch):
    monkeypatch.delenv(config.LLM_API_KEY_ENV_VAR, raising=False)
    client = GroqLLMClient()

    with pytest.raises(LLMConfigurationError) as excinfo:
        call(client)

    assert config.LLM_API_KEY_ENV_VAR in str(excinfo.value)


def test_api_key_from_environment_is_used(monkeypatch):
    monkeypatch.setenv(config.LLM_API_KEY_ENV_VAR, "dummy-env-key")
    fake = FakeGroqClient([FakeResponse("ok")])
    client = GroqLLMClient(client=fake)

    assert call(client) == "ok"
    assert client._api_key is None  # the key lives only in the environment


def test_empty_content_is_an_error_not_an_empty_answer():
    client, _, _ = make_client([FakeResponse("   ")])

    with pytest.raises(LLMResponseError):
        call(client)


def test_missing_choices_is_an_error():
    client, _, _ = make_client([type("R", (), {"choices": []})()])

    with pytest.raises(LLMResponseError):
        call(client)
