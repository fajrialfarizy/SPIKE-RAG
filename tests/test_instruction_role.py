"""
Offline tests for config.LLM_INSTRUCTION_ROLE (report K3).

Both placements must carry exactly the same instruction text; only its position
changes, and the default stays "system".
"""

import pytest

import config
from core.llm_client import GroqLLMClient, LLMConfigurationError
from tests.test_llm_client import FakeGroqClient, FakeResponse, call

PROMPT = "instruksi sistem"
USER = "pertanyaan mahasiswa"


def test_the_default_is_a_system_message():
    assert config.LLM_INSTRUCTION_ROLE == "system"

    fake = FakeGroqClient([FakeResponse("ok")])
    client = GroqLLMClient(api_key="dummy-key", client=fake, sleep_fn=lambda _: None)
    call(client)

    messages = fake.completions.requests[0]["messages"]
    assert messages == [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": USER},
    ]


def test_user_prefixed_puts_the_same_text_at_the_start_of_the_user_message():
    fake = FakeGroqClient([FakeResponse("ok")])
    client = GroqLLMClient(
        api_key="dummy-key",
        client=fake,
        sleep_fn=lambda _: None,
        instruction_role="user_prefixed",
    )
    call(client)

    messages = fake.completions.requests[0]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == f"{PROMPT}{config.LLM_INSTRUCTION_USER_SEPARATOR}{USER}"
    assert messages[0]["content"].startswith(PROMPT)


def test_an_unknown_role_is_a_configuration_error():
    with pytest.raises(LLMConfigurationError):
        GroqLLMClient(api_key="dummy-key", instruction_role="developer")
