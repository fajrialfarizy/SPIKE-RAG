"""
Groq LLM client for the Lab Prestasi RAG chat pipeline.

Facts verified against https://console.groq.com/docs on 2026-11-20:

- The model id ``openai/gpt-oss-20b`` exists on Groq.
- ``reasoning_effort`` supports ``low``, ``medium`` and ``high`` for GPT-OSS
  models (``medium`` is the server default); other values are rejected with 400.
- ``reasoning_format`` is NOT supported for GPT-OSS models. Reasoning is returned
  in ``message.reasoning``; ``include_reasoning=False`` removes it entirely.
- JSON Object Mode (``response_format={"type": "json_object"}``) and structured
  outputs (``json_schema``, ``strict: true``) are both supported for this model.

Only the final ``message.content`` is ever returned. Reasoning content is never
read, logged or surfaced. Retries are handled here (the SDK's own retry loop is
disabled) so the retry policy is exactly the one in config, and error text is
never passed through to callers or users.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import config
from core.deadline import Deadline, DeadlineExceeded

logger = logging.getLogger("core_llm")


class LLMError(RuntimeError):
    """Base class for every LLM failure. Messages are always sanitized."""


class LLMConfigurationError(LLMError):
    """Raised when the client cannot be configured (missing key/package)."""


class LLMUnavailableError(LLMError):
    """Raised when the provider is unreachable, rate limited or keeps failing."""


class LLMResponseError(LLMError):
    """Raised when the provider returns an unusable payload."""


class LLMClient(ABC):
    """Minimal LLM interface; tests inject a scripted double from tests/."""

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str,
        json_mode: bool = False,
        deadline: Optional[Deadline] = None,
    ) -> str:
        """Return the final answer content only (never reasoning)."""


def get_api_key() -> Optional[str]:
    """Read the API key from the environment. The value is never logged."""
    return os.environ.get(config.LLM_API_KEY_ENV_VAR)


class GroqLLMClient(LLMClient):
    """Chat-completions client for openai/gpt-oss-20b on Groq."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = config.LLM_MODEL_NAME,
        client: Optional[Any] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.model_name = model_name
        self._api_key = api_key
        self._client = client
        self._sleep = sleep_fn

    # -- configuration -----------------------------------------------------
    def _ensure_client(self):
        if self._client is not None:
            return

        api_key = self._api_key or get_api_key()
        if not api_key:
            raise LLMConfigurationError(
                f"{config.LLM_API_KEY_ENV_VAR} is not set, so the Groq client "
                "cannot be created. Set the environment variable; the value is "
                "never logged."
            )

        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise LLMConfigurationError(
                "The 'groq' package is not installed, so the LLM cannot be used."
            ) from exc

        # max_retries=0: this module owns the retry policy.
        self._client = Groq(
            api_key=api_key,
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )

    # -- error classification ---------------------------------------------
    @staticmethod
    def _classify_error(exc: BaseException) -> tuple[bool, str]:
        """Map an exception to (retryable, short_reason) without leaking text."""
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "response", None), "status_code", None)

        name = type(exc).__name__
        if name in {"APITimeoutError", "TimeoutError", "ReadTimeout"}:
            return True, "timeout"
        if name in {"APIConnectionError", "ConnectError", "ConnectionError"}:
            return True, "connection"

        if isinstance(status, int):
            if status in config.LLM_RETRY_STATUS_CODES:
                return True, f"http_{status}"
            return False, f"http_{status}"

        return False, "unexpected_error"

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Exponential backoff for the given 1-based attempt number."""
        delay = config.LLM_RETRY_BASE_DELAY_SECONDS * (
            config.LLM_RETRY_BACKOFF_FACTOR ** (attempt - 1)
        )
        return min(delay, config.LLM_RETRY_MAX_DELAY_SECONDS)

    # -- response handling -------------------------------------------------
    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMResponseError("LLM response contained no choices.")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("LLM returned an empty final answer.")

        # message.reasoning is deliberately never read.
        return content.strip()

    def _request_kwargs(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str,
        json_mode: bool,
        timeout_seconds: float,
    ) -> dict:
        kwargs: dict = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "include_reasoning": config.LLM_INCLUDE_REASONING,
            "stream": False,
            "timeout": timeout_seconds,
        }
        if json_mode and config.LLM_JSON_MODE_ENABLED:
            kwargs["response_format"] = dict(config.LLM_JSON_RESPONSE_FORMAT)
        return kwargs

    # -- public API --------------------------------------------------------
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str,
        json_mode: bool = False,
        deadline: Optional[Deadline] = None,
    ) -> str:
        self._ensure_client()

        max_attempts = config.LLM_MAX_RETRIES + 1

        for attempt in range(1, max_attempts + 1):
            if deadline is not None:
                timeout_seconds = deadline.call_timeout(config.LLM_TIMEOUT_SECONDS)
            else:
                timeout_seconds = config.LLM_TIMEOUT_SECONDS

            kwargs = self._request_kwargs(
                system_prompt,
                user_prompt,
                temperature,
                max_tokens,
                reasoning_effort,
                json_mode,
                timeout_seconds,
            )
            try:
                response = self._client.chat.completions.create(**kwargs)
                return self._extract_content(response)
            except LLMError:
                raise
            except Exception as exc:
                retryable, reason = self._classify_error(exc)

                if not retryable:
                    logger.warning(
                        "llm_call_failed attempt=%d/%d error=%s retryable=false",
                        attempt,
                        max_attempts,
                        reason,
                    )
                    raise LLMUnavailableError(
                        f"LLM request failed ({reason}); it is not retryable."
                    ) from exc

                if attempt == max_attempts:
                    logger.warning(
                        "llm_call_failed attempt=%d/%d error=%s retryable=true",
                        attempt,
                        max_attempts,
                        reason,
                    )
                    raise LLMUnavailableError(
                        f"LLM request failed ({reason}) after {attempt} attempts."
                    ) from exc

                delay = self._backoff_delay(attempt)
                logger.warning(
                    "llm_retry attempt=%d/%d error=%s delay=%.2fs",
                    attempt,
                    max_attempts,
                    reason,
                    delay,
                )
                if deadline is not None:
                    deadline.sleep(delay, self._sleep)
                else:
                    self._sleep(delay)

        # Unreachable: the loop either returns or raises.
        raise LLMUnavailableError("LLM request failed after all retries.")
