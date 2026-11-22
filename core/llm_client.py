"""
Groq LLM client for the Lab Prestasi RAG chat pipeline.

Facts verified against https://console.groq.com/docs on 2026-11-20:

- The model id ``openai/gpt-oss-20b`` exists on Groq.
- ``reasoning_effort`` supports ``low``, ``medium`` and ``high`` for GPT-OSS
  models (``medium`` is the server default); other values are rejected with 400.
- ``reasoning_format`` is NOT supported for GPT-OSS models. Reasoning is returned
  in ``message.reasoning``; ``include_reasoning=False`` removes it entirely.
- JSON Object Mode (``response_format={\"type\": \"json_object\"}``) and structured
  outputs (``json_schema``, ``strict: true``) are both supported for this model.

Rate limits (https://console.groq.com/docs/rate-limits, checked 2026-11-20):
every response carries ``x-ratelimit-limit-*``, ``x-ratelimit-remaining-*`` and
``x-ratelimit-reset-*`` headers; ``retry-after`` is sent only on a 429 and is the
authoritative wait time. Request counters are per day, token counters per minute.

Retry policy (owned here; the SDK's own retries are disabled):
- 429 with ``retry-after``: honour the header, bounded by the remaining pipeline
  deadline. If the wait cannot fit the budget, fail fast instead of retrying.
- 429 with a daily-style reset (no header, or a reset far beyond one minute):
  fail fast, because waiting a few seconds cannot help.
- Otherwise: the exponential backoff from config.

Only the final ``message.content`` is ever returned as the answer; reasoning
content is never read, logged or surfaced, and provider text never reaches a
caller. Token usage and per-call latency are reported alongside the content.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

import config
from core.deadline import Deadline, DeadlineExceeded, QueueDeadlineExceeded
from core.rate_limiter import (
    DailyTokenBudgetExhausted,
    MinuteTokenBudgetExhausted,
    TokenAwareRateLimiter,
    TokenBudgetExceeded,
    get_token_limiter,
)

logger = logging.getLogger("core_llm")


class LLMError(RuntimeError):
    """Base class for every LLM failure. Messages are always sanitized."""


class LLMConfigurationError(LLMError):
    """Raised when the client cannot be configured (missing key/package)."""


class LLMUnavailableError(LLMError):
    """Raised when the provider is unreachable, rate limited or keeps failing."""


class LLMRateLimitError(LLMUnavailableError):
    """The final failure cause was a rate limit (RATE_LIMITED at the boundary)."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: Optional[float] = None,
        is_daily_limit: bool = False,
    ):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.is_daily_limit = is_daily_limit


class LLMResponseError(LLMError):
    """Raised when the provider returns an unusable payload."""


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by the provider for one call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    # Fix pack H5b: prompt tokens the provider served from its cache. None means
    # the provider did not report the field (as opposed to reported zero).
    cached_tokens: Optional[int] = None

    @property
    def total(self) -> int:
        if self.total_tokens:
            return self.total_tokens
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens

    def to_dict(self) -> Dict[str, int]:
        data: Dict[str, int] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total,
        }
        if self.cached_tokens is not None:
            data["cached_tokens"] = self.cached_tokens
        return data


@dataclass(frozen=True)
class LLMResult:
    """The answer text plus the operational evidence a run needs."""

    content: str
    usage: TokenUsage = TokenUsage()
    latency_ms: float = 0.0
    attempts: int = 1
    finish_reason: Optional[str] = None
    # Fix pack H5a: waiting inside this process (token reservation plus the
    # concurrency slot) is separate from the provider's own response time.
    queue_wait_ms: float = 0.0
    model_latency_ms: float = 0.0

    # Fix pack L5b: what the token limiter saw when this call reserved its
    # budget, so a long wait can be explained from the run output.
    reservation: Dict[str, Any] = field(default_factory=dict)

    def to_metrics(self, name: str) -> Dict[str, Any]:
        return {
            "name": name,
            "latency_ms": round(self.latency_ms, 1),
            "queue_wait_ms": round(self.queue_wait_ms, 1),
            "model_latency_ms": round(self.model_latency_ms, 1),
            "attempts": self.attempts,
            "usage": self.usage.to_dict(),
            "finish_reason": self.finish_reason,
            "reservation": dict(self.reservation),
        }


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
        response_format: Optional[Mapping[str, Any]] = None,
        deadline: Optional[Deadline] = None,
    ) -> LLMResult:
        """Return the final content only (never reasoning).

        ``response_format`` carries a full Structured Outputs object (fix pack
        J1); when given it wins over ``json_mode``.
        """


class LLMConcurrencyLimiter:
    """
    Process-wide cap on simultaneous provider calls (report F1e).

    Streamlit runs sessions in threads, so concurrent questions would otherwise
    collide on the same per-minute allowance. Waiting counts against the
    question's deadline and never blocks forever.
    """

    def __init__(self, limit: int = config.LLM_MAX_CONCURRENT_CALLS):
        if limit <= 0:
            raise ValueError("LLM concurrency limit must be positive.")
        self.limit = int(limit)
        self._semaphore = threading.BoundedSemaphore(self.limit)
        self.peak = 0
        self._active = 0
        self._lock = threading.Lock()

    @contextmanager
    def slot(self, deadline: Optional[Deadline] = None):
        # Waiting for a slot is a local queue, not model time: the caller refunds
        # it to the deadline afterwards (fix pack 7).
        timeout = None if deadline is None else max(0.0, deadline.remaining())
        if timeout is not None and timeout <= 0:
            raise QueueDeadlineExceeded(
                "No budget is left to wait for an LLM slot."
            )
        acquired = self._semaphore.acquire(
            blocking=True, timeout=timeout
        )
        if not acquired:
            raise QueueDeadlineExceeded(
                "Timed out waiting for an LLM concurrency slot."
            )
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
            self._semaphore.release()


_default_limiter: Optional[LLMConcurrencyLimiter] = None
_default_limiter_lock = threading.Lock()


def get_llm_limiter() -> LLMConcurrencyLimiter:
    """The single process-wide limiter, created once from config."""
    global _default_limiter
    with _default_limiter_lock:
        if _default_limiter is None:
            _default_limiter = LLMConcurrencyLimiter(config.LLM_MAX_CONCURRENT_CALLS)
        return _default_limiter


def get_api_key() -> Optional[str]:
    """Read the API key from the environment. The value is never logged."""
    return os.environ.get(config.LLM_API_KEY_ENV_VAR)


def _optional_int_field(obj: Any, name: str) -> Optional[int]:
    """Like _int_field, but keeps "not reported" distinct from a reported zero."""
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, Mapping):
        value = obj.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_field(obj: Any, name: str) -> int:
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, Mapping):
        value = obj.get(name)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _duration_seconds(value: Any) -> Optional[float]:
    """Parse Groq's reset durations such as ``6h12m3.5s`` and plain seconds."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass

    matches = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s|d)", text)
    if not matches:
        return None
    factors = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    return sum(float(number) * factors[unit] for number, unit in matches)


class GroqLLMClient(LLMClient):
    """Chat-completions client for openai/gpt-oss-20b on Groq."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = config.LLM_MODEL_NAME,
        client: Optional[Any] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        instruction_role: str = config.LLM_INSTRUCTION_ROLE,
        limiter: Optional[LLMConcurrencyLimiter] = None,
        token_limiter: Optional[TokenAwareRateLimiter] = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        if instruction_role not in config.LLM_INSTRUCTION_ROLES:
            raise LLMConfigurationError(
                f"LLM_INSTRUCTION_ROLE '{instruction_role}' is not supported; "
                f"use one of {config.LLM_INSTRUCTION_ROLES}."
            )
        self.model_name = model_name
        self._api_key = api_key
        self._client = client
        self._sleep = sleep_fn
        self.instruction_role = instruction_role
        self._limiter = limiter
        self._token_limiter = token_limiter
        self._clock = clock

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

    def warm_up(self) -> float:
        """
        Create the provider client before the first question (fix pack 8, N4).

        ``from groq import Groq`` plus the HTTP client construction costs seconds
        on a cold process, and it happened inside ``complete`` after the call clock
        started, so it landed in the pipeline's post_ms on the first question and
        in no call metric at all. Startup calls this once and logs the duration.
        """
        started = time.perf_counter()
        self._ensure_client()
        return round((time.perf_counter() - started) * 1000.0, 1)

    @property
    def limiter(self) -> LLMConcurrencyLimiter:
        if self._limiter is None:
            self._limiter = get_llm_limiter()
        return self._limiter

    @property
    def token_limiter(self) -> TokenAwareRateLimiter:
        if self._token_limiter is None:
            self._token_limiter = get_token_limiter()
        return self._token_limiter

    # -- error classification ---------------------------------------------
    @staticmethod
    def _status_code(exc: BaseException) -> Optional[int]:
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "response", None), "status_code", None)
        try:
            return int(status) if status is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _classify_error(cls, exc: BaseException) -> tuple[bool, str]:
        """Map an exception to (retryable, short_reason) without leaking text."""
        status = cls._status_code(exc)

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
    def _response_headers(exc: BaseException) -> Dict[str, str]:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return {}
        try:
            return {str(key).lower(): str(value) for key, value in headers.items()}
        except (AttributeError, TypeError, ValueError):
            return {}

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Exponential backoff for the given 1-based attempt number."""
        delay = config.LLM_RETRY_BASE_DELAY_SECONDS * (
            config.LLM_RETRY_BACKOFF_FACTOR ** (attempt - 1)
        )
        return min(delay, config.LLM_RETRY_MAX_DELAY_SECONDS)

    @staticmethod
    def _is_daily_limit(headers: Mapping[str, str]) -> bool:
        """A daily window cannot be waited out inside one question."""
        remaining = headers.get("x-ratelimit-remaining-requests")
        reset = _duration_seconds(headers.get("x-ratelimit-reset-requests"))
        if remaining is None:
            return False
        try:
            exhausted = int(float(remaining)) <= 0
        except (TypeError, ValueError):
            return False
        return exhausted and reset is not None and reset > config.LLM_DAILY_LIMIT_MIN_RESET_SECONDS

    # -- response handling -------------------------------------------------
    @staticmethod
    def _extract(response: Any) -> LLMResult:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMResponseError("LLM response contained no choices.")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("LLM returned an empty final answer.")

        usage_obj = getattr(response, "usage", None)
        details = getattr(usage_obj, "completion_tokens_details", None)
        prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
        usage = TokenUsage(
            prompt_tokens=_int_field(usage_obj, "prompt_tokens"),
            completion_tokens=_int_field(usage_obj, "completion_tokens"),
            reasoning_tokens=_int_field(details, "reasoning_tokens"),
            total_tokens=_int_field(usage_obj, "total_tokens"),
            cached_tokens=_optional_int_field(prompt_details, "cached_tokens"),
        )

        finish_reason = getattr(choices[0], "finish_reason", None)
        if finish_reason is not None:
            finish_reason = str(finish_reason)

        # message.reasoning is deliberately never read.
        return LLMResult(
            content=content.strip(), usage=usage, finish_reason=finish_reason
        )

    # -- message assembly --------------------------------------------------
    def _build_messages(self, system_prompt: str, user_prompt: str) -> list:
        """Place the instruction block according to config.LLM_INSTRUCTION_ROLE."""
        if self.instruction_role == "system":
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        if self.instruction_role == "user_prefixed":
            return [
                {
                    "role": "user",
                    "content": (
                        f"{system_prompt}{config.LLM_INSTRUCTION_USER_SEPARATOR}"
                        f"{user_prompt}"
                    ),
                }
            ]
        raise LLMConfigurationError(
            f"LLM_INSTRUCTION_ROLE '{self.instruction_role}' is not supported; "
            f"use one of {config.LLM_INSTRUCTION_ROLES}."
        )

    def _request_kwargs(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str,
        json_mode: bool,
        timeout_seconds: float,
        response_format: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        kwargs: dict = {
            "model": self.model_name,
            "messages": self._build_messages(system_prompt, user_prompt),
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "include_reasoning": config.LLM_INCLUDE_REASONING,
            "stream": False,
            "timeout": timeout_seconds,
        }
        if response_format is not None:
            # Structured Outputs (json_schema) supersede JSON Object Mode.
            kwargs["response_format"] = dict(response_format)
        elif json_mode and config.LLM_JSON_MODE_ENABLED:
            kwargs["response_format"] = dict(config.LLM_JSON_RESPONSE_FORMAT)
        return kwargs

    @staticmethod
    def _completion_reservation(
        max_tokens: int,
        json_mode: bool,
        response_format: Optional[Mapping[str, Any]],
    ) -> int:
        """Expected completion tokens for this call (fix pack 7).

        The declared cap stays as it is (the provider's ceiling check charges it
        up front), but the local reservation uses the measured expectation per call
        type, so a 900-token cap no longer costs 900 tokens of queue time.
        """
        if response_format and response_format.get("type") == "json_schema":
            expected = config.COMPLETION_RESERVATION_TOKENS_COMPARE
        elif json_mode:
            expected = config.COMPLETION_RESERVATION_TOKENS_ANALYZER
        else:
            expected = config.COMPLETION_RESERVATION_TOKENS
        return min(int(max_tokens), int(expected))

    def _reserve_tokens(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        json_mode: bool,
        response_format: Optional[Mapping[str, Any]],
    ) -> tuple:
        """Reserve this call's estimated tokens or turn the refusal into a 429."""
        limiter = self.token_limiter
        static_extra = (
            json.dumps(response_format, ensure_ascii=False, sort_keys=True)
            if response_format
            else ""
        )
        estimate = limiter.estimate_tokens(
            system_prompt,
            user_prompt,
            self._completion_reservation(max_tokens, json_mode, response_format),
            static_extra=static_extra,
        )
        try:
            return limiter.reserve(estimate), estimate
        except DailyTokenBudgetExhausted as exc:
            logger.warning("llm_token_budget_exhausted window=daily")
            raise LLMRateLimitError(
                "The daily token budget is exhausted.",
                is_daily_limit=True,
            ) from exc
        except TokenBudgetExceeded as exc:
            retry_after = getattr(exc, "retry_after_seconds", None)
            logger.warning("llm_token_budget_exhausted window=minute retry_after=%s",
                           retry_after)
            raise LLMRateLimitError(
                "The per-minute token budget cannot be met inside the queue limit.",
                retry_after_seconds=retry_after,
            ) from exc

    @staticmethod
    def _deadline_elapsed(deadline: Optional[Deadline]) -> float:
        """Queue timing must use the deadline's own clock, not another one."""
        return 0.0 if deadline is None else deadline.elapsed()

    @staticmethod
    def _refund_deadline(deadline: Optional[Deadline], elapsed_before: float) -> None:
        """Give queue time back to a deadline that only guards model time."""
        if deadline is not None:
            deadline.extend(deadline.elapsed() - elapsed_before)

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
        response_format: Optional[Mapping[str, Any]] = None,
        deadline: Optional[Deadline] = None,
    ) -> LLMResult:
        self._ensure_client()

        max_attempts = config.LLM_MAX_RETRIES + 1
        started = self._clock()
        queue_wait_ms = 0.0
        model_latency_ms = 0.0

        for attempt in range(1, max_attempts + 1):
            wait_started = self._clock()
            tokens_before = self.token_limiter.snapshot()["tokens_used_this_minute"]
            reserve_started = self._deadline_elapsed(deadline)
            reservation, estimate = self._reserve_tokens(
                system_prompt, user_prompt, max_tokens, json_mode, response_format
            )
            # A queue wait is not model time: refund it to the deadline, so the
            # answer call still gets the full model budget (fix pack 7).
            self._refund_deadline(deadline, reserve_started)

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
                response_format,
            )
            reservation_info = {
                "tokens_used_last_60s": tokens_before,
                "estimated_tokens": estimate,
                "wait_seconds": round(self._clock() - wait_started, 2),
                "capacity_tokens_per_minute": self.token_limiter.tokens_per_minute,
            }
            try:
                slot_started = self._deadline_elapsed(deadline)
                with self.limiter.slot(deadline):
                    self._refund_deadline(deadline, slot_started)
                    queue_wait_ms += (self._clock() - wait_started) * 1000.0
                    model_started = self._clock()
                    response = self._client.chat.completions.create(**kwargs)
                    model_latency_ms += (self._clock() - model_started) * 1000.0
                result = self._extract(response)
                self.token_limiter.record(reservation, result.usage.total)
                # The estimate is replaced by the reported usage; the unused part
                # returns to the minute window and is never billed again.
                reservation_info["reserved_tokens"] = estimate
                reservation_info["actual_tokens"] = result.usage.total
                reservation_info["unused_released"] = max(
                    0, estimate - int(result.usage.total)
                )
                return LLMResult(
                    content=result.content,
                    usage=result.usage,
                    latency_ms=(self._clock() - started) * 1000.0,
                    attempts=attempt,
                    finish_reason=result.finish_reason,
                    queue_wait_ms=queue_wait_ms,
                    model_latency_ms=model_latency_ms,
                    reservation=reservation_info,
                )
            except LLMError:
                self.token_limiter.release(reservation)
                raise
            except DeadlineExceeded:
                self.token_limiter.release(reservation)
                raise
            except Exception as exc:
                # No usage was reported for this attempt; the reservation is
                # released instead of counting tokens that were never billed.
                self.token_limiter.release(reservation)
                retryable, reason = self._classify_error(exc)
                headers = self._response_headers(exc)
                self.token_limiter.observe_headers(headers)
                retry_after = _duration_seconds(headers.get("retry-after"))
                is_rate_limited = self._status_code(exc) == 429 or reason == "http_429"

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

                if is_rate_limited:
                    daily = self._is_daily_limit(headers)
                    if retry_after is not None:
                        remaining = (
                            deadline.remaining()
                            if deadline is not None
                            else config.PIPELINE_TOTAL_DEADLINE_SECONDS
                        )
                        if retry_after > remaining:
                            logger.warning(
                                "llm_rate_limited_fast_fail retry_after=%.1fs "
                                "remaining=%.1fs daily=%s",
                                retry_after,
                                remaining,
                                daily,
                            )
                            raise LLMRateLimitError(
                                "The provider asked for a wait longer than the "
                                "remaining pipeline budget.",
                                retry_after_seconds=retry_after,
                                is_daily_limit=daily
                                or retry_after > config.LLM_DAILY_LIMIT_MIN_RESET_SECONDS,
                            ) from exc
                        if attempt == max_attempts:
                            raise LLMRateLimitError(
                                f"LLM request was rate limited after {attempt} attempts.",
                                retry_after_seconds=retry_after,
                                is_daily_limit=daily,
                            ) from exc
                        logger.warning(
                            "llm_rate_limited_retry attempt=%d/%d retry_after=%.1fs",
                            attempt,
                            max_attempts,
                            retry_after,
                        )
                        if deadline is not None:
                            deadline.sleep(retry_after, self._sleep)
                        else:
                            self._sleep(retry_after)
                        continue

                    if daily:
                        logger.warning(
                            "llm_rate_limited_fast_fail daily=true attempt=%d", attempt
                        )
                        raise LLMRateLimitError(
                            "The provider's daily request allowance is exhausted.",
                            is_daily_limit=True,
                        ) from exc

                if attempt == max_attempts:
                    logger.warning(
                        "llm_call_failed attempt=%d/%d error=%s retryable=true",
                        attempt,
                        max_attempts,
                        reason,
                    )
                    if is_rate_limited:
                        raise LLMRateLimitError(
                            f"LLM request was rate limited after {attempt} attempts.",
                            retry_after_seconds=retry_after,
                            is_daily_limit=self._is_daily_limit(headers),
                        ) from exc
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
