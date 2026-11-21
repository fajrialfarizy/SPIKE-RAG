"""
Token-aware request guard in front of every Groq call (fix pack G7).

The binding constraint on this account is tokens, not requests: 8,000 tokens per
minute and 200,000 per day, against 3,800-5,800 tokens per full question. A
request-count guard therefore protects nothing. This module estimates the tokens
of every call before it is made (prompt estimate plus that call's completion cap),
keeps rolling 60-second and 24-hour counters, and refuses or queues the call:

- daily budget exhausted -> ``DailyTokenBudgetExhausted`` (the pipeline answers
  with the daily-quota text and makes no provider call),
- per-minute budget cannot be met inside the pipeline deadline ->
  ``MinuteTokenBudgetExhausted`` (the pipeline answers with the rate-limit text),
- per-minute budget can be met by waiting -> wait, bounded by the deadline.

Response headers refine the local estimate (Groq docs, checked 2026-11-21):
``x-ratelimit-remaining-tokens`` is TPM, ``x-ratelimit-reset-tokens`` is the TPM
reset, ``x-ratelimit-remaining-requests`` is RPD and
``x-ratelimit-reset-requests`` is the RPD reset. TPD is not exposed by any header,
so the daily token counter is always local. Cached prompt tokens do not count
toward the provider limits; the local estimate stays conservative and ignores
that discount.

The clock and sleep function are injectable so tests can drive the windows with a
fake clock. One instance is shared process-wide: Streamlit sessions run in
threads and all of them spend the same account allowance.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional

import config
from core.deadline import Deadline


class TokenBudgetExceeded(RuntimeError):
    """Raised when a call cannot be made inside the token allowance."""


class DailyTokenBudgetExhausted(TokenBudgetExceeded):
    """The 24-hour token budget is gone; waiting inside a question cannot help."""


class MinuteTokenBudgetExhausted(TokenBudgetExceeded):
    """The per-minute budget cannot be met before the pipeline deadline."""


def _integer_header(headers: Mapping[str, str], name: str) -> Optional[int]:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _seconds_header(headers: Mapping[str, str], name: str) -> Optional[float]:
    value = headers.get(name)
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    matches = __import__("re").findall(r"(\d+(?:\.\d+)?)(ms|h|m|s|d)", text)
    if not matches:
        return None
    factors = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    return sum(float(number) * factors[unit] for number, unit in matches)


class TokenAwareRateLimiter:
    """Rolling token and request windows for one provider account."""

    def __init__(
        self,
        limits: Optional[Mapping[str, Any]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        chars_per_token: float = config.CONTEXT_TOKEN_ESTIMATOR_CHARS_PER_TOKEN,
        minute_window_seconds: float = config.TOKEN_LIMITER_MINUTE_WINDOW_SECONDS,
        day_window_seconds: float = config.TOKEN_LIMITER_DAY_WINDOW_SECONDS,
    ):
        self.limits = dict(limits or config.GROQ_LIMITS)
        self.tokens_per_minute = int(self.limits["tokens_per_minute"])
        self.tokens_per_day = int(self.limits["tokens_per_day"])
        self.requests_per_minute = int(self.limits["requests_per_minute"])
        self.requests_per_day = int(self.limits["requests_per_day"])
        self._clock = clock
        self._sleep = sleep
        self._chars_per_token = float(chars_per_token)
        self._minute_window = float(minute_window_seconds)
        self._day_window = float(day_window_seconds)

        self._lock = threading.Lock()
        # Each entry is [timestamp, reserved_tokens]; recording the real usage
        # mutates the entry instead of appending a second one.
        self._minute: list[list[float]] = []
        self._day_tokens = 0
        self._day_started = self._clock()
        self._requests_today = 0

        self._provider_tpm_remaining: Optional[int] = None
        self._provider_tpm_reset: Optional[float] = None
        self._provider_rpd_remaining: Optional[int] = None

        self.waits = 0
        self.total_reserved = 0
        self.total_recorded = 0

    # -- estimation --------------------------------------------------------
    def estimate_tokens(
        self,
        system_prompt: str,
        user_prompt: str,
        max_completion_tokens: int,
    ) -> int:
        """
        Conservative prompt estimate plus the call's completion allowance.

        ``max_completion_tokens`` covers the visible answer and the hidden
        reasoning tokens, because Groq counts both as completion tokens.
        """
        text = f"{system_prompt or ''}{user_prompt or ''}"
        by_chars = math.ceil(len(text) / self._chars_per_token)
        by_words = len(text.split())
        return int(max_completion_tokens) + max(by_chars, by_words)

    # -- windows -----------------------------------------------------------
    def _roll_day(self, now: float) -> None:
        if now - self._day_started >= self._day_window:
            self._day_started = now
            self._day_tokens = 0
            self._requests_today = 0

    def _prune_minute(self, now: float) -> None:
        boundary = now - self._minute_window
        self._minute = [entry for entry in self._minute if entry[0] > boundary]

    def _minute_tokens(self) -> int:
        return sum(int(entry[1]) for entry in self._minute)

    def _minute_wait(self, now: float, tokens: int) -> float:
        """Seconds to wait before ``tokens`` fit the minute window (0 = fits)."""
        used = self._minute_tokens()
        capacity = self.tokens_per_minute
        if self._provider_tpm_remaining is not None:
            capacity = min(capacity, self._provider_tpm_remaining + used)
        wait_tokens = 0.0
        if used + tokens > capacity:
            needed = used + tokens - capacity
            freed = 0
            wait_tokens = self._minute_window
            for timestamp, entry_tokens in self._minute:
                freed += int(entry_tokens)
                if freed >= needed:
                    wait_tokens = max(0.0, timestamp + self._minute_window - now)
                    break

        wait_requests = 0.0
        if len(self._minute) + 1 > self.requests_per_minute and self._minute:
            wait_requests = max(
                0.0, self._minute[0][0] + self._minute_window - now
            )

        return max(wait_tokens, wait_requests)

    def wait_estimate(self, tokens: int) -> float:
        """
        Seconds a call of this size would wait, without reserving anything.

        Read-only on purpose: the app shows this before a question so the notice
        cannot itself consume the window it is describing.
        """
        tokens = max(0, int(tokens))
        with self._lock:
            now = self._clock()
            self._roll_day(now)
            self._prune_minute(now)
            return max(0.0, self._minute_wait(now, tokens))

    def _daily_exhausted(self, tokens: int) -> bool:
        # The local counters are authoritative for the daily window: TPD is not
        # exposed by any header, and a stale x-ratelimit-remaining-requests value
        # must not block the process forever after the provider window resets.
        if self._day_tokens + tokens > self.tokens_per_day:
            return True
        if self._requests_today + 1 > self.requests_per_day:
            return True
        return False

    def _drop_entry(self, entry: list[float]) -> None:
        try:
            self._minute.remove(entry)
        except ValueError:  # already pruned
            pass

    # -- reservation API ---------------------------------------------------
    def reserve(
        self, tokens: int, deadline: Optional[Deadline] = None
    ) -> list[float]:
        """
        Reserve tokens for one call, waiting if only the minute window is tight.

        Returns the reservation entry that ``record`` or ``release`` must settle.
        """
        tokens = max(0, int(tokens))

        # A single call larger than the whole minute window can never fit, no
        # matter how long it waits (Groq answers it with HTTP 413). Fail loudly
        # instead of looping.
        if tokens > self.tokens_per_minute:
            raise MinuteTokenBudgetExhausted(
                "One request needs more tokens than the per-minute limit."
            )

        while True:
            with self._lock:
                now = self._clock()
                self._roll_day(now)
                self._prune_minute(now)

                if self._daily_exhausted(tokens):
                    raise DailyTokenBudgetExhausted(
                        "The daily token budget for the model is exhausted."
                    )

                wait = self._minute_wait(now, tokens)
                if wait <= 0:
                    entry = [now, float(tokens)]
                    self._minute.append(entry)
                    self._day_tokens += tokens
                    self._requests_today += 1
                    self.total_reserved += tokens
                    return entry

            remaining = None if deadline is None else deadline.remaining()
            if remaining is not None and wait > remaining:
                raise MinuteTokenBudgetExhausted(
                    "The per-minute token budget cannot be met before the deadline."
                )
            if remaining is None and wait > self._minute_window:
                raise MinuteTokenBudgetExhausted(
                    "The per-minute token budget cannot be met."
                )

            self.waits += 1
            if deadline is not None:
                deadline.sleep(wait, self._sleep)
            else:
                self._sleep(wait)

    def record(self, entry: list[float], actual_tokens: int) -> None:
        """Replace a reservation with the usage the provider reported."""
        actual = max(0, int(actual_tokens))
        with self._lock:
            reserved = int(entry[1])
            entry[1] = float(actual)
            self._day_tokens = max(0, self._day_tokens - reserved + actual)
            self.total_recorded += actual
            if actual == 0:
                # A response that reports no usage must not keep a slot in the
                # request window.
                self._drop_entry(entry)

    def release(self, entry: list[float]) -> None:
        """Drop a reservation for a call that produced no reportable usage."""
        with self._lock:
            reserved = int(entry[1])
            entry[1] = 0.0
            self._day_tokens = max(0, self._day_tokens - reserved)
            self._drop_entry(entry)

    # -- provider feedback -------------------------------------------------
    def observe_headers(self, headers: Mapping[str, str]) -> None:
        """Refine the local windows from the x-ratelimit-* response headers."""
        if not headers:
            return
        with self._lock:
            remaining_tokens = _integer_header(headers, "x-ratelimit-remaining-tokens")
            reset_tokens = _seconds_header(headers, "x-ratelimit-reset-tokens")
            if remaining_tokens is not None:
                self._provider_tpm_remaining = max(0, remaining_tokens)
                self._provider_tpm_reset = reset_tokens

            remaining_requests = _integer_header(
                headers, "x-ratelimit-remaining-requests"
            )
            if remaining_requests is not None:
                self._provider_rpd_remaining = max(0, remaining_requests)

    # -- reporting ---------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = self._clock()
            self._roll_day(now)
            self._prune_minute(now)
            return {
                "tokens_used_today": self._day_tokens,
                "tokens_remaining_today": max(
                    0, self.tokens_per_day - self._day_tokens
                ),
                "tokens_used_this_minute": self._minute_tokens(),
                "requests_today": self._requests_today,
                "provider_tpm_remaining": self._provider_tpm_remaining,
                "provider_tpm_reset_seconds": self._provider_tpm_reset,
                "provider_rpd_remaining": self._provider_rpd_remaining,
            }

    def capacity_estimate(self, avg_tokens_per_question: float) -> Dict[str, Any]:
        """How many full questions the account can serve at this token average."""
        average = max(1.0, float(avg_tokens_per_question))
        return {
            "avg_tokens_per_question": round(average, 1),
            "questions_per_minute": round(
                min(
                    self.tokens_per_minute / average,
                    float(self.requests_per_minute),
                ),
                2,
            ),
            "questions_per_day": round(
                min(
                    self.tokens_per_day / average,
                    float(self.requests_per_day),
                ),
                1,
            ),
            "tokens_per_minute": self.tokens_per_minute,
            "tokens_per_day": self.tokens_per_day,
        }


_default_limiter: Optional[TokenAwareRateLimiter] = None
_default_limiter_lock = threading.Lock()


def get_token_limiter() -> TokenAwareRateLimiter:
    """The single process-wide limiter, created once from config."""
    global _default_limiter
    with _default_limiter_lock:
        if _default_limiter is None:
            _default_limiter = TokenAwareRateLimiter()
        return _default_limiter
