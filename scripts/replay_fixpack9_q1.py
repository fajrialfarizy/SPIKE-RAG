"""
Offline replay of the Q1 quota sequence that Fix Pack 9 changes.

The supervisor's real run: the answer model returned 12 points with no citation,
so the citation repair retry needed about 3,800 tokens while the minute window
already held 5,034. The wait was 52 s, over QUEUE_MAX_WAIT_SECONDS (45 s), so the
pipeline answered RATE_LIMITED although the model had produced a usable answer
shape, and 3,795 tokens were spent for nothing.

With the structured answer (fix pack 9) the first call already carries the cite
per point, so a "no citation at all" answer is a validation empty answer and the
one repair retry is reserved with QUEUE_MAX_WAIT_RETRY_SECONDS. This script
replays both decisions on a fake clock; no provider call and no API key.

    python scripts/replay_fixpack9_q1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core.rate_limiter import MinuteTokenBudgetExhausted, TokenAwareRateLimiter


class FakeClock:
    """One clock for the limiter's windows and its sleeps."""

    def __init__(self, start: float = 0.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def first_call_tokens() -> int:
    """Tokens the real Q1 first answer left in the minute window."""
    return 5034


def retry_tokens() -> int:
    """Tokens the real citation repair retry reserved."""
    return 3795


def replay(max_wait_seconds: float) -> dict:
    clock = FakeClock()
    limiter = TokenAwareRateLimiter(clock=clock, sleep=clock.advance)
    limiter.reserve(first_call_tokens())

    # The model call and the post-processing before the retry took 8 s; the
    # window was opened at t=0 by the first answer's reservation.
    clock.advance(8.0)
    started_at = clock.now
    try:
        limiter.reserve(retry_tokens(), max_wait_seconds=max_wait_seconds)
    except MinuteTokenBudgetExhausted as exc:
        return {
            "outcome": "RATE_LIMITED",
            "wait_seconds": exc.retry_after_seconds,
            "tokens_spent": first_call_tokens(),
        }
    return {
        "outcome": "retry served",
        "wait_seconds": round(clock.now - started_at, 1),
        "tokens_spent": first_call_tokens() + retry_tokens(),
        "tokens_left_in_window": limiter.snapshot()["tokens_used_this_minute"],
    }


def main() -> int:
    print(f"tpm={config.GROQ_LIMITS['tokens_per_minute']} "
          f"first_call={first_call_tokens()} retry={retry_tokens()}")

    old = replay(config.QUEUE_MAX_WAIT_SECONDS)
    print(
        f"before fix pack 9 (queue limit {config.QUEUE_MAX_WAIT_SECONDS:.0f}s): "
        f"outcome={old['outcome']} wait={old['wait_seconds']}s "
        f"tokens_spent={old['tokens_spent']}"
    )

    new = replay(config.QUEUE_MAX_WAIT_RETRY_SECONDS)
    print(
        f"after fix pack 9 (retry limit {config.QUEUE_MAX_WAIT_RETRY_SECONDS:.0f}s): "
        f"outcome={new['outcome']} wait={new['wait_seconds']}s "
        f"tokens_spent={new['tokens_spent']} "
        f"minute_window={new.get('tokens_left_in_window')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
