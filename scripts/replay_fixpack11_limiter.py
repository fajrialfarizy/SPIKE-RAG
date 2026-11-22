"""
Offline replay of the fix pack 10/11 limiter sequence (fix pack 11, S2c).

The real Fix Pack 10 run showed this sequence at about 8,000 tokens/minute:

    t~5s    Q1 id_route answer: reserved 3,291, actual 2,769
    t~33s   Q2 analyzer:        1,257
    t~36s   Q2 compare:         reserved ~5,400, waited 27 s, then HTTP 400
    t~95s   Q3 analyzer:        1,256
    t~96s   Q3 compare:         reserved ~5,400 -> llm_token_budget_exhausted
                                window=minute retry_after=60

This script replays that timeline with a FAKE CLOCK and the REAL
``TokenAwareRateLimiter`` (with a word-count tokenizer so no model is loaded),
and prints the window contents and the computed wait under four behaviours:

1. ``released``: the current code. A non-retryable 4xx releases the reservation.
2. ``leaked``:   the hypothesis from the run report, i.e. the failed call's
                 reservation stays in the window.
3. ``clamp-never``: a stale provider header clamp
                 (``x-ratelimit-remaining-tokens``) that never expires, which is
                 what the code did before the fix.
4. ``clamp-expires``: the same header with its documented reset, i.e. the fix.

It answers one question explicitly: does the leaked reservation explain
retry_after=60? No. A leaked 5,400-token reservation makes the Q3 compare call
wait about 29 s, which is under the 45 s queue limit, so it would have waited and
succeeded. Only a capacity clamp that never expires produces exactly 60 s,
because then no amount of waiting inside the window can make the call fit.

No model, no network, no API key.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core.rate_limiter import MinuteTokenBudgetExhausted, TokenAwareRateLimiter
from tests.stubs import WordCountTokenizer

# The measured reservations and actual usages of the real run.
Q1_ANSWER_RESERVED = 3_291
Q1_ANSWER_ACTUAL = 2_769
Q2_ANALYZER_TOKENS = 1_257
Q2_COMPARE_RESERVED = 5_400
Q3_ANALYZER_TOKENS = 1_256
Q3_COMPARE_RESERVED = 5_400

Q1_AT = 0.0
Q1_RECORDED_AT = 5.0
Q2_ANALYZER_AT = 33.0
Q2_COMPARE_AT = 36.0
Q3_ANALYZER_AT = 95.0
Q3_COMPARE_AT = 96.0
# The 400 response of the run arrived after the limiter wait.
Q2_COMPARE_FAILED_AT = 65.0
# A 400 response also carries rate-limit headers. The run's Q3 analyzer (1,256
# tokens) still succeeded, so the remaining value the 400 reported can be small
# but non-zero: 1,500 fits the analyzer and cannot fit a 5,400-token answer.
PROVIDER_REMAINING_TOKENS = 1_500
PROVIDER_RESET_SECONDS = 30.0


class FakeClock:
    """Monotonic fake clock; sleeping advances it instead of blocking."""

    def __init__(self, start: float = 0.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


def _limiter(clock: FakeClock) -> TokenAwareRateLimiter:
    return TokenAwareRateLimiter(
        clock=clock,
        sleep=clock.advance,
        token_counter=WordCountTokenizer(),
    )


def _window(limiter: TokenAwareRateLimiter) -> List[Dict[str, Any]]:
    """The minute entries as (timestamp, tokens), oldest first."""
    return [
        {"at": round(float(entry[0]), 1), "tokens": int(entry[1])}
        for entry in list(limiter._minute)  # noqa: SLF001 - diagnostic script
    ]


def simulate(outcome: str = "released", clamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Replay the real timeline.

    ``outcome`` is ``"released"`` or ``"leaked"`` for the failed compare call.
    ``clamp`` is None, ``"never"`` (old behaviour) or ``"expires"`` (the fix).
    """
    clock = FakeClock()
    limiter = _limiter(clock)
    steps: List[Dict[str, Any]] = []

    def step(label: str, at: float, action) -> Dict[str, Any]:
        clock.now = at
        record = action()
        steps.append(
            {
                "label": label,
                "at": round(at, 1),
                "window": _window(limiter),
                "used_this_minute": limiter.snapshot()["tokens_used_this_minute"],
                **record,
            }
        )
        return record

    def reserve_and_record(reserved: int, actual: int):
        def action():
            entry = limiter.reserve(reserved)
            limiter.record(entry, actual)
            return {"reserved": reserved, "actual": actual}

        return action

    def q1_answer():
        entry = limiter.reserve(Q1_ANSWER_RESERVED)
        clock.now = Q1_RECORDED_AT
        limiter.record(entry, Q1_ANSWER_ACTUAL)
        return {"reserved": Q1_ANSWER_RESERVED, "actual": Q1_ANSWER_ACTUAL}

    step("q1_answer", Q1_AT, q1_answer)

    step(
        "q2_analyzer",
        Q2_ANALYZER_AT,
        reserve_and_record(Q2_ANALYZER_TOKENS, Q2_ANALYZER_TOKENS),
    )

    compare_calls: List[Dict[str, Any]] = []

    def q2_compare():
        clock.now = Q2_COMPARE_AT
        projected = limiter.wait_estimate(Q2_COMPARE_RESERVED)
        entry = limiter.reserve(Q2_COMPARE_RESERVED)
        waited = clock.now - Q2_COMPARE_AT
        clock.now = Q2_COMPARE_FAILED_AT
        if outcome == "released":
            limiter.release(entry)
            settlement = "released"
        else:
            settlement = "kept"
        if clamp is not None:
            limiter.observe_headers(
                {
                    "x-ratelimit-remaining-tokens": str(PROVIDER_REMAINING_TOKENS),
                    "x-ratelimit-reset-tokens": f"{PROVIDER_RESET_SECONDS:.0f}s",
                }
            )
            if clamp == "never":
                # Old behaviour: the remaining value never expired, because the
                # reset the provider reported was stored and never used.
                limiter._provider_tpm_reset_at = None  # noqa: SLF001
        return {
            "projected_wait": round(projected, 1),
            "waited": round(waited, 1),
            "settlement": settlement,
        }

    compare_calls.append(step("q2_compare", Q2_COMPARE_AT, q2_compare))

    step(
        "q3_analyzer",
        Q3_ANALYZER_AT,
        reserve_and_record(Q3_ANALYZER_TOKENS, Q3_ANALYZER_TOKENS),
    )

    def q3_compare():
        clock.now = Q3_COMPARE_AT
        projected = limiter.wait_estimate(Q3_COMPARE_RESERVED)
        try:
            limiter.reserve(Q3_COMPARE_RESERVED)
        except MinuteTokenBudgetExhausted as exc:
            return {
                "projected_wait": round(projected, 1),
                "retry_after": exc.retry_after_seconds,
                "reserved": False,
            }
        return {
            "projected_wait": round(projected, 1),
            "waited": round(clock.now - Q3_COMPARE_AT, 1),
            "retry_after": None,
            "reserved": True,
        }

    q3 = step("q3_compare", Q3_COMPARE_AT, q3_compare)
    compare_calls.append(q3)

    return {
        "outcome": outcome,
        "clamp": clamp,
        "steps": steps,
        "q3_compare": q3,
        "retry_after": q3.get("retry_after"),
    }


def run_scenarios() -> Dict[str, Dict[str, Any]]:
    """The four behaviours, in the order the report presents them."""
    return {
        "released": simulate(outcome="released"),
        "leaked": simulate(outcome="leaked"),
        "clamp-never": simulate(outcome="released", clamp="never"),
        "clamp-expires": simulate(outcome="released", clamp="expires"),
    }


def main() -> int:
    print("=== limiter replay (fake clock, real TokenAwareRateLimiter)")
    print(
        f"limits: tpm={config.GROQ_LIMITS['tokens_per_minute']} "
        f"queue_max_wait={config.QUEUE_MAX_WAIT_SECONDS:.0f}s"
    )
    results = run_scenarios()
    for name, result in results.items():
        print(f"\n--- {name}")
        for item in result["steps"]:
            extra = {
                key: value
                for key, value in item.items()
                if key not in {"label", "at", "window", "used_this_minute"}
            }
            print(
                f"  t={item['at']:>5}s {item['label']:<12} "
                f"used={item['used_this_minute']:>5} "
                f"window={item['window']} {extra}"
            )
        q3 = result["q3_compare"]
        print(
            f"  => Q3 compare: projected_wait={q3['projected_wait']}s "
            f"reserved={q3['reserved']} retry_after={q3['retry_after']}"
        )

    print("\n=== answer")
    leaked = results["leaked"]["retry_after"]
    never = results["clamp-never"]["retry_after"]
    fixed = results["clamp-expires"]["retry_after"]
    print(f"leaked reservation retry_after={leaked} (60 would explain the run)")
    print(f"stale clamp that never expires retry_after={never}")
    print(f"stale clamp with its reset retry_after={fixed}")
    print(
        "\nThe leaked reservation does NOT explain retry_after=60: it frees about "
        "24 s of window, under the queue limit, so the call would have waited and "
        "succeeded. A provider clamp that never expires makes every token "
        "impossible to fit, which is exactly 60 s."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
