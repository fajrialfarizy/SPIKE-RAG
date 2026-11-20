"""
Total pipeline deadline (report K2).

One question gets a fixed budget. Every model call and every retry delay checks
the remaining budget first, so a slow provider cannot stretch a request far past
``PIPELINE_TOTAL_DEADLINE_SECONDS``. The clock is injectable so tests can move
time without sleeping.

There is no silent clamping: when the budget cannot cover the next step,
``DeadlineExceeded`` is raised and the pipeline boundary turns it into
TECHNICAL_ERROR with ``error_type=DeadlineExceeded``.
"""

from __future__ import annotations

import time
from typing import Callable

import config


class DeadlineExceeded(RuntimeError):
    """Raised when the remaining pipeline budget cannot cover the next step."""


class Deadline:
    """A monotonically measured budget for one pipeline run."""

    def __init__(
        self,
        budget_seconds: float = config.PIPELINE_TOTAL_DEADLINE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        if budget_seconds <= 0:
            raise ValueError("Deadline budget must be positive.")
        self.budget_seconds = float(budget_seconds)
        self._clock = clock
        self._started_at = clock()

    def elapsed(self) -> float:
        return self._clock() - self._started_at

    def remaining(self) -> float:
        return self.budget_seconds - self.elapsed()

    @property
    def expired(self) -> bool:
        return self.remaining() <= 0

    def require_remaining(self) -> float:
        remaining = self.remaining()
        if remaining <= 0:
            raise DeadlineExceeded(
                f"The pipeline deadline of {self.budget_seconds:.1f}s is exhausted."
            )
        return remaining

    def call_timeout(self, per_call_seconds: float) -> float:
        """Timeout for one provider call: the per-call cap, bounded by the budget."""
        return max(0.0, min(float(per_call_seconds), self.require_remaining()))

    def sleep(self, seconds: float, sleep_fn: Callable[[float], None] = time.sleep) -> None:
        """Sleep only when the budget can absorb the whole wait."""
        if seconds <= 0:
            return
        if seconds > self.remaining():
            raise DeadlineExceeded(
                f"A {seconds:.1f}s wait does not fit the remaining pipeline budget."
            )
        sleep_fn(seconds)
