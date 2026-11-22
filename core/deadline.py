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
from typing import Callable, Optional

import config


class DeadlineExceeded(RuntimeError):
    """Raised when the remaining pipeline budget cannot cover the next step."""


class QueueDeadlineExceeded(DeadlineExceeded):
    """
    A wait for a local queue (token window, concurrency slot) ran out of budget.

    Fix pack 7 keeps this apart from a real model/processing overrun: the pipeline
    answers RATE_LIMITED for this one and reserves TECHNICAL_ERROR for failures.
    """

    def __init__(self, message: str, retry_after_seconds: Optional[float] = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


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

    def extend(self, seconds: float) -> None:
        """
        Give back time spent in a local queue (fix pack 7).

        The budget guards model and processing time only, so a wait for the token
        window or the concurrency slot is added back instead of being charged to
        the model call that follows.
        """
        if seconds and seconds > 0:
            self.budget_seconds += float(seconds)

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
