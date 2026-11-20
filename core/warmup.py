"""
Embedder warm-up (fix pack F5).

Loading sentence-transformers is lazy, so the first real question would pay the
model-load cost inside the user's latency budget. The app calls this once at
startup; it embeds a dummy query through the exact production code path and
returns the measured duration.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import config

logger = logging.getLogger("core_warmup")


class WarmupError(RuntimeError):
    """Raised when the warm-up query does not produce a usable vector."""


def warm_up_embedder(embedder: Any, clock=time.perf_counter) -> float:
    """Run one dummy query embedding and return its duration in milliseconds."""
    started = clock()
    vector = embedder.embed_query(config.EMBEDDING_WARMUP_QUERY)
    elapsed_ms = (clock() - started) * 1000.0

    if not vector:
        raise WarmupError("The embedder returned an empty warm-up vector.")
    if len(vector) != config.EMBEDDING_DIMENSION:
        raise WarmupError(
            f"The warm-up vector has {len(vector)} dimensions but config expects "
            f"{config.EMBEDDING_DIMENSION}."
        )

    logger.info("embedder_warmup latency_ms=%.1f", elapsed_ms)
    return round(elapsed_ms, 1)
