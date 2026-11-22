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
from typing import Any, Dict

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


def warm_up_knowledge_base(
    knowledge_base: Any, embedder: Any, clock=time.perf_counter
) -> Dict[str, float]:
    """
    Exercise the lazily initialized retrieval paths once (fix pack 9, O6a).

    The embedder warm-up already used a real ``query: ``-prefixed sentence, but
    the first question still spent 1,240 ms in embedding_latency_ms while later
    ones took 40-55 ms: the cold paths were the first Chroma query and the
    metadata read. Both are done here, at startup, so the first question pays no
    warm-up cost. No model call is made.
    """
    started = clock()
    vector = embedder.embed_query(config.EMBEDDING_WARMUP_QUERY)
    embedder_ms = (clock() - started) * 1000.0

    started = clock()
    knowledge_base.count_chunks()
    count_ms = (clock() - started) * 1000.0

    started = clock()
    knowledge_base.query(query_embedding=vector, n_results=1)
    query_ms = (clock() - started) * 1000.0

    started = clock()
    knowledge_base.list_report_metadatas(limit=1)
    metadata_ms = (clock() - started) * 1000.0

    timings = {
        "embedder_ms": round(embedder_ms, 1),
        "count_ms": round(count_ms, 1),
        "query_ms": round(query_ms, 1),
        "metadata_ms": round(metadata_ms, 1),
    }
    logger.info(
        "retrieval_warmup embedder_ms=%.1f count_ms=%.1f query_ms=%.1f "
        "metadata_ms=%.1f",
        timings["embedder_ms"],
        timings["count_ms"],
        timings["query_ms"],
        timings["metadata_ms"],
    )
    return timings
