"""
Token accounting for the context budget (report F1d).

Two counters exist:

- ``E5TokenCounter`` uses the exact tokenizer of the configured embedding model.
  It is the accurate path used by the app wiring.
- ``EstimatorTokenCounter`` is a deterministic, deliberately conservative
  estimate (never fewer tokens than the real tokenizer is likely to produce).
  It exists so offline tests and a missing tokenizer cannot silently undercount
  into an oversized prompt.

There is no silent substitution of another model: if the real tokenizer cannot
be loaded, ``E5TokenCounter`` raises.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, Optional

import config

# One tokenizer per model for the whole process (fix pack 7): the context
# budget, the prompt estimator in the limiter and the app startup warm-up all use
# the same intfloat/multilingual-e5-small artifact, and loading it is slow.
_TOKENIZER_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE_LOCK = threading.Lock()


class TokenCounterError(RuntimeError):
    """Raised when the configured tokenizer cannot be loaded or used."""


class EstimatorTokenCounter:
    """Conservative estimate: the larger of word count and chars/ratio."""

    def __init__(
        self,
        chars_per_token: float = config.CONTEXT_TOKEN_ESTIMATOR_CHARS_PER_TOKEN,
    ):
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive.")
        self.chars_per_token = float(chars_per_token)

    def count_tokens(self, text: str) -> int:
        content = text or ""
        by_chars = math.ceil(len(content) / self.chars_per_token)
        return max(len(content.split()), by_chars)


class E5TokenCounter:
    """
    Exact token count with the tokenizer of intfloat/multilingual-e5-small.

    Counting is encode-only: the result is a number, never a model input. The e5
    sentence encoder enforces a 512-token limit, but a prompt to count can be
    longer, and transformers warns "Token indices sequence length is longer than
    the specified maximum sequence length" on every such encode (fix pack 8,
    N5a). The counting instance therefore raises ``model_max_length`` and never
    truncates, so the log stays clean and the count stays exact.
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        tokenizer: Optional[Any] = None,
    ):
        self.model_name = model_name
        self._tokenizer = tokenizer
        # Shared across Streamlit session threads; the lazy load must not race.
        self._lock = threading.Lock()

    def _ensure_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        with _TOKENIZER_CACHE_LOCK:
            cached = _TOKENIZER_CACHE.get(self.model_name)
            if cached is None:
                try:
                    from transformers import AutoTokenizer
                except ImportError as exc:  # pragma: no cover - environment dependent
                    raise TokenCounterError(
                        "transformers is not installed, so the e5 tokenizer cannot be "
                        "used for context budgeting."
                    ) from exc
                try:
                    cached = AutoTokenizer.from_pretrained(self.model_name)
                except Exception as exc:  # pragma: no cover - environment dependent
                    raise TokenCounterError(
                        f"Could not load the tokenizer for '{self.model_name}'."
                    ) from exc
                # Counting-only instance: no truncation, no 512-token warning.
                cached.model_max_length = config.TOKEN_COUNT_MAX_LENGTH
                _TOKENIZER_CACHE[self.model_name] = cached
        self._tokenizer = cached
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        with self._lock:
            tokenizer = self._ensure_tokenizer()
            return len(
                tokenizer.encode(
                    text or "", add_special_tokens=False, truncation=False
                )
            )
