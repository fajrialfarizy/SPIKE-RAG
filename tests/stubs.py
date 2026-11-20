"""
Explicit test doubles. Nothing in this module is imported by package source;
tests inject these objects only when they must stay offline and fast.
"""

import hashlib
from typing import Any, Dict, List, Optional

import numpy as np

import config
from core.llm_client import LLMClient, LLMResponseError


class WordCountTokenizer:
    """Deterministic offline token counter: one token per whitespace-separated word."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def encode(self, text: str, add_special_tokens: bool = False):
        class _Encoding:
            def __init__(self, ids):
                self.ids = ids

        return _Encoding(list(range(len(text.split()))))


class HashEmbedder:
    """
    Deterministic 384-dimension unit vectors derived from the text hash.

    Used only to exercise storage and plumbing offline. Never a production
    fallback: the real embedder fails loudly instead of silently substituting
    a different model.
    """

    def __init__(self, dimension: int = config.EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.embedded_texts: List[str] = []

    def _vector(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dimension)
        vector /= np.linalg.norm(vector)
        return vector.tolist()

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        self.embedded_texts.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._vector(f"{config.EMBEDDING_QUERY_PREFIX}{query}")


class FakeLLM(LLMClient):
    """
    Scripted LLM double used by every offline test.

    JSON-mode calls (the analyzer) and text calls (the answer) draw from separate
    queues, so a test expresses intent as data instead of guessing call order.
    Lives under tests/ only: package source never imports this module.
    """

    def __init__(
        self,
        json_responses: Optional[List[str]] = None,
        text_responses: Optional[List[str]] = None,
        fail_with: Optional[BaseException] = None,
    ):
        self.json_responses = list(json_responses or [])
        self.text_responses = list(text_responses or [])
        self.fail_with = fail_with
        self.calls: List[Dict[str, Any]] = []

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        reasoning_effort: str,
        json_mode: bool = False,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
                "json_mode": json_mode,
            }
        )

        if self.fail_with is not None:
            raise self.fail_with

        queue = self.json_responses if json_mode else self.text_responses
        if not queue:
            raise LLMResponseError("FakeLLM has no scripted response left")
        return queue.pop(0)

    @property
    def json_call_count(self) -> int:
        return sum(1 for call in self.calls if call["json_mode"])

    @property
    def text_call_count(self) -> int:
        return sum(1 for call in self.calls if not call["json_mode"])
