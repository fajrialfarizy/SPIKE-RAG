"""
Explicit test doubles. Nothing in this module is imported by package source;
tests inject these objects only when they must stay offline and fast.
"""

import hashlib
from typing import List

import numpy as np

import config


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
