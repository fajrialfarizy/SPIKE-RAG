"""
Explicit test doubles. Nothing in this module is imported by package source;
tests inject these objects only when they must stay offline and fast.
"""

import hashlib
from typing import Any, Dict, List, Optional

import numpy as np

import config
from core.llm_client import LLMClient, LLMResponseError, LLMResult, TokenUsage


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


class RecordingEmbedder:
    """Records the exact text handed to the query embedding path."""

    def __init__(self):
        self.model_name = config.EMBEDDING_MODEL_NAME
        self.queries: List[str] = []

    def embed_query(self, query: str) -> List[float]:
        self.queries.append(f"{config.EMBEDDING_QUERY_PREFIX}{query}")
        return [0.0] * config.EMBEDDING_DIMENSION


class FakeKnowledgeBase:
    """
    In-memory knowledge base double.

    It does not evaluate where filters: filter translation is unit tested
    separately, and the fake records what it received so tests can prove the
    retriever never drops a filter.
    """

    def __init__(
        self,
        matches: Optional[List[Any]] = None,
        chunks: Optional[List[Any]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        total: Optional[int] = None,
    ):
        self.matches = list(matches or [])
        self.chunks = list(chunks or [])
        self.metadatas = list(metadatas or [])
        self.total = total if total is not None else len(self.matches)
        self.query_calls: List[Dict[str, Any]] = []
        self.id_calls: List[List[str]] = []
        self.metadata_calls: List[Dict[str, Any]] = []

    def count_chunks(self) -> int:
        return self.total

    def query(self, query_embedding, where=None, n_results=1):
        self.query_calls.append(
            {
                "query_embedding": list(query_embedding),
                "where": where,
                "n_results": n_results,
            }
        )
        return self.matches[:n_results]

    def get_by_report_ids(self, report_ids):
        self.id_calls.append(list(report_ids))
        return [chunk for chunk in self.chunks if chunk.report_id in report_ids]

    def list_report_metadatas(self, where=None, limit=config.RETRIEVAL_CATALOG_FETCH_LIMIT):
        self.metadata_calls.append({"where": where, "limit": limit})
        return self.metadatas[:limit]


def _matches_where(metadata: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
    """Evaluate the subset of Chroma's where syntax the retriever emits."""
    if not where:
        return True
    if "$and" in where:
        return all(_matches_where(metadata, clause) for clause in where["$and"])

    for field, condition in where.items():
        if not isinstance(condition, dict) or "$in" not in condition:
            raise AssertionError(f"Unsupported where clause: {condition!r}")
        value = metadata.get(field)
        allowed = list(condition["$in"])
        if field == "competition_year":
            value = None if value is None else int(value)
            allowed = [int(item) for item in allowed]
        if value not in allowed:
            return False
    return True


class FilteringKnowledgeBase(FakeKnowledgeBase):
    """Fake knowledge base that actually applies the where clause."""

    def query(self, query_embedding, where=None, n_results=1):
        self.query_calls.append(
            {
                "query_embedding": list(query_embedding),
                "where": where,
                "n_results": n_results,
            }
        )
        matching = [
            match for match in self.matches if _matches_where(match.metadata, where)
        ]
        return matching[:n_results]

    def list_report_metadatas(self, where=None, limit=config.RETRIEVAL_CATALOG_FETCH_LIMIT):
        self.metadata_calls.append({"where": where, "limit": limit})
        return [
            metadata
            for metadata in self.metadatas
            if _matches_where(metadata, where)
        ][:limit]


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
        usage: Optional[TokenUsage] = None,
        text_finish_reasons: Optional[List[Optional[str]]] = None,
    ):
        self.json_responses = list(json_responses or [])
        self.text_responses = list(text_responses or [])
        self.fail_with = fail_with
        self.usage = usage or TokenUsage()
        self.text_finish_reasons = list(text_finish_reasons or [])
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
        deadline: Optional[Any] = None,
    ) -> LLMResult:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
                "json_mode": json_mode,
                "deadline": deadline,
            }
        )

        if self.fail_with is not None:
            raise self.fail_with

        queue = self.json_responses if json_mode else self.text_responses
        if not queue:
            raise LLMResponseError("FakeLLM has no scripted response left")
        finish_reason = None
        if not json_mode and self.text_finish_reasons:
            finish_reason = self.text_finish_reasons.pop(0)
        return LLMResult(
            content=queue.pop(0),
            usage=self.usage,
            latency_ms=0.0,
            finish_reason=finish_reason,
        )

    @property
    def json_call_count(self) -> int:
        return sum(1 for call in self.calls if call["json_mode"])

    @property
    def text_call_count(self) -> int:
        return sum(1 for call in self.calls if not call["json_mode"])
