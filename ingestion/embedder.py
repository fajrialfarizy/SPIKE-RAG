"""
Embedder module for Lab Prestasi Ingestion.

Produces 384-dimensional L2-normalized embeddings with
intfloat/multilingual-e5-small (Brief 3).

There is deliberately NO fallback model and NO pseudo-embedding path: if the
configured model cannot be loaded, embedding fails loudly. Offline tests inject
an explicit double from tests/stubs.py instead of relying on a silent fallback.
"""

from typing import Any, List, Optional, Sequence

import config


class EmbeddingError(RuntimeError):
    """Raised when the configured embedding model cannot be used."""


class E5Embedder:
    """Computes normalized embeddings for intfloat/multilingual-e5-small."""

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        model: Optional[Any] = None,
        expected_dimension: int = config.EMBEDDING_DIMENSION,
    ):
        self.model_name = model_name
        self.expected_dimension = expected_dimension
        self._model = model

    def _ensure_model(self):
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EmbeddingError(
                "sentence-transformers is not installed, so the configured embedding "
                f"model '{self.model_name}' cannot be loaded."
            ) from exc

        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise EmbeddingError(
                f"Could not load the configured embedding model '{self.model_name}'. "
                "Ingestion refuses to substitute another model or a pseudo-embedding. "
                "Fix the model availability or inject an explicit test double."
            ) from exc

    def _check_model_dimension(self):
        # sentence-transformers 6.x renames the accessor; support both spellings.
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        if getter is None:
            return
        dimension = getter()
        if dimension is not None and dimension != self.expected_dimension:
            raise EmbeddingError(
                f"Embedding model '{self.model_name}' produces {dimension}-dimension "
                f"vectors but config.EMBEDDING_DIMENSION is {self.expected_dimension}."
            )

    def _encode(self, texts: Sequence[str]) -> List[List[float]]:
        self._ensure_model()
        self._check_model_dimension()

        try:
            vectors = self._model.encode(
                list(texts), normalize_embeddings=True, convert_to_numpy=True
            )
        except TypeError:
            # Test doubles may not accept every encoder keyword.
            vectors = self._model.encode(list(texts))

        result: List[List[float]] = []
        for vector in vectors:
            values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            if len(values) != self.expected_dimension:
                raise EmbeddingError(
                    f"Embedding dimension mismatch: got {len(values)}, expected "
                    f"{self.expected_dimension} for model '{self.model_name}'."
                )
            result.append([float(value) for value in values])
        return result

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """Embed passage texts that already carry config.EMBEDDING_PASSAGE_PREFIX."""
        if not texts:
            return []

        missing_prefix = [
            text for text in texts if not text.startswith(config.EMBEDDING_PASSAGE_PREFIX)
        ]
        if missing_prefix:
            raise EmbeddingError(
                "Passage texts must start with "
                f"'{config.EMBEDDING_PASSAGE_PREFIX}'. Offending text starts with: "
                f"{missing_prefix[0][:60]!r}"
            )

        return self._encode(texts)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Prefix raw chunk documents with the passage prefix and embed them."""
        return self.embed_passages(
            [f"{config.EMBEDDING_PASSAGE_PREFIX}{document}" for document in documents]
        )

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query with the query prefix."""
        prefixed = f"{config.EMBEDDING_QUERY_PREFIX}{query}"
        return self._encode([prefixed])[0]
