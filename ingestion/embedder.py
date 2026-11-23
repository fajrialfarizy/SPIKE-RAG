"""
Embedder module for Lab Prestasi Ingestion.

Produces 384-dimensional L2-normalized embeddings with
intfloat/multilingual-e5-small (Brief 3).

There is deliberately NO fallback model and NO pseudo-embedding path: if the
configured model cannot be loaded, embedding fails loudly. Offline tests inject
an explicit double from tests/stubs.py instead of relying on a silent fallback.
"""

import os
import threading
import time
from typing import Any, List, Optional, Sequence

import config
from core import startup


class EmbeddingError(RuntimeError):
    """Raised when the configured embedding model cannot be used."""

    # Fix pack 13a: the code the app shows under its generic message.
    code = "EMB/load_failed"


class EmbeddingDownloadError(EmbeddingError):
    """Raised when the model weights cannot be fetched from Hugging Face."""

    code = "EMB/download_failed"


# Exception types huggingface_hub/transformers use for a network or cache failure.
# A download is told apart from a load failure by where the exception came from,
# never by reading the message (which can carry a URL).
_DOWNLOAD_ERROR_MODULES = (
    "huggingface_hub",
    "requests",
    "urllib3",
    "http",
    "socket",
    "aiohttp",
    "ssl",
)


def is_download_error(exc: BaseException) -> bool:
    """True when the failure came from the network/cache layer."""
    current: Optional[BaseException] = exc
    for _ in range(5):
        if current is None:
            return False
        module = str(type(current).__module__ or "")
        if module.split(".")[0] in _DOWNLOAD_ERROR_MODULES:
            return True
        if isinstance(current, (ConnectionError, TimeoutError)):
            return True
        current = current.__cause__ or current.__context__
    return False


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
        # Wall time spent in the lazy model load, for the app startup log.
        self.load_seconds = 0.0
        # Streamlit serves sessions in threads and they share one embedder, so
        # the lazy model load and the encode call are serialized.
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return

        # Fix pack 13a: a download that happens inside a Streamlit script run must
        # fail instead of hanging forever. huggingface_hub reads this timeout from
        # the environment, so it is set before sentence-transformers imports it.
        os.environ.setdefault(
            "HF_HUB_DOWNLOAD_TIMEOUT", str(config.EMBEDDER_DOWNLOAD_TIMEOUT_SECONDS)
        )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EmbeddingError(
                "sentence-transformers is not installed, so the configured embedding "
                f"model '{self.model_name}' cannot be loaded."
            ) from exc

        started = time.perf_counter()
        # Fix pack 13a: this stage is the only place that may download weights. The
        # wrapping happens INSIDE it, so the stage that fails is this one.
        with startup.stage(startup.STAGE_EMBEDDER_LOAD):
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # pragma: no cover - environment dependent
                if is_download_error(exc):
                    raise EmbeddingDownloadError(
                        f"Could not download the configured embedding model "
                        f"'{self.model_name}' within "
                        f"{config.EMBEDDER_DOWNLOAD_TIMEOUT_SECONDS} s."
                    ) from exc
                raise EmbeddingError(
                    f"Could not load the configured embedding model "
                    f"'{self.model_name}'. Ingestion refuses to substitute another "
                    "model or a pseudo-embedding. Fix the model availability or "
                    "inject an explicit test double."
                ) from exc
        self.load_seconds = round(time.perf_counter() - started, 2)

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
        with self._lock:
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
