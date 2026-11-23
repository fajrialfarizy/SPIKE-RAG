"""
Read-only knowledge base for the chat pipeline.

Opens a local Chroma persist directory (Phase 3 downloads the same artifact from
Drive) plus an optional manifest.json. Loading refuses to proceed when:

- the collection or the manifest records a different embedding model,
- the collection uses a different distance metric,
- the stored vector dimension differs from config,
- the manifest and the collection disagree on chunk count.

There is no fallback collection, no re-embedding and no silent repair: a
mismatch means the artifact and the code disagree, which is a deployment error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb

import config
from core import startup
from ingestion.sync import ChromaSync, ManifestError


class KnowledgeBaseError(RuntimeError):
    """Base class for knowledge base loading and integrity failures."""

    # Fix pack 13a: the code the app shows under its generic message.
    code = "DB/chroma_open"


class KnowledgeBaseNotFoundError(KnowledgeBaseError):
    """Raised when the persist directory has no usable collection."""

    code = "DB/chroma_open"


class EmbeddingContractError(KnowledgeBaseError):
    """Raised when the artifact was built with another model or dimension."""

    code = "KB/model_mismatch"


@dataclass(frozen=True)
class ChunkMatch:
    """A semantic-search hit with its cosine distance."""

    chunk_id: str
    report_id: str
    document: str
    metadata: Dict[str, Any]
    distance: float


@dataclass(frozen=True)
class StoredChunk:
    """A chunk fetched by exact field filter (no similarity involved)."""

    chunk_id: str
    report_id: str
    document: str
    metadata: Dict[str, Any]


class KnowledgeBase:
    """Read-only view over one validated Chroma collection."""

    def __init__(
        self,
        persist_dir: Path = config.CHROMA_PERSIST_DIR,
        manifest_path: Optional[Path] = None,
        collection_name: str = config.CHROMA_COLLECTION_NAME,
        embedding_model: str = config.EMBEDDING_MODEL_NAME,
        expected_dimension: int = config.EMBEDDING_DIMENSION,
        client: Optional[Any] = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.expected_dimension = expected_dimension

        with startup.stage(startup.STAGE_CHROMA_OPEN):
            if not self.persist_dir.exists():
                raise KnowledgeBaseNotFoundError(
                    f"Knowledge base directory '{self.persist_dir}' does not exist. "
                    "Run ingestion (or download the artifact) before serving questions."
                )

            self.client = client or chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self._open_collection()

        with startup.stage(startup.STAGE_MODEL_VALIDATE):
            self._validate_collection()
            self._validate_dimension()

        if manifest_path is not None:
            with startup.stage(startup.STAGE_MANIFEST_VALIDATE):
                self._validate_manifest(Path(manifest_path))

    # -- loading -----------------------------------------------------------
    def _open_collection(self):
        try:
            return self.client.get_collection(
                name=self.collection_name,
                embedding_function=None,
            )
        except Exception as exc:
            raise KnowledgeBaseNotFoundError(
                f"Collection '{self.collection_name}' was not found in "
                f"'{self.persist_dir}'. Ingestion must run before the chatbot."
            ) from exc

    def _validate_collection(self) -> None:
        metadata = dict(self.collection.metadata or {})

        stored_model = metadata.get(config.CHROMA_METADATA_EMBEDDING_MODEL_KEY)
        if stored_model != self.embedding_model:
            raise EmbeddingContractError(
                f"Collection '{self.collection_name}' was built with embedding "
                f"model '{stored_model}' but config expects "
                f"'{self.embedding_model}'. Refusing to load the knowledge base."
            )

        stored_metric = metadata.get(config.CHROMA_METADATA_METRIC_KEY)
        if stored_metric != config.CHROMA_DISTANCE_METRIC:
            raise EmbeddingContractError(
                f"Collection '{self.collection_name}' uses distance metric "
                f"'{stored_metric}' but config expects "
                f"'{config.CHROMA_DISTANCE_METRIC}'."
            )

    def _validate_dimension(self) -> None:
        sample = self.collection.get(include=["embeddings"], limit=1)
        embeddings = sample.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return

        dimension = len(embeddings[0])
        if dimension != self.expected_dimension:
            raise EmbeddingContractError(
                f"Stored vectors have {dimension} dimensions but config expects "
                f"{self.expected_dimension}. Refusing to load the knowledge base."
            )

    def _validate_manifest(self, manifest_path: Path) -> None:
        if not manifest_path.exists():
            raise KnowledgeBaseNotFoundError(
                f"Manifest '{manifest_path}' does not exist."
            )

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EmbeddingContractError(
                f"Manifest '{manifest_path}' is not readable JSON."
            ) from exc

        try:
            ChromaSync.validate_manifest(manifest, self.embedding_model)
        except ManifestError as exc:
            raise EmbeddingContractError(str(exc)) from exc

        manifest_chunks = manifest.get("chunk_count")
        collection_chunks = self.collection.count()
        if manifest_chunks is not None and int(manifest_chunks) != collection_chunks:
            raise EmbeddingContractError(
                f"Manifest records {manifest_chunks} chunks but the collection "
                f"holds {collection_chunks}. The artifact is inconsistent."
            )

        manifest_dimension = manifest.get("embedding_dimension")
        if (
            manifest_dimension is not None
            and int(manifest_dimension) != self.expected_dimension
        ):
            raise EmbeddingContractError(
                f"Manifest records {manifest_dimension}-dimension vectors but "
                f"config expects {self.expected_dimension}."
            )

    # -- reads -------------------------------------------------------------
    def count_chunks(self) -> int:
        return self.collection.count()

    def report_ids(self) -> List[str]:
        result = self.collection.get(include=["metadatas"])
        return sorted(
            {m["report_id"] for m in result["metadatas"] if m and "report_id" in m}
        )

    def query(
        self,
        query_embedding: Sequence[float],
        where: Optional[Dict[str, Any]] = None,
        n_results: int = 1,
    ) -> List[ChunkMatch]:
        """Semantic search. An empty collection returns no matches."""
        total = self.count_chunks()
        if total == 0:
            return []

        bounded = max(1, min(int(n_results), total))
        params: Dict[str, Any] = {
            "query_embeddings": [list(query_embedding)],
            "n_results": bounded,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            params["where"] = where

        result = self.collection.query(**params)

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        matches: List[ChunkMatch] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            matches.append(
                ChunkMatch(
                    chunk_id=chunk_id,
                    report_id=metadata.get("report_id", ""),
                    document=documents[index] or "",
                    metadata=dict(metadata),
                    distance=float(distances[index]),
                )
            )
        return matches

    def get_by_report_ids(self, report_ids: Sequence[str]) -> List[StoredChunk]:
        """Exact filter by report_id: no embedding, no similarity ranking."""
        ids = [report_id for report_id in report_ids if report_id]
        if not ids:
            return []

        result = self.collection.get(
            where={"report_id": {"$in": ids}},
            include=["documents", "metadatas"],
        )

        stored: List[StoredChunk] = []
        for index, chunk_id in enumerate(result.get("ids") or []):
            metadata = (result.get("metadatas") or [])[index] or {}
            stored.append(
                StoredChunk(
                    chunk_id=chunk_id,
                    report_id=metadata.get("report_id", ""),
                    document=(result.get("documents") or [])[index] or "",
                    metadata=dict(metadata),
                )
            )

        # Deterministic order: chunk_id encodes section and position.
        return sorted(stored, key=lambda chunk: chunk.chunk_id)

    def list_report_metadatas(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: int = config.RETRIEVAL_CATALOG_FETCH_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Metadata only, used by the catalog mode."""
        params: Dict[str, Any] = {"include": ["metadatas"], "limit": int(limit)}
        if where:
            params["where"] = where
        result = self.collection.get(**params)
        return [dict(metadata or {}) for metadata in (result.get("metadatas") or [])]
