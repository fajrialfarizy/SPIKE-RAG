"""
ChromaDB store management for Lab Prestasi Ingestion.

Guarantees:
- Cosine distance and L2-normalized embeddings.
- The collection records which embedding model produced its vectors; opening a
  collection written by a different model is refused instead of mixing vectors.
- Delete-then-insert per report_id, with partial chunks removed on failure.
- Exactly 8 metadata fields per chunk.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb

import config
from ingestion.chunker import DocumentChunk


class EmbeddingModelMismatch(RuntimeError):
    """Raised when a Chroma collection was written with a different embedding model."""


class ChromaStore:
    """Manages the local Chroma vector database collection."""

    def __init__(
        self,
        persist_dir: Path = config.CHROMA_PERSIST_DIR,
        embedding_model: str = config.EMBEDDING_MODEL_NAME,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        expected_metadata = {
            config.CHROMA_METADATA_METRIC_KEY: config.CHROMA_DISTANCE_METRIC,
            config.CHROMA_METADATA_EMBEDDING_MODEL_KEY: embedding_model,
        }
        # embedding_function=None: all vectors are supplied explicitly by the
        # pipeline's embedder, so Chroma must not fetch its own default model.
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata=expected_metadata,
            embedding_function=None,
        )

        stored = dict(self.collection.metadata or {})
        if config.CHROMA_METADATA_EMBEDDING_MODEL_KEY not in stored:
            # Collections created before the guard existed inherit the model now.
            stored.update(expected_metadata)
            self.collection.modify(metadata=stored)

        stored_model = stored.get(config.CHROMA_METADATA_EMBEDDING_MODEL_KEY)
        if stored_model != embedding_model:
            raise EmbeddingModelMismatch(
                f"Collection '{config.CHROMA_COLLECTION_NAME}' was built with "
                f"embedding model '{stored_model}' but config expects "
                f"'{embedding_model}'. Refusing to mix vectors from different models."
            )

        stored_metric = stored.get(config.CHROMA_METADATA_METRIC_KEY)
        if stored_metric and stored_metric != config.CHROMA_DISTANCE_METRIC:
            raise EmbeddingModelMismatch(
                f"Collection '{config.CHROMA_COLLECTION_NAME}' uses distance metric "
                f"'{stored_metric}' but config expects "
                f"'{config.CHROMA_DISTANCE_METRIC}'."
            )

    def delete_chunks_for_report(self, report_id: str) -> None:
        """Delete all existing chunks for a report_id (Brief 5.2 #10)."""
        existing = self.collection.get(where={"report_id": report_id})
        if existing and existing["ids"]:
            self.collection.delete(ids=existing["ids"])

    def _add(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Single write seam, so tests can inject a real mid-write failure."""
        self.collection.add(
            ids=list(ids),
            documents=list(documents),
            metadatas=list(metadatas),
            embeddings=[list(vector) for vector in embeddings],
        )

    def insert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        simulate_failure: bool = False,
    ) -> None:
        """
        Store chunks for one report: delete existing chunks, then insert.

        Any failure (including a partial write) removes every chunk of that
        report_id so a retry starts clean.
        """
        if not chunks:
            return

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Got {len(embeddings)} embeddings for {len(chunks)} chunks"
            )

        report_id = chunks[0].report_id
        for chunk in chunks:
            if chunk.report_id != report_id:
                raise ValueError(
                    "insert_chunks received chunks from multiple reports; "
                    "delete-then-insert must run per report_id"
                )
            if len(chunk.metadata) != 8:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} has {len(chunk.metadata)} metadata "
                    "fields, expected exactly 8"
                )

        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate chunk ids for report {report_id}")

        self.delete_chunks_for_report(report_id)

        try:
            if simulate_failure:
                self._add(
                    ids=ids[:1],
                    documents=[chunk.document_text for chunk in chunks[:1]],
                    metadatas=[chunk.metadata for chunk in chunks[:1]],
                    embeddings=embeddings[:1],
                )
                raise RuntimeError("Simulated storage failure mid-write")

            self._add(
                ids=ids,
                documents=[chunk.document_text for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                embeddings=embeddings,
            )
        except Exception:
            self.delete_chunks_for_report(report_id)
            raise

    def get_source_file_id_for_report(self, report_id: str) -> Optional[str]:
        """The source_file_id currently stored for this report_id, if any."""
        results = self.collection.get(where={"report_id": report_id}, limit=1)
        if results and results["metadatas"]:
            return results["metadatas"][0].get("source_file_id")
        return None

    def count_chunks(self) -> int:
        return self.collection.count()

    def count_chunks_for_report(self, report_id: str) -> int:
        results = self.collection.get(where={"report_id": report_id})
        return len(results["ids"]) if results else 0

    def get_all_report_ids(self) -> List[str]:
        result = self.collection.get(include=["metadatas"])
        if not result or not result["metadatas"]:
            return []
        return sorted({m["report_id"] for m in result["metadatas"] if "report_id" in m})

    def check_no_orphan_vectors(self, expected_dimension: int) -> None:
        """Refuse collections whose stored vectors do not match the configured size."""
        sample = self.collection.get(include=["embeddings"], limit=1)
        embeddings = sample.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return
        dimension = len(embeddings[0])
        if dimension != expected_dimension:
            raise EmbeddingModelMismatch(
                f"Stored vectors have {dimension} dimensions but config expects "
                f"{expected_dimension}."
            )
