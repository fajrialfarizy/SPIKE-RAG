"""
ChromaDB Store management module for Lab Prestasi Ingestion.
Uses PersistentClient with Cosine distance metric.
Enforces:
- Idempotent delete-then-insert per report_id.
- Exact 8 metadata fields.
- Atomic-like cleanup: partial chunks are removed on error.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import config
from ingestion.chunker import DocumentChunk


class ChromaStore:
    """Manages the local Chroma vector database collection."""

    def __init__(self, persist_dir: Path = config.CHROMA_PERSIST_DIR):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": config.CHROMA_DISTANCE_METRIC},
        )

    def delete_chunks_for_report(self, report_id: str):
        """Delete all existing chunks for a report_id (Brief 5.2 #10)."""
        existing = self.collection.get(where={"report_id": report_id})
        if existing and existing["ids"]:
            self.collection.delete(ids=existing["ids"])

    def insert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        simulate_failure: bool = False,
    ):
        """
        Store chunks into ChromaDB.
        If simulate_failure is True, raises an error halfway through to test rollback/cleanup.
        """
        if not chunks:
            return

        report_id = chunks[0].report_id

        # Delete existing chunks for this report_id first
        self.delete_chunks_for_report(report_id)

        try:
            ids = [c.chunk_id for c in chunks]
            documents = [c.document_text for c in chunks]
            metadatas = [c.metadata for c in chunks]

            if simulate_failure:
                # Insert first chunk, then trigger simulated failure
                self.collection.add(
                    ids=[ids[0]],
                    documents=[documents[0]],
                    metadatas=[metadatas[0]],
                    embeddings=[embeddings[0]],
                )
                raise RuntimeError("Simulasi gangguan teknis di tengah penyimpanan ChromaDB")

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        except Exception as e:
            # Delete partial chunks on failure (Brief 5.2 #10)
            self.delete_chunks_for_report(report_id)
            raise e

    def get_source_file_id_for_report(self, report_id: str) -> Optional[str]:
        """Fetch the source_file_id currently stored for this report_id in ChromaDB."""
        results = self.collection.get(where={"report_id": report_id}, limit=1)
        if results and results["metadatas"]:
            return results["metadatas"][0].get("source_file_id")
        return None

    def count_chunks(self) -> int:
        """Total chunk count in collection."""
        return self.collection.count()

    def get_all_report_ids(self) -> List[str]:
        """Get unique report_ids currently in collection."""
        res = self.collection.get(include=["metadatas"])
        if not res or not res["metadatas"]:
            return []
        ids = {m["report_id"] for m in res["metadatas"] if "report_id" in m}
        return sorted(list(ids))
