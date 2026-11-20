"""
Offline tests for the read-only knowledge base loader.

Every refusal path is exercised: missing directory, missing collection, another
embedding model, another distance metric, another vector dimension and a manifest
that disagrees with the collection.
"""

import json

import pytest

import config
from core.knowledge_base import (
    EmbeddingContractError,
    KnowledgeBase,
    KnowledgeBaseNotFoundError,
)
from enums import SectionType
from ingestion.chunker import SectionAwareChunker
from ingestion.chroma_store import ChromaStore
from ingestion.sync import ChromaSync
from tests.stubs import WordCountTokenizer


def make_chunks(report_id="RPT0001", content="Isi evaluasi tim."):
    chunker = SectionAwareChunker(tokenizer=WordCountTokenizer())
    return chunker.chunk_section(
        report_id=report_id,
        competition_name="Lomba Contoh",
        competition_year=2024,
        competition_type="Essay",
        achievement_level="Juara 1",
        source_file_id=f"DRV_{report_id}.pdf",
        judul_ide="Ide Hebat",
        section_type=SectionType.EVALUASI_TIM,
        section_content=content,
    )


class FixedDimensionEmbedder:
    """Pass-through double so a collection can hold any vector size."""

    def __init__(self, dimension=config.EMBEDDING_DIMENSION):
        self.dimension = dimension

    def embed_passages(self, texts):
        return [[0.0] * self.dimension for _ in texts]


def populated_store(persist_dir, dimension=config.EMBEDDING_DIMENSION, reports=("RPT0001",)):
    store = ChromaStore(persist_dir=persist_dir)
    embedder = FixedDimensionEmbedder(dimension)
    for report_id in reports:
        chunks = make_chunks(report_id)
        vectors = embedder.embed_passages([chunk.embed_text for chunk in chunks])
        store.insert_chunks(chunks, vectors)
    return store


def test_missing_directory_is_refused(tmp_path):
    with pytest.raises(KnowledgeBaseNotFoundError, match="does not exist"):
        KnowledgeBase(persist_dir=tmp_path / "absent")


def test_missing_collection_is_refused(tmp_path):
    (tmp_path / "chroma").mkdir()

    with pytest.raises(KnowledgeBaseNotFoundError, match="was not found"):
        KnowledgeBase(persist_dir=tmp_path / "chroma")


def test_another_embedding_model_is_refused(tmp_path):
    ChromaStore(persist_dir=tmp_path / "chroma", embedding_model="other/model-v9")

    with pytest.raises(EmbeddingContractError, match="Refusing to load"):
        KnowledgeBase(persist_dir=tmp_path / "chroma")


def test_another_distance_metric_is_refused(tmp_path):
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={
            config.CHROMA_METADATA_EMBEDDING_MODEL_KEY: config.EMBEDDING_MODEL_NAME,
            config.CHROMA_METADATA_METRIC_KEY: "l2",
        },
        embedding_function=None,
    )

    with pytest.raises(EmbeddingContractError, match="distance metric"):
        KnowledgeBase(persist_dir=tmp_path / "chroma")


def test_another_vector_dimension_is_refused(tmp_path):
    populated_store(tmp_path / "chroma", dimension=8)

    with pytest.raises(EmbeddingContractError, match="dimensions"):
        KnowledgeBase(persist_dir=tmp_path / "chroma")


def test_valid_artifact_loads_and_serves_reads(tmp_path):
    populated_store(tmp_path / "chroma", reports=("RPT0001", "RPT0002"))

    kb = KnowledgeBase(persist_dir=tmp_path / "chroma")

    assert kb.count_chunks() == 2
    assert kb.report_ids() == ["RPT0001", "RPT0002"]

    matches = kb.query([0.0] * config.EMBEDDING_DIMENSION, n_results=2)
    assert len(matches) == 2
    assert {match.report_id for match in matches} == {"RPT0001", "RPT0002"}
    assert matches[0].distance >= 0.0

    by_id = kb.get_by_report_ids(["RPT0002"])
    assert [chunk.report_id for chunk in by_id] == ["RPT0002"]
    assert by_id[0].document.startswith("Kompetisi: Lomba Contoh")

    metadatas = kb.list_report_metadatas()
    assert {metadata["report_id"] for metadata in metadatas} == {"RPT0001", "RPT0002"}


def test_query_respects_the_where_filter(tmp_path):
    populated_store(tmp_path / "chroma", reports=("RPT0001", "RPT0002"))
    kb = KnowledgeBase(persist_dir=tmp_path / "chroma")

    matches = kb.query(
        [0.0] * config.EMBEDDING_DIMENSION,
        where={"report_id": {"$in": ["RPT0002"]}},
        n_results=5,
    )

    assert [match.report_id for match in matches] == ["RPT0002"]


def test_manifest_mismatch_is_refused(tmp_path):
    store = populated_store(tmp_path / "chroma")
    manifest_path = tmp_path / "manifest.json"
    manifest = ChromaSync.generate_manifest(
        ChromaSync(store, drive_repo=None), "checksum", "chroma_backup.zip"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Consistent manifest loads.
    KnowledgeBase(persist_dir=tmp_path / "chroma", manifest_path=manifest_path)

    manifest["chunk_count"] = manifest["chunk_count"] + 5
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(EmbeddingContractError, match="inconsistent"):
        KnowledgeBase(persist_dir=tmp_path / "chroma", manifest_path=manifest_path)

    manifest["chunk_count"] = store.count_chunks()
    manifest["embedding_model"] = "sentence-transformers/all-MiniLM-L6-v2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(EmbeddingContractError, match="Refusing to load"):
        KnowledgeBase(persist_dir=tmp_path / "chroma", manifest_path=manifest_path)


def test_manifest_dimension_is_checked(tmp_path):
    populated_store(tmp_path / "chroma")
    store = ChromaStore(persist_dir=tmp_path / "chroma")
    manifest = ChromaSync.generate_manifest(
        ChromaSync(store, drive_repo=None), "checksum", "chroma_backup.zip"
    )
    manifest["embedding_dimension"] = 768
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EmbeddingContractError, match="768"):
        KnowledgeBase(persist_dir=tmp_path / "chroma", manifest_path=manifest_path)
