"""
Offline tests for the embedding contract:
- exact e5 prefixes (Brief 5.3 / PRD 7.2),
- dimension enforcement,
- refusal of a collection or manifest built with a different model.
"""

import numpy as np
import pytest

import config
from ingestion.chunker import SectionAwareChunker
from ingestion.chroma_store import ChromaStore, EmbeddingModelMismatch
from ingestion.embedder import E5Embedder, EmbeddingError
from ingestion.sync import ChromaSync, ManifestError
from enums import SectionType
from tests.stubs import WordCountTokenizer


class FakeSentenceModel:
    """Minimal stand-in for SentenceTransformer with configurable dimension."""

    def __init__(self, dimension: int = config.EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.encoded = []

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        self.encoded.append(list(texts))
        return np.array(
            [np.ones(self.dimension, dtype=float) / np.sqrt(self.dimension) for _ in texts]
        )


def test_embedding_prefixes_are_exact():
    """No leading space, one trailing space, as required by the e5 convention."""
    assert config.EMBEDDING_PASSAGE_PREFIX == "passage: "
    assert config.EMBEDDING_QUERY_PREFIX == "query: "
    assert not config.EMBEDDING_PASSAGE_PREFIX.startswith(" ")
    assert not config.EMBEDDING_QUERY_PREFIX.startswith(" ")
    assert config.EMBEDDING_PASSAGE_PREFIX.count(" ") == 1
    assert config.EMBEDDING_QUERY_PREFIX.count(" ") == 1


def test_chunks_and_queries_use_the_configured_prefixes():
    chunker = SectionAwareChunker(tokenizer=WordCountTokenizer())
    chunk = chunker.chunk_section(
        report_id="RPT0001",
        competition_name="Lomba",
        competition_year=2024,
        competition_type="Essay",
        achievement_level="Juara 1",
        source_file_id="DRV.pdf",
        judul_ide="Ide",
        section_type=SectionType.EVALUASI_TIM,
        section_content="Isi evaluasi.",
    )[0]
    assert chunk.embed_text.startswith("passage: ")
    assert not chunk.document_text.startswith("passage: ")

    model = FakeSentenceModel()
    embedder = E5Embedder(model=model)
    embedder.embed_query("kendala presentasi")
    assert model.encoded[-1] == ["query: kendala presentasi"]


def test_embedder_requires_passage_prefix():
    embedder = E5Embedder(model=FakeSentenceModel())

    with pytest.raises(EmbeddingError, match="passage: "):
        embedder.embed_passages(["teks tanpa prefix"])


def test_embedder_adds_prefix_for_documents():
    embedder = E5Embedder(model=FakeSentenceModel())
    vectors = embedder.embed_documents(["dokumen satu"])

    assert len(vectors) == 1
    assert len(vectors[0]) == config.EMBEDDING_DIMENSION
    assert embedder._model.encoded[-1] == ["passage: dokumen satu"]


def test_embedder_rejects_wrong_dimension():
    embedder = E5Embedder(model=FakeSentenceModel(dimension=128))

    with pytest.raises(EmbeddingError, match="128"):
        embedder.embed_passages(["passage: teks"])


def test_embedder_fails_loudly_when_package_missing(monkeypatch):
    """No silent fallback: a missing runtime dependency raises instead of mocking."""
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = E5Embedder()

    with pytest.raises(EmbeddingError, match="not installed"):
        embedder.embed_passages(["passage: teks"])


@pytest.mark.slow
def test_embedder_fails_loudly_when_weights_unavailable():
    import sentence_transformers

    original = sentence_transformers.SentenceTransformer

    def boom(*args, **kwargs):
        raise OSError("weights unavailable")

    sentence_transformers.SentenceTransformer = boom
    try:
        embedder = E5Embedder()
        with pytest.raises(EmbeddingError, match="refuses to substitute"):
            embedder.embed_passages(["passage: teks"])
    finally:
        sentence_transformers.SentenceTransformer = original


def make_chunks(text="Isi evaluasi tim."):
    chunker = SectionAwareChunker(tokenizer=WordCountTokenizer())
    return chunker.chunk_section(
        report_id="RPT0001",
        competition_name="Lomba",
        competition_year=2024,
        competition_type="Essay",
        achievement_level="Juara 1",
        source_file_id="DRV_RPT0001.pdf",
        judul_ide="Ide",
        section_type=SectionType.EVALUASI_TIM,
        section_content=text,
    )


def test_collection_records_cosine_and_embedding_model(tmp_path):
    store = ChromaStore(persist_dir=tmp_path / "chroma")

    metadata = dict(store.collection.metadata)
    assert metadata[config.CHROMA_METADATA_METRIC_KEY] == "cosine"
    assert metadata[config.CHROMA_METADATA_EMBEDDING_MODEL_KEY] == config.EMBEDDING_MODEL_NAME


def test_reopening_with_another_model_is_refused(tmp_path):
    ChromaStore(persist_dir=tmp_path / "chroma")

    with pytest.raises(EmbeddingModelMismatch, match="Refusing to mix"):
        ChromaStore(persist_dir=tmp_path / "chroma", embedding_model="other/model-v9")


def test_insert_is_idempotent_and_stores_eight_fields(tmp_path):
    store = ChromaStore(persist_dir=tmp_path / "chroma")
    embedder = E5Embedder(model=FakeSentenceModel())
    chunks = make_chunks()
    vectors = embedder.embed_passages([c.embed_text for c in chunks])

    store.insert_chunks(chunks, vectors)
    first_count = store.count_chunks()
    store.insert_chunks(chunks, vectors)

    assert store.count_chunks() == first_count
    stored = store.collection.get(include=["metadatas"])
    assert all(len(metadata) == 8 for metadata in stored["metadatas"])


def test_partial_write_failure_removes_created_chunks(tmp_path, monkeypatch):
    store = ChromaStore(persist_dir=tmp_path / "chroma")
    embedder = E5Embedder(model=FakeSentenceModel())
    chunks = make_chunks("\n".join("Isi evaluasi tim yang panjang." for _ in range(150)))
    vectors = embedder.embed_passages([c.embed_text for c in chunks])
    assert len(chunks) > 1

    def failing_add(ids, documents, metadatas, embeddings):
        store.collection.add(
            ids=[ids[0]],
            documents=[documents[0]],
            metadatas=[metadatas[0]],
            embeddings=[embeddings[0]],
        )
        raise RuntimeError("injected mid-write failure")

    monkeypatch.setattr(store, "_add", failing_add)
    with pytest.raises(RuntimeError, match="injected"):
        store.insert_chunks(chunks, vectors)

    assert store.count_chunks_for_report("RPT0001") == 0

    monkeypatch.undo()
    store.insert_chunks(chunks, vectors)
    assert store.count_chunks_for_report("RPT0001") == len(chunks)


def test_manifest_validation_refuses_other_model():
    good = {
        "schema_version": config.MANIFEST_SCHEMA_VERSION,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
    }
    ChromaSync.validate_manifest(good)

    bad = dict(good, embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    with pytest.raises(ManifestError, match="Refusing to load"):
        ChromaSync.validate_manifest(bad)

    with pytest.raises(ManifestError, match="schema_version"):
        ChromaSync.validate_manifest(dict(good, schema_version="0.9"))
