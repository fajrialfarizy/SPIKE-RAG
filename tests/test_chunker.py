"""
Fast, offline tests for the section-aware chunker.

The chunker is injected with a deterministic word-count tokenizer (tests/stubs.py)
so the default suite never loads model weights.
"""

import re

import pytest

import config
from enums import SectionType
from ingestion.chunker import SectionAwareChunker
from tests.stubs import WordCountTokenizer

HEADER_ARGS = dict(
    report_id="RPT0001",
    competition_name="Lomba Contoh",
    competition_year=2024,
    competition_type="Essay",
    achievement_level="Juara 1",
    source_file_id="DRV_RPT0001.pdf",
    judul_ide="Ide Hebat",
)


def make_chunker():
    return SectionAwareChunker(tokenizer=WordCountTokenizer())


def words(count: int) -> str:
    return " ".join(f"kata{i}" for i in range(count))


def test_short_section_is_one_chunk_with_context_header():
    chunker = make_chunker()
    chunks = chunker.chunk_section(
        section_type=SectionType.EVALUASI_TIM,
        section_content="Tim bekerja sangat baik dan saling melengkapi.",
        **HEADER_ARGS,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "RPT0001:evaluasi_tim:01"
    assert chunk.document_text.startswith(
        "Kompetisi: Lomba Contoh (Essay, 2024) | Judul ide: Ide Hebat | "
        "Bagian: Evaluasi Tim"
    )
    assert chunk.embed_text == f"{config.EMBEDDING_PASSAGE_PREFIX}{chunk.document_text}"
    assert chunk.token_count == chunker.count_tokens(chunk.embed_text)
    assert chunk.token_count <= config.CHUNK_HARD_LIMIT_TOKENS
    assert len(chunk.metadata) == 8
    assert chunk.metadata["competition_year"] == 2024


def test_judul_ide_is_never_a_standalone_chunk():
    chunker = make_chunker()
    for section_type in SectionType:
        chunks = chunker.chunk_section(
            section_type=section_type,
            section_content="Isi bagian.",
            **HEADER_ARGS,
        )
        assert all(c.section_type == section_type.value for c in chunks)
        assert all(c.metadata["section_type"] != "judul_ide" for c in chunks)


def test_two_sections_are_never_mixed():
    chunker = make_chunker()
    evaluasi = chunker.chunk_section(
        section_type=SectionType.EVALUASI_TIM,
        section_content=(words(120) + ".\n") * 6,
        **HEADER_ARGS,
    )
    saran = chunker.chunk_section(
        section_type=SectionType.SARAN_PENGEMBANGAN,
        section_content=(words(120) + ".\n") * 6,
        **HEADER_ARGS,
    )

    assert len(evaluasi) > 1
    assert len(saran) > 1
    for chunk in evaluasi:
        assert "Bagian: Evaluasi Tim" in chunk.document_text
        assert "Saran Pengembangan" not in chunk.document_text
    for chunk in saran:
        assert "Bagian: Saran Pengembangan" in chunk.document_text
        assert "Evaluasi Tim" not in chunk.document_text


def test_long_section_splits_on_paragraph_boundaries_within_budget():
    chunker = make_chunker()
    paragraphs = [words(100) + "." for _ in range(10)]
    content = "\n".join(paragraphs)

    chunks = chunker.chunk_section(
        section_type=SectionType.EVALUASI_TIM,
        section_content=content,
        **HEADER_ARGS,
    )

    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks] == [
        f"RPT0001:evaluasi_tim:{i:02d}" for i in range(1, len(chunks) + 1)
    ]
    for chunk in chunks:
        assert chunk.token_count <= config.CHUNK_HARD_LIMIT_TOKENS
    # Paragraphs are never cut: every chunk boundary lands between paragraphs.
    for chunk in chunks[:-1]:
        assert chunk.document_text.endswith(".")


def test_single_overlong_sentence_is_split_by_words_without_raising():
    chunker = make_chunker()
    sentence = words(900) + "."

    chunks = chunker.chunk_section(
        section_type=SectionType.SARAN_PENGEMBANGAN,
        section_content=sentence,
        **HEADER_ARGS,
    )

    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk.token_count <= config.CHUNK_HARD_LIMIT_TOKENS
    # No content is lost or duplicated by the fallback splitter.
    rebuilt = " ".join(
        c.document_text.split("\n\n", 1)[1] for c in chunks
    ).split()
    assert rebuilt == sentence.split()


def test_pathological_single_token_of_huge_length_is_handled():
    chunker = make_chunker()
    blob = "x" * 5000  # a single 'word' with no separators

    chunks = chunker.chunk_section(
        section_type=SectionType.EXECUTIVE_SUMMARY,
        section_content=blob,
        **HEADER_ARGS,
    )

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.token_count <= config.CHUNK_HARD_LIMIT_TOKENS
    assert "".join(c.document_text.split("\n\n", 1)[1] for c in chunks) == blob


def test_chunking_is_deterministic():
    chunker = make_chunker()
    content = "\n".join(words(80) + "." for _ in range(8))

    first = chunker.chunk_section(
        section_type=SectionType.EVALUASI_TIM, section_content=content, **HEADER_ARGS
    )
    second = chunker.chunk_section(
        section_type=SectionType.EVALUASI_TIM, section_content=content, **HEADER_ARGS
    )

    assert [(c.chunk_id, c.document_text) for c in first] == [
        (c.chunk_id, c.document_text) for c in second
    ]


def test_chunk_id_pattern_is_stable():
    chunker = make_chunker()
    chunks = chunker.chunk_section(
        section_type=SectionType.EXECUTIVE_SUMMARY,
        section_content=words(20),
        **HEADER_ARGS,
    )

    for chunk in chunks:
        assert re.fullmatch(r"RPT0001:executive_summary:\d{2}", chunk.chunk_id)
