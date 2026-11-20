"""
Section-aware chunker for Lab Prestasi Ingestion.

Guarantees:
- Judul Ide is injected into the context header, never a standalone chunk.
- Two sections are never merged into one chunk.
- Tokens are counted with the e5 tokenizer (or an explicitly injected counter).
- The 'passage: ' prefix and the context header count toward the token budget.
- Hard limit of 512 tokens, asserted in addition to the 350-450 target window.
- Paragraph/sentence boundaries are respected; a single overlong sentence is
  split by word windows (and, pathologically, by characters), so the assertion
  is unreachable in practice instead of aborting a batch.
- chunk_id is deterministic: ``{report_id}:{section_type}:{NN}``.
- Exactly 8 metadata fields.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import config
from enums import SectionType


@dataclass
class DocumentChunk:
    chunk_id: str
    report_id: str
    section_type: str
    document_text: str  # context header + content (stored in Chroma, no prefix)
    embed_text: str     # 'passage: ' + document_text (what gets embedded)
    token_count: int
    metadata: Dict[str, Any]


class SectionAwareChunker:
    """Chunks evaluation sections while preserving structural context and token limits."""

    SECTION_LABELS = {
        SectionType.EXECUTIVE_SUMMARY: "Executive Summary / Abstrak",
        SectionType.EVALUASI_TIM: "Evaluasi Tim",
        SectionType.SARAN_PENGEMBANGAN: "Saran Pengembangan",
    }

    def __init__(self, tokenizer: Optional[Any] = None):
        """
        `tokenizer` may be a HuggingFace tokenizer, or any object exposing
        ``count_tokens(text) -> int``. When omitted, the e5 tokenizer is loaded
        lazily from the configured model on first use.
        """
        self.tokenizer = tokenizer
        self._counter = None

    def _resolve_counter(self):
        if self._counter is not None:
            return self._counter

        if self.tokenizer is None:
            from transformers import AutoTokenizer

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config.EMBEDDING_MODEL_NAME
                )
            except Exception as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Could not load the e5 tokenizer for chunking. Chunking requires "
                    f"the tokenizer of '{config.EMBEDDING_MODEL_NAME}'. "
                    "Inject a counter explicitly for offline runs."
                ) from exc

        if hasattr(self.tokenizer, "count_tokens"):
            self._counter = self.tokenizer.count_tokens
        else:
            self._counter = lambda text: len(
                self.tokenizer.encode(text, add_special_tokens=False)
            )
        return self._counter

    def count_tokens(self, text: str) -> int:
        """Count tokens with the e5 tokenizer (or the injected counter)."""
        return self._resolve_counter()(text)

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split on sentence boundaries without breaking mid-sentence."""
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [part.strip() for part in raw if part.strip()]

    @staticmethod
    def split_into_paragraphs(text: str) -> List[str]:
        raw = text.split("\n")
        return [part.strip() for part in raw if part.strip()]

    def _split_by_words(self, text: str, max_tokens: int) -> List[str]:
        """Greedily pack words into pieces that fit the token budget."""
        pieces: List[str] = []
        current: List[str] = []

        for word in text.split():
            candidate = " ".join(current + [word])
            if current and self.count_tokens(candidate) > max_tokens:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)

        if current:
            pieces.append(" ".join(current))
        return pieces

    def _split_by_chars(self, text: str, max_tokens: int) -> List[str]:
        """Last-resort split for a single 'word' that alone exceeds the budget."""
        pieces: List[str] = []
        remaining = text.strip()
        while remaining:
            low, high, best = 1, len(remaining), 1
            while low <= high:
                mid = (low + high) // 2
                if self.count_tokens(remaining[:mid]) <= max_tokens:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            pieces.append(remaining[:best].strip())
            remaining = remaining[best:].lstrip()
        return [piece for piece in pieces if piece]

    def _section_atoms(
        self, section_content: str, budget: int
    ) -> List[Tuple[str, str]]:
        """
        Break a section into packable atoms with their joining separator.

        Paragraphs are preferred; overlong paragraphs go to sentences; overlong
        sentences go to word windows (separator ' ').
        """
        atoms: List[Tuple[str, str]] = []

        for paragraph in self.split_into_paragraphs(section_content):
            if self.count_tokens(paragraph) <= budget:
                atoms.append((paragraph, "\n\n"))
                continue

            for sentence in self.split_into_sentences(paragraph):
                if self.count_tokens(sentence) <= budget:
                    atoms.append((sentence, " "))
                    continue

                for window in self._split_by_words(sentence, budget):
                    if self.count_tokens(window) <= budget:
                        atoms.append((window, " "))
                    else:
                        for piece in self._split_by_chars(window, budget):
                            atoms.append((piece, " "))

        return atoms

    def chunk_section(
        self,
        report_id: str,
        competition_name: str,
        competition_year: int,
        competition_type: str,
        achievement_level: str,
        source_file_id: str,
        judul_ide: str,
        section_type: SectionType,
        section_content: str,
    ) -> List[DocumentChunk]:
        """
        Chunk a single section into one or more DocumentChunks.

        Context header (embedded and stored):
        ``Kompetisi: {name} ({type}, {year}) | Judul ide: {judul} | Bagian: {label}``
        """
        label = self.SECTION_LABELS[section_type]
        header = (
            f"Kompetisi: {competition_name} ({competition_type}, {competition_year}) | "
            f"Judul ide: {judul_ide} | Bagian: {label}"
        )
        passage_prefix = config.EMBEDDING_PASSAGE_PREFIX
        header_tokens = self.count_tokens(f"{passage_prefix}{header}\n\n")
        budget = config.CHUNK_TARGET_MAX_TOKENS - header_tokens
        hard_budget = config.CHUNK_HARD_LIMIT_TOKENS - header_tokens

        if budget <= 0:
            raise ValueError(
                f"Context header for {report_id}:{section_type.value} consumes "
                f"{header_tokens} tokens, leaving no room inside the "
                f"{config.CHUNK_TARGET_MAX_TOKENS}-token target."
            )

        def build_chunk(content_text: str, chunk_number: int) -> DocumentChunk:
            document_text = f"{header}\n\n{content_text}"
            embed_text = f"{passage_prefix}{document_text}"
            token_count = self.count_tokens(embed_text)

            assert token_count <= config.CHUNK_HARD_LIMIT_TOKENS, (
                f"Chunk {report_id}:{section_type.value}:{chunk_number:02d} has "
                f"{token_count} tokens, above the hard limit "
                f"{config.CHUNK_HARD_LIMIT_TOKENS}"
            )

            chunk_id = f"{report_id}:{section_type.value}:{chunk_number:02d}"
            metadata = {
                "report_id": report_id,
                "competition_name": competition_name,
                "competition_year": int(competition_year),
                "achievement_level": achievement_level,
                "competition_type": competition_type,
                "section_type": section_type.value,
                "source_file_id": source_file_id,
                "chunk_id": chunk_id,
            }
            assert len(metadata) == 8, (
                f"Metadata must have exactly 8 fields, got {len(metadata)}"
            )

            return DocumentChunk(
                chunk_id=chunk_id,
                report_id=report_id,
                section_type=section_type.value,
                document_text=document_text,
                embed_text=embed_text,
                token_count=token_count,
                metadata=metadata,
            )

        atoms = self._section_atoms(section_content, budget)
        chunks: List[DocumentChunk] = []
        current_text = ""
        chunk_number = 1

        for atom, separator in atoms:
            candidate = f"{current_text}{separator}{atom}" if current_text else atom
            if current_text and self.count_tokens(candidate) > budget:
                chunks.append(build_chunk(current_text, chunk_number))
                chunk_number += 1
                current_text = atom
            else:
                current_text = candidate

        if current_text:
            chunks.append(build_chunk(current_text, chunk_number))

        return chunks
