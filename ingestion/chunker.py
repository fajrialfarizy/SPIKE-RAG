"""
Section-aware chunker for Lab Prestasi Ingestion.
Enforces:
- Judul Ide is injected into context header, NEVER a standalone chunk.
- Never blends two sections in one chunk.
- Token counts measured with intfloat/multilingual-e5-small tokenizer.
- Prefix 'passage: ' and header included in token count.
- Hard limit <= 512 tokens (assertion + test).
- Splits by paragraphs/sentences without splitting mid-sentence.
- Deterministic chunk_id format: {report_id}:{section_type}:{2-digit-index}.
- Exactly 8 metadata fields.
"""

from typing import List, Dict, Any, Optional
import glob
import os
import re
from dataclasses import dataclass
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from enums import SectionType
import config


@dataclass
class DocumentChunk:
    chunk_id: str
    report_id: str
    section_type: str
    document_text: str  # Context Header + Content (stored in Chroma without 'passage: ')
    embed_text: str     # 'passage: ' + document_text (used for embedding calculation)
    token_count: int
    metadata: Dict[str, Any]


class SectionAwareChunker:
    """Chunks evaluation sections while preserving structural context and token limits."""

    def __init__(self, tokenizer: Optional[Any] = None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            # Fast local loading via tokenizers from HF cache, falling back to AutoTokenizer
            cache_pattern = os.path.expanduser(
                "~/.cache/huggingface/hub/models--intfloat--multilingual-e5-small/snapshots/*/tokenizer.json"
            )
            matches = glob.glob(cache_pattern)
            if matches:
                self.tokenizer = Tokenizer.from_file(matches[0])
                self._is_fast = True
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
                self._is_fast = False

    def count_tokens(self, text: str) -> int:
        """Count tokens using E5 tokenizer."""
        if hasattr(self, "_is_fast") and self._is_fast:
            return len(self.tokenizer.encode(text).ids)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences cleanly without breaking mid-sentence."""
        # Split on sentence boundaries (.!? followed by whitespace or newline)
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        raw_paras = text.split("\n")
        paras = [p.strip() for p in raw_paras if p.strip()]
        return paras

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
        Embeds context header:
        Kompetisi: {competition_name} ({competition_type}, {competition_year}) | Judul ide: {judul_ide} | Bagian: {label}
        """
        section_label_map = {
            SectionType.EXECUTIVE_SUMMARY: "Executive Summary / Abstrak",
            SectionType.EVALUASI_TIM: "Evaluasi Tim",
            SectionType.SARAN_PENGEMBANGAN: "Saran Pengembangan",
        }
        label = section_label_map[section_type]

        header = (
            f"Kompetisi: {competition_name} ({competition_type}, {competition_year}) | "
            f"Judul ide: {judul_ide} | Bagian: {label}"
        )
        passage_prefix = config.EMBEDDING_PASSAGE_PREFIX

        # Calculate base tokens taken by prefix and header
        header_tokens = self.count_tokens(f"{passage_prefix}{header}\n\n")
        available_tokens = config.CHUNK_TARGET_MAX_TOKENS - header_tokens
        hard_available_tokens = config.CHUNK_HARD_LIMIT_TOKENS - header_tokens

        paragraphs = self.split_into_paragraphs(section_content)
        chunks: List[DocumentChunk] = []
        current_paras: List[str] = []

        def build_chunk(content_text: str, chunk_num: int) -> DocumentChunk:
            doc_text = f"{header}\n\n{content_text}"
            embed_text = f"{passage_prefix}{doc_text}"
            tok_count = self.count_tokens(embed_text)

            # Strict assertion: No chunk ever exceeds 512 tokens
            assert tok_count <= config.CHUNK_HARD_LIMIT_TOKENS, (
                f"Chunk token count {tok_count} exceeds hard limit {config.CHUNK_HARD_LIMIT_TOKENS}!"
            )

            cid = f"{report_id}:{section_type.value}:{chunk_num:02d}"

            # Strictly 8 metadata fields as specified in PRD 7.3 & Brief 5.3
            metadata = {
                "report_id": report_id,
                "competition_name": competition_name,
                "competition_year": int(competition_year),
                "achievement_level": achievement_level,
                "competition_type": competition_type,
                "section_type": section_type.value,
                "source_file_id": source_file_id,
                "chunk_id": cid,
            }
            assert len(metadata) == 8, f"Metadata must have exactly 8 fields, got {len(metadata)}"

            return DocumentChunk(
                chunk_id=cid,
                report_id=report_id,
                section_type=section_type.value,
                document_text=doc_text,
                embed_text=embed_text,
                token_count=tok_count,
                metadata=metadata,
            )

        chunk_counter = 1

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If a single paragraph is too large to fit, break by sentences
            if para_tokens > available_tokens:
                sentences = self.split_into_sentences(para)
                for sent in sentences:
                    candidate = " ".join(current_paras + [sent])
                    cand_tokens = self.count_tokens(candidate)
                    if cand_tokens > available_tokens and current_paras:
                        # Flush current
                        chunks.append(build_chunk(" ".join(current_paras), chunk_counter))
                        chunk_counter += 1
                        current_paras = [sent]
                    else:
                        current_paras.append(sent)
            else:
                candidate = "\n\n".join(current_paras + [para])
                cand_tokens = self.count_tokens(candidate)
                if cand_tokens > available_tokens and current_paras:
                    chunks.append(build_chunk("\n\n".join(current_paras), chunk_counter))
                    chunk_counter += 1
                    current_paras = [para]
                else:
                    current_paras.append(para)

        if current_paras:
            chunks.append(build_chunk("\n\n".join(current_paras), chunk_counter))

        return chunks
