"""
PDF parser module for Lab Prestasi Ingestion.

Uses PyMuPDF to extract text in reading order, removes page numbers and repeating
page headers/footers, normalizes whitespace, and separates the four evaluation
sections with tolerant heading detection (Brief 5.2 #4 and #6).

All patterns live in config.py.
"""

from collections import Counter
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pymupdf

import config


class PDFParser:
    """Extracts, cleans, and separates text sections from evaluation PDFs."""

    def __init__(self, min_text_chars: int = config.PDF_MIN_TEXT_CHARS):
        self.min_text_chars = min_text_chars

    @staticmethod
    def is_heading(line: str) -> bool:
        """True when the line looks like one of the configured section headings."""
        return any(pattern.match(line) for pattern in config.SECTION_PATTERNS.values())

    @staticmethod
    def _read_pages(pdf_path: Path) -> List[List[str]]:
        """Return non-empty, stripped lines per page, with page-number lines removed."""
        document = pymupdf.open(str(pdf_path))
        pages: List[List[str]] = []
        try:
            for page in document:
                text = page.get_text("text")
                text = config.PATTERN_PAGE_NUMBER.sub("", text)
                lines = [line.strip() for line in text.split("\n")]
                pages.append([line for line in lines if line])
        finally:
            document.close()
        return pages

    @classmethod
    def _detect_repeated_margins(cls, pages: List[List[str]]) -> Set[str]:
        """
        Find lines repeated in the top/bottom margin of most pages (running
        headers/footers). Section headings are never treated as repeatable.
        """
        if len(pages) < config.REPEATED_MARGIN_MIN_PAGES:
            return set()

        edge_lines = config.REPEATED_MARGIN_LINES
        counts: Counter[str] = Counter()
        for lines in pages:
            counts.update(set(lines[:edge_lines] + lines[-edge_lines:]))

        threshold = max(
            config.REPEATED_MARGIN_MIN_PAGES,
            ceil(len(pages) * config.REPEATED_MARGIN_RATIO),
        )
        repeated = {
            line
            for line, count in counts.items()
            if count >= threshold
            and 3 < len(line) <= config.REPEATED_MARGIN_MAX_CHARS
        }
        # Never drop a section heading, even if it looks like a running header.
        return {line for line in repeated if not cls.is_heading(line)}

    def extract_and_clean_text(self, pdf_path: Path) -> str:
        """Extract selectable text from every page and normalize it."""
        pages = self._read_pages(pdf_path)
        repeated = self._detect_repeated_margins(pages)

        kept_lines: List[str] = []
        for lines in pages:
            kept_lines.extend(line for line in lines if line not in repeated)

        return "\n".join(kept_lines)

    def parse_sections(
        self, cleaned_text: str
    ) -> Tuple[bool, Optional[Dict[str, str]], str]:
        """
        Detect the four mandatory sections using tolerant heading matching.

        Returns: (success, sections_dict or None, reason_if_failed).
        Text outside the four sections is ignored (Brief 5.2 #6).
        """
        if len(cleaned_text.strip()) < self.min_text_chars:
            return False, None, "PDF tidak memiliki teks (kemungkinan hasil scan)"

        matches: Dict[str, Tuple[int, int]] = {}
        for section_name, pattern in config.SECTION_PATTERNS.items():
            match = pattern.search(cleaned_text)
            if not match:
                return (
                    False,
                    None,
                    f"Bagian dokumen tidak lengkap: '{section_name}' tidak ditemukan",
                )
            matches[section_name] = (match.start(), match.end())

        ordered = sorted(matches.items(), key=lambda item: item[1][0])

        extracted: Dict[str, str] = {}
        for position, (section_name, (_, heading_end)) in enumerate(ordered):
            content_end = (
                ordered[position + 1][1][0]
                if position + 1 < len(ordered)
                else len(cleaned_text)
            )
            extracted[section_name] = cleaned_text[heading_end:content_end].strip()

        for section_name, text in extracted.items():
            if not text:
                return False, None, f"Bagian dokumen '{section_name}' kosong"

        return True, extracted, "OK"
