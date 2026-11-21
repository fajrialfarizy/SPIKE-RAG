"""
PDF parser module for Lab Prestasi Ingestion.

Uses PyMuPDF to extract text in reading order, removes page numbers and repeating
page headers/footers, reflows wrapped PDF lines into real paragraphs, and
separates the four evaluation sections with tolerant heading detection
(Brief 5.2 #4 and #6).

PyMuPDF emits one newline per visual line and no paragraph marker, so paragraph
breaks come from block boundaries and from real blank lines; everything else is a
wrap and is joined back into one paragraph (fix pack L1).

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

    @classmethod
    def is_heading_line(cls, line: str) -> bool:
        """True only when the whole line is a heading, not a sentence starting like one."""
        stripped = line.strip()
        return any(
            pattern.match(stripped) and pattern.match(stripped).end() == len(stripped)
            for pattern in config.SECTION_PATTERNS.values()
        )

    @staticmethod
    def _read_pages(pdf_path: Path) -> List[List[str]]:
        """Lines per page, page numbers removed, with one empty marker per block."""
        document = pymupdf.open(str(pdf_path))
        pages: List[List[str]] = []
        try:
            for page in document:
                lines: List[str] = []
                for block in page.get_text("blocks"):
                    # Image blocks carry no text.
                    if len(block) > 6 and block[6] != 0:
                        continue
                    block_lines = []
                    for line in str(block[4]).split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        if config.PATTERN_PAGE_NUMBER.sub("", line).strip():
                            block_lines.append(line)
                    if block_lines:
                        lines.extend(block_lines)
                        # A block boundary is a real paragraph break, unlike a wrap.
                        lines.append("")
                while lines and not lines[-1]:
                    lines.pop()
                pages.append(lines)
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
            non_empty = [line for line in lines if line]
            counts.update(set(non_empty[:edge_lines] + non_empty[-edge_lines:]))

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
        return {line for line in repeated if not cls.is_heading_line(line)}

    def extract_and_clean_text(self, pdf_path: Path) -> str:
        """Extract selectable text from every page and normalize it."""
        pages = self._read_pages(pdf_path)
        repeated = self._detect_repeated_margins(pages)

        kept_pages: List[str] = []
        for lines in pages:
            kept = [line for line in lines if line not in repeated]
            while kept and not kept[0]:
                kept.pop(0)
            while kept and not kept[-1]:
                kept.pop()
            kept_pages.append("\n".join(kept))

        # A page boundary is a continuation, not a paragraph break: the last line
        # of one page and the first line of the next belong together.
        return "\n".join(kept_pages)

    @classmethod
    def reflow_section(cls, text: str) -> str:
        """
        Rebuild paragraphs from wrapped PDF lines (fix pack L1).

        Joins consecutive lines with a single space; keeps a paragraph break on a
        blank line, on a list-item marker and before/after a heading; joins a
        hyphenated wrap (``terburu-`` + ``buru``) into one word without deleting
        the hyphen in any other case.
        """
        paragraphs: List[str] = []
        current = ""

        def flush() -> None:
            nonlocal current
            if current:
                paragraphs.append(current)
                current = ""

        for raw_line in (text or "").split("\n"):
            line = " ".join(raw_line.split())
            if not line:
                flush()
                continue
            if cls.is_heading_line(line):
                flush()
                paragraphs.append(line)
                continue
            if config.PATTERN_PDF_LIST_ITEM.match(line):
                flush()
                current = line
                continue
            if not current:
                current = line
                continue
            if (
                current.endswith(config.PDF_HYPHEN_BREAK_CHARS)
                and line[:1].islower()
            ):
                # The hyphen belongs to the word ("terburu-" + "buru"), so it is
                # kept and only the line break disappears.
                current = f"{current}{line}"
            else:
                current = f"{current}{config.PDF_LINE_JOIN_SEPARATOR}{line}"

        flush()
        return config.PDF_PARAGRAPH_SEPARATOR.join(paragraphs)

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
            extracted[section_name] = self.reflow_section(
                cleaned_text[heading_end:content_end]
            )

        for section_name, text in extracted.items():
            if not text:
                return False, None, f"Bagian dokumen '{section_name}' kosong"

        return True, extracted, "OK"
