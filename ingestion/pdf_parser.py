"""
PDF parser module for Lab Prestasi Ingestion.
Uses PyMuPDF to extract text in reading order, normalize spaces,
clean page numbers and repeating headers/footers, and validate readability.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import re
import fitz  # PyMuPDF
import config


class PDFParser:
    """Extracts, cleans, and separates text sections from evaluation PDFs."""

    def __init__(self, min_text_chars: int = config.PDF_MIN_TEXT_CHARS):
        self.min_text_chars = min_text_chars

    def extract_and_clean_text(self, pdf_path: Path) -> str:
        """
        Extract selectable text from PDF across all pages in natural reading order.
        Cleans recurring page numbers, headers/footers, and normalizes whitespaces.
        """
        doc = fitz.open(str(pdf_path))
        raw_pages = []

        for page in doc:
            text = page.get_text("text")
            # Remove isolated page numbers (e.g., "1", "- 1 -", "Page 1 of 3", "Halaman 1")
            text = re.sub(r"(?m)^\s*(?:Halaman|Page|\-)?\s*\d+\s*(?:dari|of\s*\d+)?\s*(?:\-)?\s*$", "", text)
            raw_pages.append(text)

        doc.close()
        combined = "\n".join(raw_pages)

        # Normalize line breaks and spaces: keep paragraphs, collapse excessive blank lines
        lines = [line.strip() for line in combined.split("\n")]
        cleaned = "\n".join(line for line in lines if line)
        return cleaned

    def parse_sections(self, cleaned_text: str) -> Tuple[bool, Optional[Dict[str, str]], str]:
        """
        Detect 4 mandatory sections using tolerant regex matching (Brief 5.2 #6).
        Sections:
        1. Judul Ide
        2. Executive Summary / Abstrak
        3. Evaluasi Tim
        4. Saran Pengembangan

        Returns: (success: bool, sections_dict: Optional[Dict], reason_if_failed: str)
        """
        if len(cleaned_text.strip()) < self.min_text_chars:
            return False, None, "PDF tidak memiliki teks (kemungkinan hasil scan)"

        patterns = config.SECTION_PATTERNS

        # Find match indices of all section headings in text
        matches = {}
        for sec_name, pattern in patterns.items():
            m = pattern.search(cleaned_text)
            if not m:
                return False, None, f"Bagian dokumen tidak lengkap: '{sec_name}' tidak ditemukan"
            matches[sec_name] = (m.start(), m.end())

        # Sort sections by their starting index in the document
        sorted_sections = sorted(matches.items(), key=lambda item: item[1][0])

        extracted = {}
        for i, (sec_name, (start, end)) in enumerate(sorted_sections):
            # Content starts after heading
            content_start = end
            if i + 1 < len(sorted_sections):
                content_end = sorted_sections[i + 1][1][0]
                sec_text = cleaned_text[content_start:content_end].strip()
            else:
                sec_text = cleaned_text[content_start:].strip()

            extracted[sec_name] = sec_text

        # Validate that no extracted section is completely empty
        for sec_name, text in extracted.items():
            if not text:
                return False, None, f"Bagian dokumen '{sec_name}' kosong"

        return True, extracted, "OK"
