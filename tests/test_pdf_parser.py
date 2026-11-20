"""
Tests for the PDF parser: tolerant heading detection, page-number removal,
repeated header/footer removal, and section boundaries. All offline and fast.
"""

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import config
from ingestion.pdf_parser import PDFParser


def build_pdf(path: Path, page_lines):
    """Write a simple PDF with one list of text lines per page."""
    pdf = canvas.Canvas(str(path), pagesize=A4)
    _, height = A4
    for lines in page_lines:
        y = height - 60
        for line in lines:
            pdf.drawString(50, y, line)
            y -= 18
        pdf.showPage()
    pdf.save()
    return path


def test_heading_detection_tolerance():
    parser = PDFParser()
    text = (
        "1. JUDUL IDE: Inovasi Solar Cell\n"
        "Ini adalah penjelasan judul ide yang sangat inovatif.\n"
        "2) abstrak :\n"
        "Ringkasan eksekutif mengenai panel surya generasi ketiga.\n"
        "III - Evaluasi Tim :\n"
        "Kerjasama tim berjalan sangat baik selama masa persiapan kompetisi.\n"
        "4.   Saran Pengembangan  \n"
        "Disarankan menambah kapasitas baterai penyimpanan daya.\n"
    )

    success, sections, err = parser.parse_sections(text)

    assert success is True, err
    assert "Inovasi Solar Cell" in sections["judul_ide"]
    assert "Ringkasan eksekutif" in sections["executive_summary"]
    assert "Kerjasama tim" in sections["evaluasi_tim"]
    assert "Disarankan menambah" in sections["saran_pengembangan"]


@pytest.mark.parametrize(
    "heading",
    [
        "Judul Ide",
        "JUDUL IDE:",
        "1. Judul Ide",
        "2) Judul Ide",
        "IV. Judul Ide",
        "III - Judul Ide",
        "   Judul   Ide   ",
    ],
)
def test_judul_ide_heading_variants(heading):
    assert config.SECTION_PATTERNS["judul_ide"].match(heading) is not None


@pytest.mark.parametrize(
    "heading",
    ["Saran Pengembangan", "4. Saran Pengembangan", "4) saran pengembangan:", "V - Saran Pengembangan -"],
)
def test_saran_heading_variants(heading):
    assert config.SECTION_PATTERNS["saran_pengembangan"].match(heading) is not None


def test_sections_detected_in_any_order():
    parser = PDFParser()
    text = (
        "Saran Pengembangan\n"
        "Perbaiki dokumentasi teknis.\n"
        "2. Evaluasi Tim\n"
        "Tim solid namun manajemen waktu lemah.\n"
        "Abstrak\n"
        "Ringkasan proyek.\n"
        "Judul Ide\n"
        "Sistem Monitoring Kualitas Udara.\n"
    )

    success, sections, err = parser.parse_sections(text)

    assert success is True, err
    assert sections["judul_ide"] == "Sistem Monitoring Kualitas Udara."
    assert sections["executive_summary"] == "Ringkasan proyek."
    assert sections["evaluasi_tim"] == "Tim solid namun manajemen waktu lemah."
    assert sections["saran_pengembangan"] == "Perbaiki dokumentasi teknis."


def test_text_outside_sections_is_ignored():
    parser = PDFParser()
    text = (
        "LAPORAN EVALUASI KOMPETISI MAHASISWA\n"
        "Dokumen administratif yang tidak boleh masuk basis pengetahuan.\n"
        "Judul Ide\n"
        "Ide brilian.\n"
        "Executive Summary\n"
        "Ringkasan.\n"
        "Evaluasi Tim\n"
        "Evaluasi.\n"
        "Saran Pengembangan\n"
        "Saran.\n"
        "Lampiran administratif di bagian akhir.\n"
    )

    success, sections, err = parser.parse_sections(text)

    assert success is True, err
    # Preamble before the first heading is discarded.
    assert "LAPORAN EVALUASI" not in "".join(sections.values())
    assert "Dokumen administratif" not in "".join(sections.values())
    # Known limitation: text after the LAST section has no closing boundary and
    # is attributed to that section. Template documents end with Saran
    # Pengembangan, so this is accepted and documented.
    assert "Lampiran administratif" in sections["saran_pengembangan"]


def test_missing_section_reports_clear_reason():
    parser = PDFParser()
    text = (
        "Judul Ide\nIde.\n"
        "Executive Summary\nRingkasan panjang yang memadai untuk melewati ambang.\n"
        "Evaluasi Tim\nEvaluasi.\n"
    )

    success, sections, reason = parser.parse_sections(text)

    assert success is False
    assert sections is None
    assert "saran_pengembangan" in reason


def test_page_numbers_and_repeated_margins_are_removed(tmp_path):
    repeated_header = "LAB PRESTASI - LAPORAN EVALUASI"
    repeated_footer = "Dokumen internal, jangan disebarkan"
    pages = []
    for page_number in range(1, 4):
        pages.append(
            [
                repeated_header,
                f"Isi halaman {page_number} baris pertama.",
                f"Isi halaman {page_number} baris kedua.",
                f"- {page_number} -",
                repeated_footer,
            ]
        )
    pdf_path = build_pdf(tmp_path / "margins.pdf", pages)

    parser = PDFParser()
    text = parser.extract_and_clean_text(pdf_path)

    assert repeated_header not in text
    assert repeated_footer not in text
    assert "Isi halaman 1 baris pertama." in text
    assert "Isi halaman 3 baris kedua." in text
    assert "\n- 1 -\n" not in text
    assert "\n1\n" not in text


def test_single_page_document_keeps_all_lines(tmp_path):
    pdf_path = build_pdf(
        tmp_path / "single.pdf",
        [["Judul Ide", "Ide.", "Executive Summary", "Ringkasan yang cukup panjang."]],
    )

    text = PDFParser().extract_and_clean_text(pdf_path)

    assert "Judul Ide" in text
    assert "Ringkasan yang cukup panjang." in text


def test_scanned_pdf_has_no_extractable_text(tmp_path):
    from PIL import Image, ImageDraw

    image_path = tmp_path / "scan.png"
    image = Image.new("RGB", (600, 800), "white")
    ImageDraw.Draw(image).text((40, 40), "Hasil scan tanpa teks", fill="black")
    image.save(image_path)

    pdf = canvas.Canvas(str(tmp_path / "scan.pdf"), pagesize=A4)
    pdf.drawImage(str(image_path), 40, 200, width=480, height=620)
    pdf.save()

    text = PDFParser().extract_and_clean_text(tmp_path / "scan.pdf")
    success, sections, reason = PDFParser().parse_sections(text)

    assert len(text.strip()) < config.PDF_MIN_TEXT_CHARS
    assert success is False
    assert "scan" in reason.lower()
