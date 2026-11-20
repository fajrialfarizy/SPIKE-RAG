"""
PII scrubbing tests (supervisor decisions Q2/Q3).

Positive cases must disappear; the legitimate funding amount must survive.
"""

import pytest

import config
from ingestion.pii_scrubber import PIIScrubber

REDACTED = config.PII_REDACTION_REPLACEMENT


@pytest.fixture
def scrubber(tmp_path):
    denylist = tmp_path / "pii_denylist.txt"
    denylist.write_text(
        "# dummy names\nBudi Santoso\nAhmad Fauzi\nHendra\n", encoding="utf-8"
    )
    return PIIScrubber(denylist_path=denylist)


@pytest.mark.parametrize(
    "text",
    [
        "Ketua tim Ahmad Fauzi (NIM: 13520099) mengkoordinir pengujian.",
        "NPM 13520099 terdaftar sebagai peserta.",
        "NRP: 13520099",
        "nomor induk 13520099",
        "Mahasiswa dengan kode 13520099 mengikuti lomba.",
    ],
)
def test_nim_is_always_redacted(scrubber, text):
    result, counts = scrubber.scrub(text)

    assert "13520099" not in result
    assert REDACTED in result
    assert counts["nim"] >= 1


def test_label_is_preserved_when_nim_is_redacted(scrubber):
    result, _ = scrubber.scrub("NIM: 13520099")

    assert result == f"NIM: {REDACTED}"


@pytest.mark.parametrize(
    "text",
    [
        "Anggaran pilot project disetujui sebesar Rp 25.000.000 dari hibah riset.",
        "nominal dana: 25000000 dari hibah kementerian.",
        "Biaya operasional 15000000 sudah cair.",
        "Total anggaran 90000000 dialokasikan untuk produksi.",
        "Hadiah 10000000 diberikan kepada juara satu.",
        "Nilai kontrak senilai 75000000 rupiah.",
    ],
)
def test_legitimate_amounts_survive(scrubber, text):
    result, counts = scrubber.scrub(text)

    assert REDACTED not in result
    assert counts["nim"] == 0
    # The figures themselves are untouched.
    assert any(ch.isdigit() for ch in result)


def test_funding_amount_with_both_formats_survives(scrubber):
    text = (
        "Sebagai catatan transparansi alokasi, anggaran dana operasional pilot "
        "project disetujui sebesar Rp 25.000.000 (nominal dana: 25000000) dari "
        "hibah riset kementerian."
    )

    result, counts = scrubber.scrub(text)

    assert "Rp 25.000.000" in result
    assert "25000000" in result
    assert REDACTED not in result
    assert counts["nim"] == 0


def test_phone_is_detected_before_nim(scrubber):
    result, counts = scrubber.scrub("Hubungi 081234567890 untuk konfirmasi.")

    assert "081234567890" not in result
    assert counts["phone"] == 1
    assert counts["nim"] == 0


def test_email_is_redacted(scrubber):
    result, counts = scrubber.scrub("Email: budi.santoso2024@gmail.com")

    assert "budi.santoso2024@gmail.com" not in result
    assert counts["email"] == 1


def test_name_after_role_label_is_redacted(scrubber):
    result, counts = scrubber.scrub(
        "Ketua tim Ahmad Fauzi mengkoordinir pengujian sensor elektroda."
    )

    assert "Ahmad Fauzi" not in result
    assert "Ketua tim" in result
    assert REDACTED in result
    assert counts["name"] >= 1


def test_name_with_honorifics_is_redacted(scrubber):
    result, counts = scrubber.scrub("Pengujian dibantu oleh mentor Dr. Ir. Hendra.")

    assert "Hendra" not in result
    assert REDACTED in result
    assert counts["name"] >= 1


def test_name_with_academic_suffix_is_redacted(scrubber):
    result, _ = scrubber.scrub("Pendamping lapangan Rina Kusuma, S.T., M.T. hadir.")

    assert "Rina Kusuma" not in result
    assert "S.T." not in result


def test_denylisted_name_is_redacted_without_title_or_role(scrubber):
    result, counts = scrubber.scrub("Catatan tambahan dari Hendra pada rapat akhir.")

    assert "Hendra" not in result
    assert counts["name"] >= 1


def test_full_pii_document_sample(scrubber):
    text = (
        "Untuk informasi lebih lanjut mengenai prototipe ini, hubungi kontak "
        "pelapor Budi Santoso di 081234567890 atau melalui surat elektronik "
        "budi.santoso2024@gmail.com. Ketua tim Ahmad Fauzi (NIM: 13520099) "
        "mengkoordinir pengujian sensor elektroda. Pengujian dibantu oleh mentor "
        "Dr. Ir. Hendra di nomor +6281198765432 dan email resmi "
        "hendra@institusi.ac.id."
    )

    result, counts = scrubber.scrub(text)

    for leaked in (
        "Budi Santoso",
        "Ahmad Fauzi",
        "Hendra",
        "13520099",
        "081234567890",
        "+6281198765432",
        "budi.santoso2024@gmail.com",
        "hendra@institusi.ac.id",
    ):
        assert leaked not in result

    assert counts["email"] == 2
    assert counts["phone"] == 2
    assert counts["nim"] == 1
    assert counts["name"] >= 3


def test_counts_are_reported_per_category(scrubber):
    result, counts = scrubber.scrub(
        "NIM: 13520099, telepon 081234567890, surel a@b.co, ketua tim Ahmad Fauzi."
    )

    assert set(counts) == {"nim", "phone", "email", "name"}
    assert counts["nim"] == 1
    assert counts["phone"] == 1
    assert counts["email"] == 1
    assert counts["name"] == 1
    assert "13520099" not in result
