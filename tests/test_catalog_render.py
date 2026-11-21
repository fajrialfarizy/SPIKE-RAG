"""Offline tests for the backend-rendered catalog answer (fix pack G5)."""

import pytest

import config
from core.catalog import achievement_group_label, render_catalog_answer


def report(
    index,
    competition_type="Business Plan",
    year=2024,
    level="Juara 1",
):
    return {
        "report_id": f"RPT{index:04d}",
        "competition_name": f"Kompetisi Nomor {index}",
        "competition_type": competition_type,
        "competition_year": year,
        "achievement_level": level,
    }


@pytest.mark.parametrize(
    "level,label",
    [
        ("Juara 1", "juara"),
        ("Juara 2", "juara"),
        ("Juara 3", "juara"),
        ("Juara Harapan", "juara"),
        ("Most Aspiring dan penghargaan serupa", "juara"),
        ("Finalis", "belum juara"),
    ],
)
def test_every_achievement_level_maps_to_its_configured_group(level, label):
    assert achievement_group_label(level) == label


def test_the_row_list_is_capped_and_the_totals_are_stated():
    reports = [report(index) for index in range(1, 36)]

    text = render_catalog_answer(reports)

    rows = [line for line in text.splitlines() if line.startswith("- RPT")]
    assert len(rows) == config.CATALOG_MAX_ROWS
    assert "Menampilkan 30 dari 35 laporan; persempit dengan jenis kompetisi, tahun, atau pencapaian." in text


def test_the_summary_counts_type_year_and_achievement_group():
    reports = [
        report(1, competition_type="Business Plan", year=2024, level="Juara 1"),
        report(2, competition_type="Essay", year=2023, level="Finalis"),
    ]

    text = render_catalog_answer(reports)

    assert "- Jenis kompetisi: Business Plan (1); Essay (1)" in text
    assert "- Tahun: 2023 (1); 2024 (1)" in text
    assert "- Pencapaian: belum juara (1); juara (1)" in text


def test_rows_are_ordered_by_report_id_and_show_the_documented_fields():
    reports = [report(3), report(1, competition_type="Essay", year=2022, level="Finalis")]

    text = render_catalog_answer(reports)

    first_row = next(line for line in text.splitlines() if line.startswith("- RPT"))
    assert first_row == (
        "- RPT0001 — Kompetisi Nomor 1 (Essay, 2022, Finalis)"
    )


def test_a_short_catalog_has_no_truncation_line():
    text = render_catalog_answer([report(1)])

    assert "Menampilkan" not in text


def test_an_empty_catalog_is_an_error_not_an_empty_answer():
    with pytest.raises(ValueError):
        render_catalog_answer([])
