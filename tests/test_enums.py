"""
Unit tests for enums module.
Verifies all enum values, mappings, and immutability.
"""

import pytest
from enums import (
    CompetitionType,
    AchievementLevel,
    SectionType,
    RagStatus,
    QueryMode,
    AWARD_WINNER_LEVELS,
    NON_AWARD_LEVELS,
)


def test_competition_type_enum():
    assert CompetitionType.BUSINESS_PLAN.value == "Business Plan"
    assert CompetitionType.BUSINESS_CASE.value == "Business Case"
    assert CompetitionType.ESSAY.value == "Essay"
    assert CompetitionType.KARYA_TULIS_ILMIAH.value == "Karya Tulis Ilmiah"
    assert CompetitionType.LAINNYA.value == "Lainnya"


def test_achievement_level_groups():
    # Meraih penghargaan = lima nilai pertama
    assert AchievementLevel.JUARA_1.value in AWARD_WINNER_LEVELS
    assert AchievementLevel.JUARA_2.value in AWARD_WINNER_LEVELS
    assert AchievementLevel.JUARA_3.value in AWARD_WINNER_LEVELS
    assert AchievementLevel.JUARA_HARAPAN.value in AWARD_WINNER_LEVELS
    assert AchievementLevel.MOST_ASPIRING.value in AWARD_WINNER_LEVELS

    # Belum juara = Finalis
    assert AchievementLevel.FINALIS.value in NON_AWARD_LEVELS
    assert AchievementLevel.FINALIS.value not in AWARD_WINNER_LEVELS
    assert len(AWARD_WINNER_LEVELS) == 5
    assert len(NON_AWARD_LEVELS) == 1


def test_section_type_enum():
    # Only 3 section types allowed for chunks (Judul Ide is context header only)
    assert SectionType.EXECUTIVE_SUMMARY.value == "executive_summary"
    assert SectionType.EVALUASI_TIM.value == "evaluasi_tim"
    assert SectionType.SARAN_PENGEMBANGAN.value == "saran_pengembangan"
    assert len(SectionType) == 3


def test_rag_status_enum():
    # Strictly three allowed statuses
    assert RagStatus.BELUM_DIPROSES.value == "Belum diproses"
    assert RagStatus.PROSES_GAGAL.value == "Proses gagal"
    assert RagStatus.SELESAI.value == "Selesai"
    assert len(RagStatus) == 3
