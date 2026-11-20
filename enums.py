"""
Enum definitions for Lab Prestasi RAG System.
Strictly maps domain values and prevents invalid values from entering vector store or filters.
"""

from enum import Enum


class CompetitionType(str, Enum):
    BUSINESS_PLAN = "Business Plan"
    BUSINESS_CASE = "Business Case"
    ESSAY = "Essay"
    KARYA_TULIS_ILMIAH = "Karya Tulis Ilmiah"
    LAINNYA = "Lainnya"


class AchievementLevel(str, Enum):
    JUARA_1 = "Juara 1"
    JUARA_2 = "Juara 2"
    JUARA_3 = "Juara 3"
    JUARA_HARAPAN = "Juara Harapan"
    MOST_ASPIRING = "Most Aspiring dan penghargaan serupa"
    FINALIS = "Finalis"


# Kelompok achievement level berdasarkan Bagian 2 Brief
AWARD_WINNER_LEVELS = frozenset({
    AchievementLevel.JUARA_1.value,
    AchievementLevel.JUARA_2.value,
    AchievementLevel.JUARA_3.value,
    AchievementLevel.JUARA_HARAPAN.value,
    AchievementLevel.MOST_ASPIRING.value,
})

NON_AWARD_LEVELS = frozenset({
    AchievementLevel.FINALIS.value,
})


class SectionType(str, Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    EVALUASI_TIM = "evaluasi_tim"
    SARAN_PENGEMBANGAN = "saran_pengembangan"


class RagStatus(str, Enum):
    BELUM_DIPROSES = "Belum diproses"
    PROSES_GAGAL = "Proses gagal"
    SELESAI = "Selesai"


class QueryMode(str, Enum):
    SPECIFIC = "specific"
    CROSS_REPORT = "cross_report"
    CATALOG = "catalog"
