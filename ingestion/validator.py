"""
Validator module for Lab Prestasi Ingestion.
Enforces metadata integrity, enum membership, section character limits, and business rules.
"""

from typing import Tuple, List, Dict
from enums import CompetitionType, AchievementLevel, RagStatus
import config


class IngestionValidator:
    """Validates metadata, enums, and checks character limits for warnings."""

    @staticmethod
    def validate_metadata(
        competition_type_raw: str,
        achievement_level_raw: str,
        competition_year_raw: any,
    ) -> Tuple[bool, str]:
        """
        Validate metadata against strict enum specifications.
        Returns: (is_valid: bool, error_reason: str)
        """
        # competition_type = Lainnya must be skipped with warning (Brief 5.2 #3)
        if competition_type_raw == CompetitionType.LAINNYA.value:
            return False, "competition_type = 'Lainnya' tidak diproses ke RAG (dilewati)"

        try:
            CompetitionType(competition_type_raw)
        except ValueError:
            return False, f"competition_type tidak valid: {competition_type_raw}"

        try:
            AchievementLevel(achievement_level_raw)
        except ValueError:
            return False, f"achievement_level tidak valid: {achievement_level_raw}"

        try:
            year = int(competition_year_raw)
            if year < 2000 or year > 2100:
                return False, f"competition_year di luar rentang wajar: {year}"
        except (ValueError, TypeError):
            return False, f"competition_year harus bilangan bulat valid: {competition_year_raw}"

        return True, "OK"

    @staticmethod
    def check_character_limits(sections: Dict[str, str]) -> List[str]:
        """
        Check character limits per section (PRD 6.2).
        Exceeding characters results ONLY in a warning log, document continues (Brief 5.2 #7).
        """
        warnings = []
        limits = config.SECTION_CHAR_LIMITS

        for sec_name, text in sections.items():
            if sec_name in limits:
                char_count = len(text)
                limit = limits[sec_name]
                if char_count > limit:
                    warnings.append(
                        f"Peringatan: Bagian '{sec_name}' memiliki {char_count} karakter (melebihi batas {limit} karakter)."
                    )
        return warnings
