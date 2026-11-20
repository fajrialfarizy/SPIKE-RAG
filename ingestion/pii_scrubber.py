"""
PII scrubber for Lab Prestasi Ingestion (Brief 5.2 #8, PRD 6.3).

Redacts with [DIREDAKSI] and returns per-category counts for the run log.

Order and rules (supervisor decisions Q2/Q3):
1. email,
2. phone (checked first among the number patterns),
3. labelled NIM/NPM/NRP/"nomor induk" - always redacted,
4. unlabelled 8-14 digit numbers - redacted unless an amount context precedes
   them (Rp, dana, biaya, ...), so legitimate funding figures survive,
5. personal names via honourifics, role labels, academic suffixes and an
   optional deny-list.

Trade-off: unlabelled digit strings near an amount word are kept, and a name
without a title, role label, suffix or deny-list entry cannot be detected.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config


class PIIScrubber:
    """Scrubs Indonesian PII (NIM, phone, email, names) from evaluation text."""

    CATEGORIES = ("nim", "phone", "email", "name")

    def __init__(self, denylist_path: Optional[Path] = None):
        self.replacement = config.PII_REDACTION_REPLACEMENT
        path = Path(denylist_path) if denylist_path else config.PII_NAME_DENYLIST_PATH
        self.denylist = self._load_denylist(path)

    @staticmethod
    def _load_denylist(path: Path) -> List[str]:
        if not path.exists():
            return []
        names = []
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if entry and not entry.startswith("#"):
                names.append(entry)
        # Longest first so "Budi Santoso" wins over "Budi".
        return sorted(set(names), key=len, reverse=True)

    def _sub_count(self, pattern, text: str) -> Tuple[str, int]:
        matches = pattern.findall(text)
        if not matches:
            return text, 0
        return pattern.sub(self.replacement, text), len(matches)

    def _scrub_labeled_nim(self, text: str) -> Tuple[str, int]:
        """Redact the number but keep its label ("NIM: [DIREDAKSI]")."""
        matches = config.PATTERN_PII_NIM_LABELED.findall(text)
        if not matches:
            return text, 0

        def replace(match: "re.Match[str]") -> str:
            return match.group(0).replace(
                match.group("number"), self.replacement
            )

        return config.PATTERN_PII_NIM_LABELED.sub(replace, text), len(matches)

    def _scrub_unlabeled_nim(self, text: str) -> Tuple[str, int]:
        """Redact 8-14 digit numbers unless an amount context word precedes them."""
        pieces: List[str] = []
        cursor = 0
        count = 0

        for match in config.PATTERN_PII_NIM.finditer(text):
            window_start = max(0, match.start() - config.PII_CURRENCY_CONTEXT_WINDOW)
            context = text[window_start : match.start()].lower()
            if any(word in context for word in config.PII_CURRENCY_CONTEXT_WORDS):
                continue

            pieces.append(text[cursor : match.start()])
            pieces.append(self.replacement)
            cursor = match.end()
            count += 1

        pieces.append(text[cursor:])
        return "".join(pieces), count

    def _scrub_names(self, text: str) -> Tuple[str, int]:
        count = 0

        for pattern in (
            config.PATTERN_PII_NAME_WITH_TITLE,
            config.PATTERN_PII_NAME_AFTER_ROLE,
            config.PATTERN_PII_NAME_WITH_SUFFIX,
        ):
            occurrences = len(pattern.findall(text))
            if not occurrences:
                continue

            def replace(match: "re.Match[str]") -> str:
                groups = match.groupdict()
                if groups.get("name"):
                    return match.group(0).replace(groups["name"], self.replacement)
                return self.replacement

            text = pattern.sub(replace, text)
            count += occurrences

        for name in self.denylist:
            pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)")
            occurrences = len(pattern.findall(text))
            if occurrences:
                text = pattern.sub(self.replacement, text)
                count += occurrences

        return text, count

    def scrub(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Redact PII and return the scrubbed text plus per-category counts."""
        counts = {category: 0 for category in self.CATEGORIES}

        text, counts["email"] = self._sub_count(config.PATTERN_PII_EMAIL, text)
        text, counts["phone"] = self._sub_count(config.PATTERN_PII_PHONE, text)

        text, labeled = self._scrub_labeled_nim(text)
        text, unlabeled = self._scrub_unlabeled_nim(text)
        counts["nim"] = labeled + unlabeled

        text, counts["name"] = self._scrub_names(text)

        return text, counts
