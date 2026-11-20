"""
PII Scrubber module for Lab Prestasi Ingestion.
Redacts personal data (NIM, phone, email) from text with [DIREDAKSI].
Logs counts and types of redactions for auditing.
"""

import re
from typing import Tuple, Dict
import config


class PIIScrubber:
    """Scrubs Indonesian PII (NIM, Phone, Email) from evaluation text."""

    def __init__(self):
        self.nim_pattern = config.PATTERN_PII_NIM
        self.phone_pattern = config.PATTERN_PII_PHONE
        self.email_pattern = config.PATTERN_PII_EMAIL
        self.replacement = config.PII_REDACTION_REPLACEMENT

    def scrub(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Redact NIM, phone, and email from text.
        Returns: (scrubbed_text, counts_dict)
        """
        counts = {"nim": 0, "phone": 0, "email": 0}

        # 1. Scrub Email
        emails = self.email_pattern.findall(text)
        if emails:
            counts["email"] = len(emails)
            text = self.email_pattern.sub(self.replacement, text)

        # 2. Scrub Phone numbers
        phones = self.phone_pattern.findall(text)
        if phones:
            counts["phone"] = len(phones)
            text = self.phone_pattern.sub(self.replacement, text)

        # 3. Scrub NIM
        nims = self.nim_pattern.findall(text)
        if nims:
            counts["nim"] = len(nims)
            text = self.nim_pattern.sub(self.replacement, text)

        return text, counts
