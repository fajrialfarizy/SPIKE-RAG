"""
Guardrails enforced in backend code, not only by the system prompt (Brief 5.7).

Input side:
- ``detect_privacy_request`` is a narrow, targeted detector for requests about
  personal or administrative data (NIM/NPM, phone/WA, email, address, or the
  identity of the reporter, leader, member, mentor or supervisor). It is
  deliberately conservative: questions that merely mention a role stay allowed.
- ``detect_instruction_override`` catches obvious attempts to extract the
  system prompt or to override the instructions, and refuses them before any
  model call. It needs both an action verb and a target phrase, so a legitimate
  question that merely contains the word "instruksi" or "prompt" passes.

Output side:
- every answer is scrubbed with the ingestion PII sieve (including the name
  scrubber and the funding-amount exemption),
- an answer that reproduces long spans of the system prompt is replaced with a
  refusal,
- incidents are counted; their text is never logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import config
from core import prompts
from ingestion.pii_scrubber import PIIScrubber


def detect_instruction_override(question: str) -> Optional[str]:
    """Return a category when a question clearly attacks the instructions."""
    text = question or ""

    if config.PATTERN_SECURITY_PROMPT_EXTRACT.search(text):
        return "prompt_extraction"
    if config.PATTERN_SECURITY_INSTRUCTION_OVERRIDE.search(text):
        return "instruction_override"
    if config.PATTERN_SECURITY_IGNORE_PREVIOUS.search(text):
        return "instruction_override"
    return None


def detect_privacy_request(question: str) -> Optional[str]:
    """Return a category when a question asks for personal/administrative data."""
    text = question or ""

    if config.PATTERN_REQUEST_PII_ID_NUMBER.search(text):
        return "id_number"
    if config.PATTERN_REQUEST_PII_ENGLISH.search(text):
        return "english_pii"
    if config.PATTERN_REQUEST_PII_PERSONAL_DATA.search(text):
        return "personal_data"
    if config.PATTERN_REQUEST_PII_IDENTITY.search(text):
        return "identity"
    if config.PATTERN_REQUEST_PII_CONTACT.search(
        text
    ) and config.PATTERN_REQUEST_PII_INTENT.search(text):
        return "contact"
    return None


@dataclass(frozen=True)
class GuardResult:
    text: str
    incidents: Dict[str, int] = field(default_factory=dict)
    blocked: bool = False


class OutputGuard:
    """Final answer filter: PII masking plus system-prompt leak detection."""

    def __init__(
        self,
        scrubber: Optional[PIIScrubber] = None,
        system_prompt: str = prompts.SYSTEM_PROMPT,
    ):
        self.scrubber = scrubber or PIIScrubber()
        self.system_prompt = system_prompt
        self._prompt_shingles = self._shingles(system_prompt)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    @classmethod
    def _shingles(cls, text: str) -> set:
        words = cls._normalize(text).split()
        size = config.LEAK_NGRAM_SIZE
        if len(words) < size:
            return set()
        return {
            tuple(words[index : index + size])
            for index in range(len(words) - size + 1)
        }

    def detect_prompt_leak(self, answer: str) -> bool:
        """True when a sizeable share of the answer repeats the system prompt."""
        shingles = self._shingles(answer)
        if len(shingles) < config.LEAK_MIN_MATCHED_SHINGLES:
            return False

        matched = shingles & self._prompt_shingles
        if len(matched) < config.LEAK_MIN_MATCHED_SHINGLES:
            return False

        return (len(matched) / len(shingles)) >= config.LEAK_OVERLAP_RATIO_THRESHOLD

    def apply(self, answer: str) -> GuardResult:
        incidents = {"pii_masked": 0, "prompt_leak": 0}

        if self.detect_prompt_leak(answer):
            incidents["prompt_leak"] = 1
            return GuardResult(
                text=prompts.SECURITY_REFUSAL_RESPONSE,
                incidents=incidents,
                blocked=True,
            )

        scrubbed, counts = self.scrubber.scrub(answer)
        incidents["pii_masked"] = sum(counts.values())
        return GuardResult(text=scrubbed, incidents=incidents, blocked=False)
