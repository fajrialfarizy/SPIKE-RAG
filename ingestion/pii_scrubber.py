"""Backwards-compatible import path for the PII scrubber.

The pure-regex implementation lives in ``core.pii`` so the app runtime never
imports the ingestion package for it (fix pack P1).
"""

from core.pii import PIIScrubber

__all__ = ["PIIScrubber"]
