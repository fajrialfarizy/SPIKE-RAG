"""
Import hygiene for the app runtime (fix pack P1).

The app is deployed with requirements.txt only. This test runs a FRESH interpreter
(no pytest, no conftest imports), imports the app and every core module, and fails
if an ingestion-only or test-only package is already loaded.

torch is expected to appear only when the embedder actually loads its model, which
is lazy, so this test does not require it. No network call is made.
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BANNED_MODULES = ("pymupdf", "fitz", "gspread", "reportlab", "pytest", "onnxruntime")

PROBE = """
import json
import sys

import app  # noqa: F401  (entry point: pulls in every core module it uses)
import core.guardrails  # noqa: F401
import core.kb_sync  # noqa: F401
import core.knowledge_base  # noqa: F401
import core.pipeline  # noqa: F401

banned = {banned!r}
print(json.dumps({{
    "loaded": sorted(name for name in banned if name in sys.modules),
    "torch": "torch" in sys.modules,
}}))
"""


def test_the_app_runtime_does_not_import_ingestion_or_test_packages():
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(banned=BANNED_MODULES)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["loaded"] == []
