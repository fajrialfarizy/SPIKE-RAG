"""
Offline tests for scripts/print_artifact_info.py (fix pack 12, T3).

The script is the deploy checklist's own proof: it reports the zip name, size
and SHA-256 plus the manifest summary the app validates, and it must never print
anything outside its whitelist (a credential in the manifest stays unread).
"""

import importlib.util
import json
from pathlib import Path

import pytest

ZIP_NAME = "chroma_backup.zip"
MANIFEST_NAME = "manifest.json"


def _load_script():
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "print_artifact_info.py"
    )
    spec = importlib.util.spec_from_file_location("print_artifact_info", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(checksum, **extra):
    payload = {
        "schema_version": "1.0",
        "timestamp": "2026-11-21T09:12:02.274142+00:00",
        "embedding_model": "intfloat/multilingual-e5-small",
        "embedding_dimension": 384,
        "report_count": 13,
        "chunk_count": 44,
        "report_ids": ["RPT0001"],
        "zip_filename": ZIP_NAME,
        "zip_checksum": checksum,
    }
    payload.update(extra)
    return payload


def _folder(tmp_path, *, checksum="match", extra=None):
    zip_path = tmp_path / ZIP_NAME
    zip_path.write_bytes(b"dummy artifact bytes")
    digest = _load_script().sha256_of(zip_path) if checksum == "match" else "0" * 64
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps(_manifest(digest, **(extra or {}))), encoding="utf-8"
    )
    return tmp_path


def test_a_matching_folder_is_reported_as_ready(tmp_path):
    script = _load_script()
    folder = _folder(tmp_path)

    summary = script.artifact_summary(folder)

    assert summary["problems"] == []
    assert summary["zip_name"] == ZIP_NAME
    assert summary["zip_bytes"] == len(b"dummy artifact bytes")
    assert summary["checksum_matches"] is True
    assert summary["manifest"]["report_count"] == 13
    assert summary["manifest"]["chunk_count"] == 44
    assert summary["manifest"]["embedding_dimension"] == 384


def test_the_printed_lines_carry_the_numbers_and_no_secret(tmp_path, capsys):
    script = _load_script()
    folder = _folder(tmp_path, extra={"api_key": "gsk_DO_NOT_PRINT_ME"})

    exit_code = script.main([str(folder)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert ZIP_NAME in output
    assert "sha256:" in output
    assert "schema_version=1.0" in output
    assert "embedding_model=intfloat/multilingual-e5-small" in output
    assert "embedding_dimension=384" in output
    assert "reports=13" in output
    assert "chunks=44" in output
    assert "checksum: the zip matches" in output
    assert "gsk_DO_NOT_PRINT_ME" not in output
    assert "api_key" not in output


def test_a_checksum_mismatch_is_a_problem(tmp_path, capsys):
    script = _load_script()
    folder = _folder(tmp_path, checksum="wrong")

    exit_code = script.main([str(folder)])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "checksum: MISMATCH" in output
    assert "upload the zip first" in output


def test_a_missing_zip_is_a_problem(tmp_path):
    script = _load_script()
    folder = _folder(tmp_path)
    (folder / ZIP_NAME).unlink()

    summary = script.artifact_summary(folder)

    assert any("is missing from the folder" in item for item in summary["problems"])


def test_a_duplicate_artifact_stops_the_run(tmp_path, capsys):
    script = _load_script()
    folder = _folder(tmp_path)
    (folder / "chroma_backup (1).zip").write_bytes(b"second copy")

    exit_code = script.main([str(folder)])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "more than one zip in the folder" in output


def test_a_broken_manifest_is_reported(tmp_path):
    script = _load_script()
    folder = _folder(tmp_path)
    (folder / MANIFEST_NAME).write_text("{not json", encoding="utf-8")

    summary = script.artifact_summary(folder)

    assert any("is not readable JSON" in item for item in summary["problems"])


@pytest.mark.parametrize("missing", ["zip", "manifest"])
def test_a_missing_file_is_reported(tmp_path, missing):
    script = _load_script()
    folder = _folder(tmp_path)
    (folder / (ZIP_NAME if missing == "zip" else MANIFEST_NAME)).unlink()

    summary = script.artifact_summary(folder)

    assert summary["problems"]


def test_a_missing_folder_is_reported(tmp_path):
    script = _load_script()

    summary = script.artifact_summary(tmp_path / "absent")

    assert summary["problems"] == [f"artifact folder not found: {tmp_path / 'absent'}"]
