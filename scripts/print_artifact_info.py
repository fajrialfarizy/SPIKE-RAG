"""
Print the deploy artifact summary (fix pack 12, T3).

For one artifact folder (default ``data/artifacts``) this prints the zip name,
its size and SHA-256, and the manifest fields the app validates: schema version,
timestamp, embedding model and dimension, report count and chunk count. It also
says whether the zip matches the checksum the manifest recorded.

The script is read-only and offline: no Drive call, no app runtime, no model.
It prints ONLY the whitelisted manifest fields above, so a manifest that somehow
carries a credential can never leak through this output. It fails loudly when
the folder does not hold exactly one zip and one manifest, because that is the
same condition the app refuses to start on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Run as ``python scripts/print_artifact_info.py``: the project root must be
# importable for ``config``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

DEFAULT_ARTIFACT_DIR = config.DATA_DIR / "artifacts"
DEFAULT_ZIP_NAME = "chroma_backup.zip"
# Only these manifest fields are ever printed. Everything else stays unread.
MANIFEST_SUMMARY_FIELDS = (
    "schema_version",
    "timestamp",
    "embedding_model",
    "embedding_dimension",
    "report_count",
    "chunk_count",
    "zip_filename",
    "zip_checksum",
)
HASH_CHUNK_BYTES = 1024 * 1024


def sha256_of(path: Path) -> str:
    """SHA-256 of one file, read in bounded blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _duplicate_warnings(folder: Path) -> List[str]:
    warnings: List[str] = []
    zips = sorted(item.name for item in folder.glob("*.zip"))
    manifests = sorted(item.name for item in folder.glob("*.json"))
    if len(zips) > 1:
        warnings.append(
            "more than one zip in the folder (" + ", ".join(zips) + "); "
            "the app fails loudly on duplicates"
        )
    if len(manifests) > 1:
        warnings.append(
            "more than one manifest in the folder (" + ", ".join(manifests) + "); "
            "the app fails loudly on duplicates"
        )
    return warnings


def artifact_summary(
    folder: Path,
    *,
    zip_name: str = DEFAULT_ZIP_NAME,
    manifest_name: str = config.MANIFEST_FILE_NAME,
) -> Dict[str, Any]:
    """
    The deploy-relevant facts about one artifact folder.

    ``problems`` is empty only when the folder can be uploaded as-is: the zip and
    the manifest exist, the manifest parses, and the checksum matches.
    """
    folder = Path(folder)
    problems: List[str] = []
    summary: Dict[str, Any] = {
        "folder": str(folder),
        "zip_name": None,
        "zip_bytes": None,
        "zip_sha256": None,
        "manifest": {},
        "checksum_matches": None,
        "problems": problems,
    }

    if not folder.is_dir():
        problems.append(f"artifact folder not found: {folder}")
        return summary

    problems.extend(_duplicate_warnings(folder))

    zip_path = folder / zip_name
    if not zip_path.is_file():
        problems.append(f"{zip_name} is missing from the folder")
    else:
        summary["zip_name"] = zip_path.name
        summary["zip_bytes"] = zip_path.stat().st_size
        summary["zip_sha256"] = sha256_of(zip_path)

    manifest_path = folder / manifest_name
    if not manifest_path.is_file():
        problems.append(f"{manifest_name} is missing from the folder")
        return summary

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        problems.append(f"{manifest_name} is not readable JSON")
        return summary
    if not isinstance(manifest, dict):
        problems.append(f"{manifest_name} is not a JSON object")
        return summary

    summary["manifest"] = {
        field: manifest.get(field) for field in MANIFEST_SUMMARY_FIELDS
    }

    recorded = manifest.get("zip_checksum")
    if isinstance(recorded, str) and summary["zip_sha256"]:
        summary["checksum_matches"] = recorded.lower() == summary["zip_sha256"].lower()
        if not summary["checksum_matches"]:
            problems.append(
                "the zip checksum does not match zip_checksum in the manifest; "
                "upload the zip first, then the matching manifest"
            )
    elif zip_path.is_file():
        problems.append("the manifest records no zip_checksum, so the zip is unverified")

    return summary


def _human_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value / (1024 * 1024):.1f} MiB"


def format_summary(summary: Dict[str, Any]) -> List[str]:
    """The printable lines, in order. Never contains a non-whitelisted field."""
    manifest = summary["manifest"]
    lines = [
        f"artifact folder: {summary['folder']}",
        (
            f"zip: {summary['zip_name'] or 'missing'} | "
            f"{_human_bytes(summary['zip_bytes'])} "
            f"({summary['zip_bytes'] if summary['zip_bytes'] is not None else 'unknown'} bytes)"
        ),
        f"sha256: {summary['zip_sha256'] or 'unknown'}",
        (
            "manifest: "
            f"schema_version={manifest.get('schema_version')} "
            f"timestamp={manifest.get('timestamp')} "
            f"embedding_model={manifest.get('embedding_model')} "
            f"embedding_dimension={manifest.get('embedding_dimension')} "
            f"reports={manifest.get('report_count')} "
            f"chunks={manifest.get('chunk_count')}"
        ),
        (
            "manifest zip entry: "
            f"{manifest.get('zip_filename') or 'unknown'} "
            f"checksum={manifest.get('zip_checksum') or 'unknown'}"
        ),
    ]
    if summary["checksum_matches"] is True:
        lines.append("checksum: the zip matches zip_checksum in the manifest")
    elif summary["checksum_matches"] is False:
        lines.append("checksum: MISMATCH between the zip and zip_checksum")
    else:
        lines.append("checksum: not verified")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the deploy artifact summary (offline, read-only)."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(DEFAULT_ARTIFACT_DIR),
        help="artifact folder (default: data/artifacts)",
    )
    parser.add_argument("--zip-name", default=DEFAULT_ZIP_NAME)
    parser.add_argument("--manifest-name", default=config.MANIFEST_FILE_NAME)
    args = parser.parse_args(argv)

    summary = artifact_summary(
        Path(args.folder),
        zip_name=args.zip_name,
        manifest_name=args.manifest_name,
    )
    for line in format_summary(summary):
        print(line)
    for problem in summary["problems"]:
        print(f"problem: {problem}")
    return 1 if summary["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
