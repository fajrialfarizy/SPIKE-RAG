"""
Ingestion entry point used by the Colab notebook and by local runs.

No ingestion logic lives here: this module only wires configured dependencies and
prints the run summary. Credentials are read from the environment, never logged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from enums import RagStatus
from ingestion.chroma_store import ChromaStore
from ingestion.embedder import E5Embedder
from ingestion.pipeline import IngestionPipeline
from ingestion.repos.base import DriveRepo, SheetsRepo
from ingestion.repos.local import LocalDriveRepo, LocalSheetsRepo
from ingestion.sync import ChromaSync


def _local_repos(args) -> Tuple[SheetsRepo, DriveRepo]:
    sheets = LocalSheetsRepo(
        competition_reports_csv=Path(args.competition_reports_csv),
        report_documents_csv=Path(args.report_documents_csv),
    )
    drive = LocalDriveRepo(
        local_pdf_dir=Path(args.pdf_dir),
        artifacts_dir=Path(args.artifacts_dir),
    )
    return sheets, drive


def _google_repos() -> Tuple[SheetsRepo, DriveRepo]:
    from ingestion.repos.google import GoogleDriveRepo, GoogleSheetsRepo

    credentials_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    spreadsheet_id = os.environ.get("SHEET_ID")
    folder_id = os.environ.get("DRIVE_FOLDER_ID")

    missing = [
        name
        for name, value in (
            ("GCP_SERVICE_ACCOUNT_JSON", credentials_json),
            ("SHEET_ID", spreadsheet_id),
            ("DRIVE_FOLDER_ID", folder_id),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing required environment variables for --source google: "
            + ", ".join(missing)
        )

    credentials = json.loads(credentials_json)
    return (
        GoogleSheetsRepo(spreadsheet_id, credentials),
        GoogleDriveRepo(folder_id, credentials),
    )


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Lab Prestasi ingestion.")
    parser.add_argument("--source", choices=["local", "google"], default="local")
    parser.add_argument("--competition-reports-csv", default=str(config.FIXTURES_DIR / "competition_reports.csv"))
    parser.add_argument("--report-documents-csv", default=str(config.FIXTURES_DIR / "report_documents.csv"))
    parser.add_argument("--pdf-dir", default=str(config.DUMMY_PDF_DIR))
    parser.add_argument("--artifacts-dir", default=str(config.DATA_DIR / "artifacts"))
    parser.add_argument("--chroma-dir", default=str(config.CHROMA_PERSIST_DIR))
    parser.add_argument("--temp-dir", default=str(config.DATA_DIR / "temp_downloads"))
    parser.add_argument("--log-dir", default=str(config.LOG_DIR))
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip ChromaDB zip/manifest upload after the run.",
    )
    args = parser.parse_args(argv)

    sheets, drive = (
        _google_repos() if args.source == "google" else _local_repos(args)
    )

    store = ChromaStore(persist_dir=Path(args.chroma_dir))
    pipeline = IngestionPipeline(
        sheets_repo=sheets,
        drive_repo=drive,
        chroma_store=store,
        embedder=E5Embedder(),
        temp_download_dir=Path(args.temp_dir),
        log_dir=Path(args.log_dir),
    )

    summary = pipeline.run()

    print(
        f"processed={summary.processed_count} skipped={summary.skipped_count} "
        f"failed={summary.failed_count} chunks={sum(summary.chunks_per_report.values())} "
        f"max_tokens={summary.max_tokens_recorded}"
    )
    for entry in summary.failed_reports:
        print(f"FAILED {entry['report_id']}: {entry['reason']}")
    for entry in summary.skipped_reports:
        print(f"SKIPPED {entry['report_id']}: {entry['reason']}")
    print(f"log: {summary.log_path}")
    print(f"summary: {summary.summary_path}")

    if not args.no_sync:
        sync = ChromaSync(chroma_store=store, drive_repo=drive)
        work_dir = Path(args.log_dir)
        manifest = sync.sync_to_drive(
            work_dir / "chroma_backup.zip", work_dir / config.MANIFEST_FILE_NAME
        )
        print(
            f"manifest: reports={manifest['report_count']} "
            f"chunks={manifest['chunk_count']} model={manifest['embedding_model']}"
        )

    return 0 if summary.failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
