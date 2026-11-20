"""
ChromaDB sync and manifest module for Lab Prestasi Ingestion.
Creates a zip archive of the ChromaDB directory and writes manifest.json.
Supports checking manifest TTL and restoring/reloading vector database.
"""

import json
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import config
from ingestion.chroma_store import ChromaStore
from ingestion.repos.base import DriveRepo


class ManifestError(RuntimeError):
    """Raised when a manifest is unusable or was produced by a different model."""


class ChromaSync:
    """Handles zipping, manifest creation, and Drive synchronization."""

    def __init__(self, chroma_store: ChromaStore, drive_repo: DriveRepo):
        self.chroma_store = chroma_store
        self.drive_repo = drive_repo

    @staticmethod
    def calculate_file_hash(path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def create_zip_archive(self, output_zip_path: Path) -> str:
        """Zip the persistent chroma_db directory into a single file."""
        persist_dir = self.chroma_store.persist_dir
        output_zip_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in persist_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(persist_dir)
                    zipf.write(file_path, arcname)

        return self.calculate_file_hash(output_zip_path)

    def generate_manifest(self, zip_checksum: str, zip_filename: str) -> Dict[str, Any]:
        """Generate manifest.json per Brief 5.4 specifications."""
        reports = self.chroma_store.get_all_report_ids()
        total_chunks = self.chroma_store.count_chunks()

        manifest = {
            "schema_version": config.MANIFEST_SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "report_count": len(reports),
            "chunk_count": total_chunks,
            "report_ids": reports,
            "zip_filename": zip_filename,
            "zip_checksum": zip_checksum,
        }
        return manifest

    @staticmethod
    def validate_manifest(
        manifest: Dict[str, Any],
        expected_model: str = config.EMBEDDING_MODEL_NAME,
    ) -> None:
        """
        Refuse a manifest produced by another embedding model (Brief 5.4).
        The application calls this at startup before loading the collection.
        """
        if not isinstance(manifest, dict):
            raise ManifestError("Manifest is not a JSON object")

        model = manifest.get("embedding_model")
        if model != expected_model:
            raise ManifestError(
                f"Manifest was built with embedding model '{model}' but config "
                f"expects '{expected_model}'. Refusing to load the knowledge base."
            )

        schema_version = manifest.get("schema_version")
        if schema_version != config.MANIFEST_SCHEMA_VERSION:
            raise ManifestError(
                f"Manifest schema_version '{schema_version}' is not supported "
                f"(expected '{config.MANIFEST_SCHEMA_VERSION}')"
            )

    def sync_to_drive(self, zip_path: Path, manifest_path: Path) -> Dict[str, Any]:
        """
        Zip ChromaDB, write manifest.json, and upload both via DriveRepo.
        """
        zip_hash = self.create_zip_archive(zip_path)
        manifest_data = self.generate_manifest(zip_hash, zip_path.name)

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # Upload artifacts to Drive
        self.drive_repo.upload_artifact(str(zip_path), zip_path.name)
        self.drive_repo.upload_artifact(str(manifest_path), manifest_path.name)

        return manifest_data

    @staticmethod
    def restore_from_zip(zip_path: Path, destination_dir: Path):
        """Extract ChromaDB zip into destination directory."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(destination_dir)
