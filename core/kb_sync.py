"""
Knowledge-base synchronisation (Phase 3, Brief 5.4).

Two sources:

- ``local``: an existing ``chroma_db`` plus its ``manifest.json``; validated once
  and served as-is.
- ``drive``: a private Drive folder holding ``manifest.json`` and the artifact zip.
  The manifest is re-read at most every ``KB_MANIFEST_TTL_SECONDS``; when its
  checksum changes the artifact is downloaded to a temp dir, its SHA-256 is
  verified, it is extracted into a NEW versioned directory (zip-slip and size
  guarded), validated with the same loader the pipeline uses, and only then does
  the process atomically switch to it.

Any failure keeps the previous version serving and logs the error type only. With
no previous version the failure is raised, so the app can show its technical-error
screen instead of serving an unvalidated collection.

Nothing here imports streamlit: the app passes its settings in.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from googleapiclient.http import MediaIoBaseDownload

import config
from core.knowledge_base import KnowledgeBase, KnowledgeBaseError
from core.settings import KBSettings
from ingestion.sync import ChromaSync, ManifestError

logger = logging.getLogger("core_kb_sync")


class KBSyncError(RuntimeError):
    """Raised when no validated knowledge base can be served."""


class KBIntegrityError(KBSyncError):
    """Raised when the downloaded artifact does not match the manifest."""


class KBZipSafetyError(KBSyncError):
    """Raised when an archive tries to escape the extraction directory."""


@dataclass(frozen=True)
class KBVersion:
    """One validated, immutable knowledge base version."""

    persist_dir: Path
    manifest: Dict[str, Any]
    checksum: str
    handle: Any = None

    @property
    def version(self) -> str:
        return str(self.manifest.get("timestamp") or self.checksum)

    @property
    def chunk_count(self) -> int:
        try:
            return int(self.manifest.get("chunk_count") or 0)
        except (TypeError, ValueError):
            return 0


def sha256_file(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Extract an archive, refusing path traversal and absurd sizes."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    try:
        archive = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile as exc:
        raise KBIntegrityError("The knowledge-base artifact is not a valid zip.") from exc

    with archive:
        members = archive.infolist()
        if len(members) > config.KB_ZIP_MAX_MEMBERS:
            raise KBZipSafetyError(
                f"The artifact has {len(members)} members, above the "
                f"{config.KB_ZIP_MAX_MEMBERS} limit."
            )

        total_bytes = sum(member.file_size for member in members)
        if total_bytes > config.KB_ZIP_MAX_UNCOMPRESSED_BYTES:
            raise KBZipSafetyError(
                f"The artifact expands to {total_bytes} bytes, above the "
                f"{config.KB_ZIP_MAX_UNCOMPRESSED_BYTES}-byte limit."
            )

        for member in members:
            name = member.filename
            if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
                raise KBZipSafetyError(f"Absolute path in the artifact: {name!r}")

            target = (destination / name).resolve()
            if target != root and root not in target.parents:
                raise KBZipSafetyError(
                    f"Path traversal in the artifact: {name!r}"
                )

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink)


def _load_manifest_file(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise KBSyncError(f"Manifest '{path}' does not exist.") from exc
    except (OSError, ValueError) as exc:
        raise KBSyncError(f"Manifest '{path}' is not readable JSON.") from exc
    if not isinstance(payload, dict):
        raise KBSyncError(f"Manifest '{path}' is not a JSON object.")
    return payload


class LocalKBSource:
    """Serve an existing chroma_db directory and its manifest."""

    def __init__(
        self,
        local_dir: Path,
        manifest_name: str = config.MANIFEST_FILE_NAME,
        manifest_path: Optional[Path] = None,
    ):
        self.local_dir = Path(local_dir)
        # run_ingestion.py writes the manifest next to its artifact bundle, not
        # next to chroma_db, so the path is configurable.
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self.local_dir.parent / manifest_name
        )

    def fetch_manifest(self) -> Dict[str, Any]:
        if not self.local_dir.exists():
            raise KBSyncError(
                f"Local knowledge base directory '{self.local_dir}' does not exist."
            )
        return _load_manifest_file(self.manifest_path)

    def download_artifact(self, manifest: Mapping[str, Any], destination: Path) -> None:
        raise KBSyncError("The local source has no artifact to download.")


class DriveKBSource:
    """Read the manifest and artifact from a private Drive folder (read-only)."""

    def __init__(
        self,
        folder_id: str,
        service_account_info: Mapping[str, Any],
        manifest_name: str = config.KB_DRIVE_MANIFEST_NAME,
    ):
        if not folder_id:
            raise KBSyncError("A Drive folder id is required for the drive source.")
        self.folder_id = folder_id
        self.manifest_name = manifest_name
        self._service_account_info = dict(service_account_info)
        self._service: Optional[Any] = None

    def _get_service(self) -> Any:
        if self._service is not None:
            return self._service
        if not self._service_account_info:
            raise KBSyncError(
                "The [gcp_service_account] section is required for the drive source."
            )
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise KBSyncError(
                "google-api-python-client is not installed, so Drive cannot be read."
            ) from exc

        credentials = service_account.Credentials.from_service_account_info(
            self._service_account_info, scopes=config.KB_DRIVE_SCOPES
        )
        self._service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        return self._service

    def _find_file_id(self, name: str) -> str:
        query = (
            f"name = '{name}' and '{self.folder_id}' in parents and trashed = false"
        )
        response = (
            self._get_service()
            .files()
            .list(
                q=query,
                spaces="drive",
                fields="files(id, name, modifiedTime, size)",
                pageSize=10,
            )
            .execute()
        )
        files = response.get("files") or []
        if not files:
            raise KBSyncError(f"'{name}' was not found in the knowledge-base folder.")
        if len(files) > 1:
            raise KBSyncError(
                f"'{name}' appears {len(files)} times in the knowledge-base folder."
            )
        return files[0]["id"]

    def _download_to_file(self, file_id: str, destination: Path) -> None:
        request = self._get_service().files().get_media(fileId=file_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "wb") as handle:
            downloader = MediaIoBaseDownload(handle, request, chunksize=1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    def fetch_manifest(self) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="kb_manifest_") as tmp:
            path = Path(tmp) / self.manifest_name
            self._download_to_file(self._find_file_id(self.manifest_name), path)
            return _load_manifest_file(path)

    def download_artifact(self, manifest: Mapping[str, Any], destination: Path) -> None:
        name = str(manifest.get("zip_filename") or config.KB_DRIVE_ARTIFACT_NAME)
        self._download_to_file(self._find_file_id(name), destination)


class KBSynchronizer:
    """Owns the active knowledge base version and its refresh policy."""

    def __init__(
        self,
        settings: KBSettings,
        source: Optional[Any] = None,
        *,
        loader: Callable[..., Any] = KnowledgeBase,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.settings = settings
        self.clock = clock
        self._loader = loader
        self.source = source or self._build_source(settings)
        self._lock = threading.RLock()
        self._active: Optional[KBVersion] = None
        self._last_check: Optional[float] = None

    @staticmethod
    def _build_source(settings: KBSettings) -> Any:
        if settings.source == config.KB_SOURCE_DRIVE:
            raise KBSyncError(
                "The drive source needs its service-account credentials; build a "
                "DriveKBSource and pass it in."
            )
        return LocalKBSource(settings.local_dir, manifest_path=settings.manifest_path)

    # -- public API --------------------------------------------------------
    def current(self) -> KBVersion:
        """Return a validated version, refreshing it when the TTL has elapsed."""
        with self._lock:
            if self._active is None:
                self._active = self._load_initial()
                self._last_check = self.clock()
                return self._active

            if not self._is_due():
                return self._active

            self._last_check = self.clock()
            try:
                self._refresh()
            except Exception as exc:
                # Keep serving the previous version; only the error type is logged.
                logger.warning(
                    "kb_refresh_failed error_type=%s", type(exc).__name__
                )
            return self._active

    def force_refresh(self) -> KBVersion:
        """Refresh now (used by tests and an operator action)."""
        with self._lock:
            if self._active is None:
                self._active = self._load_initial()
            else:
                self._refresh()
            self._last_check = self.clock()
            return self._active

    # -- internals ---------------------------------------------------------
    def _is_due(self) -> bool:
        if self._last_check is None:
            return True
        return (self.clock() - self._last_check) >= self.settings.manifest_ttl_seconds

    def _load_initial(self) -> KBVersion:
        manifest = self.source.fetch_manifest()
        if self.settings.source == config.KB_SOURCE_LOCAL:
            return self._validate_local(manifest)
        return self._install(manifest, previous=None)

    def _validate_local(self, manifest: Mapping[str, Any]) -> KBVersion:
        manifest_path = self.source.manifest_path
        try:
            ChromaSync.validate_manifest(dict(manifest), config.EMBEDDING_MODEL_NAME)
        except ManifestError as exc:
            raise KBIntegrityError(str(exc)) from exc

        loaded = self._loader(
            persist_dir=self.settings.local_dir,
            manifest_path=manifest_path,
        )
        self._check_counts(loaded, manifest)
        checksum = str(manifest.get("zip_checksum") or "")
        logger.info(
            "kb_loaded source=local chunks=%d", int(manifest.get("chunk_count") or 0)
        )
        return KBVersion(
            persist_dir=Path(self.settings.local_dir),
            manifest=dict(manifest),
            checksum=checksum,
            handle=loaded,
        )

    def _refresh(self) -> None:
        if self.settings.source == config.KB_SOURCE_LOCAL:
            return

        manifest = self.source.fetch_manifest()
        checksum = str(manifest.get("zip_checksum") or "")
        if (
            self._active is not None
            and checksum
            and checksum == self._active.checksum
        ):
            logger.info("kb_manifest_unchanged")
            return

        self._active = self._install(manifest, previous=self._active)

    def _install(
        self, manifest: Mapping[str, Any], previous: Optional[KBVersion]
    ) -> KBVersion:
        manifest = dict(manifest)
        try:
            ChromaSync.validate_manifest(manifest, config.EMBEDDING_MODEL_NAME)
        except ManifestError as exc:
            raise KBIntegrityError(str(exc)) from exc

        checksum = str(manifest.get("zip_checksum") or "")
        if not checksum:
            raise KBIntegrityError("The manifest has no zip_checksum.")

        work_dir = Path(self.settings.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        versions_dir = work_dir / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)

        staging = Path(tempfile.mkdtemp(prefix="kb_staging_", dir=str(work_dir)))
        try:
            artifact = staging / "artifact.zip"
            self.source.download_artifact(manifest, artifact)

            actual = sha256_file(artifact)
            if actual != checksum:
                raise KBIntegrityError(
                    "The downloaded artifact does not match the manifest checksum."
                )

            extracted = staging / "extracted"
            safe_extract_zip(artifact, extracted)

            target = versions_dir / (
                f"{config.KB_VERSION_DIR_PREFIX}{uuid.uuid4().hex[:12]}"
            )
            shutil.move(str(extracted), str(target))

            manifest_copy = target / "downloaded_manifest.json"
            manifest_copy.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            loaded = self._loader(persist_dir=target, manifest_path=manifest_copy)
            self._check_counts(loaded, manifest)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        shutil.rmtree(staging, ignore_errors=True)

        # Atomic switch: the new directory is complete and validated before it
        # becomes the active one.
        version = KBVersion(
            persist_dir=target, manifest=manifest, checksum=checksum, handle=loaded
        )
        self._remove_old_versions(versions_dir, keep=target)
        logger.info(
            "kb_switched version=%s chunks=%d", version.version, version.chunk_count
        )
        return version

    @staticmethod
    def _check_counts(loaded: Any, manifest: Mapping[str, Any]) -> None:
        expected_chunks = manifest.get("chunk_count")
        if expected_chunks is not None:
            actual = int(loaded.count_chunks())
            if int(expected_chunks) != actual:
                raise KBIntegrityError(
                    f"Manifest records {expected_chunks} chunks but the collection "
                    f"holds {actual}."
                )

        expected_reports = manifest.get("report_count")
        if expected_reports is not None:
            actual_reports = len(loaded.report_ids())
            if abs(int(expected_reports) - actual_reports) > config.KB_REPORT_COUNT_TOLERANCE:
                raise KBIntegrityError(
                    f"Manifest records {expected_reports} reports but the collection "
                    f"holds {actual_reports}."
                )

    @staticmethod
    def _remove_old_versions(versions_dir: Path, keep: Path) -> None:
        for child in versions_dir.iterdir():
            if child == keep:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    os.remove(child)
                except OSError:
                    pass
