"""
Offline tests for knowledge-base synchronisation (Phase 3).

The Drive source is replaced by an in-memory double; the loader is a fake that
returns manifest-consistent counts. No network, no Chroma, no model weights.
"""

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from core.kb_sync import (
    KBIntegrityError,
    KBSyncError,
    KBSynchronizer,
    KBZipSafetyError,
    LocalKBSource,
    safe_extract_zip,
    sha256_file,
)
from core.knowledge_base import EmbeddingContractError
from core.settings import KBSettings

ARTIFACT_NAME = "chroma_backup.zip"


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_zip(members=None) -> bytes:
    buffer = io.BytesIO()
    payload = members or {"chroma.sqlite3": b"sqlite-bytes", "index/uuid.bin": b"index"}
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def manifest_for(
    checksum,
    *,
    model=config.EMBEDDING_MODEL_NAME,
    dimension=config.EMBEDDING_DIMENSION,
    chunks=3,
    reports=2,
    timestamp="2026-01-01T00:00:00+00:00",
):
    return {
        "schema_version": config.MANIFEST_SCHEMA_VERSION,
        "timestamp": timestamp,
        "embedding_model": model,
        "embedding_dimension": dimension,
        "report_count": reports,
        "chunk_count": chunks,
        "report_ids": [f"RPT{index:04d}" for index in range(1, reports + 1)],
        "zip_filename": ARTIFACT_NAME,
        "zip_checksum": checksum,
    }


class FakeSource:
    def __init__(self, manifest, artifact=None):
        self.manifest = manifest
        self.artifact = artifact if artifact is not None else make_zip()
        self.downloads = 0

    def fetch_manifest(self):
        return dict(self.manifest)

    def download_artifact(self, manifest, destination):
        self.downloads += 1
        Path(destination).write_bytes(self.artifact)


class FakeLoader:
    def __init__(self, chunks=3, reports=2, fail_with=None):
        self.chunks = chunks
        self.reports = reports
        self.fail_with = fail_with
        self.calls = []

    def __call__(self, persist_dir, manifest_path=None):
        self.calls.append(Path(persist_dir))
        if self.fail_with is not None:
            raise self.fail_with
        assert (Path(persist_dir) / "chroma.sqlite3").exists()
        return SimpleNamespace(
            count_chunks=lambda: self.chunks,
            report_ids=lambda: [f"RPT{index:04d}" for index in range(1, self.reports + 1)],
        )


def drive_settings(tmp_path):
    return KBSettings(
        source=config.KB_SOURCE_DRIVE,
        local_dir=tmp_path / "chroma_db",
        drive_folder_id="folder-123",
        work_dir=tmp_path / "kb_work",
        manifest_ttl_seconds=300.0,
    )


def build(tmp_path, manifest, loader=None, artifact=None, clock=None):
    source = FakeSource(manifest, artifact)
    synchronizer = KBSynchronizer(
        drive_settings(tmp_path),
        source=source,
        loader=loader or FakeLoader(),
        clock=clock or FakeClock(),
    )
    return synchronizer, source


# -- happy path -----------------------------------------------------------
def test_the_first_load_downloads_and_installs(tmp_path):
    artifact = make_zip()
    manifest = manifest_for(sha256_file_from_bytes(artifact))
    synchronizer, source = build(tmp_path, manifest, artifact=artifact)

    version = synchronizer.current()

    assert source.downloads == 1
    assert version.checksum == manifest["zip_checksum"]
    assert (version.persist_dir / "chroma.sqlite3").exists()
    assert version.chunk_count == 3
    assert (version.persist_dir / "downloaded_manifest.json").exists()


def test_an_unchanged_manifest_never_downloads_again(tmp_path):
    artifact = make_zip()
    manifest = manifest_for(sha256_file_from_bytes(artifact))
    clock = FakeClock()
    synchronizer, source = build(tmp_path, manifest, artifact=artifact, clock=clock)

    synchronizer.current()
    clock.advance(config.KB_MANIFEST_TTL_SECONDS + 1)
    synchronizer.current()

    assert source.downloads == 1


def test_a_changed_manifest_switches_and_removes_the_old_version(tmp_path):
    first = make_zip({"chroma.sqlite3": b"one"})
    clock = FakeClock()
    manifest = manifest_for(sha256_file_from_bytes(first), timestamp="v1")
    synchronizer, source = build(tmp_path, manifest, artifact=first, clock=clock)
    first_version = synchronizer.current()

    second = make_zip({"chroma.sqlite3": b"two"})
    source.artifact = second
    source.manifest = manifest_for(sha256_file_from_bytes(second), timestamp="v2")
    clock.advance(config.KB_MANIFEST_TTL_SECONDS + 1)

    second_version = synchronizer.current()

    assert source.downloads == 2
    assert second_version.persist_dir != first_version.persist_dir
    assert (second_version.persist_dir / "chroma.sqlite3").read_bytes() == b"two"
    assert not first_version.persist_dir.exists()


def test_the_ttl_delays_the_manifest_check(tmp_path):
    artifact = make_zip()
    manifest = manifest_for(sha256_file_from_bytes(artifact))
    clock = FakeClock()
    synchronizer, source = build(tmp_path, manifest, artifact=artifact, clock=clock)
    synchronizer.current()

    source.manifest = manifest_for("different-checksum", timestamp="v2")
    clock.advance(config.KB_MANIFEST_TTL_SECONDS - 1)
    synchronizer.current()

    assert source.downloads == 1


# -- failures keep the old version ---------------------------------------
def test_a_checksum_mismatch_keeps_the_old_version(tmp_path, caplog):
    first = make_zip({"chroma.sqlite3": b"one"})
    clock = FakeClock()
    manifest = manifest_for(sha256_file_from_bytes(first), timestamp="v1")
    synchronizer, source = build(tmp_path, manifest, artifact=first, clock=clock)
    old = synchronizer.current()

    source.artifact = make_zip({"chroma.sqlite3": b"tampered"})
    source.manifest = manifest_for("deadbeef", timestamp="v2")
    clock.advance(config.KB_MANIFEST_TTL_SECONDS + 1)

    result = synchronizer.current()

    assert result.persist_dir == old.persist_dir
    assert "KBIntegrityError" in caplog.text


def test_a_model_mismatch_keeps_the_old_version(tmp_path, caplog):
    first = make_zip()
    clock = FakeClock()
    manifest = manifest_for(sha256_file_from_bytes(first), timestamp="v1")
    synchronizer, source = build(tmp_path, manifest, artifact=first, clock=clock)
    old = synchronizer.current()

    source.manifest = manifest_for(
        "other-checksum", model="intfloat/multilingual-e5-base", timestamp="v2"
    )
    clock.advance(config.KB_MANIFEST_TTL_SECONDS + 1)

    assert synchronizer.current().persist_dir == old.persist_dir
    assert "KBIntegrityError" in caplog.text


def test_a_dimension_mismatch_from_the_loader_keeps_the_old_version(tmp_path):
    first = make_zip()
    clock = FakeClock()
    manifest = manifest_for(sha256_file_from_bytes(first), timestamp="v1")
    loader = FakeLoader()
    synchronizer, source = build(
        tmp_path, manifest, loader=loader, artifact=first, clock=clock
    )
    old = synchronizer.current()

    source.manifest = manifest_for("changed", timestamp="v2")
    source.artifact = make_zip({"chroma.sqlite3": b"new"})
    loader.fail_with = EmbeddingContractError("stored vectors have 768 dimensions")
    clock.advance(config.KB_MANIFEST_TTL_SECONDS + 1)

    assert synchronizer.current().persist_dir == old.persist_dir


def test_a_chunk_count_mismatch_keeps_the_old_version(tmp_path):
    first = make_zip()
    clock = FakeClock()
    manifest = manifest_for(sha256_file_from_bytes(first), timestamp="v1")
    loader = FakeLoader(chunks=2)
    synchronizer, source = build(
        tmp_path, manifest, loader=loader, artifact=first, clock=clock
    )

    # The manifest says 3 chunks, the loader sees 2: the first load must fail.
    with pytest.raises(KBIntegrityError):
        synchronizer.current()


def test_a_first_load_failure_raises_instead_of_serving_nothing(tmp_path):
    synchronizer, _ = build(tmp_path, manifest_for("nope"))

    with pytest.raises(KBIntegrityError):
        synchronizer.current()


def test_an_empty_manifest_checksum_is_refused(tmp_path):
    manifest = manifest_for("")
    manifest["zip_checksum"] = ""
    synchronizer, _ = build(tmp_path, manifest)

    with pytest.raises(KBIntegrityError):
        synchronizer.current()


# -- zip safety -----------------------------------------------------------
def test_zip_slip_is_rejected(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../evil.txt", b"owned")
    archive_path = tmp_path / "evil.zip"
    archive_path.write_bytes(buffer.getvalue())

    with pytest.raises(KBZipSafetyError):
        safe_extract_zip(archive_path, tmp_path / "out")

    assert not (tmp_path / "evil.txt").exists()


def test_an_absolute_path_is_rejected(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("/etc/passwd", b"x")
    archive_path = tmp_path / "absolute.zip"
    archive_path.write_bytes(buffer.getvalue())

    with pytest.raises(KBZipSafetyError):
        safe_extract_zip(archive_path, tmp_path / "out")


def test_an_absurd_member_count_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "KB_ZIP_MAX_MEMBERS", 2)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index in range(5):
            archive.writestr(f"f{index}.txt", b"x")
    archive_path = tmp_path / "many.zip"
    archive_path.write_bytes(buffer.getvalue())

    with pytest.raises(KBZipSafetyError):
        safe_extract_zip(archive_path, tmp_path / "out")


def test_a_valid_archive_extracts_inside_the_destination(tmp_path):
    archive_path = tmp_path / "ok.zip"
    archive_path.write_bytes(make_zip())

    safe_extract_zip(archive_path, tmp_path / "out")

    assert (tmp_path / "out" / "chroma.sqlite3").read_bytes() == b"sqlite-bytes"


# -- local source ---------------------------------------------------------
def test_the_local_source_validates_the_existing_directory(tmp_path):
    persist_dir = tmp_path / "chroma_db"
    persist_dir.mkdir()
    (persist_dir / "chroma.sqlite3").write_bytes(b"x")
    manifest = manifest_for("checksum")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loader = FakeLoader()
    settings = KBSettings(
        source=config.KB_SOURCE_LOCAL,
        local_dir=persist_dir,
        drive_folder_id=None,
        work_dir=tmp_path / "kb_work",
    )
    synchronizer = KBSynchronizer(settings, source=LocalKBSource(persist_dir), loader=loader)

    version = synchronizer.current()

    assert version.persist_dir == persist_dir
    assert loader.calls == [persist_dir]


def test_the_drive_source_requires_credentials_at_construction(tmp_path):
    settings = drive_settings(tmp_path)

    with pytest.raises(KBSyncError):
        KBSynchronizer(settings)


def sha256_file_from_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def test_the_local_source_reads_the_configured_manifest_path(tmp_path):
    chroma = tmp_path / "chroma_db"
    chroma.mkdir()
    manifest = tmp_path / "artifacts" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")

    settings = KBSettings(
        source=config.KB_SOURCE_LOCAL,
        local_dir=chroma,
        manifest_path=manifest,
        drive_folder_id=None,
    )
    source = KBSynchronizer(settings)._build_source(settings)

    assert source.manifest_path == manifest
    assert source.fetch_manifest() == {}
