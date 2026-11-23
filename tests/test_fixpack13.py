"""
Fix pack 13a: the post-login runtime build must name its cause.

The failure happens after a successful login, while the knowledge base and the
model are being loaded, and it used to reach the log as nothing at all. These
tests cover the diagnostics only: the stage lines, the environment line, the
single error line and the short code under the generic message. Every test uses a
sentinel secret to prove that no value, id or file name is ever logged or shown.
"""

import hashlib
import io
import json
import logging
import os
import sys
import types
import zipfile
from pathlib import Path

import pytest

import app_texts as T
import config
from core import startup
from core.kb_sync import DriveKBSource, KBSyncError, KBSynchronizer
from core.settings import KBSettings, SettingsError, load_settings
from core.sqlite_compat import SQLiteVersionError
from core.warmup import WarmupError
from ingestion.embedder import (
    E5Embedder,
    EmbeddingDownloadError,
    EmbeddingError,
)
from streamlit.testing.v1 import AppTest

from tests.test_app import login, start_app

SENTINEL = "SECRET-SENTINEL-42"


# -- doubles ---------------------------------------------------------------
def zip_bytes(members=None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in (members or {"chroma.sqlite3": b"sqlite"}).items():
            archive.writestr(name, data)
    return buffer.getvalue()


def manifest_for(checksum, **overrides):
    manifest = {
        "schema_version": config.MANIFEST_SCHEMA_VERSION,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "embedding_dimension": config.EMBEDDING_DIMENSION,
        "report_count": 2,
        "chunk_count": 3,
        "report_ids": ["RPT0001", "RPT0002"],
        "zip_filename": "chroma_backup.zip",
        "zip_checksum": checksum,
    }
    manifest.update(overrides)
    return manifest


class FakeSource:
    def __init__(self, manifest, artifact):
        self.manifest = manifest
        self.artifact = artifact

    def fetch_manifest(self):
        return dict(self.manifest)

    def download_artifact(self, manifest, destination):
        Path(destination).write_bytes(self.artifact)


class FakeLoader:
    def __call__(self, persist_dir, manifest_path=None):
        return types.SimpleNamespace(
            count_chunks=lambda: 3, report_ids=lambda: ["RPT0001", "RPT0002"]
        )


class FakeResponse:
    def __init__(self, status):
        self.status = status


class FakeHttpError(Exception):
    """Shaped like ``googleapiclient.errors.HttpError``."""

    __module__ = "googleapiclient.errors"

    def __init__(self, status, reason=None):
        super().__init__(f"raw provider detail {SENTINEL}")
        self.resp = FakeResponse(status)
        payload = {"error": {"errors": [{"reason": reason}] if reason else []}}
        self.content = json.dumps(payload).encode("utf-8")


class FakeRequest:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    def execute(self):
        if self.error is not None:
            raise self.error
        return self.payload


class FakeFiles:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    def list(self, **kwargs):
        return FakeRequest(self.payload, self.error)

    def get_media(self, fileId):  # noqa: N803 - mirrors the client signature
        return FakeRequest({"id": fileId})


def drive_settings(tmp_path):
    return KBSettings(
        source=config.KB_SOURCE_DRIVE,
        local_dir=tmp_path / "chroma_db",
        drive_folder_id=f"folder-{SENTINEL}",
        work_dir=tmp_path / "kb_work",
        manifest_ttl_seconds=300.0,
    )


def drive_source(files):
    source = DriveKBSource(
        folder_id=f"folder-{SENTINEL}",
        service_account_info={"client_email": f"{SENTINEL}@example.test"},
    )
    source._service = types.SimpleNamespace(files=lambda: files)
    return source


# -- stage lines -----------------------------------------------------------
def test_a_stage_logs_start_and_done_with_seconds_and_rss(caplog):
    caplog.set_level(logging.INFO, logger="core_startup")

    with startup.stage(startup.STAGE_SETTINGS):
        pass

    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == "stage_start name=settings"
    assert messages[1].startswith("stage_done name=settings seconds=")
    assert "rss_mb=" in messages[1]


def test_only_whitelisted_stage_names_are_accepted():
    with pytest.raises(ValueError):
        with startup.stage("Free-form name"):
            pass


def test_the_innermost_stage_names_the_failure():
    inner = None
    try:
        with startup.stage(startup.STAGE_KB_LOAD):
            with startup.stage(startup.STAGE_CHECKSUM):
                raise RuntimeError("boom")
    except RuntimeError as exc:
        inner = startup.describe_failure(exc)

    assert inner.stage == startup.STAGE_CHECKSUM
    assert inner.code == "KB/checksum_mismatch"


def test_the_kb_install_logs_its_stages_in_install_order(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="core_startup")
    artifact = zip_bytes()
    manifest = manifest_for(hashlib.sha256(artifact).hexdigest())
    synchronizer = KBSynchronizer(
        drive_settings(tmp_path),
        source=FakeSource(manifest, artifact),
        loader=FakeLoader(),
    )

    synchronizer.current()

    names = [
        record.getMessage().split()[1].split("=")[1]
        for record in caplog.records
        if record.getMessage().startswith("stage_start")
    ]
    assert names == [
        "manifest_validate",
        "zip_download",
        "checksum",
        "extract",
        "manifest_validate",
    ]


# -- the single error line -------------------------------------------------
def test_the_error_line_never_carries_the_message(caplog):
    caplog.set_level(logging.INFO)
    failure = startup.describe_failure(RuntimeError(f"raw {SENTINEL} detail"))
    startup.log_failure(failure)

    line = startup.format_failure(failure)
    assert line.startswith("error_stage=")
    for key in ("error_class=RuntimeError", "http_status=-", "reason=-"):
        assert key in line
    assert failure.code == "UNK/RuntimeError"
    assert SENTINEL not in line
    assert SENTINEL not in caplog.text


def test_a_settings_error_names_the_key_but_never_the_value():
    with pytest.raises(SettingsError) as excinfo:
        load_settings({}, environ={})

    failure = startup.describe_failure(excinfo.value)
    assert failure.code == "CFG/missing_key:auth.username"
    assert SENTINEL not in startup.format_failure(failure)


def test_an_invalid_kb_source_names_the_key_only():
    secrets = {
        "auth": {"username": "u", "password_hash": "h"},
        "groq": {"api_key": "k"},
        "kb": {"source": SENTINEL},
    }

    with pytest.raises(SettingsError) as excinfo:
        load_settings(secrets, environ={})

    failure = startup.describe_failure(excinfo.value)
    assert failure.code == "CFG/missing_key:kb.source"
    assert SENTINEL not in startup.format_failure(failure)


# -- Drive mapping ---------------------------------------------------------
@pytest.mark.parametrize(
    "status,reason,expected",
    [
        (403, "accessNotConfigured", "KB/drive_api_disabled"),
        (403, "SERVICE_DISABLED", "KB/drive_api_disabled"),
        (403, "insufficientPermissions", "KB/drive_forbidden"),
        (403, None, "KB/drive_forbidden"),
        (401, None, "KB/drive_forbidden"),
        (404, None, "KB/drive_not_found"),
        (500, None, None),
    ],
)
def test_google_status_and_reason_map_to_one_code(status, reason, expected):
    assert startup.google_drive_code(status, reason) == expected


def test_a_drive_404_reaches_the_code_and_the_error_line():
    source = drive_source(FakeFiles(error=FakeHttpError(404)))

    with pytest.raises(KBSyncError) as excinfo:
        source.fetch_manifest()

    failure = startup.describe_failure(excinfo.value)
    assert failure.code == "KB/drive_not_found"
    assert failure.http_status == 404
    assert SENTINEL not in startup.format_failure(failure)


def test_a_disabled_drive_api_maps_to_its_own_code():
    source = drive_source(
        FakeFiles(error=FakeHttpError(403, "accessNotConfigured"))
    )

    with pytest.raises(KBSyncError) as excinfo:
        source.fetch_manifest()

    failure = startup.describe_failure(excinfo.value)
    assert failure.code == "KB/drive_api_disabled"
    assert failure.http_status == 403
    assert failure.reason == "accessNotConfigured"
    assert SENTINEL not in startup.format_failure(failure)


def test_a_missing_manifest_maps_to_missing_artifact():
    source = drive_source(FakeFiles(payload={"files": []}))

    with pytest.raises(KBSyncError) as excinfo:
        source.fetch_manifest()

    assert startup.describe_failure(excinfo.value).code == "KB/missing_artifact"


def test_a_duplicate_manifest_maps_to_duplicate_artifact():
    payload = {"files": [{"id": "1"}, {"id": "2"}]}
    source = drive_source(FakeFiles(payload=payload))

    with pytest.raises(KBSyncError) as excinfo:
        source.fetch_manifest()

    assert startup.describe_failure(excinfo.value).code == "KB/duplicate_artifact"


# -- artifact integrity ----------------------------------------------------
def _install_failure(tmp_path, artifact, manifest):
    synchronizer = KBSynchronizer(
        drive_settings(tmp_path),
        source=FakeSource(manifest, artifact),
        loader=FakeLoader(),
    )
    with pytest.raises(Exception) as excinfo:
        synchronizer.current()
    return startup.describe_failure(excinfo.value)


def test_a_checksum_mismatch_maps_to_its_code(tmp_path):
    failure = _install_failure(
        tmp_path, zip_bytes(), manifest_for("deadbeef" * 8)
    )

    assert failure.code == "KB/checksum_mismatch"


def test_a_bad_zip_maps_to_extract_failed(tmp_path):
    artifact = b"not a zip at all"
    failure = _install_failure(
        tmp_path, artifact, manifest_for(hashlib.sha256(artifact).hexdigest())
    )

    assert failure.code == "KB/extract_failed"


def test_a_manifest_model_mismatch_maps_to_model_mismatch(tmp_path):
    artifact = zip_bytes()
    failure = _install_failure(
        tmp_path,
        artifact,
        manifest_for(
            hashlib.sha256(artifact).hexdigest(),
            embedding_model="intfloat/multilingual-e5-base",
        ),
    )

    assert failure.code == "KB/model_mismatch"


def test_a_manifest_schema_mismatch_maps_to_manifest_mismatch(tmp_path):
    artifact = zip_bytes()
    failure = _install_failure(
        tmp_path,
        artifact,
        manifest_for(hashlib.sha256(artifact).hexdigest(), schema_version="9.9"),
    )

    assert failure.code == "KB/manifest_mismatch"


# -- model / database / warm-up -------------------------------------------
def test_a_missing_chroma_directory_maps_to_chroma_open(tmp_path):
    from core.knowledge_base import KnowledgeBaseNotFoundError, KnowledgeBase

    with pytest.raises(KnowledgeBaseNotFoundError) as excinfo:
        KnowledgeBase(persist_dir=tmp_path / "missing")

    assert startup.describe_failure(excinfo.value).code == "DB/chroma_open"


def test_a_warmup_failure_maps_to_warmup():
    failure = startup.describe_failure(WarmupError("empty vector"))

    assert failure.code == "RT/warmup"


def test_a_sqlite_failure_maps_to_sqlite_version():
    failure = startup.describe_failure(SQLiteVersionError("too old"))

    assert failure.code == "DB/sqlite_version"


def _stub_sentence_transformers(monkeypatch, error):
    class Boom:
        def __init__(self, *args, **kwargs):
            raise error

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = Boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_a_download_failure_maps_to_its_code_and_sets_the_timeout(monkeypatch):
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    network_error = type(
        "LocalEntryNotFoundError",
        (Exception,),
        {"__module__": "huggingface_hub.errors"},
    )
    _stub_sentence_transformers(monkeypatch, network_error(SENTINEL))

    with pytest.raises(EmbeddingDownloadError) as excinfo:
        E5Embedder().embed_query("halo")

    failure = startup.describe_failure(excinfo.value)
    assert failure.code == "EMB/download_failed"
    assert failure.stage == "embedder_load"
    assert SENTINEL not in startup.format_failure(failure)
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == str(
        config.EMBEDDER_DOWNLOAD_TIMEOUT_SECONDS
    )


def test_a_model_load_failure_maps_to_load_failed(monkeypatch):
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    _stub_sentence_transformers(monkeypatch, ValueError(SENTINEL))

    with pytest.raises(EmbeddingError) as excinfo:
        E5Embedder().embed_query("halo")

    failure = startup.describe_failure(excinfo.value)
    assert failure.code == "EMB/load_failed"
    assert SENTINEL not in startup.format_failure(failure)


# -- environment line ------------------------------------------------------
def test_the_environment_line_is_written_once(monkeypatch, caplog):
    monkeypatch.setattr(startup, "_ENVIRONMENT_LOGGED", False)
    monkeypatch.setenv("GROQ_API_KEY", SENTINEL)
    caplog.set_level(logging.INFO, logger="core_startup")

    startup.log_environment_once()
    startup.log_environment_once()

    lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("startup_env")
    ]
    assert len(lines) == 1
    for key in (
        "python=",
        "platform=",
        "torch=",
        "cuda=",
        "sqlite=",
        "sqlite_shim=",
        "rss_mb=",
        "vm_total_mb=",
        "vm_available_mb=",
        "cgroup_limit_mb=",
        "cpu_count=",
        "temp_free_mb=",
    ):
        assert key in lines[0]
    assert SENTINEL not in lines[0]


def test_cgroup_memory_is_unavailable_instead_of_failing():
    limit, usage = startup._cgroup_memory_mb()

    assert isinstance(limit, str)
    assert isinstance(usage, str)


# -- the code under the message -------------------------------------------
def _raising_factory(exc):
    def factory():
        raise exc

    return factory


def test_a_post_login_failure_shows_a_code_under_the_message(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    at, _ = start_app(
        tmp_path,
        factory=_raising_factory(KBSyncError(f"raw {SENTINEL}")),
        monkeypatch=monkeypatch,
    )

    login(at)

    errors = [error.value for error in at.error]
    assert T.TECHNICAL_ERROR in errors[0]
    assert "Kode: UNK/KBSyncError" in errors[0]
    assert SENTINEL not in " ".join(errors)
    assert not at.exception

    error_lines = [
        record.getMessage()
        for record in caplog.records
        if "error_stage=" in record.getMessage()
    ]
    assert len(error_lines) == 1
    assert SENTINEL not in error_lines[0]


def test_a_classified_post_login_failure_shows_its_own_code(tmp_path, monkeypatch):
    at, _ = start_app(
        tmp_path,
        factory=_raising_factory(SQLiteVersionError("too old")),
        monkeypatch=monkeypatch,
    )

    login(at)

    assert any("Kode: DB/sqlite_version" in error.value for error in at.error)
    assert not at.exception


def test_the_app_starts_cleanly_with_no_runtime_override(tmp_path, monkeypatch):
    """The login screen must not raise now that stage logging is wired in."""
    at, _ = start_app(tmp_path, monkeypatch=monkeypatch)

    assert at.title[0].value == T.LOGIN_TITLE
    assert not at.exception
