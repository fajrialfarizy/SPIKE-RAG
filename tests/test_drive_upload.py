"""
Mocked Drive artifact handling (fix pack P2).

No Google credentials and no network: a small fake service records what the repo
calls, so the tests can prove upload order, in-place replacement, and loud failure
on zero or several files with the same name.
"""

from pathlib import Path

import pytest

from enums import RagStatus
from ingestion.repos.base import DriveRepo
from ingestion.repos.google import DriveArtifactError, GoogleDriveRepo
from ingestion.sync import ChromaSync

FOLDER = "folder123"
ZIP_NAME = "chroma_backup.zip"
MANIFEST_NAME = "manifest.json"


class HttpErrorStub(Exception):
    """Stands in for googleapiclient.errors.HttpError."""

    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


class FakeRequest:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def execute(self):
        if self._error is not None:
            raise self._error
        return self._result


class FakeFiles:
    def __init__(self, service):
        self.service = service

    def list(self, **kwargs):
        self.service.calls.append(("list", kwargs))
        return FakeRequest(
            {"files": [dict(item) for item in self.service.listing]},
            self.service.list_error,
        )

    def create(self, **kwargs):
        self.service.calls.append(("create", kwargs))
        body = kwargs.get("body") or {}
        item = {"id": f"created_{len(self.service.calls)}", "name": body.get("name")}
        self.service.listing.append(item)
        return FakeRequest({"id": item["id"]}, self.service.write_error)

    def update(self, **kwargs):
        self.service.calls.append(("update", kwargs))
        return FakeRequest({"id": kwargs.get("fileId")}, self.service.write_error)


class FakeService:
    def __init__(self, listing=None, list_error=None, write_error=None):
        self.listing = list(listing or [])
        self.list_error = list_error
        self.write_error = write_error
        self.calls = []

    def files(self):
        return FakeFiles(self)


def repo_with(service):
    repo = GoogleDriveRepo(FOLDER, {})
    repo._service = service
    return repo


def local_artifact(tmp_path, name=ZIP_NAME):
    path = tmp_path / name
    path.write_text("artifact bytes", encoding="utf-8")
    return path


def test_a_missing_artifact_is_created_once(tmp_path):
    service = FakeService()
    repo = repo_with(service)
    path = local_artifact(tmp_path)

    file_id = repo.upload_artifact(str(path), ZIP_NAME)

    assert file_id.startswith("created_")
    assert [call[0] for call in service.calls] == ["list", "create"]


def test_an_existing_artifact_is_replaced_in_place(tmp_path):
    service = FakeService(listing=[{"id": "existing_id", "name": ZIP_NAME}])
    repo = repo_with(service)
    path = local_artifact(tmp_path)

    file_id = repo.upload_artifact(str(path), ZIP_NAME)

    assert file_id == "existing_id"
    assert [call[0] for call in service.calls] == ["list", "update"]
    update = service.calls[1][1]
    assert update["fileId"] == "existing_id"
    assert len(service.listing) == 1


def test_two_uploads_leave_exactly_one_file(tmp_path):
    service = FakeService()
    repo = repo_with(service)
    path = local_artifact(tmp_path)

    repo.upload_artifact(str(path), ZIP_NAME)
    repo.upload_artifact(str(path), ZIP_NAME)

    assert [call[0] for call in service.calls] == ["list", "create", "list", "update"]
    assert len(service.listing) == 1


def test_duplicate_artifacts_fail_loudly(tmp_path):
    service = FakeService(
        listing=[
            {"id": "a", "name": ZIP_NAME},
            {"id": "b", "name": ZIP_NAME},
        ]
    )
    repo = repo_with(service)
    path = local_artifact(tmp_path)

    with pytest.raises(DriveArtifactError, match="appears 2 times"):
        repo.upload_artifact(str(path), ZIP_NAME)

    assert [call[0] for call in service.calls] == ["list"]


def test_a_missing_local_file_is_a_file_error(tmp_path):
    repo = repo_with(FakeService())

    with pytest.raises(FileNotFoundError):
        repo.upload_artifact(str(tmp_path / "nope.zip"), ZIP_NAME)


def test_a_missing_artifact_download_fails_loudly(tmp_path):
    repo = repo_with(FakeService())

    with pytest.raises(DriveArtifactError, match="was not found"):
        repo.download_artifact(ZIP_NAME, tmp_path / "out.zip")


def test_a_duplicated_artifact_download_fails_loudly(tmp_path):
    service = FakeService(
        listing=[{"id": "a", "name": ZIP_NAME}, {"id": "b", "name": ZIP_NAME}]
    )
    repo = repo_with(service)

    with pytest.raises(DriveArtifactError, match="appears 2 times"):
        repo.download_artifact(ZIP_NAME, tmp_path / "out.zip")


def test_a_403_from_drive_is_not_swallowed(tmp_path):
    service = FakeService(list_error=HttpErrorStub(403))
    repo = repo_with(service)

    with pytest.raises(HttpErrorStub):
        repo.upload_artifact(str(local_artifact(tmp_path)), ZIP_NAME)


def test_a_404_on_upload_is_not_swallowed(tmp_path):
    service = FakeService(write_error=HttpErrorStub(404))
    repo = repo_with(service)

    with pytest.raises(HttpErrorStub):
        repo.upload_artifact(str(local_artifact(tmp_path)), ZIP_NAME)


# -- upload order ----------------------------------------------------------
class RecordingDriveRepo(DriveRepo):
    def __init__(self):
        self.uploads = []
        self.fail_on = None

    def download_pdf(self, file_id, destination_path):
        return False

    def upload_artifact(self, local_path, remote_filename):
        if remote_filename == self.fail_on:
            raise RuntimeError("manifest upload failed")
        self.uploads.append(Path(remote_filename).name)
        return remote_filename

    def download_artifact(self, remote_filename, destination_path):
        return False


class StubChromaStore:
    def __init__(self, tmp_path):
        self.persist_dir = tmp_path / "chroma_db"
        self.persist_dir.mkdir()
        (self.persist_dir / "chroma.sqlite3").write_text("db", encoding="utf-8")

    def get_all_report_ids(self):
        return ["RPT0001"]

    def count_chunks(self):
        return 3


def test_the_zip_is_uploaded_before_the_manifest(tmp_path):
    drive = RecordingDriveRepo()
    sync = ChromaSync(StubChromaStore(tmp_path), drive)

    sync.sync_to_drive(tmp_path / ZIP_NAME, tmp_path / MANIFEST_NAME)

    assert drive.uploads == [ZIP_NAME, MANIFEST_NAME]


def test_a_failed_manifest_upload_leaves_the_old_manifest_in_place(tmp_path):
    drive = RecordingDriveRepo()
    drive.fail_on = MANIFEST_NAME
    sync = ChromaSync(StubChromaStore(tmp_path), drive)

    with pytest.raises(RuntimeError):
        sync.sync_to_drive(tmp_path / ZIP_NAME, tmp_path / MANIFEST_NAME)

    # The zip was replaced; the old manifest still points at the old checksum, so
    # the app refuses the mismatch instead of serving a half-updated artifact.
    assert drive.uploads == [ZIP_NAME]
